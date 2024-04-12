# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from peft import LoraConfig, get_peft_model
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.generation.utils import GenerationConfig
from trl import SFTTrainer

# Define and parse arguments.
@dataclass
class ScriptArguments:
    # traindata parameters
    train_data: Optional[str] = field(
        default="/data2/huzheng/train_dpo/data/hh-rlhf", metadata={"help": "训练数据的位置"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="gpt2", metadata={"help": "the model name"})

    max_length: Optional[int] = field(
        default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(
        default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(
        default=-100, metadata={"help": "label for non response tokens"})

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


def get_dataset(train_data_path: str, silent: bool = False, cache_dir: str = None) -> Tuple[Dataset, Dataset]:

    datasetall = load_dataset(
        path="json",
        data_files={
            train_data_path
        },
        cache_dir=cache_dir,
    )

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        answers = sample["output"]
        instruction = sample["instruction"]
        return {
            "prompt": instruction + answers[0],
        }
    datasetall = datasetall.map(split_prompt_and_responses)
    train_test_split = datasetall["train"].train_test_split(test_size=0.8)
    dataset_train = train_test_split['test']
    dataset_test = train_test_split['train']

    return dataset_train, dataset_test


if __name__ == "__main__":
    parser = HfArgumentParser(dataclass_types=(ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=script_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype='auto',
        # device_map='auto'
    )
    model.generation_config = GenerationConfig.from_pretrained(
        pretrained_model_name=script_args.model_name_or_path)

    # 1.1 laod peft model
    LORA_R = 32
    # LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    # TARGET_MODULES = ["c_attn", "c_proj", "w1", "w2"]  # Qwen
    TARGET_MODULES = ["W_pack", "o_proj", "gate_proj", "down_proj"]
    config = LoraConfig(
        r=LORA_R,
        # lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model=model, peft_config=config)
    model.print_trainable_parameters()

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=script_args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eod_id

    # 3. Load train and Load evaluation dataset
    with training_args.main_process_first(desc="loading and tokenization"):
        train_dataset, eval_dataset = get_dataset(
            train_data_path=script_args.train_data)

    # 4. initialize the SFT trainer
    sft_trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=script_args.max_length,
        dataset_text_field = "prompt"
    )

    sft_trainer.train()
