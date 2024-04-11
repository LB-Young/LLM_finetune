# 0. imports
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from peft import LoraConfig, get_peft_model

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.generation.utils import GenerationConfig

from trl import KTOTrainer, KTOConfig, ModelConfig, get_peft_config


# Define and parse arguments.
@dataclass
class ScriptArguments:
    # training parameters
    model_name_or_path: Optional[str] = field(
        default="gpt2", metadata={"help": "the model name"})
    
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train in 1000 samples"})
    # traindata parameters
    train_data: Optional[str] = field(
        default="xx", metadata={"help": "训练数据的位置"})

def get_data(train_data_path: str, silent: bool = False, cache_dir: str = None) -> Tuple[Dataset, Dataset]:
    datasetall = load_dataset(
        "json",
        data_files={
            train_data_path
        },
        cache_dir=cache_dir,
    )

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        answers = sample["output"]
        instruction = sample["instruction"]
        return {
            "prompt": instruction,
            "completion": answers,
            "label": [1, 0],
        }
    datasetall = datasetall.map(split_prompt_and_responses)
    train_test_split = datasetall["train"].train_test_split(test_size=0.8)
    dataset_train = train_test_split['test']
    dataset_test = train_test_split['train']

    return dataset_train, dataset_test


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    script_args, kto_args, model_args = parser.parse_args_into_dataclasses()  # [0]

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype='auto',
        # device_map='auto'
    )
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype='auto',
        # device_map='auto'
    )
    model.generation_config = GenerationConfig.from_pretrained(
        script_args.model_name_or_path)

    # 1.1 laod peft model
    LORA_R = 32
    # LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["c_attn", "c_proj", "w1", "w2"]
    # TARGET_MODULES = ["W_pack", "o_proj", "gate_proj", "down_proj"]
    config = LoraConfig(
        r=LORA_R,
        # lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # model = get_peft_model(model, config)
    # model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eod_id

    # 2. Load train and Load evaluation dataset
    train_dataset, eval_dataset = get_data(
        train_data_path=script_args.train_data)

    # 5. initialize the DPO trainer
    kto_trainer = KTOTrainer(
        model,
        model_ref,
        args=kto_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config = get_peft_config(model_args)
    )

    # 6. train
    kto_trainer.train()
