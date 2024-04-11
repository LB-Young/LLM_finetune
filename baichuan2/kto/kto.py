# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Run the KTO training script with the following command with some example arguments.
In general, the optimal configuration for KTO will be similar to that of DPO:

# regular:
python examples/scripts/kto.py \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="kto_anthropic_hh" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

# peft:
python examples/scripts/kto.py \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="kto_anthropic_hh" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""

from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the KTO training script.
    """

    # debugging
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})

def get_dataset(sanity_check: bool = False, silent: bool = False, cache_dir: str = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'completion': List[str],
        'label': List[bool],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset(
        "json",
        data_files={
            "/mnt/data3/xxxxxxxxxxxxxxx/comparison_gpt4_data_zh.json", 
        },
        cache_dir=cache_dir
        )
    
    # flat_data = {
    #     "prompt": [],
    #     "completion": [],
    #     "label": [],
    # }
    # for sample in dataset['train']:
    #     flat_data['prompt'].append(sample["instruction"])
    #     flat_data['prompt'].append(sample["instruction"])
    #     flat_data['completion'].append(sample["output"][0])
    #     flat_data['completion'].append(sample["output"][1])
    #     flat_data['label'].append(1)
    #     flat_data['label'].append(0)
    
    # return dataset.from_json(flat_data)

    def split_prompt_and_responses(sample):
        answers = sample["output"]
        instruction = sample["instruction"]
        ans = len(instruction)%2
        bal = (len(instruction)+1)%2
        if bal == 1:
            bal = True
        else:
            bal = False
        return  {
            "prompt": instruction,
            "completion": answers[ans],
            "label": bal,
        }
    dataset = dataset.map(split_prompt_and_responses)
    train_test_split = dataset["train"].train_test_split(test_size=0.8)
    dataset_train = train_test_split['test']
    dataset_test = train_test_split['train']
    # breakpoint()
    return dataset_train, dataset_test


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    script_args, kto_args, model_args = parser.parse_args_into_dataclasses()

    # 1. Load the Anthropic Helpful-Harmless dataset
    train_dataset, eval_dataset = get_dataset(sanity_check=script_args.sanity_check)

    # 2. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model_ref = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eod_id

    # 3. 配置peft参数
    peft_config = get_peft_config(model_args)
    peft_config.target_modules = ['W_pack', 'o_proj', 'gate_proj']

    # TARGET_MODULES = ["c_attn", "c_proj", "w1", "w2"]
    # peft_config.target_modules = TARGET_MODULES
    # 4. initialize the KTO trainer
    # breakpoint()
    # kto_args.accelerator_config = AcceleratorConfig(split_batches=False, dispatch_batches=None, even_batches=True)
    breakpoint()
    kto_trainer = KTOTrainer(
        model,
        model_ref,
        args=kto_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # 5. train
    kto_trainer.train()
