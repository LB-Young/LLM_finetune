import os
# os.environ['http_proxy'] = "http:://127.0.0.1:10808"
# os.environ['https_proxy'] = "http:://127.0.0.1:10808"

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

train_data_path = "F:\Cmodels\LLM_finetune\components\school_math_0.25M.json"
datasetall = load_dataset(
    "json",
    data_files={
        train_data_path
    }
)
print(datasetall)
print("-"*100)

def format_data(example):
    text = f"### Question: {example['instruction']}\n ### Answer: {example['output']}"
    return {
            "prompt": text,
        }

format_dataset = datasetall.map(format_data)
print(format_dataset['train'])

model = AutoModelForCausalLM.from_pretrained("F:\Cmodels\model_weights\opt-350m", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("F:\Cmodels\model_weights\opt-350m", trust_remote_code=True)

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
trainer = SFTTrainer(
    model,
    train_dataset=format_dataset['train'],
    formatting_func=format_data,
    data_collator=collator,
    dataset_text_field = "prompt"
)

trainer.train()