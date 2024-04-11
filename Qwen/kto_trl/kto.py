from dataclasses import dataclass

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, setup_chat_format


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the KTO training script.
    """

    dataset_name: str = "/mnt/data3/xxxxxxxx/kto-mix-14k"


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    script_args, kto_args, model_args = parser.parse_args_into_dataclasses()

    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model_ref = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # If we are aligning a base model, we use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name)

    # Apply chat template
    def format_dataset(example):
        example["prompt"] = tokenizer.apply_chat_template(example["prompt"], tokenize=False)
        example["completion"] = tokenizer.apply_chat_template(example["completion"], tokenize=False)
        return example

    formatted_dataset = dataset.map(format_dataset)

    # Initialize the KTO trainer
    peft_config = get_peft_config(model_args)
    breakpoint()
    peft_config.target_modules = set(['c_attn', 'c_proj'])
    kto_trainer = KTOTrainer(
        model,
        model_ref,
        args=kto_args,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["test"],
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Train and push the model to the Hub
    kto_trainer.train()
    kto_trainer.save_model(kto_args.output_dir)
    kto_trainer.push_to_hub()