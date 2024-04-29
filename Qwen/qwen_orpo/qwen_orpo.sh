export CUDA_VISIBLE_DEVICES=5
python qwen_orpo.py \
    --model_name_or_path "/mnt/data3/models/qwen/Qwen1___5-0___5B/" \
    --per_device_train_batch_size 2 \
    --max_steps 1000 \
    --learning_rate 8e-5 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir "qwen-lora-aligned-orpo" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to tensorboard \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=2 \
    --lora_alpha=16