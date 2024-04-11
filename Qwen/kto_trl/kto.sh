export CUDA_VISIBLE_DEVICES=6
python kto.py \
    --model_name_or_path="/mnt/data3/models/Qwen-7B-Chat/" \
    --per_device_train_batch_size 2 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type=cosine \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir=kto-aligned-model-lora \
    --warmup_ratio 0.1 \
    --report_to tensorboard \
    --bf16 \
    --logging_first_step \
    --use_peft \
    --load_in_4bit \
    --lora_target_modules=all-linear \
    --lora_r=2 \
    --lora_alpha=16