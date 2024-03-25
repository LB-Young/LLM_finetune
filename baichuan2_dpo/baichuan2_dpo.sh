# deepspeed --num_nodes 1 --num_gpus 8 train_dpo.py \ --include localhost:2,3,5,6
deepspeed --include localhost:7 train_dpo_Young.py \
    --deepspeed default_offlload_zero2.json \
    --model_name_or_path xxxxxxx/Baichuan2-13B-Chat-v2/ \
    --train_data /mnt/data3/liu/LLaMA-Factory-20240317/LLaMA-Factory-main/data/comparison_gpt4_data_zh.json \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --max_length 1024 \
    --max_prompt_length 128 \
    --max_target_length 512 \
    --report_to tensorboard \
    --save_strategy steps \
    --save_steps 500 \
    --logging_steps 10 \
    --save_total_limit 2 \
    --output_dir ./test # --max_steps 2000 \
