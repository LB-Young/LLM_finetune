# deepspeed --num_nodes 1 --num_gpus 8 train_dpo.py \ --include localhost:2,3,5,6
# deepspeed --include localhost:6 train_kto_Young.py \
#     --deepspeed default_offlload_zero2.json \
#     --model_name_or_path /mnt/data1/lyt/pretrain_model/Baichuan2-13B-Chat-v2/\
#     --train_data /mnt/data3/liu/LLaMA-Factory-20240317/LLaMA-Factory-main/data/comparison_gpt4_data_zh.json \
#     --learning_rate 2e-4 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 1 \
#     --report_to tensorboard \
#     --logging_steps 10 \
#     --max_steps 1000 \
#     --eval_steps 500 \
#     --optim rmsprop \
#     --warmup_steps 150 \
#     --bf16 \
#     --logging_first_step \
#     --no_remove_unused_columns \
#     --use_peft \
#     --lora_r 16\
#     --lora_alpha 16 \
#     --output_dir ./test # --max_steps 2000 \

export CUDA_VISIBLE_DEVICES=5
python kto.py \
    --model_name_or_path="/mnt/data3/models/Qwen-7B-Chat/" \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="./test" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to tensorboard \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=1 \
    --lora_alpha=16
