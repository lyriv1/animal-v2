# deepspeed ./main.py \
#     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
#     --deepspeed ./deepspeed/deepspeed_zero3.yaml \
#     --model_name_or_path lmsys/vicuna-13b-v1.5 \
#     --version v1 \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.6-7b-pretrain/mm_projector.bin \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-v1.5-7b-lora \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb



#!/bin/bash

# deepspeed llava/train/train_mem.py \
#     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path ./model \
#     --version v1 \
#     --data_path ./playground/data/llava_v1_5_mix665k.json \
#     --image_folder ./playground/data \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-v1.5-13b-task-lora \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb



#  deepspeed main.py \
#     --deepspeed ./scripts/zero3.json \
#     --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --output_dir sft-llava-1.6-7b-hf-customer2batch \
#     --torch_dtype bfloat16 \
#     --gradient_checkpointing \
#     --num_train_epochs 20 \
#     --save_strategy "steps" \
#     --save_steps 100 \
#     --warmup_ratio 0.03 \
#     --per_device_eval_batch_size 8



accelerate launch --config_file=./scripts/deepspeed_zero3.yaml \
    train.py \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --output_dir sft-Qwen2-VL-2b \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --num_train_epochs 2 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --learning_rate 2e-5 \
    --save_total_limit 3 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --optim adamw_bnb_8bit \
    --per_device_eval_batch_size 4 \
    --logging_steps 10 \
    --logging_dir "./logs" \
    --report_to "tensorboard" 