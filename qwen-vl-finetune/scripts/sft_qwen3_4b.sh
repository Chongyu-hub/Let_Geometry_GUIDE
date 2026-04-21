#!/bin/bash
export WANDB_API_KEY=
# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}  # Default to 8 GPUs if not set

# DeepSpeed configuration
deepspeed=/storage/chongyu/Qwen3-VL/qwen-vl-finetune/scripts/zero2.json

# Model configuration
llm=/storage/chongyu/Qwen3-VL/ckpt/Qwen3-VL-4B-Instruct

# Training hyperparameters
lr=1e-5
batch_size=1
grad_accum_steps=8

# Training entry point
entry_file=/storage/chongyu/Qwen3-VL/qwen-vl-finetune/qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=scanrefer,scan2cap,scannet_det

# Output configuration
run_name="qwen3_4b_vggt_1e-5_0331_vggt_deepstack_7_10_13_16_19_22_global_importance_gate_all"
output_dir=/storage/chongyu/Qwen3-VL/ckpt/qwen3_4b_vggt_1e-5_0331_vggt_deepstack_7_10_13_16_19_22_global_importance_gate_all
mkdir -p ${output_dir}
# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --optim adamw_torch \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 268324 \
    --min_pixels 8192 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 12800 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --run_name ${run_name} \
    --report_to wandb \
    --use_geometry_inputs True \
    --use_geometry_encoder True \
    --use_feature_fusion_module True \
    --use_patch_size_alin True \
    --geometry_deepstack_indexes_pro "7:0,10:1,13:2,16:3,19:4,22:5" \
    --use_deepstack_importance_gate "all" \
    --use_deepstack_global_gate "all" \
    --geometry_encoder_type vggt \
    --geometry_encoder_path /storage/chongyu/Qwen3-VL/ckpt/VGGT-1B \
    --group_by_modality_length True "

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} > ${output_dir}/train_$(date +%Y%m%d_%H%M%S).log 2>&1