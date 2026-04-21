set -e
export LMMS_EVAL_LAUNCHER="accelerate"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0
benchmark=viewspatial # choices: [cv_bench, cv_bench_2d, cv_bench_3d]
output_path=logs/$(TZ="Asia/Shanghai" date "+%Y%m%d%H%M%S")
model_path=/storage/chongyu/Qwen3-VL/ckpt/qwen3_4b_vggt_sr_1e-5_0323_vggt_only_deepstack_7_10_13_16_19_22_lobal_importance_gate_all

model_args_str="pretrained=$model_path,max_pixels=268324,min_pixels=8192,attn_implementation=flash_attention_2"

accelerate launch --num_processes=8  --main_process_port 29537 -m lmms_eval \
    --model qwen3_vl_my \
    --model_args "$model_args_str" \
    --tasks ${benchmark} \
    --batch_size 1 \
    --log_samples_suffix original \
    --log_samples \
    --output_path $output_path \
    --force_simple
