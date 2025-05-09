# export HF_DATASETS_CACHE 下载缓存的数据集放到这里
export HF_DATASETS_CACHE="/data5/zhangenwei/rubbish"

CUDA_VISIBLE_DEVICES=7 \
swift sft \
    --model_type deepseek_r1_distill \
    --check_model false \
    --model /data5/zhangenwei/BigModel/DeepSeek-R1-Distill-Qwen-7B \
    --dataset /data5/zhangenwei/Datasets/ClassicalChinese \
    --train_type lora \
    --output_dir /data5/zhangenwei/Datasets/Output/ClassicalChinese \
    --num_train_epochs 1 \
    --max_length 2048 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 8 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --device_map auto \
    --torch_dtype float16
