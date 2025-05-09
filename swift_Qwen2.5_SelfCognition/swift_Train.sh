# export HF_DATASETS_CACHE 下载缓存的数据集放到这里
export HF_DATASETS_CACHE=/data5/zhangenwei/rubbish

CUDA_VISIBLE_DEVICES=7 swift sft \
    --model /data5/zhangenwei/BigModel/Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset /data5/zhangenwei/Datasets/Alpaca-gpt4-data-en#500 \
              /data5/zhangenwei/Datasets/Alpaca-gpt4-data-zh#500 \
              /data5/zhangenwei/Datasets/SelfCognition#500 \
    --torch_dtype float16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir /data5/zhangenwei/Datasets/Output/SelfCognition \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --device_map auto \
    --model_name 小黄 'Xiao Huang' \
    --model_author '魔搭' 'ModelScope'