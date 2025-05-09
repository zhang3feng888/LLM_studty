# 模型推理.sh文件版
# ./swift_Qwen2.5-VL_coco_2014/swift_sft.sh

# 最重要的是做出来/data5/zhangenwei/Datasets/coco_2014_caption/data_vl.json
# 那两个py文件做的（DataToCsv.py、CsvToJson.py），做出来后可直接用swift_sft.sh训练

export HF_DATASETS_CACHE=/data5/zhangenwei/rubbish
export MODELSCOPE_CACHE=/data5/zhangenwei/rubbish

CUDA_VISIBLE_DEVICES=7 \
swift sft \
    --model /data5/zhangenwei/BigModel/Qwen/Qwen2.5-VL-7B-Instruct \
    --train_type lora \
    --dataset '/data5/zhangenwei/Datasets/coco_2014_caption/data_vl.json' \
    --output_dir /data5/zhangenwei/Datasets/Output/swift_coco_2014 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 8 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --system '说出图片包含的内容。' \
    --model_author zew \
    --model_name baby
