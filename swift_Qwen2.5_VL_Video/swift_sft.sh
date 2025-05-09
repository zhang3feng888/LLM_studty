# 模型推理.sh文件版
# ./swift_Qwen2.5_VL_Video/swift_sft.sh

export HF_DATASETS_CACHE=/data5/zhangenwei/rubbish

SIZE_FACTOR=4 NFRAMES=10 CUDA_VISIBLE_DEVICES="4,5,6,7" swift sft \
    --model /data5/zhangenwei/BigModel/Qwen/Qwen2-VL-2B-Instruct \
    --train_type lora \
    --dataset /data5/llm/fune-turning/dataset/video_label/train_video_add_file_name.jsonl \
    --val_dataset /data5/llm/fune-turning/dataset/video_label/val_video_add_file_name.jsonl \
    --output_dir /data5/zhangenwei/Datasets/Output/VideoChatGPT \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --max_length 4096 
