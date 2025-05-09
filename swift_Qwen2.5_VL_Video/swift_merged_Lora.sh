# 模型合并.sh文件版 通过swift sft微调训练的模型不需要填原始模型
# ./swift_Qwen2.5-VL_coco_2014/swift_merged_Lora.sh
CUDA_VISIBLE_DEVICES=7 \
swift export \
    --adapters /data5/zhangenwei/Datasets/Output/swift_coco_2014/v15-20250501-152910/checkpoint-61 \
    --merge_lora true \
    --output_dir /data5/zhangenwei/BigModel/Qwen/swift_Qwen2.5-VL-7B-Instruct
