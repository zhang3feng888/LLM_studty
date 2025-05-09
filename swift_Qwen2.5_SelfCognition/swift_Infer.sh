# --merge_lora  true        合并LoRA模型
# --infer_backend  vllm     加速推理过程。
# 注意是/data5/zhangenwei   开始有个/
# --ckpt_dir    基础模型权重目录
# --adapters    LoRA 微调权重目。使用了LoRA进行微调要加上

CUDA_VISIBLE_DEVICES=7 \
swift infer \
    --ckpt_dir /data5/zhangenwei/BigModel/Qwen/Qwen2.5-7B-Instruct \
    --adapters /data5/zhangenwei/Datasets/Output/SelfCognition/v7-20250430-111426/checkpoint-93 \
    --stream true \
    --infer_backend vllm \
    --max_model_len 2048 \
    --torch_dtype float16

