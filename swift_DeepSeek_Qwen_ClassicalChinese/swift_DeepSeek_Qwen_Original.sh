# 直接运行，可连续对话

CUDA_VISIBLE_DEVICES=7 swift infer \
    --model BigModel/DeepSeek-R1-Distill-Qwen-7B \
    --stream true \
    --infer_backend vllm \
    --max_model_len 2048
