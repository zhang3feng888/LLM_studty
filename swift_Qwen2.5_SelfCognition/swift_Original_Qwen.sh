# 使用/Qwen/Qwen2.5-7B-Instruct原始模型
# 注意是/data5/zhangenwei   开始有个/
# --model：指定完整模型的路径（就是你现在本地下载好的模型目录）。

CUDA_VISIBLE_DEVICES=7 \
swift infer \
    --model /data5/zhangenwei/BigModel/Qwen/Qwen2.5-7B-Instruct \
    --stream true \
    --infer_backend vllm \
    --max_model_len 2048 \
    --torch_dtype float16