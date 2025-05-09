# ClassicalChinese  /data5/zhangenwei/Datasets/Output/ClassicalChinese/v2-20250430-094413/checkpoint-206

CUDA_VISIBLE_DEVICES=7 swift infer \
  --ckpt_dir /data5/zhangenwei/Datasets/Output/ClassicalChinese/v2-20250430-094413/checkpoint-206 \
  --torch_dtype float16 \
  --infer_backend vllm \
  --max-model-len 2048
