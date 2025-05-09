#!/bin/bash

# 模型评估
# pip install pycocoevalcap 需要pycocoevalcap
# ./swift_Qwen2.5_VL_Video/swift_eval.sh

# 原始模型/data5/zhangenwei/BigModel/Qwen/Qwen2.5-VL-7B-Instruct    最差
# sh微调后的/data5/zhangenwei/BigModel/Qwen/swift_Qwen2.5-VL-7B-Instruct    最好

export HF_DATASETS_CACHE=/data5/zhangenwei/rubbish
export MODELSCOPE_CACHE=/data5/zhangenwei/rubbish

MAX_PIXELS=50000 VIDEO_MAX_PIXELS=10000 FPS_MAX_FRAMES=4 CUDA_VISIBLE_DEVICES=4,5,6,7 swift eval \
    --model /data5/zhangenwei/BigModel/Qwen/Qwen2.5-VL-7B-Instruct \
    --eval_backend VLMEvalKit \
    --infer_backend pt \
    --eval_limit 5 \
    --eval_dataset MMBench-Video \
    --max_new_tokens 2048