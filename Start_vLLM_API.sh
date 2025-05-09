#!/bin/bash

cd BigModel

CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
    --model DeepSeek-R1-Distill-Qwen-7B \
    --max-model-len=2048 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype=half
