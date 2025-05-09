# MCP 的API

cd BigModel

CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --max-model-len=2048 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype=half \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
