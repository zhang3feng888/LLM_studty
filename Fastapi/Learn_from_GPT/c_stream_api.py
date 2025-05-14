from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading
import json
import os
import torch

# 指定可见 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

app = FastAPI(
    title="简易 Chat Completion 接口（流式输出版）",
    description="这是一个模拟 OpenAI Chat Completion 接口的简化版本，仅支持流式输出",
    version="1.0.0"
)

# 请求体定义，始终 stream=True
class ChatRequest(BaseModel):
    messages: List[Dict] = Field(..., description="对话消息列表")
    model: str = Field(..., description="使用的模型名称，例如 Qwen3-4b")
    temperature: float = Field(1.0, ge=0.0, le=2.0, description="采样温度，值越高越随机")
    stream: Literal[True] = Field(..., description="必须为 True，启用流式输出")

# 加载本地模型
MODEL_PATH = "/data6/zhangenwei/BigModel/Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def generate_stream(request: ChatRequest):
    # 1. 把历史消息编码成 input_ids
    encoded = tokenizer.apply_chat_template(
        request.messages,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)

    # 2. 初始化 TextIteratorStreamer，用于流式输出
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # 3. 在后台线程启动 model.generate，让它把 token 推到 streamer
    thread = threading.Thread(
        target=model.generate,
        kwargs=dict(
            input_ids=encoded,
            max_new_tokens=1024,
            do_sample=True,
            temperature=request.temperature,
            streamer=streamer
        ),
        daemon=True
    )
    thread.start()

    # 4. 按 SSE 格式 yield 每个 token
    for token in streamer:
        chunk = {
            "id": "chatcmpl-001",
            "object": "chat.completion.chunk",
            "choices": [{
                "delta": {"content": token},
                "index": 0,
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    # 5. 输出结束标志
    finish = {
        "id": "chatcmpl-001",
        "object": "chat.completion.chunk",
        "choices": [{
            "delta": {},
            "index": 0,
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(finish, ensure_ascii=False)}\n\n"

@app.post("/v1/chat/completions", summary="生成聊天回复（流式）")
def chat_stream(request: ChatRequest):
    return StreamingResponse(
        generate_stream(request),
        media_type="text/event-stream"   # SSE 协议
    )

@app.get("/v1/models", summary="获取模型列表", description="获取支持的模型列表")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "Qwen3-4b", "object": "model"},
            {"id": "gpt-4",     "object": "model"},
            {"id": "baichuan2", "object": "model"},
            {"id": "gpt-3.5-turbo", "object": "model"},
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
