# http://localhost:8000/docs

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

app = FastAPI(
    title="简易 Chat Completion 接口",
    description="这是一个模拟 OpenAI Chat Completion 接口的简化版本。",
    version="1.0.0"
)

# 加载本地模型（假设 Qwen 是 causal LM）
MODEL_PATH = "/data6/zhangenwei/BigModel/Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"  # 自动分配 GPU
)
model.eval()# 将模型设置为评估模式

def generate_response(messages: List[Dict], temperature: float = 1.0) -> str:
    # -> str ：表示 generate_response 函数返回的结果是字符串    
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=temperature
        )

    # 只保留新生成的内容
    generated = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

# 单条消息  {"role": "user", "content": "你好"}
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(..., description="消息发送者角色")
    content: str = Field(..., description="消息内容")

# 请求体
class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    model: str = Field(..., description="使用的模型名称，例如 gpt-3.5-turbo")
    temperature: float = Field(1.0, ge=0.0, le=2.0, description="采样温度，值越高越随机")

# 定义每个模型的信息
class ModelCard(BaseModel):
    id: str
    object: str = "model"

# 定义模型列表结构
class ModelList(BaseModel):
    object: str = 'list'
    data: List[ModelCard] = []

@app.post("/v1/chat/completions", summary="生成聊天回复", description="根据历史消息生成 AI 的回答")
def chat(request: ChatRequest):
    result = generate_response(request.messages, request.temperature)
    return {
        "id": "chatcmpl-001",
        "object": "chat.completion",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": result
            },
            "finish_reason": "stop",
            "index": 0
        }],
        "model": request.model
    }

@app.get("/v1/models", summary="获取模型列表", description="获取支持的模型列表")
def list_models():
    model_ids = ["Qwen3-4b", "gpt-4", "baichuan2", "gpt-3.5-turbo"]
    model_List = [ModelCard(id=mid) for mid in model_ids]
    return ModelList(data=model_List)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
curl -X 'POST' \
  'http://localhost:8000/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    {"role": "system", "content": "你是一个乐于助人的 AI 助手。"},
    {"role": "user", "content": "你好，学挖掘机学校哪家强？"}
  ],
  "model": "Qwen3-4b",
  "temperature": 0.8
}'

curl -X 'GET' \
  'http://localhost:8000/v1/models' \
  -H 'accept: application/json'
"""