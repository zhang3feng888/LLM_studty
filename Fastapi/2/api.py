import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

app = FastAPI()

# 用于接收客户端发送的数据
class Query(BaseModel):
    text: str

# 模型加载
path = "/data6/zhangenwei/BigModel/Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path, 
    trust_remote_code=True
    )

@app.post("/chat/")
async def chat(query: Query):
    # 编码输入，添加 attention_mask
    inputs = tokenizer(
        query.text,
        return_tensors="pt"
    ).to(model.device)

    # 生成响应
    output = model.generate(
        **inputs,
        do_sample=True
        )

    # 解码输出
    response = tokenizer.decode(
        output.cpu()[0],  # 只取生成的新内容
        skip_special_tokens=True
    )
    return {"result": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6665)
