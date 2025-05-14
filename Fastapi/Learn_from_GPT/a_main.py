# http://localhost:8000/docs

from fastapi import FastAPI
from pydantic import BaseModel
from a_model import generate_response

app = FastAPI()

# 接收客户端发送的数据结构
class RequestModel(BaseModel):
    prompt: str
    max_tokens: int = 128  # 可选参数

# 返回给客户端的数据结构
class ResponseModel(BaseModel):
    result: str
    # {"result": "这是模型的输出"}

# 用户发 POST 请求到 http://你的域名/chat 会触发chat函数
@app.post("/chat", response_model=ResponseModel)
# response_model=ResponseModel : 对函数返回的结果 {"result": result} 进行校验和自动转换为 JSON
def chat(request: RequestModel):
    # request:  表示“客户端发送来的,RequestModel格式的，数据”。
    result = generate_response(request.prompt, request.max_tokens)
    return {"result": result}

# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# curl -X POST http://localhost:8000/chat     -H "Content-Type: application/json"     -d '{"prompt": "我刚刚问的什么问题", "max_tokens": 100}'
