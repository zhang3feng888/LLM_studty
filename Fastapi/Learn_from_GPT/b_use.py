# 使用API，只有预定好的问题-学挖掘机技术哪家强？

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-xxx", # 随便填写，只是为了通过接口参数校验
)

chat_outputs = client.chat.completions.create(
    model="Qwen3-4b",
    messages=[
        {"role": "system", "content": "你是一个职业技术学习专家。"},
        {"role": "user", "content": "学挖掘机技术哪家强？"},
    ]
)
print(chat_outputs)