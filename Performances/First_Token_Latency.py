# DeepSeek-R1-Distill-Qwen-7B 的首字延迟
# 先./Start_Qwen_7B_API.sh 启动模型的API

import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-xxx",  # 随便写
)

start_time = time.time()

stream = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Qwen-7B",
    messages=[
        {"role": "system", "content": "你是一个职业技术学习专家。"},
        {"role": "user", "content": "学挖掘机技术哪家强？"},
    ],
    # max_tokens=256,  # 最大生成长度，越小越快
    # temperature=0.7,  # 控制随机性，越小越快
    stream=True  # 流式输出，生成一个字出一个字
)

first_token_received = False

for chunk in stream:
    if not first_token_received:
        first_token_time = time.time()
        latency = first_token_time - start_time
        print(f"首字延迟: {latency:.4f} s")
        first_token_received = True
    break  # 收到第一个 token 后就退出