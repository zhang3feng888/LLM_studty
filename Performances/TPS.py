# DeepSeek-R1-Distill-Qwen-7B 的TPS
# 先./Start_Qwen_7B_API.sh 启动模型的API

import time
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-xxx",  # 可随便填
)

# 记录开始时间
start_time = time.time()
total_tokens = 0

# 发起请求（流式返回）
stream = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Qwen-7B",
    messages=[
        {"role": "system", "content": "你是一个职业技术学习专家。"},
        {"role": "user", "content": "学挖掘机技术哪家强？"},
    ],
    stream=True
)

# 统计返回的 tokens
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
        token = chunk.choices[0].delta.content
        total_tokens += 1  # 每个 delta.content 就是一个 token

# 记录结束时间
end_time = time.time()
elapsed_time = end_time - start_time

# 计算 TPS
tps = total_tokens / elapsed_time if elapsed_time > 0 else 0

print(f"输出 token 总数: {total_tokens}")
print(f"总耗时: {elapsed_time:.4f} 秒")
print(f"吞吐量 (TPS): {tps:.2f} tokens/s")
