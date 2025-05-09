# DeepSeek-R1-Distill-Qwen-7B 的所有性能测试
# 先./Start_Qwen_7B_API.sh 启动模型的API

import requests
import threading
import time
from typing import List
import json

# === 配置部分 ===
API_URL = "http://localhost:8000/v1/chat/completions"  # 替换为你的 OpenAI 兼容服务地址
API_KEY = "sk-..."  # # 随便填写，只是为了通过接口参数校验, 如果没有，可以留空
MODEL_NAME = "DeepSeek-R1-Distill-Qwen-7B"

PROMPT = "我现在想学挖掘机技术。"
CONCURRENCY = 5  # 并发线程数
REQUESTS_PER_THREAD = 5  # 每个线程请求几次

# === 结果记录 ===
all_latencies = []
first_token_latencies = []
total_tokens = 0
lock = threading.Lock()


def call_api_stream():
    global total_tokens
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    for _ in range(REQUESTS_PER_THREAD):
        data = {
            "model": MODEL_NAME,
            "stream": True,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0.7,
        }

        first_token_time = None
        start_time = time.time()
        token_count = 0

        try:
            response = requests.post(API_URL, headers=headers, json=data, stream=True, timeout=60)
            for line in response.iter_lines():
                if line:
                    if line.startswith(b"data: "):
                        payload = line[6:]
                        if payload == b"[DONE]":
                            break
                        chunk = json.loads(payload.decode("utf-8"))# AI回答的结果
                        token_count += 1
                        if first_token_time is None:
                            first_token_time = time.time()
        except Exception as e:
            print(f"[Error] {e}")
            continue

        end_time = time.time()

        with lock:
            latency = end_time - start_time
            if first_token_time:
                first_token_latencies.append(first_token_time - start_time)
            all_latencies.append(latency)
            total_tokens += token_count


# === 主入口 ===
start_all = time.time()
threads = [threading.Thread(target=call_api_stream) for _ in range(CONCURRENCY)]
[t.start() for t in threads]
[t.join() for t in threads]
end_all = time.time()

# === 结果汇总 ===
total_time = end_all - start_all
avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
avg_first_token_latency = sum(first_token_latencies) / len(first_token_latencies) if first_token_latencies else 0
tps = total_tokens / total_time if total_time else 0
rps = CONCURRENCY * REQUESTS_PER_THREAD / total_time

print(f"=========== 性能测试结果 ===========")
print(f"总请求数:\t\t线程数 {CONCURRENCY} * 每线程请求数 {REQUESTS_PER_THREAD} = {CONCURRENCY * REQUESTS_PER_THREAD}")
print(f"总用时:\t\t\t{total_time:.2f} s")
print(f"Requests (RPS):\t\t{rps:.2f} requests/s")
print(f"平均总延迟:\t\t{avg_latency:.2f} s")
print(f"Token (TPS):\t\t{tps:.2f} tokens/s")
print(f"平均首字延迟:\t\t{avg_first_token_latency:.2f} s")
