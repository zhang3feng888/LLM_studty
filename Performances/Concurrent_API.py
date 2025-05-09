# DeepSeek-R1-Distill-Qwen-7B 的并发
# 先./Start_Qwen_7B_API.sh 启动模型的API

from openai import OpenAI
import concurrent.futures
import time

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-xxx",  # 随便填，满足接口校验
)

# 单个线程做的事情：发5次请求
def thread_worker(thread_id, requests_per_thread):
    results = []
    for i in range(requests_per_thread):
        try:
            response = client.chat.completions.create(
                model="DeepSeek-R1-Distill-Qwen-7B",
                messages=[
                    {"role": "system", "content": "你是一个职业技术学习专家。"},
                    {"role": "user", "content": f"学挖掘机技术哪家强？线程{thread_id} 第{i+1}次提问。"},
                ]
            )
            results.append(response.choices[0].message.content)
        except Exception as e:
            results.append(f"请求失败: {e}")
    return results

# 并发测试
def test_concurrency(num_threads, requests_per_thread):
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(thread_worker, thread_id, requests_per_thread) for thread_id in range(num_threads)]
        concurrent.futures.as_completed(futures)
        
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    num_threads=2
    requests_per_thread=5
    test_concurrency(num_threads, requests_per_thread)
