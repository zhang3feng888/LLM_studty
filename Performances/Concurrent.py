# 并发测试

import os
import time
from vllm import LLM, SamplingParams
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

model_dir = '/data5/zhangenwei/BigModel/DeepSeek-R1-Distill-Qwen-7B'

text = "你好，帮我介绍一下什么时大语言模型。"
batch_size = 16 # 并行数
texts = [text] * batch_size

# 初始化LLM
llm = LLM(
    model = model_dir,
    dtype = "half", # 使用 FP16 精度
    tensor_parallel_size = 1,
    max_model_len=2048
)

# 超参数
sampling_params = SamplingParams(temperature=0.7, top_p=0.8,max_tokens=512)

# 测试并行推理时间
start_time = time.time()
outputs = llm.generate(texts, sampling_params)
end_time = time.time()

total_time = end_time - start_time
print(f"\nBatch size: {batch_size}")
print(f"总推理耗时: {total_time:.4f}秒")
print(f"平均每条耗时: {total_time / batch_size:.4f}秒")
