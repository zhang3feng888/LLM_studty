# 本地直接调用，只能预定好的问题

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
from transformers import AutoTokenizer
from vllm import LLM,SamplingParams

model_dir = 'BigModel/DeepSeek-R1-Distill-Qwen-7B'

"""
# 带提示词的
# Tokenizer初始化分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    local_files_only = True, # 使用本地模型
)
# Prompt 提示词
messages = [
    {'role': 'system','content':'你是一个智能问答助手.'},
    {'role':'system','content':'学挖掘机技术哪家强?'}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True,
)
"""
text = ["你好，帮我介绍一下什么时大语言模型。",
        "可以给我将一个有趣的童话故事吗？"]

# 超参数：最多512Token
sampling_params = SamplingParams(temperature=0.7, top_p=0.8,max_tokens=512)

# 初始化大模型
llm = LLM(
    model = model_dir,
    dtype = "half", # 使用 FP16 精度
    tensor_parallel_size = 1,
    max_model_len=2048
)

#模型推理输出
outputs = llm.generate(text, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f'Prompt提示词：{prompt!r},大模型推理输出：{generated_text!r}')
