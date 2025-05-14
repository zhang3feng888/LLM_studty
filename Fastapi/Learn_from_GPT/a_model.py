from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

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

def generate_response(prompt: str, max_new_tokens: int = 128) -> str:
    # -> str ：表示 generate_response 函数返回的结果是字符串
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)
