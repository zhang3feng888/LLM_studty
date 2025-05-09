# 不使用API，创建模型，连续对话

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 根据你的GPU编号修改

class QwenTextChatBot:
    def __init__(self, model_path):
        print("加载模型中...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.messages = []  # 用于记录历史对话
        print("模型加载完成！")

    def chat(self, user_input):
        self.messages.append({"role": "user", "content": user_input})

        # 使用 tokenizer 构造多轮对话格式
        prompt = self.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 模型生成回答
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.8,
            top_p=0.95
        )

        generated_ids_trimmed = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # 记录 assistant 的回答
        self.messages.append({"role": "assistant", "content": response})
        return response

if __name__ == "__main__":
    model_path = "/data5/zhangenwei/BigModel/Qwen/Qwen2.5-7B-Instruct"  # 修改为你本地的纯文本模型路径
    bot = QwenTextChatBot(model_path)

    print("欢迎使用 Qwen2.5 多轮对话机器人，输入 'exit' 退出。")
    while True:
        user_input = input("你：")
        if user_input.strip().lower() == "exit":
            print("对话结束，再见！")
            break
        reply = bot.chat(user_input)
        print("助手：", reply)
