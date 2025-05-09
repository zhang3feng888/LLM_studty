from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class QwenVLChatBot:
    def __init__(self, base_model_path):
        print("加载模型中...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(base_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.messages = []  # 用于记录完整历史对话
        print("模型加载完成！")
    def chat(self, image_path):
        # 处理用户输入并添加到历史记录
        self.messages.append({"role": "user", "content": [{"type": "image", "image": image_path}]})

        # 构造输入
        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(self.messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # 生成回复
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 将模型回复加入历史
        self.messages.append({"role": "assistant", "content": output_text})
        return output_text

# 示例使用
if __name__ == "__main__":
    bot = QwenVLChatBot(
        base_model_path="/data5/zhangenwei/BigModel/Qwen/Qwen2.5-VL-7B-Instruct"
    )

    while True:
        image_path = input("请输入图片路径 (或者输入 'exit' 退出): ")
        if image_path.lower() == "exit":
            print("退出对话")
            break
        
        result = bot.chat(image_path=image_path)
        print("生成结果:", result)
