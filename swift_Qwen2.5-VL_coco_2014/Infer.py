# 第56行换图片。只执行一次，要重复加载模型

from peft import PeftModel, LoraConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# 加载原始模型
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/data5/zhangenwei/BigModel/Qwen/Qwen2.5-VL-7B-Instruct",
    device_map="auto",
    trust_remote_code=True,
)

# 加载微调的LoRA参数（这里是你保存的checkpoint路径）
model = PeftModel.from_pretrained(
    base_model,
    model_id="/data5/zhangenwei/Datasets/Output/coco_2014/checkpoint-62",
)

# 加载tokenizer和processor
tokenizer = AutoTokenizer.from_pretrained("/data5/zhangenwei/BigModel/Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("/data5/zhangenwei/BigModel/Qwen/Qwen2.5-VL-7B-Instruct")

def predict(messages, model):
    # 准备推理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/data5/zhangenwei/Datasets/coco_2014_caption/coco_2014_caption_images/790.jpg"},
            {"type": "text", "text": "图中是什么内容？"}
        ]
    }
]

result = predict(messages, model)
print("生成结果:", result)

