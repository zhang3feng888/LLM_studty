import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel

def merge_lora(base_model_path: str, lora_checkpoint_path: str, output_path: str):
    # 1. 加载基座模型
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 2. 加载 LoRA adapter
    peft_model = PeftModel.from_pretrained(
        base_model,
        lora_checkpoint_path,
        is_trainable=False
    )

    # 3. 合并 LoRA 权重
    merged_model = peft_model.merge_and_unload()

    # 4. 保存合并模型
    merged_model.save_pretrained(output_path)

    # 5. 保存 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_path)

    # ✅ 6. 保存 processor（图像/多模态配置）
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    processor.save_pretrained(output_path)

if __name__ == "__main__":
    merge_lora(
        base_model_path='/data5/zhangenwei/BigModel/Qwen/Qwen2.5-VL-7B-Instruct',
        lora_checkpoint_path='/data5/zhangenwei/Datasets/Output/coco_2014/checkpoint-62',
        output_path='/data5/zhangenwei/BigModel/Qwen/py_Qwen2.5-VL-7B-Instruct'
    )
