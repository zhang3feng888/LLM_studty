# 模型评估
# pip install pycocoevalcap 需要pycocoevalcap
# ./swift_Qwen2.5-VL_coco_2014/swift_eval.sh

# 原始模型/data5/zhangenwei/BigModel/Qwen/Qwen2.5-VL-7B-Instruct    最差
# py微调后的/data5/zhangenwei/BigModel/Qwen/py_Qwen2.5-VL-7B-Instruct   第二好
# sh微调后的/data5/zhangenwei/BigModel/Qwen/swift_Qwen2.5-VL-7B-Instruct    最好


CUDA_VISIBLE_DEVICES=7 \
swift eval \
    --model /data5/zhangenwei/BigModel/Qwen/swift_Qwen2.5-VL-7B-Instruct \
    --eval_backend VLMEvalKit \
    --infer_backend pt \
    --eval_limit 10 \
    --eval_dataset COCO_VAL