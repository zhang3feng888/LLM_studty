# ./swift_Qwen2.5_VL_Video/swift_Deploy.sh

CUDA_VISIBLE_DEVICES=4,5,6,7 swift deploy \
    --model /data5/zhangenwei/BigModel/Qwen/Qwen2.5-VL-7B-Instruct \
    --port 8000