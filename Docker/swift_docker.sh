# 创建一个swift容器

docker run -it --gpus all --name swift_zew \
  --security-opt seccomp=unconfined \
  -v /data5/zhangenwei:/data5/zhangenwei \
  modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-vllm0.8.3-modelscope1.25.0-swift3.3.0.post1
