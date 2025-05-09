# 创建nvidia/cuda:12.4.0-base-ubuntu20.04容器
# 先创建一个.dockerignore，防止上传文件。使用挂载的方式同步文件

# 使用 NVIDIA CUDA 12.4 作为基础镜像
FROM nvidia/cuda:12.4.0-base-ubuntu20.04

# 设置工作目录
WORKDIR /data5/zhangenwei

# 安装 Miniconda
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && /opt/conda/bin/conda init

# 设置 PATH 环境变量，以便 Miniconda 可用
ENV PATH=/opt/conda/bin:$PATH

# 安装所需的 CUDA 工具包和其他依赖（如果需要）
RUN apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    libsm6 libxext6 libxrender-dev

# 设置默认命令
CMD ["bash"]

#创建
# docker build -t zew .
#运行挂载
# docker run --gpus all -it --name zew_container -v /data5/zhangenwei:/data5/zhangenwei zew
