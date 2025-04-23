# 使用官方Python镜像作为基础
FROM python:3.11-slim

# 设置python镜像
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# 设置工作目录
WORKDIR /app

# 跳过安全更新，加快构建速度
RUN echo 'APT::Get::AllowUnauthenticated "true";' > /etc/apt/apt.conf.d/99allow-unauthenticated
RUN echo 'Acquire::AllowReleaseInfoChange::Suite "true";' > /etc/apt/apt.conf.d/99allow-releaseinfo-change

# 设置debian镜像
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*
RUN echo "\
deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free\n\
deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free\n\
deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-backports main contrib non-free\n\
deb https://security.debian.org/debian-security bookworm-security main contrib non-free" > /etc/apt/sources.list

# 安装系统依赖（包括cron）
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    cron \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 升级cmake
RUN pip install --upgrade cmake
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

# 安装Python依赖
RUN pip install uv
RUN uv sync

# 创建日志目录和模型目录
RUN mkdir -p /var/log/cron

# 下载LLM模型 (如果使用本地LLM)
RUN if [ "$USE_LLM_API" = "0" ]; then \
    mkdir -p /app/models \
    wget https://huggingface.co/Qwen/Qwen1.5-3B-Instruct-GGUF/resolve/main/qwen1.5-3b-instruct-q4_k_m.gguf -O /app/models/qwen.gguf; \
    fi

# 设置容器启动命令（由compose覆盖）
CMD ["cd /app && /usr/local/bin/uv run main.py"]
