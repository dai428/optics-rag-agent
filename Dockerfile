# ============================================================
# rag-demo 的容器图纸（Dockerfile）
# 用法（在 D:\Desktop\rag-demo 目录下执行）：
#   docker build -t rag-demo:v1 .
#   docker run -d --name rag -p 8000:8000 --env-file .env \
#              -v rag_models:/models rag-demo:v1
# 然后浏览器打开 http://localhost:8000/health
# ============================================================

# ① 基础镜像：别人的模具。术业有专攻，直接拿现成的 python 运行环境
FROM python:3.11-slim

# ② 环境变量：写进镜像的「运行参数」
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/models

WORKDIR /app

# ③ 先只拷依赖清单，再装依赖
#    这是 Docker 层的复用技巧：requirements.txt 没变，这层直接用上次的缓存，
#    改代码时不必重新装一遍依赖（能省几分钟）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ④ 再拷真正的代码和数据
COPY app/ ./app/
COPY data/ ./data/

# ⑤ 声明容器对外提供 8000 端口（只是文档性声明，真正映射靠 docker run -p）
EXPOSE 8000

# ⑥ 容器启动时执行的命令
#    ⚠️ 必须是 0.0.0.0 不能是 127.0.0.1 —— 容器里的 127.0.0.1 只指容器自己，
#       主机访问不到。这点和你在本机跑的习惯相反，是第一个必踩的坑。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
