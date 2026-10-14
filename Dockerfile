# ============================================================
#  多阶段构建：先把前端编译成静态文件，再放进运行时镜像。
#
#  为什么用多阶段？
#    前端构建需要 Node 与几百 MB 的 node_modules，而运行时完全不需要。
#    分成两阶段后，最终镜像里只有编译好的 dist，体积小很多。
# ============================================================

# ── 阶段 1：构建前端 ──────────────────────────────────────────
FROM node:20-slim AS frontend

WORKDIR /build
# 先只拷依赖清单：依赖没变时这层直接命中缓存，改代码不必重装依赖
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build


# ── 阶段 2：Python 运行时 ─────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/models \
    CORPUS_DIR=/app/data/docs

WORKDIR /app

# 先装依赖（利用层缓存），再拷代码
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY tests/ ./tests/
COPY data/eval_set.json ./data/
# 公开示例语料随镜像提供：这样容器起来就能提问，不需要用户先准备数据
COPY data/sample_docs/ ./data/docs/
COPY data/sample_eval_set.json ./data/

# 前端构建产物（由阶段 1 产出，FastAPI 会直接托管）
COPY --from=frontend /build/dist ./frontend/dist/

# 数据目录：索引、SQLite、父块仓库都写在这里，需挂卷持久化
RUN mkdir -p /app/data/chroma_db /app/data/parent_store

# 以非 root 用户运行，降低容器逃逸后的影响面
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /models 2>/dev/null || chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 健康检查：用 /health（轻量探针，不加载模型、不查索引）
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# ⚠️ 必须是 0.0.0.0 不能是 127.0.0.1 —— 容器里的 127.0.0.1 只指容器自己，
#    主机访问不到。这点和本机跑的习惯相反，是第一个必踩的坑。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
