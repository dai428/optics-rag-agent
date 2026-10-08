"""
RAG 真实版（API + 本地 Embedding）—— 张万森专用

架构：
  · 生成回答：AMD Radeon Cloud 的 DeepSeek-V4.1-Flash（云端 API，需 Key）
  · 向量嵌入：BAAI/bge-small-zh-v1.5（本地跑，免费，不需 Key）

跑法：
  cd "C:/Users/45064/WorkBuddy/工作/rag-demo"
  ./.venv/Scripts/python.exe run_api_demo.py
"""
import os
import sys
from pathlib import Path

# ---------- 读取 .env ----------
def load_env(path=".env"):
    env = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

ENV = load_env()
API_KEY = ENV.get("AMD_API_KEY") or os.getenv("AMD_API_KEY")
BASE_URL = ENV.get("AMD_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1")
LLM_MODEL = ENV.get("AMD_LLM_MODEL", "DeepSeek-V4.1-Flash")
EMBED_MODEL = ENV.get("LOCAL_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")

if not API_KEY:
    sys.exit("❌ 未找到 AMD_API_KEY，请检查 .env 文件")

# ---------- 1. 本地 Embedding ----------
print(f"[1/5] 加载本地嵌入模型 {EMBED_MODEL} ...（首次运行会自动下载约 100MB）")
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("      ✅ 嵌入模型就绪")

# ---------- 2. 云端 LLM ----------
print(f"[2/5] 连接云端 LLM {LLM_MODEL} ...")
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.1,
    timeout=60,
)
print("      ✅ LLM 客户端就绪")

# ---------- 3. 加载并切片 ----------
print("[3/5] 加载知识库并切片 ...")
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs_dir = Path("docs")
all_docs = []
for f in docs_dir.glob("*.txt"):
    all_docs.extend(TextLoader(str(f), encoding="utf-8").load())
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, length_function=len)
chunks = splitter.split_documents(all_docs)
print(f"      共 {len(all_docs)} 篇文档 → {len(chunks)} 个片段")

# ---------- 4. 构建向量库 ----------
print("[4/5] 构建向量库（内存版，无需 Chroma 持久化）...")
from langchain_core.vectorstores import InMemoryVectorStore
vector_db = InMemoryVectorStore(embedding=embeddings)
vector_db.add_documents(chunks)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
print(f"      ✅ {len(chunks)} 条向量已入库")

# ---------- 5. 构建 RAG 链 ----------
print("[5/5] 构建 RAG 链 ...\n")
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("""
你是专业的光学领域问答助手，必须严格基于以下上下文信息回答用户问题。
如果上下文没有相关内容，直接回复"根据知识库，未找到相关答案"，禁止编造信息。
回答要条理清晰、专业准确。
上下文信息：
{context}
用户问题：
{question}
""")

rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    questions = [
        "LCVR 是什么？它相比传统旋转波片有什么优势？",
        "偏振仪为什么要做定标？定标不准会怎样？",
        "偏振测量的误差来源有哪些？怎么降低？",
    ]
    for q in questions:
        print("=" * 64)
        print(f"❓ 提问：{q}")
        hits = retriever.invoke(q)
        print(f"🔍 检索：命中 {len(hits)} 个片段")
        print(f"💡 回答：")
        try:
            ans = rag_chain.invoke(q)
            print(ans)
        except Exception as e:
            print(f"   ⚠️ 调用失败：{e}")
        print()
    print("=" * 64)
    print("✅ 真实 RAG 链路运行完成：本地嵌入检索 + 云端大模型生成")
