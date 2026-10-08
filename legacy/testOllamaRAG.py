from langchain_community.document_loaders import TextLoader  # 加载 TXT 文档（支持其他格式，见下文扩展）
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文档分割器
from langchain_core.prompts import ChatPromptTemplate  # 提示词模板
from langchain_core.runnables import RunnablePassthrough  # 流程透传
from langchain_core.output_parsers import StrOutputParser  # 输出解析器
 
# ===================== 1. 初始化模型（Ollama 本地模型）=====================
# 嵌入模型：embeddinggemma（用于生成文本向量）
embeddings = OllamaEmbeddings(
    model="embeddinggemma",  # 必须与本地拉取的模型名一致
    base_url="http://localhost:11434"  # Ollama 默认端口（无需修改，除非自定义）
)
 
# 大语言模型：用于生成最终回答（示例用 deepseek-r1:1.5b 等）
llm = OllamaLLM(
    model="deepseek-r1:1.5b",  # 确保已通过 `ollama run deepseek-r1:1.5b` 下载并启动
    temperature=0.1,  # 温度越低，回答越精准（0.1~0.3 适合事实性问答）
    base_url="http://localhost:11434"
)
 
# ===================== 2. 加载并处理文档（核心：分割为小片段）=====================
# 1. 加载本地 TXT 文档（替换为你的文档路径，支持多个文档）
loader = TextLoader("./docs/《三国演义》.txt", encoding="utf-8")  # Windows 下建议显式指定编码，避免默认 gbk 解码失败
documents = loader.load()
 
# 2. 分割文档：将长文档拆分为短片段（避免嵌入时丢失语义，提升检索精度）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # 每个片段最大字符数（根据文档复杂度调整）
    chunk_overlap=200,  # 片段重叠字符数（保持上下文连贯性）
    length_function=len  # 按字符长度计算
)
split_documents = text_splitter.split_documents(documents)
 
# ===================== 3. 构建/加载 Chroma 向量库（核心替换）=====================
# Chroma 配置：本地存储路径、集合名称（可自定义）
CHROMA_DB_PATH = "./chroma_db"  # 向量库本地存储文件夹
COLLECTION_NAME = "knowledge_base"  # 集合名称（类似数据库的表）
 
# 方式1：新建 Chroma 向量库（首次运行，处理文档并写入）
vector_db = Chroma.from_documents(
    documents=split_documents,
    embedding=embeddings,
    persist_directory=CHROMA_DB_PATH,  # 本地持久化路径
    collection_name=COLLECTION_NAME,  # 集合名称（便于多知识库隔离）
    collection_metadata={"hnsw:space": "cosine"}  # 相似度计算方式（余弦相似度，适合嵌入向量）
)
 
 
# # 方式2：加载已保存的 Chroma 向量库（后续运行，跳过文档处理）
# vector_db = Chroma(
#     collection_name=COLLECTION_NAME,
#     embedding_function=embeddings,
#     persist_directory=CHROMA_DB_PATH
# )
 
# ===================== （可选）4. 动态添加新文档到 Chroma（增量更新）=====================
# 场景：后续新增文档，无需重建向量库，直接添加
def add_new_documents(new_doc_paths):
    """
    动态添加新文档到 Chroma 向量库
    new_doc_paths: 新文档路径列表（如 ["new_doc1.txt", "new_doc2.pdf"]）
    """
    for path in new_doc_paths:
        # 加载新文档（根据后缀选择对应的 Loader）
        if path.endswith(".txt"):
            new_loader = TextLoader(path, encoding="utf-8")
        elif path.endswith(".pdf"):
            from langchain_community.document_loaders import PyPDFLoader
            new_loader = PyPDFLoader(path)
        else:
            print(f"不支持的文档格式：{path}")
            continue
 
        new_docs = new_loader.load()
        new_split_docs = text_splitter.split_documents(new_docs)
 
        # 添加到 Chroma 向量库（自动增量更新，无需重建）
        vector_db.add_documents(documents=new_split_docs)
        print(f"已添加文档：{path}")
 
 
# 示例：添加新文档（实际使用时取消注释）
# add_new_documents(["new_document.txt", "tech_manual.pdf"])
 
# ===================== 5. 构建 RAG 检索链（与之前一致）=====================
# 检索器：从 Chroma 中检索最相关的 3 个文档片段
retriever = vector_db.as_retriever(
    search_kwargs={"k": 10}  # k=3 表示取 top3 相关片段
)
 
# 提示词模板：基于检索到的上下文回答，不编造信息
prompt = ChatPromptTemplate.from_template("""
你是专业的问答助手，必须严格基于以下上下文信息回答用户问题。
如果上下文没有相关内容，直接回复"根据知识库，未找到相关答案"，禁止编造信息。
上下文信息：
{context}
用户问题：
{question}
""")
 
# 构建 RAG 链：检索 → 格式化上下文 → 生成回答
rag_chain = (
        {
            # 检索并拼接上下文（将多个文档片段合并为字符串）
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
            "question": RunnablePassthrough()  # 透传用户问题
        }
        | prompt  # 注入提示词和上下文
        | llm  # 本地 LLM 生成回答
        | StrOutputParser()  # 解析输出为字符串
)
 
# ===================== 6. 测试 RAG 效果 =====================
if __name__ == "__main__":
    # 测试问题（基于 《三国演义》.txt 中的内容）
    user_question = "诸葛亮是谁，简单介绍下他的功绩"
 
    # 执行 RAG 链
    result = rag_chain.invoke(user_question)
 
    # 输出结果
    print("=== RAG 回答 ===")
    print(result)
 
    # （可选）查看检索到的相关文档片段（调试用）
    retrieved_docs = retriever.invoke(user_question)
    print("\n=== 检索到的相关文档 ===")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n【片段 {i}】")
        print(doc.page_content)