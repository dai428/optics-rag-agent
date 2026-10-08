"""
RAG 链路「离线验证版」——不需要 Ollama、不需要任何 API Key。

作用：让你在没装大模型的情况下，也能把整条 RAG 流水线跑通、看清每一步。
原理：用「极简的本地假嵌入」+「假 LLM」替代真实模型，验证：
    文档加载 → 切片 → 向量化 → 存储 → 检索 → 拼上下文 → 生成

想跑真实版本，装好 Ollama 后运行 testOllamaRAG.py 即可。
"""
import re
import math
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore

# ============ 1. 假嵌入：把中文按「字/词」散列成向量（仅用于演示原理）============
class FakeHashEmbeddings(Embeddings):
    """用字符哈希做嵌入，使相同关键词的文本在向量空间中更接近。
    真实项目请换成 OllamaEmbeddings / OpenAIEmbeddings。"""
    DIM = 256

    def _vec(self, text: str):
        v = [0.0] * self.DIM
        # 中文按字切，英文/数字按词切
        tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())
        for t in tokens:
            v[hash(t) % self.DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


# ============ 2. 假 LLM：把检索到的上下文回显出来，证明链路通了 ============
class EchoLLM(LLM):
    """不做真实生成，只把 prompt 里的检索结果摘要回显。
    真实项目请换成 OllamaLLM(model='deepseek-r1:1.5b')。"""

    @property
    def _llm_type(self):
        return "echo"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        # 从 prompt 中提取「上下文信息」段落，模拟"基于上下文回答"
        m = re.search(r"上下文信息：(.*?)用户问题：", prompt, re.S)
        ctx = m.group(1).strip() if m else "(未捕获到上下文)"
        first = ctx.replace("\n", " ")[:160]
        return f"[演示回答] 根据知识库检索到的内容，最相关的段落开头是：{first}..."


# ============ 3. 加载文档并切片 ============
loader = TextLoader("./docs/偏振测量基础知识.txt", encoding="utf-8")
documents = loader.load()
print(f"[步骤1] 加载文档: {len(documents)} 篇, 共 {sum(len(d.page_content) for d in documents)} 字符")

splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30, length_function=len)
chunks = splitter.split_documents(documents)
print(f"[步骤2] 切片完成: {len(chunks)} 个片段")

# ============ 4. 建向量库 ============
embeddings = FakeHashEmbeddings()
vector_db = InMemoryVectorStore(embedding=embeddings)
vector_db.add_documents(chunks)
print(f"[步骤3] 向量库构建完成: {len(chunks)} 条向量已入库 (维度 {FakeHashEmbeddings.DIM})")

# ============ 5. 构建 RAG 链 ============
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template("""
你是专业的问答助手，必须严格基于以下上下文信息回答用户问题。
如果上下文没有相关内容，直接回复"根据知识库，未找到相关答案"，禁止编造信息。
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
    | EchoLLM()
    | StrOutputParser()
)

# ============ 6. 运行测试 ============
if __name__ == "__main__":
    questions = [
        "LCVR 是什么？它相比传统旋转波片有什么优势？",
        "斯托克斯参量怎么描述偏振光？",
        "偏振测量的误差来源有哪些？",
    ]
    for q in questions:
        print("\n" + "=" * 62)
        print(f"【提问】{q}")
        hits = retriever.invoke(q)
        print(f"【检索】命中 {len(hits)} 个片段，最相关片段预览：")
        print("   " + hits[0].page_content.replace("\n", " ")[:100] + "...")
        print(f"【生成】{rag_chain.invoke(q)}")
    print("\n" + "=" * 62)
    print("✅ 整条 RAG 链路验证通过：加载→切片→向量化→检索→拼上下文→生成")
    print("下一步：装好 Ollama 后运行 testOllamaRAG.py，替换为真实大模型。")
