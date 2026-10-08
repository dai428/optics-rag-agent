# D:\Desktop\rag-demo\app\knowledge_base.py
"""知识库构建（父子分层切块 → 本地 BGE 嵌入 → Chroma 持久化 → 混合检索 + 重排）

数据流
------
  data/docs 下的语料
      ↓  DocumentChunker（Markdown 标题切父块 → 字符切子块）
  父块 → data/parent_store/*.json（按 parent_id 回查，不参与向量检索）
  子块 → 本地 BGE 嵌入 → Chroma（真正参与检索的粒度）
      ↓
  检索：dense（向量）+ sparse（BM25）→ EnsembleRetriever 融合 → CrossEncoder 精排
      ↓
  回溯父块 → 拿完整上下文喂给 LLM

对外接口
--------
  get_retriever(k)        向量检索器（兼容旧调用）
  get_hybrid_retriever(k) 混合检索器（dense + BM25）
  search(query, k)        带重排的完整检索（推荐入口）
  get_parent_store()      父块仓库（回溯上下文用）
  corpus_info()           知识库概况（供 /documents）

配置来源：app/config.py（不再硬编码模型名与参数）
"""

import logging
import os
from pathlib import Path

# ── 环境自举：不依赖外部 shell / IDE 配置 ──────────────────────────────
# HF 走国内镜像 + 缓存到 D 盘；no_proxy=* 绕过已失效的系统代理
# （否则 huggingface_hub → requests → Windows 注册表代理 会卡死下载）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "D:/DevCache/huggingface")
os.environ.setdefault("no_proxy", "*")
os.environ.setdefault("NO_PROXY", "*")

try:  # 包方式：app.knowledge_base
    from .config import settings
    from .chunker import DocumentChunker, ParentStore
except ImportError:  # 扁平方式：cd app && python knowledge_base.py
    from config import settings
    from chunker import DocumentChunker, ParentStore

logger = logging.getLogger(__name__)

# 说明：重型依赖（torch / sentence_transformers / chromadb）一律**延迟导入**，
# 只在真正用到时才加载，避免拖慢 FastAPI 冷启动（torch 顶层导入约 10.5s）。

_embeddings = None
_vs = None
_parent_store = None
_bm25_retriever = None
_reranker = None

# 支持的语料类型
# ⚠️ 坑：pathlib 的 "**/*" 只匹配「至少一层子目录」下的文件，**不匹配 docs 根目录下的文件**。
#    所以必须再加一条顶层的 "*.md" 等，否则往 data/docs/ 根目录扔文件会「凭空消失」。
_PATTERNS = ("*.md", "*.txt", "*.csv", "**/*.md", "**/*.txt", "**/*.csv")


def get_embeddings():
    """本地嵌入模型（免费、无需 Key），单例复用。"""
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings  # 延迟导入（连带 torch，重）

        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _iter_source_files():
    """遍历 data/docs 下的所有语料文件（去重）。返回 [(绝对路径, 相对路径), ...]。"""
    seen: set[str] = set()  # 去重：顶层的 "*.md" 与递归的 "**/*.md" 会重叠匹配子目录文件
    out = []
    for pattern in _PATTERNS:
        for f in sorted(settings.docs_dir.glob(pattern)):
            key = str(f)
            if key in seen:
                continue
            seen.add(key)
            out.append((f, f.relative_to(settings.docs_dir).as_posix()))
    return out


def _load_docs():
    """读取语料文件（不切块）。返回 [(Document, 相对路径), ...]。"""
    from langchain_community.document_loaders import TextLoader

    out = []
    for path, rel in _iter_source_files():
        try:
            loaded = TextLoader(str(path), encoding="utf-8").load()
            for d in loaded:
                out.append((d, rel))
        except Exception as e:  # noqa: BLE001 —— 单个文件失败不应中断整体建库
            logger.warning("读取失败，已跳过 %s：%s", path.name, e)
    return out


def _split_all() -> tuple[list, list]:
    """把语料全部切成父块 + 子块。返回 (parent_pairs, children)。"""
    chunker = DocumentChunker(
        min_parent_size=settings.parent_chunk_size_min,
        max_parent_size=settings.parent_chunk_size_max,
        child_chunk_size=settings.child_chunk_size,
        child_chunk_overlap=settings.child_chunk_overlap,
    )
    all_parents, all_children = [], []
    for doc, rel in _load_docs():
        file_name = Path(rel).name
        parents, children = chunker.split_document(doc.page_content, file_name, rel)
        all_parents.extend(parents)
        all_children.extend(children)
        logger.debug(
            "切块 %s：父块 %d / 子块 %d", file_name, len(parents), len(children)
        )
    return all_parents, all_children


def get_parent_store() -> ParentStore:
    """父块仓库单例。"""
    global _parent_store
    if _parent_store is None:
        _parent_store = ParentStore(settings.parent_store_dir)
    return _parent_store


def build_vectorstore(force: bool = False):
    """构建（或复用）Chroma 向量库。force=True 时重建。

    重建时会同时刷新父块仓库（两者必须同源，否则 parent_id 会对不上）。
    """
    from langchain_chroma import Chroma  # 延迟导入（连带 chromadb）

    emb = get_embeddings()

    # 已有库且非强制重建 → 直接复用
    if settings.chroma_dir.exists() and not force:
        vs = Chroma(
            collection_name=settings.collection_name,
            persist_directory=str(settings.chroma_dir),
            embedding_function=emb,
        )
        try:
            if vs._collection.count() > 0:
                logger.info("复用已有向量库：%d 个子块", vs._collection.count())
                return vs
        except Exception:  # noqa: BLE001 —— 空库或异常 → 走重建
            pass

    # ── 重建：先清空旧库与旧父块仓库，避免新旧 parent_id 混在一起 ──
    if settings.chroma_dir.exists():
        try:
            old = Chroma(
                collection_name=settings.collection_name,
                persist_directory=str(settings.chroma_dir),
                embedding_function=emb,
            )
            old.delete_collection()
            logger.info("已删除旧的 Chroma collection：%s", settings.collection_name)
        except Exception as e:  # noqa: BLE001
            logger.warning("删除旧 collection 失败（继续重建）：%s", e)
    store = get_parent_store()
    store.clear()

    parents, children = _split_all()
    store.save_many(parents)

    vs = Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=emb,
    )
    if children:
        # 分批写入：一次性提交 800+ 文档时，Chroma 的单次请求体过大容易超时
        batch = 200
        for i in range(0, len(children), batch):
            vs.add_documents(children[i : i + batch])
            logger.info("写入进度：%d / %d", min(i + batch, len(children)), len(children))
        logger.info("向量库构建完成：父块 %d / 子块 %d", len(parents), len(children))
    else:
        logger.warning("语料为空，向量库未写入任何片段。请检查 %s", settings.docs_dir)
    return vs


def get_vectorstore(force: bool = False):
    """取向量库单例。"""
    global _vs
    if _vs is None or force:
        _vs = build_vectorstore(force=force)
    return _vs


# ══════════════════════════════════════════════════════════════════
#  检索：向量（兼容旧接口）
# ══════════════════════════════════════════════════════════════════

def get_retriever(k: int | None = None):
    """纯向量检索器（保留旧接口，legacy 脚本与 test_rag.py 仍在使用）。"""
    k = k or settings.retrieval_k
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


# ══════════════════════════════════════════════════════════════════
#  检索：混合（dense + BM25）
# ══════════════════════════════════════════════════════════════════

def _get_all_children() -> list:
    """从 Chroma 把所有子块取出来（BM25 需要全量语料建索引）。"""
    vs = get_vectorstore()
    data = vs._collection.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document

    docs = []
    for content, meta in zip(data["documents"], data["metadatas"]):
        docs.append(Document(page_content=content, metadata=meta or {}))
    return docs


def get_bm25_retriever(k: int | None = None):
    """BM25 关键词检索器（中文用 jieba 分词）。

    为什么需要它？
      向量检索懂「语义」，但对**数字、缩写、专有名词**不敏感。
      例：Q11 问「总共多少点」，答案是 396；向量算出来的是「这段在讲点的数量」
      这种模糊语义，而 BM25 直接匹配 "396" 这个 token，一击即中。
      你的语料里表格密布（条件数、标准差、种子号），BM25 是不可缺的另一条腿。
    """
    global _bm25_retriever
    k = k or settings.sparse_k
    if _bm25_retriever is None:
        from langchain_community.retrievers import BM25Retriever

        docs = _get_all_children()
        if not docs:
            return None
        retr = BM25Retriever.from_documents(docs, preprocess_func=_tokenize_zh)
        retr.k = k
        _bm25_retriever = retr
    _bm25_retriever.k = k
    return _bm25_retriever


def _tokenize_zh(text: str) -> list[str]:
    """中文分词函数（供 BM25 使用）。

    ⚠️ 关键：BM25 默认按空格切词，中文整句会被当成一个 token → 完全失效。
    用 jieba 的搜索引擎模式（cut_for_search）——它会把长词再切出短词
    （「偏振测量」→「偏振」「测量」「偏振测量」），提高召回。
    """
    import jieba

    tokens = [t.strip() for t in jieba.cut_for_search(text) if t.strip()]
    # 数字、英文缩写单独保留（表格里的 396、1.3889、CN、BCPN 全靠它们命中）
    return tokens


class _RRFRetriever:
    """把 RRF 融合逻辑包装成「有 invoke 方法的检索器对象」。

    为什么不直接返回函数？
      LangGraph 的 retrieve 节点、以及 LangChain 生态里的 `create_retriever_tool`
      都要求传入的对象有 `.invoke(query)` 方法。返回裸函数会在调用处报
      `'function' object has no attribute 'invoke'`。
    """

    def __init__(self, k: int):
        self.k = k
        self.dense_k = max(k, settings.dense_k)
        self.sparse_k = max(k, settings.sparse_k)

    def invoke(self, query: str, config=None) -> list:
        dense = get_vectorstore().as_retriever(search_kwargs={"k": self.dense_k})
        sparse = get_bm25_retriever(self.sparse_k)

        ranked_lists = [(_safe_invoke(dense, query), settings.dense_weight)]
        if sparse is not None:
            ranked_lists.append((_safe_invoke(sparse, query), settings.sparse_weight))
        else:
            logger.warning("BM25 语料为空，退化为纯向量检索")

        K0 = 60  # RRF 原论文常数：平滑作用，防止第 1 名分数压倒性领先
        scores: dict[str, float] = {}
        store_map: dict[str, object] = {}
        for docs, weight in ranked_lists:
            for rank, doc in enumerate(docs, start=1):
                key = _doc_key(doc)
                store_map[key] = doc
                scores[key] = scores.get(key, 0.0) + weight / (K0 + rank)

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        out = []
        for key, sc in ordered[: max(self.dense_k, self.sparse_k)]:
            d = store_map[key]
            d.metadata["rrf_score"] = round(sc, 6)
            out.append(d)
        return out

    # 兼容 LangChain 的 Runnable 调用习惯（有些地方会直接 retriever(query)）
    def __call__(self, query: str, config=None) -> list:
        return self.invoke(query, config=config)


def get_hybrid_retriever(k: int | None = None) -> _RRFRetriever:
    """混合检索器：向量 + BM25，用 RRF（倒数排名融合）合并。

    为什么不用 LangChain 的 EnsembleRetriever？
      本项目的 langchain-community 0.4.2 已 sunset，该类被移除。自己实现
      反而更透明：融合逻辑摆在明面上，面试时能逐行讲清楚。

    RRF 算法
    --------
        score(doc) = Σ  weight_i / (k0 + rank_i(doc))
        k0 = 60（原论文常数）

    为什么用「排名」而不是「原始分」融合？
      向量给的是余弦相似度（0~1），BM25 给的是 TF-IDF 分（0~几十），
      两者量纲完全不可比。直接加权求和需要先归一化，而归一化又会受
      离群值影响。只看排名则天然免疫 —— 无论分量纲怎么变，排名关系稳定。
    """
    return _RRFRetriever(k or settings.retrieval_k)


def _safe_invoke(retriever, query: str) -> list:
    """调用检索器，异常时返回空列表（单条腿瘸了不该拖垮整条链路）。"""
    try:
        return retriever.invoke(query)
    except Exception as e:  # noqa: BLE001
        logger.warning("检索器调用失败：%s", e)
        return []


def _doc_key(doc) -> str:
    """文档去重键：父块 id + 内容指纹。

    为什么不用整段内容做 key？答案可能很长，做字典键浪费内存。
    用「父块 id + 内容前 80 字」既保证唯一性，又轻量。
    """
    pid = doc.metadata.get("parent_id", "")
    return f"{pid}::{hash(doc.page_content)}"


# ══════════════════════════════════════════════════════════════════
#  检索：CrossEncoder 重排
# ══════════════════════════════════════════════════════════════════

def get_reranker():
    """CrossEncoder 重排模型单例（首次调用会从 hf-mirror 下载，约 1 GB）。

    Bi-Encoder（就是嵌入模型）vs CrossEncoder 的区别：
      Bi-Encoder  ：问题和文档**分别**编码成向量，再算余弦。快，但两者
                    从未「见过面」，细粒度的措辞匹配抓不住。
      CrossEncoder：把「问题 + 文档」拼成一句话**一起**喂进模型，直接输出
                    相关性分数。慢，但准得多 —— 适合在小候选集（10~20 条）上做精排。
    这就是「先粗召回、再精排」两阶段检索的原理。
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        logger.info("加载重排模型 %s（首次需下载，请稍候）", settings.rerank_model)
        _reranker = CrossEncoder(settings.rerank_model, device="cpu", max_length=512)
    return _reranker


def _source_weight(doc) -> float:
    """按文件名给文档一个来源先验权重（见 config.source_prior 的说明）。

    匹配规则：文件名包含关键词就乘对应系数；命中多个关键词时取乘积
    （如「补充实验_C档汇总」同时含「实验」→ 加权）。
    """
    name = doc.metadata.get("file_name", "") or doc.metadata.get("source", "")
    w = 1.0
    for kw, factor in settings.source_prior.items():
        if kw in name:
            w *= factor
    return w


def rerank(query: str, docs: list, top_n: int | None = None, aggregate: bool = True) -> list:
    """对候选文档做 CrossEncoder 精排 + 来源加权，返回前 top_n 条。

    打分公式：
        final = CrossEncoder(question, content) × source_weight(file_name)

    参数
    ----
    aggregate : 是否做「父块聚合」（同一父块只保留最高分子块）。
                为什么需要？同一父块会切出 5~8 个子块，若它们同时挤进候选，
                会占满 top_n 名额，让「其他文档」完全没机会露脸。
    """
    if not docs:
        return []
    top_n = top_n or settings.rerank_top_n
    model = get_reranker()
    pairs = [(query, d.page_content) for d in docs]
    scores = model.predict(pairs)

    scored = []
    for doc, raw in zip(docs, scores):
        final = float(raw) * _source_weight(doc)
        doc.metadata["rerank_score"] = round(final, 4)
        doc.metadata["rerank_raw"] = round(float(raw), 4)
        scored.append((doc, final))

    if aggregate:
        # 父块聚合：同一父块只保留最高分子块
        best: dict[str, tuple] = {}
        for doc, s in scored:
            key = doc.metadata.get("parent_id") or str(hash(doc.page_content))
            if key not in best or s > best[key][1]:
                best[key] = (doc, s)
        items = sorted(best.values(), key=lambda x: x[1], reverse=True)
    else:
        items = sorted(scored, key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in items[:top_n]]


# ══════════════════════════════════════════════════════════════════
#  推荐入口：检索 + 重排 + 回溯父块
# ══════════════════════════════════════════════════════════════════

def search(query: str, k: int | None = None, fetch_parent: bool = True) -> list:
    """完整检索入口：混合召回 → 重排 → （可选）回溯父块。

    参数
    ----
    query : 检索问句
    k     : 最终返回条数（重排后的 top_n），默认 settings.retrieval_k
    fetch_parent : 是否把命中的子块替换为其父块内容

    返回
    ----
    Document 列表。metadata 中带 file_name / section_path / rerank_score，
    供回答节点标注引用来源。
    """
    k = k or settings.rerank_top_n
    retriever = get_hybrid_retriever()
    candidates = retriever.invoke(query)

    if settings.use_rerank:
        try:
            hits = rerank(query, candidates, top_n=k)
        except Exception as e:  # noqa: BLE001 —— 重排失败不能拖垮检索，退化为融合结果
            logger.warning("重排失败，退化为融合排序：%s", e)
            hits = candidates[:k]
    else:
        hits = candidates[:k]

    if not fetch_parent:
        return hits

    # ── 回溯父块：用子块命中，但喂给 LLM 的是父块（上下文完整） ──
    store = get_parent_store()
    out, seen = [], set()
    for h in hits:
        pid = h.metadata.get("parent_id")
        if not pid or pid in seen:
            # 没有 parent_id（理论上不该发生）→ 原样返回子块
            if not pid:
                out.append(h)
            continue
        seen.add(pid)
        data = store.load(pid)
        if not data:
            out.append(h)
            continue
        from langchain_core.documents import Document

        out.append(
            Document(
                page_content=data["page_content"],
                metadata={
                    **data["metadata"],
                    "rerank_score": h.metadata.get("rerank_score"),
                    "matched_child": h.page_content[:120],  # 保留命中的那一小段，便于调试
                },
            )
        )
    return out


def corpus_info() -> dict:
    """知识库概况（/documents 接口用，无需 API Key）。"""
    files = [rel for _, rel in _iter_source_files()]
    vs = get_vectorstore()
    try:
        n = vs._collection.count()
    except Exception:  # noqa: BLE001
        n = -1
    return {
        "documents": sorted(files),
        "chunks": n,
        "parent_chunks": get_parent_store().count(),
        "embed_model": settings.embed_model,
        "retrieval": {
            "mode": "hybrid(dense+bm25)",
            "dense_weight": settings.dense_weight,
            "sparse_weight": settings.sparse_weight,
            "rerank_model": settings.rerank_model if settings.use_rerank else None,
            "retrieval_k": settings.retrieval_k,
        },
        "chroma_dir": str(settings.chroma_dir),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    info = corpus_info()
    print("[知识库]", {k: v for k, v in info.items() if k != "documents"})
    print(f"[知识库] 文件数 = {len(info['documents'])}")
    hits = search("评估网格用了多少个波长点和温度点？总共多少点？")
    print(f"\n检索命中 {len(hits)} 条：")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {h.metadata.get('file_name')} | {h.metadata.get('section_path')}")
        print(f"      score={h.metadata.get('rerank_score')}  len={len(h.page_content)}")
