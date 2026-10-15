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
import math
import os
from dataclasses import dataclass
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
    from .embeddings import get_embeddings, get_reranker
    from .ingestion import indexer as _indexer
    from .ingestion import loaders as _loaders
except ImportError:  # 扁平方式：cd app && python knowledge_base.py
    from config import settings
    from chunker import DocumentChunker, ParentStore
    from embeddings import get_embeddings, get_reranker

    # 扁平方式下 app/ 不是包，需要把 ingestion 目录挂进来
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).parent))
    from ingestion import indexer as _indexer
    from ingestion import loaders as _loaders

logger = logging.getLogger(__name__)

# 说明：重型依赖（torch / sentence_transformers / chromadb）一律**延迟导入**，
# 只在真正用到时才加载，避免拖慢 FastAPI 冷启动（torch 顶层导入约 10.5s）。

_vs = None
_parent_store = None
_bm25_retriever = None

# 支持的语料类型（与 app/ingestion/loaders.py 的 SUPPORTED_SUFFIXES 保持一致）
_PATTERNS = settings.corpus_patterns


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
    """读取语料文件（不切块）。返回 [(ParsedDocument, ...)]。

    统一走 app/ingestion 的解析器，从而支持 PDF / DOCX / CSV，
    并与索引构建使用完全相同的解析路径（避免「建库用 A 解析、评测用 B 解析」）。
    """
    parse_bytes = _loaders.parse_bytes

    out = []
    for path, rel in _iter_source_files():
        try:
            out.append(parse_bytes(path.read_bytes(), rel))
        except Exception as e:  # noqa: BLE001 —— 单个文件失败不应中断整体建库
            logger.warning("读取失败，已跳过 %s：%s", path.name, e)
    return out


def _split_all() -> tuple[list, list]:
    """把语料全部切成父块 + 子块。返回 (parent_pairs, children)。

    实际切块逻辑复用 ingestion.indexer，保证手动建库与 API 建库结果一致。
    """
    _chunk_documents = _indexer._chunk_documents

    documents = _load_docs()
    return _chunk_documents(settings, documents)


def get_parent_store() -> ParentStore:
    """父块仓库单例。"""
    global _parent_store
    if _parent_store is None:
        _parent_store = ParentStore(settings.parent_store_dir)
    return _parent_store


def index_status() -> dict:
    """索引与语料的一致性状态，供 /ready 与前端展示。"""
    compute_diff = _indexer.compute_diff
    current_manifest = _indexer.current_manifest

    manifest = current_manifest(settings)
    try:
        diff, failures = compute_diff(settings)
    except Exception as e:  # noqa: BLE001
        return {
            "status": "unknown",
            "detail": f"无法扫描语料：{type(e).__name__}: {e}",
            "stale": True,
        }

    if manifest is None:
        return {
            "status": "missing",
            "detail": "尚未建立索引，请先上传文档或运行建库命令",
            "stale": True,
            "failed_files": failures,
        }

    compatible, reason = manifest.is_compatible(
        settings.embed_model,
        {
            "parent_min": settings.parent_chunk_size_min,
            "parent_max": settings.parent_chunk_size_max,
            "child_size": settings.child_chunk_size,
            "child_overlap": settings.child_chunk_overlap,
        },
    )
    if not compatible:
        return {
            "status": "incompatible",
            "detail": reason,
            "stale": True,
            "built_at": manifest.built_at,
        }

    stale = bool(diff and diff.needs_rebuild)
    return {
        "status": "stale" if stale else "ok",
        "detail": diff.summary() if diff else "",
        "stale": stale,
        "built_at": manifest.built_at,
        "files": len(manifest.files),
        "child_count": manifest.child_count,
        "parent_count": manifest.parent_count,
        "diff": (
            {
                "added": diff.added,
                "changed": diff.changed,
                "removed": diff.removed,
            }
            if diff
            else None
        ),
        "failed_files": failures,
    }


def build_vectorstore(force: bool = False):
    """构建（或复用）向量库。

    走 ingestion 的「staging → 原子切换」流程：构建失败时旧索引完好无损，
    不会再出现「删了旧库、新库没建成」的中间态。

    force=True 时无条件全量重建；否则先看清单判断语料是否变化。
    """
    build_index = _indexer.build_index

    report = build_index(settings, force=force, incremental=True)
    if not report.ok:
        logger.warning("索引构建未完成：%s", "；".join(report.warnings) or "未知原因")
    if report.warnings:
        for warning in report.warnings:
            logger.info("建库提示：%s", warning)

    # 无论本次是否重建，都要重新读一次正式集合（可能已被切换）
    global _vs
    _vs = None
    return get_vectorstore()


def get_vectorstore(force: bool = False):
    """取向量库单例。

    与旧实现的关键区别：**不再无条件复用**。每次首次获取时校验清单，
    若语料已变化、嵌入模型或切块参数已变更，则先重建再返回，
    避免「查的是旧索引却毫无提示」。
    """
    global _vs
    if _vs is not None and not force:
        return _vs

    from langchain_chroma import Chroma

    build_index = _indexer.build_index
    compute_diff = _indexer.compute_diff
    current_manifest = _indexer.current_manifest

    emb = get_embeddings()

    if force:
        build_index(settings, force=True, incremental=False)
    else:
        manifest = current_manifest(settings)
        need_build = manifest is None
        if manifest is not None:
            compatible, _ = manifest.is_compatible(
                settings.embed_model,
                {
                    "parent_min": settings.parent_chunk_size_min,
                    "parent_max": settings.parent_chunk_size_max,
                    "child_size": settings.child_chunk_size,
                    "child_overlap": settings.child_chunk_overlap,
                },
            )
            if not compatible:
                need_build = True
            else:
                diff, _ = compute_diff(settings)
                need_build = bool(diff and diff.needs_rebuild)
        if need_build:
            logger.info("检测到索引缺失或语料变化，正在自动重建索引…")
            build_index(settings, force=manifest is None, incremental=True)

    vs = Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=emb,
    )
    try:
        count = vs._collection.count()
        logger.info("向量库就绪：%d 个子块", count)
        if count == 0:
            logger.warning(
                "向量库为空。请确认 %s 下有 .md/.txt/.csv/.pdf/.docx 文件并完成建库。",
                settings.docs_dir,
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("读取向量库计数失败：%s", e)
    _vs = vs
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
    # 数字与英文缩写必须原样保留（表格里的 1.732、396、LCVR 这类 token
    # 一旦被切碎就再也命中不了，而它们往往正是答案本身）
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

def _source_weight(doc) -> float:
    """按文件名给文档一个来源先验权重（见 config.source_prior 的说明）。

    匹配规则：文件名包含关键词就乘对应系数；命中多个关键词时取乘积
    （如「实验记录_温度标定」同时含「实验」与「数据表」→ 两次加权）。
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
        final = CrossEncoder_logit(question, content) + log(source_weight(file_name))

    CrossEncoder 输出可能是负 logit。若直接相乘，负数乘 1.03 会变得更小，
    “加权”反而成为降权；转到 log-prior 后，无论原始 logit 正负，权重方向都正确。

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
        final = float(raw) + math.log(_source_weight(doc))
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

@dataclass
class RetrievalResult:
    """一次检索的完整结果，含「证据是否充分」的判断。

    为什么要把 abstain 单独拿出来
    ----------------------------
    RAG 最危险的失败不是「答不出」，而是「没有相关资料时仍然自信地编」。
    重排分数天然提供了一个判据：所有候选的最高分都低于阈值时，说明知识库里
    确实没有相关内容，此时应当明确拒答，而不是把无关片段硬塞给模型。
    """

    query: str
    docs: list
    top_score: float | None
    abstain: bool
    reason: str = ""
    candidate_count: int = 0

    def citations(self) -> list[dict]:
        """把命中文档转成结构化引用，供前端渲染与程序核验。

        ⚠️ 拒答时返回空列表：证据不足的片段不该出现在引用卡片里，
           否则前端会显示「引用 N 条」，而回答本身是「没找到证据」——
           自相矛盾，比不显示更误导。
        """
        if self.abstain:
            return []
        return build_citations(self.docs)


def _top_score(docs: list) -> float | None:
    scores = [
        d.metadata.get("rerank_score")
        for d in docs
        if d.metadata.get("rerank_score") is not None
    ]
    return max(scores) if scores else None


def retrieve(query: str, k: int | None = None) -> RetrievalResult:
    """检索并给出证据充分性判断（不做父块回溯，保留命中的子块原文）。

    与 search() 的分工：
      retrieve() → 评测与拒答判断用，返回子块（精确命中位置 + 原始分数）
      search()   → 生成回答用，返回父块（上下文完整）
    """
    k = k or settings.rerank_top_n
    candidates = get_hybrid_retriever().invoke(query)

    if settings.use_rerank:
        try:
            hits = rerank(query, candidates, top_n=k)
        except Exception as e:  # noqa: BLE001 —— 重排失败不能拖垮检索
            logger.warning("重排失败，退化为融合排序：%s", e)
            hits = candidates[:k]
    else:
        hits = candidates[:k]

    score = _top_score(hits)
    threshold = settings.abstain_threshold
    abstain = False
    reason = ""
    if not hits:
        abstain = True
        reason = "知识库中没有检索到任何相关内容"
    elif threshold is not None and score is not None and score < threshold:
        abstain = True
        reason = (
            f"最高相关度 {score:.2f} 低于拒答阈值 {threshold:.2f}，"
            "判断知识库中没有足够证据回答该问题"
        )

    return RetrievalResult(
        query=query,
        docs=hits,
        top_score=score,
        abstain=abstain,
        reason=reason,
        candidate_count=len(candidates),
    )


def search(query: str, k: int | None = None, fetch_parent: bool = True) -> list:
    """完整检索入口：混合召回 → 重排 → （可选）回溯父块。

    参数
    ----
    query : 检索问句
    k     : 最终返回条数（重排后的 top_n），默认 settings.rerank_top_n
    fetch_parent : 是否把命中的子块替换为其父块内容

    返回
    ----
    Document 列表。metadata 中带 file_name / section_path / rerank_score，
    供回答节点标注引用来源。
    """
    result = retrieve(query, k=k)
    hits = result.docs

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

        # 命中的子块可能位于父块中部或尾部。若只截取父块开头，答案会被切掉，
        # 所以这里把父块内容重新排序：先放命中窗口附近的上下文，再补其余部分。
        content = _window_around(data["page_content"], h.page_content)
        out.append(
            Document(
                page_content=content,
                metadata={
                    **data["metadata"],
                    "rerank_score": h.metadata.get("rerank_score"),
                    "rerank_raw": h.metadata.get("rerank_raw"),
                    "matched_child": h.page_content[:200],
                    "page": h.metadata.get("page") or data["metadata"].get("page"),
                },
            )
        )
    return out


def _window_around(parent_text: str, child_text: str, budget: int = 3000) -> str:
    """以命中片段为中心截取父块内容，保证证据不被截断。

    原实现固定取父块前 3000 字。若答案在父块尾部（长表格的最后一节很常见），
    命中的子块进去了、答案却被截掉，模型只能答「资料未涉及」。
    这里改为：先定位命中片段在父块中的位置，围绕它取窗口。
    """
    if len(parent_text) <= budget:
        return parent_text

    needle = child_text[:120]
    index = parent_text.find(needle) if needle else -1
    if index < 0:
        return parent_text[:budget] + "\n…（片段过长已截断）"

    start = max(0, index - budget // 3)
    end = min(len(parent_text), start + budget)
    start = max(0, end - budget)

    prefix = "…（前文省略）\n" if start > 0 else ""
    suffix = "\n…（后文省略）" if end < len(parent_text) else ""
    return prefix + parent_text[start:end] + suffix


def build_citations(docs: list) -> list[dict]:
    """把命中文档转成结构化引用列表。

    每条引用都带 source / section / page / chunk_id / quote / score，
    使前端可以定位原文，也使「引用是否支持结论」可以被程序检查。
    """
    strip_page_markers = _loaders.strip_page_markers

    citations: list[dict] = []
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        quote_source = meta.get("matched_child") or doc.page_content
        quote = strip_page_markers(str(quote_source)).replace("\n", " ").strip()
        citations.append(
            {
                "index": index,
                "source": meta.get("file_name") or meta.get("source") or "未知来源",
                "rel_path": meta.get("source", ""),
                "section": meta.get("section_path", ""),
                "page": meta.get("page"),
                "chunk_id": meta.get("parent_id", ""),
                "file_type": meta.get("file_type", ""),
                "score": meta.get("rerank_score"),
                "raw_score": meta.get("rerank_raw"),
                "quote": quote[:300],
            }
        )
    return citations


def corpus_info() -> dict:
    """知识库概况（/documents 接口用，无需 API Key）。"""
    files = [rel for _, rel in _iter_source_files()]
    vs = get_vectorstore()
    try:
        n = vs._collection.count()
    except Exception:  # noqa: BLE001
        n = -1
    status = index_status()
    return {
        "documents": sorted(files),
        "document_count": len(files),
        "chunks": n,
        "parent_chunks": get_parent_store().count(),
        "embed_model": settings.embed_model,
        "retrieval": {
            "mode": "hybrid(dense+bm25)",
            "dense_weight": settings.dense_weight,
            "sparse_weight": settings.sparse_weight,
            "rerank_model": settings.rerank_model if settings.use_rerank else None,
            "retrieval_k": settings.retrieval_k,
            "abstain_threshold": settings.abstain_threshold,
        },
        "index": status,
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
