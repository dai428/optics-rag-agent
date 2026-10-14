"""嵌入模型与重排模型：单例管理 + 延迟加载。

为什么单独一个模块
------------------
索引构建（app/ingestion）与在线检索（app/knowledge_base）都需要嵌入模型。
如果放在 knowledge_base 里，ingestion 就要反向 import knowledge_base，形成循环依赖。
把「模型加载」这件与业务无关的事抽出来，两边都只依赖本模块。

延迟加载的理由：torch 顶层导入约 10.5 秒。FastAPI 冷启动时不该付这个代价，
所以只有真正用到时才 import。
"""

from __future__ import annotations

import logging

try:
    from .config import settings
except ImportError:  # 扁平方式（cd app && python xxx.py）
    from config import settings

logger = logging.getLogger(__name__)

_embeddings = None
_reranker = None


def get_embeddings():
    """本地嵌入模型（免费、无需 Key），单例复用。"""
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("加载嵌入模型 %s", settings.embed_model)
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_reranker():
    """CrossEncoder 重排模型单例（首次调用会下载，约 1 GB）。

    Bi-Encoder（嵌入模型）把问题和文档分别编码再算余弦，快但抓不住细粒度措辞；
    CrossEncoder 把「问题 + 文档」拼在一起喂进模型直接打分，慢但准得多。
    所以用它做「小候选集（10~25 条）上的精排」。
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        logger.info("加载重排模型 %s（首次需下载，请稍候）", settings.rerank_model)
        _reranker = CrossEncoder(settings.rerank_model, device="cpu", max_length=512)
    return _reranker


def reset_models() -> None:
    """释放模型单例（测试与配置热变更后使用）。"""
    global _embeddings, _reranker
    _embeddings = None
    _reranker = None
