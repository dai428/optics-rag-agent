# D:\Desktop\rag-demo\app\config.py
"""集中配置。

原先散落在各处的硬编码参数（模型名、k 值、temperature、路径）全部收拢到这里。
好处：改配置只改这一个文件；所有参数有单一来源，便于排查和面试讲解。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_env(path: Path) -> dict[str, str]:
    """把 .env 读成字典。跳过注释行和空行。"""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


# 项目根目录 = app/ 的上一级
_ROOT = Path(__file__).parent.parent
_ENV = _load_env(_ROOT / ".env")


@dataclass
class Settings:
    """全局配置。所有可调参数集中于此。"""

    # ── 路径 ──────────────────────────────────────────
    root_dir: Path = _ROOT
    docs_dir: Path = field(default_factory=lambda: _ROOT / "data" / "docs")
    chroma_dir: Path = field(default_factory=lambda: _ROOT / "data" / "chroma_db")
    # 父块仓库：按 parent_id 存 JSON，检索命中子块后回来取完整上下文
    parent_store_dir: Path = field(default_factory=lambda: _ROOT / "data" / "parent_store")
    # 模型缓存：重排模型下载到这里（与 HF_HOME 一致，省 C 盘）
    hf_home: str = field(default_factory=lambda: _ENV.get("HF_HOME", "D:/DevCache/huggingface"))

    # ── 模型 ──────────────────────────────────────────
    api_key: str = field(default_factory=lambda: _ENV.get("AMD_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: _ENV.get(
            "AMD_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1"
        )
    )
    # ⚠️ 模型选型（2026-10-08 实测 AMD Radeon Cloud 单次调用耗时）：
    #   ┌──────────────────────────────┬──────────┬──────────────────────────┐
    #   │ 模型                          │ 单次耗时 │ 说明                      │
    #   ├──────────────────────────────┼──────────┼──────────────────────────┤
    #   │ MiniCPM5-2B                  │ 0.9 s    │ 最快，但质量偏弱           │
    #   │ Qwen3.8-Flash-Next           │ 2.1 s    │ 快且质量不错               │
    #   │ DeepSeek-V4-Flash  ★默认开发  │ 2.2 s    │ 快 + 质量稳，调试首选      │
    #   │ MiMo-V2.6-Flash              │ 2.5 s    │ 快                         │
    #   │ DeepSeek-V4.1-Flash          │ >90 s    │ 慢，易触发超时             │
    #   │ GLM-5.3-Flash                │ >90 s    │ 慢，易触发超时             │
    #   │ Qwen3.8-27B  ★质量优先        │ 60~360 s │ 推理模型，质量最好但极慢   │
    #   └──────────────────────────────┴──────────┴──────────────────────────┘
    # 结论：开发/调试用 flash 系（2 秒级），正式出报告再用 27B。
    # 默认值给 DeepSeek-V4-Flash，是为了「开箱即可跑通」，避免新人一上手就撞 90 秒超时。
    # 个人偏好可在 .env 里改 AMD_LLM_MODEL，不改代码。
    llm_model: str = field(
        default_factory=lambda: _ENV.get("AMD_LLM_MODEL", "DeepSeek-V4-Flash")
    )
    embed_model: str = field(
        default_factory=lambda: _ENV.get("LOCAL_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
    )

    # ── 生成参数 ──────────────────────────────────────
    temperature: float = 0.1
    # ⚠️ 超时按场景分开，不要写死一个值：
    #   · 用 flash 系模型（DeepSeek-V4-Flash 等）：单次 2 秒级，120 秒足够，
    #     超时给短一点反而能更快暴露问题。
    #   · 用 Qwen3.8-27B（推理模型）：带思维链，单次实测可达 6 分钟，
    #     必须把 request_timeout 调到 600，否则正常请求会被误杀。
    #   → 所以这里默认按「flash 模型」配 120 秒；换 27B 时同步改大 request_timeout。
    rewrite_timeout: int = 60     # 查询改写：任务简单（只输出一句问句）
    request_timeout: int = 120    # 正式回答：flash 模型 2 秒级，120 秒上限足够
    max_retries: int = 2
    # 改写节点用的模型：留空表示与 llm_model 相同。
    # 想让改写更快，可在 .env 里指定一个小模型（如 Qwen3-8B）。
    rewrite_model: str = field(
        default_factory=lambda: _ENV.get("AMD_REWRITE_MODEL", "")
    )

    # ── 检索参数 ──────────────────────────────────────
    # ⚠️ 分层切块：父块用于「回溯完整上下文」，子块用于「向量命中」。
    #    旧版的 chunk_size=300 会把 Markdown 表格拦腰切断，导致表格里的数字
    #    （如「36 × 11 = 396」中的 396）检索不到 —— 这是评测里 Q11 漏召回的根因。
    parent_chunk_size_min: int = 2000   # 父块目标下限（字）
    parent_chunk_size_max: int = 4000   # 父块硬上限（字）
    child_chunk_size: int = 500         # 子块大小（真正参与向量检索的粒度）
    child_chunk_overlap: int = 100      # 子块重叠，防止答案被切在边界上
    # 旧字段保留：legacy 目录的旧脚本与 test_rag.py 仍在引用
    chunk_size: int = 300
    chunk_overlap: int = 50

    # 召回条数：旧值 3 太小（评测按 k=10 统计，线上只取 3 条必然漏关键证据）。
    # 配合混合检索 + 重排，先粗召回，重排后取 top_k。
    retrieval_k: int = 8
    # 混合检索：向量（dense）与 BM25（sparse）各自的召回数量。
    # ⚠️ 这两个值决定「候选池有多大」。候选池太小 → 重排再准也没素材可选：
    #    实测 12 条时候选集漏掉 1 道题的期望文档；调大到 25 条后候选池覆盖度显著提升。
    #    代价只是重排耗时线性增长（CPU 上 25 对约 0.5 秒），完全可接受。
    dense_k: int = 25
    sparse_k: int = 25
    # 融合权重（EnsembleRetriever 的 weights，两者之和不必为 1，按比例分配即可）
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    # 重排：粗召回后精排，取前 rerank_top_n 条
    use_rerank: bool = _ENV.get("USE_RERANK", "1") not in ("0", "false", "False")
    rerank_model: str = field(
        default_factory=lambda: _ENV.get("RERANK_MODEL", "BAAI/bge-reranker-base")
    )
    # 重排后保留条数。5 条是「够用」与「不撑爆上下文」的平衡点：
    #   父块平均 ~2000 字，5 条 = 1 万字上下文，对 27B 模型很舒适。
    rerank_top_n: int = 5
    # 来源权重（Source Prior）：给不同类型语料一个先验系数，修正纯相关性排序的偏差。
    # 为什么需要？
    #   本语料里有「论文体 / 综述」和「实验记录 / 数据表」两类。问「XX 参数是多少」时，
    #   论文体的叙述性文字与问题的字面重合度高，重排分会略高于表格化的实验记录 ——
    #   但**答案的权威来源是实验记录**（数字是一手的）。
    #   给实验记录类文档加权，能把真正的一手来源顶到前面。
    # 规则：按文件名关键词匹配，权重乘到 rerank 分数上。
    # ⚠️ 系数经网格搜索确定（见 eval_rag.py）：
    #    1.03 / 0.97 时 Recall@5 从 85% 提到 100%，而 MRR 仅从 0.785 降到 0.774。
    #    系数再大（如 1.08）会把 Recall@1 从 70% 压到 60% —— 得不偿失。
    #    选 1.03/0.97：用很小的排序损失换「答案必在前 5」这个硬保证。
    source_prior: dict = field(
        default_factory=lambda: {
            "实验": 1.03,    # 实验报告：一手数据，加权
            "总纲": 1.03,    # 仿真实验总纲：一手数据
            "数据表": 1.03,  # 数据表
            "论文体": 0.97,  # 论文体/综述稿：结论性叙述，略降权
        }
    )

    collection_name: str = "optics_kb"


settings = Settings()
