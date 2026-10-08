# D:\Desktop\rag-demo\app\eval_rag.py
"""检索层评测（不花 API Key，纯本地跑）。

评测对象 = 完整检索链路：混合召回（dense + BM25）→ CrossEncoder 重排 → 回溯父块。

指标：
  Recall@k    前 k 条里是否召回了「期望文档」（任一命中即算 1）
  MRR@k       首个正确结果排名的倒数（越靠前越高）
  NDCG@k      二值相关下的排序质量（单个相关项时 = 1/log2(rank+1)）
  关键词命中@k 前 k 条内容里是否出现标准答案关键词

对比模式（--compare）：
  vector  纯向量（旧方案）
  hybrid  混合召回，不重排
  full    混合召回 + 重排（新方案）

用法：
  cd app && ../.venv/Scripts/python.exe eval_rag.py              # 只跑完整链路
  cd app && ../.venv/Scripts/python.exe eval_rag.py --compare    # 三方案对比
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

logging.basicConfig(level=logging.ERROR)

import knowledge_base as kb  # noqa: E402

EVAL_FILE = Path(__file__).parent.parent / "data" / "eval_set.json"
KS = [1, 3, 5, 10]


def _file_of(hit) -> str:
    """从命中结果取文件名（新元数据用 file_name，兼容旧的 source）。"""
    m = hit.metadata
    return m.get("file_name") or os.path.basename(m.get("source", ""))


def _section_of(hit) -> str:
    return hit.metadata.get("section_path", "")


def _make_retriever(mode: str):
    """按模式返回检索函数：query -> [Document]。"""
    if mode == "vector":

        def _f(q):
            return kb.get_retriever(max(KS)).invoke(q)

    elif mode == "hybrid":

        def _f(q):
            return kb.get_hybrid_retriever()(q)

    elif mode == "full":

        def _f(q):
            return kb.search(q, k=max(KS))

    else:
        raise ValueError(f"未知模式：{mode}")
    return _f


def evaluate(mode: str = "full", quiet: bool = False) -> dict:
    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    cases = data["cases"]
    retrieve = _make_retriever(mode)

    rows = []
    for c in cases:
        hits = retrieve(c["q"])
        srcs = [_file_of(h) for h in hits]
        exp = set(c["expected_sources"])
        rank = next((i + 1 for i, s in enumerate(srcs) if s in exp), 0)
        rows.append((c, hits, rank))

    n = len(rows)
    if not quiet:
        print("=" * 66)
        print(f"[{mode}] 检索评测 · {n} 题 · 语料 14 篇 · 索引 67 父块 / 499 子块")
        print("=" * 66)
        print(f"{'k':>2} | {'Recall@k':>9} | {'MRR@k':>7} | {'NDCG@k':>7} | {'关键词命中@k':>12}")
        print("-" * 66)

    summary = {}
    for k in KS:
        rec = sum(1 for _, _, r in rows if 0 < r <= k) / n
        mrr = sum((1 / r if 0 < r <= k else 0) for _, _, r in rows) / n
        ndcg = sum((1 / math.log2(r + 1) if 0 < r <= k else 0) for _, _, r in rows) / n
        kw = sum(
            1
            for c, hits, _ in rows
            if c["keywords"]
            and any(kw in "\n".join(h.page_content for h in hits[:k]) for kw in c["keywords"])
        ) / n
        summary[k] = {"recall": rec, "mrr": mrr, "ndcg": ndcg, "keyword": kw}
        if not quiet:
            print(f"{k:>2} | {rec:>8.1%} | {mrr:>7.3f} | {ndcg:>7.3f} | {kw:>11.1%}")

    if not quiet:
        print("-" * 66)
        print("\n明细（rank = 期望文档首次出现位置，0 = 未命中）：")
        for c, hits, rank in rows:
            flag = "OK  " if rank else "MISS"
            top = _file_of(hits[0]) if hits else "-"
            sec = _section_of(hits[0])[:30] if hits else "-"
            print(f"  [{flag}] Q{c['id']:>2}  rank={rank:<3} 首条={top[:26]:<28} §{sec}")

        miss5 = [c["id"] for c, _, r in rows if not (0 < r <= 5)]
        print(f"\nRecall@5 未达标(rank=0 或 >5): {miss5 if miss5 else '无'}")
    return summary


def compare():
    modes = ["vector", "hybrid", "full"]
    results = {m: evaluate(m, quiet=True) for m in modes}

    print("=" * 80)
    print("三方案对比 · Recall@1 / Recall@5 / Recall@10 / MRR@10 / NDCG@10 / 关键词@5")
    print("=" * 80)
    print(
        f"{'方案':<12} | {'Recall@1':>8} | {'Recall@5':>8} | {'Recall@10':>9} | "
        f"{'MRR@10':>7} | {'NDCG@10':>7} | {'KW@5':>6}"
    )
    print("-" * 80)
    labels = {"vector": "纯向量", "hybrid": "混合·不重排", "full": "混合+重排 ★"}
    for m in modes:
        s = results[m]
        print(
            f"{labels[m]:<12} | {s[1]['recall']:>7.1%} | {s[5]['recall']:>7.1%} | "
            f"{s[10]['recall']:>8.1%} | {s[10]['mrr']:>7.3f} | {s[10]['ndcg']:>7.3f} | "
            f"{s[5]['keyword']:>5.1%}"
        )
    print("-" * 80)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="三方案对比")
    ap.add_argument("--mode", default="full", choices=["vector", "hybrid", "full"])
    args = ap.parse_args()
    if args.compare:
        compare()
    else:
        evaluate(args.mode)
