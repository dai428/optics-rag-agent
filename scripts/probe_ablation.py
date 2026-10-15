# -*- coding: utf-8 -*-
"""消融探针：用**当前代码**逐项关掉一个特性，重建可复现的贡献分解表。

背景
----
README「优化贡献分解」表里的中间步骤数字（0.694 / 0.716 / 0.744 / 0.785）
是分阶段开发时测的，代码后来演进过，未必还能复现。本探针在**同一份代码、
同一套 20 题评测集**上，用「关掉一个特性」的方式重测，给出诚实数字。

注意：**表格展开已实现在 chunker.py 内，没有开关，无法作为消融项单独测量**。
因此 README 不再为它列数字（旧版给过一个不可复现的中间值，已移除）。

用法
----
    cd rag-demo
    .venv/Scripts/python.exe scripts/probe_ablation.py

输出
----
    控制台表格 + data/probe_ablation.json
"""
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from app.config import settings  # noqa: E402

eval_rag = importlib.import_module("app.eval_rag")

DEFAULT_PRIOR = {"实验": 1.03, "总纲": 1.03, "数据表": 1.03, "论文体": 0.97}


def run(label: str, mode: str = "full") -> dict:
    out = eval_rag.evaluate(mode, quiet=True)["metrics"]
    m5, m10 = out[5], out[10]
    row = {
        "label": label,
        "r1": out[1]["recall"],
        "r5": m5["recall"],
        "r10": m10["recall"],
        "mrr": m10["mrr"],
        "ndcg": m10["ndcg"],
        "kw5": m5["keyword"],
    }
    print(f"{label:<34} | R@1 {row['r1']:>5.1%} | R@5 {row['r5']:>5.1%} | "
          f"R@10 {row['r10']:>5.1%} | MRR {row['mrr']:.4f} | NDCG {row['ndcg']:.4f}")
    return row


print("=" * 110)
print("当前代码 · 可复现消融（20 题私有评测集 · full 模式）")
print("=" * 110)
rows = []

# ① 基线：纯向量（旧方案）
rows.append(run("① 基线 纯向量 chunk=300", mode="vector"))

# ② 混合召回，不重排
saved_rerank = settings.use_rerank
settings.use_rerank = False
settings.source_prior = {}
rows.append(run("② 混合召回 RRF（不重排·不加权）", mode="hybrid"))
settings.source_prior = dict(DEFAULT_PRIOR)
rows.append(run("③ 混合 + 来源加权（不重排）", mode="hybrid"))

# ③ 召回池 12（重排 + 加权，候选池缩小）
saved_dense, saved_sparse = settings.dense_k, settings.sparse_k
settings.use_rerank = True
settings.dense_k, settings.sparse_k = 12, 12
rows.append(run("④ 重排+加权 · 但召回池 12"))
settings.dense_k, settings.sparse_k = saved_dense, saved_sparse
rows.append(run("⑤ 重排+加权 · 召回池 25（当前）"))

# ⑥ 关掉来源加权（当前召回池）
settings.source_prior = {}
rows.append(run("⑥ 重排 · 召回池 25 · 无来源加权"))
settings.source_prior = dict(DEFAULT_PRIOR)
rows.append(run("⑦ 重排 · 召回池 25 · 有来源加权 ★"))

settings.use_rerank = saved_rerank

print("=" * 110)
print("关键差值（用于 README / 简历 / 逐字稿统一口径）")
print("=" * 110)
d = {r["label"]: r for r in rows}
base = d["① 基线 纯向量 chunk=300"]
full = d["⑦ 重排 · 召回池 25 · 有来源加权 ★"]
no_prior = d["⑥ 重排 · 召回池 25 · 无来源加权"]
pool12 = d["④ 重排+加权 · 但召回池 12"]
hyb = d["② 混合召回 RRF（不重排·不加权）"]

def line(name, a, b, key, pct=True):
    va, vb = a[key], b[key]
    if pct:
        print(f"  {name:<26} {va:>7.1%} → {vb:>7.1%}  ({(vb - va) * 100:+.1f} pt)")
    else:
        print(f"  {name:<26} {va:>7.4f} → {vb:>7.4f}  ({vb - va:+.4f})")

print("【基线 → 最终】")
line("Recall@1", base, full, "r1")
line("Recall@5", base, full, "r5")
line("Recall@10", base, full, "r10")
line("MRR@10", base, full, "mrr", pct=False)
line("NDCG@10", base, full, "ndcg", pct=False)
print("【来源加权的代价（⑥ → ⑦）】")
line("Recall@1", no_prior, full, "r1")
line("Recall@5", no_prior, full, "r5")
line("MRR@10", no_prior, full, "mrr", pct=False)
line("NDCG@10", no_prior, full, "ndcg", pct=False)
print("【召回池 12 → 25（④ → ⑦）】")
line("Recall@5", pool12, full, "r5")
line("MRR@10", pool12, full, "mrr", pct=False)
print("【重排的收益（③ → ⑥，同为无加权）】")
line("Recall@5", d["③ 混合 + 来源加权（不重排）"], no_prior, "r5")

dst = ROOT / "data" / "probe_ablation.json"
dst.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n结果已写入 {dst}")
