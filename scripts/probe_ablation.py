# -*- coding: utf-8 -*-
"""消融探针：用**当前代码**逐项关掉一个特性，重建可复现的贡献分解表。

背景
----
README「优化贡献分解」表里的中间步骤数字（0.694 / 0.716 / 0.744 / 0.785）
是分阶段开发时测的，代码后来演进过，未必还能复现。本探针在**同一份代码、
同一套 20 题评测集**上，用「关掉一个特性」的方式重测，给出诚实数字。

⚠️ 只测**互不相同**的配置（2026-10-15 修订）
   上一版测 7 轮，其中两轮是恒等重复，已删除：
     · 「混合 + 来源加权（不重排）」≡「混合召回（不重排）」—— 来源权重只乘在
       重排分数上，关闭重排后 source_prior 不参与任何计算，两轮数字必然全等
       （实测都是 R@1 55% / R@5 85% / MRR 0.6937）。
     · 「重排+加权·池25（当前）」≡「重排·池25·有加权 ★」—— 两者是同一配置。
   现为 5 轮，每轮配置互不相同，省掉约 2 轮（~5 分钟）的无意义计算。

   注意：**表格展开已实现在 chunker.py 内，没有开关，无法作为消融项单独测量**。

⚠️ 关于「看起来卡死」
   CPU 上 CrossEncoder 重排很慢（本机实测约 1.2 秒/对，一轮 20 题约 2~3 分钟）。
   `evaluate(quiet=True)` 中途不打印进度，若 stdout 又是块缓冲，进度会全部压在
   缓冲区里不落盘 —— 表现就是「输出文件停在表头、CPU 却满载」。
   本脚本已给所有 print 加 flush=True，并逐轮打印耗时；命令行请加 `-u`。

用法
----
    cd rag-demo
    .venv/Scripts/python.exe -u scripts/probe_ablation.py

输出
----
    控制台表格 + data/probe_ablation.json
"""
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from app.config import settings  # noqa: E402

eval_rag = importlib.import_module("app.eval_rag")

DEFAULT_PRIOR = {"实验": 1.03, "总纲": 1.03, "数据表": 1.03, "论文体": 0.97}

ROUNDS = 5


def run(idx: int, label: str, mode: str) -> dict:
    """跑一轮评测，mode 必须显式给出（vector / hybrid / full）。"""
    t0 = time.perf_counter()
    out = eval_rag.evaluate(mode, quiet=True)["metrics"]
    m5, m10 = out[5], out[10]
    row = {
        "label": label,
        "mode": mode,
        "rerank": settings.use_rerank,
        "pool": (settings.dense_k, settings.sparse_k),
        "prior_on": bool(settings.source_prior),
        "secs": round(time.perf_counter() - t0, 1),
        "r1": out[1]["recall"],
        "r5": m5["recall"],
        "r10": m10["recall"],
        "mrr": m10["mrr"],
        "ndcg": m10["ndcg"],
        "kw5": m5["keyword"],
    }
    print(
        f"[{idx}/{ROUNDS}] {label:<28} | R@1 {row['r1']:>5.1%} | R@5 {row['r5']:>5.1%} | "
        f"R@10 {row['r10']:>5.1%} | MRR {row['mrr']:.4f} | NDCG {row['ndcg']:.4f} | "
        f"耗时 {row['secs']:>5.1f}s",
        flush=True,
    )
    return row


print("=" * 118, flush=True)
print("当前代码 · 可复现消融（20 题私有评测集）", flush=True)
print("每行配置互不相同；CPU 重排较慢，约 2~3 分钟/轮", flush=True)
print("=" * 118, flush=True)

saved_rerank = settings.use_rerank
saved_dense, saved_sparse = settings.dense_k, settings.sparse_k
saved_prior = dict(settings.source_prior)

rows = []

# ① 基线：纯向量（旧方案，无混合、无重排）
rows.append(run(1, "① 基线 纯向量 chunk=300", "vector"))

# ② 混合召回 RRF，不重排
#    此时来源加权不参与计算（权重只作用于重排分数），故不为它单跑一轮
settings.use_rerank = False
settings.source_prior = {}
rows.append(run(2, "② 混合召回 RRF（不重排）", "hybrid"))

# ③ 重排 + 来源加权，但把召回池缩到 12
settings.use_rerank = True
settings.source_prior = dict(DEFAULT_PRIOR)
settings.dense_k, settings.sparse_k = 12, 12
rows.append(run(3, "③ 重排+加权 · 召回池 12", "full"))

# ④ 重排 + 池 25，但**关掉**来源加权（隔离加权的净效应）
settings.dense_k, settings.sparse_k = 25, 25
settings.source_prior = {}
rows.append(run(4, "④ 重排 · 池 25 · 无来源加权", "full"))

# ⑤ 当前线上配置：重排 + 池 25 + 来源加权 ★
settings.source_prior = dict(DEFAULT_PRIOR)
rows.append(run(5, "⑤ 重排+加权 · 池 25（当前）★", "full"))

# 还原现场
settings.use_rerank = saved_rerank
settings.dense_k, settings.sparse_k = saved_dense, saved_sparse
settings.source_prior = saved_prior

base, hyb, pool12, no_prior, full = rows


def line(name, a, b, key, pct=True):
    va, vb = a[key], b[key]
    if pct:
        print(f"  {name:<24} {va:>7.1%} → {vb:>7.1%}   ({(vb - va) * 100:+.1f} pt)", flush=True)
    else:
        print(f"  {name:<24} {va:>7.4f} → {vb:>7.4f}   ({vb - va:+.4f})", flush=True)


print("=" * 118, flush=True)
print("贡献分解（用于 README / 简历 / 逐字稿统一口径）", flush=True)
print("=" * 118, flush=True)
print("【① 纯向量 → ② 混合召回】", flush=True)
line("Recall@1", base, hyb, "r1")
line("Recall@5", base, hyb, "r5")
line("MRR@10", base, hyb, "mrr", pct=False)
line("NDCG@10", base, hyb, "ndcg", pct=False)
print("【② 混合 → ④ 加重排（同为无来源加权）】", flush=True)
line("Recall@1", hyb, no_prior, "r1")
line("Recall@5", hyb, no_prior, "r5")
line("MRR@10", hyb, no_prior, "mrr", pct=False)
line("NDCG@10", hyb, no_prior, "ndcg", pct=False)
print("【④ 无加权 → ⑤ 有加权（隔离来源加权）】", flush=True)
line("Recall@1", no_prior, full, "r1")
line("Recall@5", no_prior, full, "r5")
line("MRR@10", no_prior, full, "mrr", pct=False)
line("NDCG@10", no_prior, full, "ndcg", pct=False)
print("【③ 池12 → ⑤ 池25（隔离召回池）】", flush=True)
line("Recall@1", pool12, full, "r1")
line("Recall@5", pool12, full, "r5")
line("MRR@10", pool12, full, "mrr", pct=False)
print("【① 纯向量 → ⑤ 最终方案（总收益）】", flush=True)
line("Recall@1", base, full, "r1")
line("Recall@5", base, full, "r5")
line("Recall@10", base, full, "r10")
line("MRR@10", base, full, "mrr", pct=False)
line("NDCG@10", base, full, "ndcg", pct=False)

dst = ROOT / "data" / "probe_ablation.json"
dst.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n结果已写入 {dst}", flush=True)
