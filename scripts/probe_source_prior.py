"""来源加权消融探针：把「来源加权」关掉，测出它的真实代价。

背景
----
项目文档里曾同时存在两个互相矛盾的代价数字：
    README     ：MRR 0.785 → 0.783（−0.002）
    config.py  ：MRR 0.785 → 0.774（−0.011）
本探针把 source_prior 置空（等价于所有权重 =1.0），在**同一份代码、同一套评测集**
上重跑检索评测，钉死真实数字，避免文档继续自相矛盾。

用法
----
    cd rag-demo
    .venv/Scripts/python.exe scripts/probe_source_prior.py

输出
----
    控制台表格 + data/probe_source_prior.json
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


def _metrics() -> dict:
    m = eval_rag.evaluate("full", quiet=True)["metrics"]
    return {k: {
        "recall": m[k]["recall"], "mrr": m[k]["mrr"],
        "ndcg": m[k]["ndcg"], "keyword": m[k]["keyword"],
    } for k in (1, 3, 5, 10)}


print("=" * 76)
print("来源加权消融探针 · full 模式 · 当前代码")
print("=" * 76)

print("\n[对照 A] source_prior = {}  →  所有权重 = 1.0（等价于关闭来源加权）")
settings.source_prior = {}
off = _metrics()
print(f"  R@1={off[1]['recall']:.1%}  R@5={off[5]['recall']:.1%}  R@10={off[10]['recall']:.1%}")
print(f"  MRR@10={off[10]['mrr']:.4f}  NDCG@10={off[10]['ndcg']:.4f}  KW@5={off[5]['keyword']:.1%}")

print(f"\n[对照 B] source_prior = {DEFAULT_PRIOR}（当前默认）")
settings.source_prior = dict(DEFAULT_PRIOR)
on = _metrics()
print(f"  R@1={on[1]['recall']:.1%}  R@5={on[5]['recall']:.1%}  R@10={on[10]['recall']:.1%}")
print(f"  MRR@10={on[10]['mrr']:.4f}  NDCG@10={on[10]['ndcg']:.4f}  KW@5={on[5]['keyword']:.1%}")

print("\n" + "=" * 76)
print("来源加权的真实代价（供文档/简历统一口径）")
print("=" * 76)
print(f"  Recall@1 : {off[1]['recall']:.1%} → {on[1]['recall']:.1%}  "
      f"({(on[1]['recall'] - off[1]['recall']) * 100:+.1f} pt)")
print(f"  Recall@5 : {off[5]['recall']:.1%} → {on[5]['recall']:.1%}  "
      f"({(on[5]['recall'] - off[5]['recall']) * 100:+.1f} pt)")
print(f"  MRR@10   : {off[10]['mrr']:.4f} → {on[10]['mrr']:.4f}  "
      f"({on[10]['mrr'] - off[10]['mrr']:+.4f})")
print(f"  NDCG@10  : {off[10]['ndcg']:.4f} → {on[10]['ndcg']:.4f}  "
      f"({on[10]['ndcg'] - off[10]['ndcg']:+.4f})")

dst = ROOT / "data" / "probe_source_prior.json"
dst.write_text(
    json.dumps({"off": off, "on": on}, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"\n结果已写入 {dst}")
