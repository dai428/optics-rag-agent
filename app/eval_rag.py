# D:\Desktop\rag-demo\app\eval_rag.py
"""检索层评测（不花 API Key，纯本地跑）。

评测对象 = 完整检索链路：混合召回（dense + BM25）→ CrossEncoder 重排 → 回溯父块。

指标
----
  Recall@k      前 k 条里是否召回了「期望文档」（任一命中即算 1）
  MRR@k         首个正确结果排名的倒数
  NDCG@k        多相关文档下的折损累计增益（正确区分「命中 1 篇」与「命中 3 篇」）
  关键词命中@k   前 k 条内容里是否**全部**出现该题的关键词

⚠️ 关键词口径的变更（2026-10-14）
   原实现用 `any(...)`：只要命中任意一个关键词就算通过。这会让「答错口径」也通过
   （例如新旧两套数字并存时，命中旧值即算对）。现改为 **all**：全部关键词都要出现。
   若某题确实需要「任一即可」，请拆成多道题，而不是放宽判定。

⚠️ 关于调参集与测试集
   source_prior 等系数是在评测集上选出来的，因此**同一套题上的分数必然偏乐观**。
   评测集 JSON 里的 `split` 字段区分 dev / test：报告结论时只应引用 test 集结果。
   用法：`--split test` 只跑标记为 test 的题目。

用法
----
  cd app && ../.venv/Scripts/python.exe eval_rag.py                  # 跑当前语料评测集
  cd app && ../.venv/Scripts/python.exe eval_rag.py --compare        # 三方案对比
  cd app && ../.venv/Scripts/python.exe eval_rag.py --sample         # 用公开示例语料
  cd app && ../.venv/Scripts/python.exe eval_rag.py --split test     # 只看测试集
  cd app && ../.venv/Scripts/python.exe eval_rag.py --json out.json  # 结果落盘
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

logging.basicConfig(level=logging.ERROR)

try:
    from .config import settings
except ImportError:  # 扁平方式
    from config import settings

try:
    from . import knowledge_base as kb
except ImportError:
    import knowledge_base as kb  # noqa: E402

KS = [1, 3, 5, 10]


def _eval_file(sample: bool) -> Path:
    """选择评测集。

    ⚠️ 私有评测集（data/eval_set.json）含未发表论文的实验数值，**不入 Git**，
    只存在于本地。因此公开克隆里默认评测集是不存在的 —— 这时自动回退到
    公开示例评测集，让「clone 后直接跑评测」这件事成立。
    """
    if sample:
        return settings.sample_eval_set_path

    private = settings.eval_set_path
    if private.exists():
        return private
    if settings.sample_eval_set_path.exists():
        print(
            f"提示：未找到私有评测集 {private.name}，"
            f"自动改用公开示例评测集 {settings.sample_eval_set_path.name}"
        )
        return settings.sample_eval_set_path
    return private


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
            return kb.search(q, k=max(KS), fetch_parent=False)

    else:
        raise ValueError(f"未知模式：{mode}")
    return _f


def _keyword_hit(case: dict, hits: list, k: int) -> bool:
    """判断前 k 条内容是否满足该题的关键词要求。

    默认要求**全部**关键词都出现（all），避免「只答对一半也算通过」。
    若某题确实属于「新旧口径任一对即可」（例如同一份材料里先后存在两套评估网格
    规模），必须在评测集里显式写 `keyword_mode: "any"` —— 让放宽判定成为
    一个可见的、需要理由的决定，而不是评测脚本的默认行为。
    """
    mode = case.get("keyword_mode", "all")
    text = "\n".join(h.page_content for h in hits[:k])
    checks = [kw in text for kw in case["keywords"]]
    return any(checks) if mode == "any" else all(checks)


def _ndcg_at_k(ranks: list[int], k: int, relevant_total: int) -> float:
    """多相关文档下的 NDCG@k（二值相关）。

    ranks 是该题所有相关文档的排名（可能为空）。理想排序是把 min(relevant_total, k)
    篇相关文档全部排在最前面，即 DCG = Σ 1/log2(i+1)。
    """
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks if 0 < r <= k)
    ideal_n = min(relevant_total, k)
    if ideal_n == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


def _load_cases(sample: bool, split: str | None) -> tuple[list[dict], dict]:
    path = _eval_file(sample)
    if not path.exists():
        raise SystemExit(f"找不到评测集：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"]
    if split:
        cases = [c for c in cases if (c.get("split") or data.get("meta", {}).get("split")) == split]
    return cases, data.get("meta", {})


def evaluate(mode: str = "full", quiet: bool = False, sample: bool = False,
             split: str | None = None) -> dict:
    cases, meta = _load_cases(sample, split)
    if not cases:
        raise SystemExit("筛选后没有可评测的题目（检查 --split 参数）")
    retrieve = _make_retriever(mode)

    rows = []
    for c in cases:
        hits = retrieve(c["q"])
        srcs = [_file_of(h) for h in hits]
        expected = set(c["expected_sources"])
        # 每个「期望来源」只取它的最佳排名。
        # ⚠️ 不能把同一文件的多个片段各自算作一个相关文档：父块/子块会让同一篇文档
        #    在前 10 条里出现多次，DCG 就会被重复累加而超过理想值（NDCG > 1）。
        ranks = []
        for source in expected:
            positions = [i + 1 for i, s in enumerate(srcs) if s == source]
            if positions:
                ranks.append(min(positions))
        rows.append((c, hits, ranks))

    n = len(rows)
    if not quiet:
        corpus = settings.docs_dir.name
        print("=" * 74)
        print(f"[{mode}] 检索评测 · {n} 题 · 语料 {corpus} · 指标集 {_eval_file(sample).name}")
        if split:
            print(f"         （仅统计 split={split} 的题目）")
        print("=" * 74)
        print(f"{'k':>2} | {'Recall@k':>9} | {'MRR@k':>7} | {'NDCG@k':>7} | {'关键词命中@k':>12}")
        print("-" * 74)

    summary = {}
    for k in KS:
        rec = sum(1 for _, _, ranks in rows if any(0 < r <= k for r in ranks)) / n
        mrr = sum(
            (1 / min(r for r in ranks if 0 < r <= k) if any(0 < r <= k for r in ranks) else 0)
            for _, _, ranks in rows
        ) / n
        ndcg = sum(_ndcg_at_k(ranks, k, len(c["expected_sources"])) for c, _, ranks in rows) / n
        kw = sum(
            1
            for c, hits, _ in rows
            if c["keywords"] and _keyword_hit(c, hits, k)
        ) / n
        summary[k] = {"recall": rec, "mrr": mrr, "ndcg": ndcg, "keyword": kw}
        if not quiet:
            print(f"{k:>2} | {rec:>8.1%} | {mrr:>7.3f} | {ndcg:>7.3f} | {kw:>11.1%}")

    if not quiet:
        print("-" * 74)
        print("\n明细（rank = 期望文档首次出现位置，0 = 未命中）：")
        for c, hits, ranks in rows:
            rank = min(ranks) if ranks else 0
            flag = "OK  " if rank else "MISS"
            top = _file_of(hits[0]) if hits else "-"
            sec = _section_of(hits[0])[:30] if hits else "-"
            print(f"  [{flag}] Q{c['id']:>2}  rank={rank:<3} 首条={top[:26]:<28} §{sec}")

        miss5 = [c["id"] for c, _, ranks in rows if not any(0 < r <= 5 for r in ranks)]
        print(f"\nRecall@5 未达标(rank=0 或 >5): {miss5 if miss5 else '无'}")
    return {"n": n, "metrics": summary}


def compare(sample: bool = False, split: str | None = None) -> dict:
    modes = ["vector", "hybrid", "full"]
    results = {m: evaluate(m, quiet=True, sample=sample, split=split) for m in modes}

    print("=" * 84)
    print("三方案对比 · Recall@1 / Recall@5 / Recall@10 / MRR@10 / NDCG@10 / 关键词@5")
    print("=" * 84)
    print(
        f"{'方案':<12} | {'Recall@1':>8} | {'Recall@5':>8} | {'Recall@10':>9} | "
        f"{'MRR@10':>7} | {'NDCG@10':>7} | {'KW@5':>6}"
    )
    print("-" * 84)
    labels = {"vector": "纯向量", "hybrid": "混合·不重排", "full": "混合+重排 ★"}
    for m in modes:
        s = results[m]["metrics"]
        print(
            f"{labels[m]:<12} | {s[1]['recall']:>7.1%} | {s[5]['recall']:>7.1%} | "
            f"{s[10]['recall']:>8.1%} | {s[10]['mrr']:>7.3f} | {s[10]['ndcg']:>7.3f} | "
            f"{s[5]['keyword']:>5.1%}"
        )
    print("-" * 84)

    # 同时给出「推荐方案」的扁平指标，便于 CI 门禁与报告直接取用，
    # 不必再记住 JSON 的嵌套层级（曾经因为层级搞错导致门禁读不到数据）。
    return {
        "recommended": "full",
        "modes": results,
        "metrics": results["full"]["metrics"],
        "n": results["full"]["n"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="三方案对比")
    ap.add_argument("--mode", default="full", choices=["vector", "hybrid", "full"])
    ap.add_argument("--sample", action="store_true", help="使用公开示例语料评测集")
    ap.add_argument("--split", default=None, help="只评测指定 split（dev/test）")
    ap.add_argument("--json", default=None, help="把结果写入指定 JSON 文件")
    args = ap.parse_args()

    if args.compare:
        out = compare(sample=args.sample, split=args.split)
    else:
        out = evaluate(args.mode, sample=args.sample, split=args.split)

    if args.json:
        Path(args.json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n结果已写入 {args.json}")
