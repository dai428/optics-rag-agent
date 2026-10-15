# -*- coding: utf-8 -*-
"""BM25 召回贡献探针：验证「混合检索的收益来自重排还是扩召回」。

背景
----
README 宣称「混合检索（向量 + BM25）把 R@5 从 80% 抬到 85%」。
但当前参数下存在一个结构性问题，本脚本用来把它量化出来：

  _RRFRetriever.__init__ 里 dense_k = sparse_k = max(k, settings.dense_k) = 25
  invoke 末尾又截断到 max(dense_k, sparse_k) = 25
  —— 与「纯向量单独返回 25 条」的条数完全相同。

更关键的是权重折算：
  只看 BM25 的文档最高分 0.4/(60+1)  = 0.00656
  只看向量的文档最低分 0.6/(60+25) = 0.00706
  上限 < 下限 ⇒ BM25 独有文档**必然**被截断在 25 名之外。

因此预期结论：融合结果集合 == 纯向量集合，BM25 新增召回 = 0，
hybrid 相对 vector 的提升全部来自「两路重叠文档被 RRF 重新排序」。

用法
----
    cd D:\\Desktop\\rag-demo
    .venv\\Scripts\\python.exe scripts\\probe_bm25_recall.py

输出：逐题比对 + 汇总；结论写入 stdout（不修改任何数据）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# 环境变量在 app 模块顶部已 setdefault，这里无需重复设置
from app.knowledge_base import (  # noqa: E402
    _RRFRetriever,
    _safe_invoke,
    get_bm25_retriever,
    get_vectorstore,
    settings,
)

EVAL_SET = _ROOT / "data" / "eval_set.json"


def _keys(docs) -> set[str]:
    """与 _RRFRetriever 内部一致的去重键，保证同口径比较。"""
    return {f"{d.metadata.get('parent_id')}::{hash(d.page_content)}" for d in docs}


def main() -> int:
    if not EVAL_SET.exists():
        print(f"[跳过] 未找到评测集 {EVAL_SET}")
        print("       公开仓库里只有脱敏样例；真实评测集涉及未发表数据，未入库。")
        return 0

    data = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    cases = data.get("cases", data if isinstance(data, list) else [])
    print(f"评测集: {EVAL_SET.name}  共 {len(cases)} 题")
    print(f"dense_k={settings.dense_k}  sparse_k={settings.sparse_k}  "
          f"rerank_top_n={settings.rerank_top_n}")

    # _RRFRetriever(k) 内部会做 dense_k = sparse_k = max(k, settings.dense_k) = 25
    retr = _RRFRetriever(settings.retrieval_k)
    print(f"融合器内部: dense_k={retr.dense_k}  sparse_k={retr.sparse_k}  "
          f"输出上限={max(retr.dense_k, retr.sparse_k)}")

    s_only = settings.sparse_weight / (60 + 1)
    d_only = settings.dense_weight / (60 + 25)
    print(f"\n权重折算：BM25 独有上限 {s_only:.5f}  vs  向量独有下限 {d_only:.5f}")
    print("→ " + ("BM25 独有文档必然被截断（上限 < 下限）"
                  if s_only < d_only else "BM25 独有文档有机会进入结果"))

    same = 0
    added_total = 0
    print("\n逐题比对（融合结果 vs 纯向量召回，**同为子块粒度**）:")
    print("说明：必须用向量检索器的原始 top-25 子块作对照。")
    print("      不能用 retrieve() —— 它会回溯父块并去重，条数天然更少，")
    print("      与融合结果不同粒度，比出来的『新增』是假的。")
    print(f"{'#':>3} | {'融合n':>5} | {'向量n':>5} | {'BM25独有':>8} | "
          f"{'融合新增':>8} | 相同?")
    print("-" * 62)

    for i, case in enumerate(cases, 1):
        # 评测集字段名是 q（不是 question）
        q = case.get("q") or case.get("question") or case.get("query")
        if not q:
            continue
        fused = retr.invoke(q)
        # 纯向量：直接取向量库 top-25 子块（与融合器同一粒度）
        dense = _safe_invoke(
            get_vectorstore().as_retriever(search_kwargs={"k": retr.dense_k}), q
        )

        fk, dk = _keys(fused), _keys(dense)
        try:
            sparse_hits = _safe_invoke(get_bm25_retriever(retr.sparse_k), q)
            sk = _keys(sparse_hits)
        except Exception:  # noqa: BLE001
            sk = set()
        sparse_only = sk - dk
        added = fk - dk
        added_total += len(added)
        is_same = fk == dk
        same += int(is_same)
        print(f"{i:>3} | {len(fk):>5} | {len(dk):>5} | {len(sparse_only):>8} | "
              f"{len(added):>8} | {'是' if is_same else '否'}")

    n = max(1, len([c for c in cases if c.get("q") or c.get("question") or c.get("query")]))
    print("-" * 62)
    print(f"融合集合 == 纯向量集合 : {same} / {n}")
    print(f"BM25 实际新增召回的文档总数: {added_total}")
    print()
    if added_total == 0:
        print("结论：BM25 在当前参数下【没有参与召回】，只通过 RRF 参与了重复文档的重排。")
        print("      hybrid 相对 vector 的指标提升来自重排，不是召回扩展。")
        print("      修法：融合后取并集（≤ dense_k+sparse_k）再交给重排；")
        print("            或降低 K0、提高 sparse_weight，让 BM25 独有文档不被截断。")
    else:
        print(f"结论：BM25 补充了 {added_total} 条向量未召回的文档，起到了扩召回作用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
