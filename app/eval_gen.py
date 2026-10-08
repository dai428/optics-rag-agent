# D:\Desktop\rag-demo\app\eval_gen.py
"""生成层评测（RAGAS）—— 补齐「检索好 ≠ 回答好」这一环。

与 eval_rag.py 的分工
---------------------
  eval_rag.py  → **检索层**：只关心「该文档有没有被召回」，指标是 Recall / MRR / NDCG。
                 不需要调用大模型，跑得快，适合每次改检索就回归。
  eval_gen.py  → **生成层**：关心「拿到资料后，回答得对不对、有没有编」，
                 指标来自 RAGAS：faithfulness（忠实度）/ answer_relevancy（答案相关性）/
                 context_precision（上下文精度）。
                 要调大模型，慢且费额度，适合阶段性评测。

为什么两个分开？
  混在一起跑，一次评测要十几分钟，改一行检索参数都要等这么久 —— 没人会愿意跑。
  分开后：检索层秒级回归，生成层阶段验收。

RAGAS 三个指标什么意思（给不熟悉的人）
---------------------------------------
  · faithfulness      ：答案里的每句话，能不能在检索到的资料里找到依据？
                        低 = 幻觉（模型自己编的）→ 这是科研场景最致命的指标。
  · answer_relevancy  ：答案有没有正面回答用户的问题？（不跑题）
  · context_precision ：检索到的资料里，真正有用的占多大比例？（有没有凑数的）

用法
----
    cd D:/Desktop/rag-demo/app
    ../.venv/Scripts/python.exe eval_gen.py                 # 跑全部
    ../.venv/Scripts/python.exe eval_gen.py --limit 5       # 只跑前 5 题（省额度）
    ../.venv/Scripts/python.exe eval_gen.py --no-ragas      # 只采集数据不评分（调试用）
"""

import argparse
import asyncio
import json
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "D:/DevCache/huggingface")
os.environ.setdefault("no_proxy", "*")
os.environ.setdefault("NO_PROXY", "*")

from pathlib import Path

from langchain_core.messages import HumanMessage

import config
import rag_graph as g

ROOT = Path(__file__).parent.parent
EVAL_SET = ROOT / "data" / "eval_set.json"
OUT_DIR = ROOT / "data"


# ── 数据采集：跑图，收集 (问题, 答案, 检索上下文) ──────────────
def collect(limit: int | None = None) -> list[dict]:
    """对评测集每题跑一遍完整图，收集 RAGAS 需要的三元组。

    ⚠️ 为什么不用 graph.invoke 一次拿全部？
       invoke 返回的最终 state 里，retrieved_docs 是**所有轮次累加**的，
       多子问题并行时还会混在一起。评测需要的是「这道题最终用了哪些上下文」，
       所以这里用 stream 逐步收集，并在 respond 节点触发后定格。
       简化起见：直接取最终 state 的 retrieved_docs（去重后就是本题用到的资料）。
    """
    raw = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    cases = raw["cases"]
    if limit:
        cases = cases[:limit]

    print(f"评测集：{len(cases)} 题（{EVAL_SET.name}）\n")
    graph = g.build_graph_with_memory()
    rows = []

    for i, case in enumerate(cases, 1):
        # ⚠️ 字段名是 q（不是 question）；评测集没有标准答案字段（检索集只标了期望来源+关键词）
        q = case["q"]
        t0 = time.time()
        try:
            # ⚠️ thread_id 用纯 ASCII：中文会让 invoke 挂死（见 rag_graph.build_graph 注释）
            state = graph.invoke(
                {"messages": [HumanMessage(content=q)]},
                config={"configurable": {"thread_id": f"eval-gen-{i}"}},
            )
            answer = _last_ai_text(state.get("messages", []))
            docs = state.get("retrieved_docs", [])
            contexts = [d.page_content for d in docs]
            elapsed = time.time() - t0
            print(f"[{i:2d}/{len(cases)}] {elapsed:5.1f}s  资料{len(contexts):2d}条  {q[:40]}")
            rows.append(
                {
                    "question": q,
                    "answer": answer,
                    "contexts": contexts,
                    # 评测集未提供标准答案 → 留空；RAGAS 的 faithfulness / answer_relevancy
                    # 不需要 reference，context_precision 在无 reference 时也退化为「有用上下文占比」。
                    "ground_truth": "",
                    "elapsed": round(elapsed, 1),
                }
            )
        except Exception as e:
            print(f"[{i:2d}/{len(cases)}] ❌ {type(e).__name__}: {e}")
            rows.append(
                {
                    "question": q,
                    "answer": "",
                    "contexts": [],
                    "ground_truth": "",
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    return rows


def _last_ai_text(messages) -> str:
    """取最后一条 AI 消息的文本内容。"""
    for m in reversed(messages):
        if m.__class__.__name__ == "AIMessage" and getattr(m, "content", None):
            c = m.content
            return c if isinstance(c, str) else str(c)
    return ""


# ── RAGAS 评分 ────────────────────────────────────────────────
def score_with_ragas(rows: list[dict]) -> dict:
    """用 RAGAS 对采集到的三元组打分。

    ⚠️ 版本说明：本项目锁 ragas==0.2.15（0.3+ 需要新版 langchain-community，
       与项目的 sunset 版 0.4.2 冲突）。0.2.x 的 API 是：
         evaluate(dataset, metrics, llm=LangchainLLMWrapper(...), embeddings=...)

    ⚠️ 裁判模型说明：RAGAS 需要一个「裁判 LLM」来打分。这里复用项目自己的 AMD 模型
       （通过 LangChain 的 ChatOpenAI 封装，走 OpenAI 兼容接口），不额外申请新 Key。
       注意：裁判模型和被评模型是同一个，属于「自评」，绝对分数仅供参考，
       **横向对比（改前 vs 改后）才有意义** —— 这点写进报告里，不夸大。
    """
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness

    # ⚠️ 为什么只用 faithfulness + answer_relevancy 两个指标？
    #   ragas 0.2.15 里，context_precision / context_recall / answer_correctness
    #   都强制要求数据集含 `reference`（标准答案）列。而本项目的 eval_set.json 是
    #   **检索评测集**（只标了「期望来源 + 关键词」），没有逐题标准答案 —— 硬凑答案
    #   既费时又容易失真，反而污染评测。
    #   好在 faithfulness（忠实度）和 answer_relevancy（答案相关性）**不需要 reference**，
    #   且恰好覆盖科研问答最关键的两点：「有没有编」和「跑不跑题」。
    #   → 若将来补了标准答案，把 context_precision 加回 metrics 列表即可。

    # 裁判模型：复用 AMD 端点
    # ⚠️ 裁判的超时要单独放大（这里给 600s，不套用主流程的 request_timeout）：
    #   faithfulness 要把答案逐句拆开、逐句判断「这句有没有资料支撑」，
    #   一次评分会发起十几次裁判调用。实测用 120s 时会有几条超时 →
    #   该指标直接算成 nan（平均值缺项）。给足时间是唯一可靠解法。
    judge_llm = ChatOpenAI(
        model=config.settings.llm_model,
        api_key=config.settings.api_key,
        base_url=config.settings.base_url,
        temperature=0.0,
        timeout=600,
        max_retries=3,
    )
    ragas_llm = LangchainLLMWrapper(judge_llm)

    # 裁判嵌入：answer_relevancy 需要算「原问题 vs 生成问题」的语义相似度。
    # 复用项目本地 BGE 嵌入（免费、无需 Key）。
    from knowledge_base import get_embeddings

    ragas_emb = LangchainEmbeddingsWrapper(get_embeddings())

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r.get("ground_truth", "") or None,
        )
        for r in rows
        if r.get("answer") and r.get("contexts")
    ]
    skipped = len(rows) - len(samples)
    if skipped:
        print(f"⚠️ {skipped} 题因无答案/无上下文被跳过")

    if not samples:
        print("❌ 没有可用样本，无法评分")
        return {}

    ds = EvaluationDataset(samples=samples)
    print(f"\n开始 RAGAS 评分（{len(samples)} 题 × 2 指标，需要调用裁判模型，请耐心等）...")
    # ⚠️ timeout 是「单个评分任务」的超时（含多次裁判调用），给 900s；
    #    max_workers 压到 2，避免并发太高打满 AMD 共享模型的 16 并发上限（会返回 429）。
    from ragas.run_config import RunConfig

    result = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=RunConfig(timeout=900, max_workers=2),
    )
    # 0.2.x 返回 EvaluationResult，转成普通 dict
    scores = result.scores if hasattr(result, "scores") else dict(result)
    if isinstance(scores, list):  # 有些版本给的是逐样本列表
        import statistics

        agg = {}
        for k in scores[0].keys():
            vals = [s[k] for s in scores if s.get(k) is not None]
            if vals:
                agg[k] = statistics.mean(vals)
        return agg
    return dict(scores) if not isinstance(scores, dict) else scores


# ── 主流程 ────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只评测前 N 题（省额度）")
    ap.add_argument("--no-ragas", action="store_true", help="只采集数据，不做 RAGAS 评分")
    ap.add_argument(
        "--from-raw",
        action="store_true",
        help="跳过采集，直接对已有的 eval_gen_raw.json 评分（改指标时省时间，不用重跑图）",
    )
    args = ap.parse_args()

    print(f"裁判/生成模型：{config.settings.llm_model}")

    if args.from_raw:
        raw_path = OUT_DIR / "eval_gen_raw.json"
        if not raw_path.exists():
            print(f"❌ 找不到 {raw_path}，请先去掉 --from-raw 跑一次采集")
            return
        rows = json.loads(raw_path.read_text(encoding="utf-8"))
        print(f"复用已有数据：{len(rows)} 题（{raw_path.name}）")
    else:
        rows = collect(args.limit)
        # 落盘原始数据（无论评不评分，都留一份，方便离线看）
        raw_path = OUT_DIR / "eval_gen_raw.json"
        raw_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n原始数据已存：{raw_path}")

    if args.no_ragas:
        print("（--no-ragas：跳过评分）")
        return

    try:
        scores = score_with_ragas(rows)
    except ImportError as e:
        print(f"\n❌ 缺少依赖：{e}")
        print("   请先安装：pip install ragas")
        return

    if scores:
        print("\n" + "=" * 60)
        print("  RAGAS 生成层评分")
        print("=" * 60)
        for k, v in scores.items():
            print(f"  {k:24s} {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")

        # 存一份汇总
        summary = OUT_DIR / "eval_gen_report.json"
        summary.write_text(
            json.dumps(
                {"model": config.settings.llm_model, "n": len(rows), "scores": {k: str(v) for k, v in scores.items()}},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n评分已存：{summary}")


if __name__ == "__main__":
    main()
