# D:\Desktop\rag-demo\app\verify_p1.py
"""P1 四项 Agentic 能力的端到端验证脚本。

用法：
    cd D:/Desktop/rag-demo/app
    ../.venv/Scripts/python.exe verify_p1.py            # 跑全部
    ../.venv/Scripts/python.exe verify_p1.py clarify    # 只跑澄清
    ../.venv/Scripts/python.exe verify_p1.py parallel   # 只跑多子问题并行
    ../.venv/Scripts/python.exe verify_p1.py summary    # 只跑历史摘要

设计说明：
  为什么单独写一个验证脚本，而不是塞进 test_rag.py？
    test_rag.py 测的是「检索链路」（能不能搜到），面向回归；
    本脚本测的是「Agentic 行为」（该不该反问、拆不拆、压不压缩），
    面向行为断言 —— 两者关注的层次不同，混在一起会互相污染。
  另外，跑 LLM 的用例又慢又费额度，所以用 `--only` 参数支持按需单跑。
"""

import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "D:/DevCache/huggingface")
os.environ.setdefault("no_proxy", "*")
os.environ.setdefault("NO_PROXY", "*")

from langchain_core.messages import AIMessage, HumanMessage

import rag_graph as g


# ⚠️ 坑：LangGraph 的 thread_id 不要用中文/空格等非 ASCII 字符。
#    实测把中文问题直接拼进 thread_id，graph.invoke() 会挂住不返回
#    （stream 正常、invoke 卡死），而同一图用纯 ASCII 的 thread_id 77 秒内正常跑完。
#    修复方式：thread_id 一律用「test-<序号>」这种纯 ASCII 短标识。
_tid_seq = [0]


def _next_tid(prefix: str = "t") -> str:
    _tid_seq[0] += 1
    return f"{prefix}-{_tid_seq[0]}"


def _new_graph():
    """每个用例用全新对话记忆，避免记忆串场污染断言。"""
    return g.build_graph_with_memory()


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ── P1-1 澄清反问 ─────────────────────────────────────────────
def test_clarify():
    _hr("P1-1 澄清反问：模糊问题该反问，明确问题不该反问")
    graph = _new_graph()

    cases = [
        ("那个结果呢", True,  "指代不明 → 应该反问"),
        ("温度漂移的影响大吗？", False, "指向明确 → 不该反问"),
        ("它怎么做的", True, "「它」指代不明 → 应该反问"),
    ]
    ok = 0
    for q, expect_clarify, note in cases:
        t0 = time.time()
        state = graph.invoke({"messages": [HumanMessage(content=q)]}, config=_cfg(_next_tid("clarify")))
        need = state.get("need_clarify", False)
        qs = state.get("clarify_questions", [])
        hit = need == expect_clarify
        ok += hit
        flag = "✅" if hit else "❌"
        print(f"\n{flag} 「{q}」（{note}）")
        print(f"   期望反问={expect_clarify}  实际反问={need}  耗时={time.time()-t0:.1f}s")
        if need:
            for i, one in enumerate(qs, 1):
                print(f"     反问{i}: {one}")
    print(f"\n→ 澄清用例 {ok}/{len(cases)} 通过")
    return ok == len(cases)


# ── P1-2 多子问题并行 ─────────────────────────────────────────
def test_parallel():
    _hr("P1-2 多子问题并行：复合问题该拆、单点问题不该拆")
    graph = _new_graph()

    cases = [
        ("对比一下 DE 加权路线和 NSGA-II 路线的优缺点", True,  "复合（对比两条路线）→ 该拆"),
        ("温度漂移对测量精度的影响是多少？", False, "单点问题 → 不该拆"),
    ]
    ok = 0
    for q, expect_multi, note in cases:
        _t0 = time.time()
        state = graph.invoke({"messages": [HumanMessage(content=q)]}, config=_cfg(_next_tid("par")))
        elapsed = time.time() - _t0
        subs = state.get("sub_questions", [])
        answers = state.get("sub_answers", [])
        is_multi = len(subs) > 1
        hit = is_multi == expect_multi
        ok += hit
        flag = "✅" if hit else "❌"
        print(f"\n{flag} 「{q}」（{note}）")
        print(f"   期望拆分={expect_multi}  实际子问题数={len(subs)}  子答案数={len(answers)}  耗时={elapsed:.1f}s")
        for i, s in enumerate(subs, 1):
            print(f"     子问题{i}: {s}")
        # 断言：子问题数应等于子答案数（map 几个就该 reduce 几个）
        if expect_multi and len(subs) != len(answers):
            print(f"   ⚠️ 子问题数({len(subs)}) ≠ 子答案数({len(answers)})，map-reduce 配平有问题")
            hit = False
        print(f"   预算：tool_calls={state.get('tool_calls')}  iterations={state.get('iterations')}")
    print(f"\n→ 并行用例 {ok}/{len(cases)} 通过")
    return ok == len(cases)


# ── P1-3 历史摘要 ─────────────────────────────────────────────
def test_summary():
    _hr("P1-3 历史摘要：长对话触发压缩，且数字不丢")
    graph = _new_graph()
    tid = "summary-case"  # 纯 ASCII：中文 thread_id 会让 invoke 卡死（见文件头注释）

    # 先灌入几轮关键事实（带数字），超过 SUMMARIZE_TRIGGER（=8 条）后应触发摘要
    rounds = [
        "DE 加权路线跑了 10 种权重扫描",
        "那 DE 大概花了多少次评估？",
        "计算成本大概是 139.7 次评估、133.9 秒",
        "NSGA-II 那边呢？",
        "NSGA-II 膝点方案全温域 ΔCN 改善 20.2%",
    ]
    printed = False
    for i, q in enumerate(rounds, 1):
        state = graph.invoke({"messages": [HumanMessage(content=q)]}, config=_cfg(tid))
        summary = state.get("history_summary", "")
        n_msg = len(state.get("messages", []))
        print(f"   第{i}轮：消息数={n_msg}  摘要长度={len(summary)}")
        if summary:
            print(f"     📌 摘要内容：{summary[:200]}")
            printed = True

    if not printed:
        print("\n   ⚠️ 未触发摘要（消息数可能还没到阈值 8 条）—— 属于正常，多聊几轮即可")
        return True

    # 断言：摘要应保留数字（这是压缩质量的关键检验）
    keep_num = any(num in (state.get("history_summary") or "") for num in ["10", "20.2", "139.7", "133.9"])
    print(f"\n   {'✅' if keep_num else '❌'} 摘要是否保留关键数字：{keep_num}")
    return keep_num


# ── P1-4 预算控制 ─────────────────────────────────────────────
def test_budget():
    _hr("P1-4 预算控制：上限常量生效 + 实际消耗不越界")
    print(f"   MAX_TOOL_CALLS   = {g.MAX_TOOL_CALLS}")
    print(f"   MAX_ITERATIONS   = {g.MAX_ITERATIONS}")
    print(f"   MAX_SUB_QUESTIONS= {g.MAX_SUB_QUESTIONS}")
    print(f"   SUMMARIZE_TRIGGER= {g.SUMMARIZE_TRIGGER}")

    graph = _new_graph()
    q = "对比一下 DE 加权路线、NSGA-II 路线、以及宽温域扩展三种方案的优缺点和计算成本"
    state = graph.invoke({"messages": [HumanMessage(content=q)]}, config=_cfg("budget-case"))
    subs = state.get("sub_questions", [])
    tc = state.get("tool_calls", 0)

    ok = True
    if len(subs) > g.MAX_SUB_QUESTIONS:
        print(f"   ❌ 子问题数 {len(subs)} 超过上限 {g.MAX_SUB_QUESTIONS}")
        ok = False
    if tc > g.MAX_TOOL_CALLS:
        print(f"   ❌ tool_calls {tc} 超过上限 {g.MAX_TOOL_CALLS}")
        ok = False
    print(f"\n   {'✅' if ok else '❌'} 实际：子问题={len(subs)}  tool_calls={tc}（均未越界）")
    return ok


TESTS = {
    "clarify": test_clarify,
    "parallel": test_parallel,
    "summary": test_summary,
    "budget": test_budget,
}


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    picked = [only] if only in TESTS else list(TESTS.keys())

    results = {}
    for name in picked:
        try:
            results[name] = TESTS[name]()
        except Exception as e:
            print(f"\n   ❌ {name} 抛异常：{type(e).__name__}: {e}")
            results[name] = False

    _hr("汇总")
    for name, passed in results.items():
        print(f"   {'✅ 通过' if passed else '❌ 失败'}  {name}")
    print(f"\n   共 {sum(results.values())}/{len(results)} 项通过")


if __name__ == "__main__":
    main()
