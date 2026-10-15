"""Phase 5 测试：拒答（证据不足时不得编造）的行为契约。

这个契约最容易在重构中被悄悄破坏：阈值量纲变了、状态字段没写回、
或者流式路径忘了透传 —— 三种情况都会让「无证据时仍然自信作答」重新出现，
而那是科研场景最不可接受的失败模式。
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app import rag_graph
from app.config import settings
from app.knowledge_base import RetrievalResult, build_citations


def _doc(score: float, name: str = "sample.md", text: str = "证据内容 1.732") -> Document:
    return Document(
        page_content=text,
        metadata={
            "file_name": name,
            "section_path": "示例章节",
            "parent_id": f"{name}_p0",
            "rerank_score": score,
            "rerank_raw": score,
            "page": 3,
        },
    )


class RetrievalAbstentionTests(unittest.TestCase):
    """检索层的拒答判定。"""

    def test_abstains_below_threshold(self):
        low = RetrievalResult(query="q", docs=[_doc(0.03)], top_score=0.03, abstain=True,
                              reason="低于阈值")
        self.assertTrue(low.abstain)
        self.assertIn("低于阈值", low.reason)

    def test_answers_above_threshold(self):
        high = RetrievalResult(query="q", docs=[_doc(0.97)], top_score=0.97, abstain=False)
        self.assertFalse(high.abstain)

    def test_threshold_is_probability_scale(self):
        """阈值必须是 0~1 的概率量纲。

        bge-reranker-base 输出 sigmoid 概率而非 logit。曾把阈值设成 -6.0
        （误以为是 logit 量纲），结果拒答永不触发 —— 用这个断言把量纲锁住。
        """
        threshold = settings.abstain_threshold
        self.assertIsNotNone(threshold)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)

    def test_citations_carry_verifiable_fields(self):
        cites = build_citations([_doc(0.95)])
        self.assertEqual(len(cites), 1)
        cite = cites[0]
        for field in ("source", "section", "chunk_id", "quote", "score", "index"):
            self.assertIn(field, cite)
        self.assertEqual(cite["source"], "sample.md")
        self.assertEqual(cite["chunk_id"], "sample.md_p0")
        self.assertTrue(cite["quote"])


class GraphAbstentionTests(unittest.TestCase):
    """图层的拒答行为：证据不足时必须走拒答，而不是让模型自由发挥。"""

    def _run(self, retrieval: RetrievalResult):
        """跑一次图，并只记录「真正生成答案」的模型调用。

        ⚠️ 不能笼统统计所有模型调用：改写、澄清检查、问题拆分都会调用模型，
        它们属于辅助步骤，与「是否基于证据作答」无关。这里按提示词内容区分，
        只统计子问题作答 / 汇总这两类生成调用。
        """
        generation_calls: list[str] = []

        class _FakeLLM:
            def invoke(self, prompt, config=None):
                system = ""
                if isinstance(prompt, list) and prompt:
                    system = getattr(prompt[0], "content", "") or ""
                if "基于【检索到的资料】回答【子问题】" in system:
                    generation_calls.append("sub_answer")
                    return type("R", (), {"content": "模型编造的内容"})()
                if "回答汇总器" in system:
                    generation_calls.append("aggregate")
                    return type("R", (), {"content": "模型编造的汇总"})()
                # 其余为辅助步骤：改写返回原句、澄清返回「清晰」、拆分返回原问题
                if "提问质量审查器" in system:
                    return type("R", (), {"content": "清晰"})()
                if "问题拆解器" in system:
                    return type("R", (), {"content": "一个措辞具体但没有资料的问题？"})()
                if "对话摘要器" in system:
                    return type("R", (), {"content": "摘要"})()
                return type("R", (), {"content": "一个措辞具体但没有资料的问题？"})()

        def fake_invoke_llm(llm, prompt, config, attempts=2):
            system = ""
            if isinstance(prompt, list) and prompt:
                system = getattr(prompt[0], "content", "") or ""
            if "基于【检索到的资料】回答【子问题】" in system:
                generation_calls.append("sub_answer")
                return "模型编造的内容"
            if "回答汇总器" in system:
                generation_calls.append("aggregate")
                return "模型编造的汇总"
            return "辅助结果"

        with patch.object(rag_graph, "_llm", return_value=_FakeLLM()), patch.object(
            rag_graph, "_invoke_llm", side_effect=fake_invoke_llm
        ), patch("app.knowledge_base.retrieve", return_value=retrieval), patch(
            "app.knowledge_base.search", return_value=retrieval.docs
        ):
            graph = rag_graph.build_graph_with_memory()
            state = graph.invoke(
                {"messages": [HumanMessage(content="一个措辞具体但没有资料的问题？")]},
                config={"configurable": {"thread_id": "abstain-contract"}},
            )
        return state, generation_calls

    def test_abstains_without_calling_model(self):
        """证据不足时不应发生「生成答案」的调用 —— 发生了就说明拒答没生效。"""
        retrieval = RetrievalResult(
            query="q", docs=[_doc(0.03)], top_score=0.03, abstain=True,
            reason="最高相关度 0.03 低于拒答阈值 0.40",
        )
        state, generation_calls = self._run(retrieval)

        self.assertTrue(state.get("abstained"))
        self.assertIn("0.03", state.get("abstain_reason", ""))
        self.assertEqual(
            generation_calls, [], f"证据不足却调用了生成：{generation_calls}"
        )
        answer = state["messages"][-1].content
        self.assertIn("没有找到", answer)
        self.assertNotIn("模型编造的内容", answer)

    def test_answers_when_evidence_sufficient(self):
        retrieval = RetrievalResult(
            query="q", docs=[_doc(0.95)], top_score=0.95, abstain=False
        )
        state, generation_calls = self._run(retrieval)

        self.assertFalse(state.get("abstained"))
        self.assertTrue(generation_calls, "证据充分时应当发生生成调用")


class VerifyNumberIntegrationTests(unittest.TestCase):
    """数字核验接入图之后的行为。"""

    def test_unsupported_numbers_are_flagged(self):
        from app.tools.optics import verify_numbers

        answer = "根据资料，条件数下界是 1.732，最优测量次数是 42。"
        evidence = ["条件数的理论下界是 1.732。"]
        result = verify_numbers(answer, evidence)

        self.assertIn("42", result["unsupported"])
        self.assertNotIn("1.732", result["unsupported"])
        self.assertLess(result["supported_ratio"], 1.0)


class AbstainReducerTests(unittest.TestCase):
    """abstained / abstain_reason 的 reducer 语义。

    这两个字段没有 reducer 时，多子问题并行分支同时写会抛 InvalidUpdateError，
    对外表现是「知识库里没有的多子问题提问 → HTTP 500」。
    这组用例把这个契约钉死在单元层，避免回归。
    """

    def test_or_bool_keeps_true_on_any_true(self):
        from app.state import or_bool

        self.assertTrue(or_bool(False, True))
        self.assertTrue(or_bool(True, True))
        self.assertFalse(or_bool(False, False))
        # 空历史（初次进入）不得被误判为拒答
        self.assertFalse(or_bool(False, False))

    def test_or_bool_false_is_a_vote_not_a_reset(self):
        """False 是「我这支没拒答」的合法否定票，不能抹掉同伴的 True。

        ⚠️ 这条曾经设计错：把 False 当重置信号后，reducer 的合并结果
           取决于分支写入顺序（update([True, False]) → False），
           而并行顺序由调度决定 —— 就是一个偶现、难复现的拒答丢失 bug。
        """
        from app.state import or_bool

        self.assertTrue(or_bool(True, False))

    def test_or_bool_none_resets_for_new_turn(self):
        """None 是唯一的重置哨兵：新一轮开始时由 begin_turn 单独写入。"""
        from app.state import or_bool

        self.assertFalse(or_bool(True, None))
        self.assertFalse(or_bool(False, None))

    def test_reducer_survives_parallel_writes_in_one_step(self):
        """直接用 LangGraph 的真实 channel 验证「同一 step 多次写入」不抛错。

        这是 500 的直接机制：BinaryOperatorAggregate 会在一个 step 内把
        多个分支的写入逐个喂给 reducer。缺 reducer 的 LastValue 会抛
        InvalidUpdateError —— 此处断言修复后不会。
        """
        from langgraph.channels import BinaryOperatorAggregate

        from app.state import or_bool

        channel = BinaryOperatorAggregate(bool, or_bool)
        channel.update([True, True])          # 两个分支同时拒答
        self.assertTrue(channel.get())

        # 顺序无关：真值不会因为兄弟分支投了 False 而消失
        for writes in ([True, False], [False, True]):
            ch = BinaryOperatorAggregate(bool, or_bool)
            ch.update(writes)
            self.assertTrue(ch.get(), f"写入顺序 {writes} 下拒答真值丢失")

        # 重置哨兵能把上一轮的 True 清掉
        reset = BinaryOperatorAggregate(bool, or_bool)
        reset.update([True])
        reset.update([None])
        self.assertFalse(reset.get())

    def test_reason_merge_is_deduped_and_stable(self):
        from app.state import merge_abstain_reason

        a = "最高相关度 0.03 低于拒答阈值 0.40"
        b = "知识库中没有检索到任何相关内容"
        # 顺序无关：并行分支的写入顺序由调度决定，输出必须一致
        self.assertEqual(merge_abstain_reason(a, b), merge_abstain_reason(b, a))
        # 去重：同一分支重复写不产生重复文案
        self.assertEqual(merge_abstain_reason(a, a), a)

    def test_reason_empty_resets(self):
        from app.state import merge_abstain_reason

        self.assertEqual(merge_abstain_reason("上一轮的原因", ""), "")


class MultiSubQuestionAbstainTests(unittest.TestCase):
    """回归：库外「多子问题」提问曾整条 500。

    根因是 abstained 字段缺 reducer —— 两个并行 retrieve_one 分支在同一个 step
    里各写一次，LangGraph 抛 InvalidUpdateError。用例必须在**多子问题**下跑，
    单子问题掩盖了这个缺陷。
    """

    def _run(self, abstain: bool, subs: list[str]):
        """跑一次图；subs 为拆解出的子问题，全部返回同一拒答判定。"""
        generation_calls: list[str] = []

        class _FakeLLM:
            def invoke(self, prompt, config=None):
                system = ""
                if isinstance(prompt, list) and prompt:
                    system = getattr(prompt[0], "content", "") or ""
                if "问题拆解器" in system:
                    return type("R", (), {"content": "\n".join(subs)})()
                if "提问质量审查器" in system:
                    return type("R", (), {"content": "清晰"})()
                if "检索问句改写器" in system:
                    return type("R", (), {"content": "改写后的问句？"})()
                if "对话摘要器" in system:
                    return type("R", (), {"content": "摘要"})()
                return type("R", (), {"content": "辅助结果"})()

        def fake_invoke_llm(llm, prompt, config, attempts=2):
            system = ""
            if isinstance(prompt, list) and prompt:
                system = getattr(prompt[0], "content", "") or ""
            if "基于【检索到的资料】回答【子问题】" in system:
                generation_calls.append("sub_answer")
                return "模型编造的内容"
            if "回答汇总器" in system:
                generation_calls.append("aggregate")
                return "模型编造的汇总"
            return "辅助结果"

        def fake_retrieve(query, k=None):
            return RetrievalResult(
                query=query,
                docs=[_doc(0.03)],
                top_score=0.03,
                abstain=abstain,
                reason="最高相关度 0.03 低于拒答阈值 0.40" if abstain else "",
            )

        with patch.object(rag_graph, "_llm", return_value=_FakeLLM()), patch.object(
            rag_graph, "_invoke_llm", side_effect=fake_invoke_llm
        ), patch("app.knowledge_base.retrieve", side_effect=fake_retrieve), patch(
            "app.knowledge_base.search", return_value=[_doc(0.95)]
        ):
            graph = rag_graph.build_graph_with_memory()
            state = graph.invoke(
                {"messages": [HumanMessage(content="一个知识库里没有的复合问题？")]},
                config={"configurable": {"thread_id": f"multi-abstain-{abstain}-{len(subs)}"}},
            )
        return state, generation_calls

    def test_multi_subquestion_abstain_does_not_raise(self):
        """两个子问题同时拒答：不得抛 InvalidUpdateError（曾经直接 500）。"""
        state, generation_calls = self._run(True, ["子问题甲？", "子问题乙？"])

        self.assertTrue(state.get("abstained"))
        self.assertIn("0.03", state.get("abstain_reason", ""))
        self.assertEqual(generation_calls, [], f"证据不足却调用了生成：{generation_calls}")
        self.assertEqual(len(state.get("ungrounded_questions") or []), 2)
        # 拒答文案不进 sub_answers，避免被 reduce 润色成「有结论」
        self.assertEqual([a for a in (state.get("sub_answers") or []) if a], [])
        answer = state["messages"][-1].content
        self.assertIn("没有找到", answer)
        self.assertNotIn("模型编造", answer)

    def test_multi_subquestion_answers_when_grounded(self):
        """证据充分的多子问题仍应正常汇总（防止修复把正常路径一起挡掉）。"""
        state, generation_calls = self._run(False, ["子问题甲？", "子问题乙？"])

        self.assertFalse(state.get("abstained"))
        self.assertIn("aggregate", generation_calls)

    def test_abstain_does_not_leak_into_next_turn(self):
        """跨轮污染回归：第一轮拒答后，第二轮证据充分不得仍报 abstained。

        checkpointer 会把 abstained=True 留在 state 里，begin_turn 不重置就会黏住。
        """
        fake = _FakeLLMForChitchat()
        with patch.object(rag_graph, "_llm", return_value=fake):
            graph = rag_graph.build_graph_with_memory()
            cfg = {"configurable": {"thread_id": "abstain-cross-turn"}}

            def fake_retrieve_abstain(query, k=None):
                return RetrievalResult(
                    query=query, docs=[_doc(0.03)], top_score=0.03,
                    abstain=True, reason="最高相关度 0.03 低于拒答阈值 0.40",
                )

            def fake_retrieve_ok(query, k=None):
                return RetrievalResult(
                    query=query, docs=[_doc(0.95)], top_score=0.95, abstain=False
                )

            with patch("app.knowledge_base.retrieve", side_effect=fake_retrieve_abstain), patch(
                "app.knowledge_base.search", return_value=[_doc(0.95)]
            ):
                first = graph.invoke(
                    {"messages": [HumanMessage(content="知识库里没有的问题？")]}, config=cfg
                )
            with patch("app.knowledge_base.retrieve", side_effect=fake_retrieve_ok), patch(
                "app.knowledge_base.search", return_value=[_doc(0.95)]
            ):
                second = graph.invoke(
                    {"messages": [HumanMessage(content="一个证据充分的问题？")]}, config=cfg
                )

        self.assertTrue(first.get("abstained"))
        self.assertFalse(second.get("abstained"), "上一轮的拒答状态泄漏到了这一轮")
        self.assertEqual(second.get("abstain_reason"), "")
        self.assertEqual(second.get("ungrounded_questions"), [])

    def test_abstained_retrieval_result_hides_citations(self):
        """拒答时不应给出引用卡片：『引用 N 条』与『没找到证据』自相矛盾。"""
        low = RetrievalResult(
            query="q", docs=[_doc(0.03)], top_score=0.03, abstain=True, reason="低于阈值"
        )
        self.assertEqual(low.citations(), [])

        ok = RetrievalResult(query="q", docs=[_doc(0.95)], top_score=0.95, abstain=False)
        self.assertEqual(len(ok.citations()), 1)


class _FakeLLMForChitchat:
    """跨轮用例用的确定性假模型：只保证辅助步骤不干扰主流程。"""

    def invoke(self, prompt, config=None):
        system = getattr(prompt[0], "content", "") if isinstance(prompt, list) and prompt else ""
        if "提问质量审查器" in system:
            return type("R", (), {"content": "清晰"})()
        if "问题拆解器" in system:
            return type("R", (), {"content": "改写后的问句？"})()
        if "检索问句改写器" in system:
            return type("R", (), {"content": "改写后的问句？"})()
        return type("R", (), {"content": "本轮答案"})()


if __name__ == "__main__":
    unittest.main()
