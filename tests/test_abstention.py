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


if __name__ == "__main__":
    unittest.main()
