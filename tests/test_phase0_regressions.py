"""Phase 0 regression tests: configuration, package imports, and turn-state isolation."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app import rag_graph
from app.config import env_value
from app.state import add_docs, add_queries, sum_int


class _Reply:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    """Deterministic model used to test graph wiring without an external API."""

    def invoke(self, prompt, config=None):
        system = getattr(prompt[0], "content", "") if isinstance(prompt, list) and prompt else ""
        if "检索问句改写器" in system:
            latest = getattr(prompt[-1], "content", "").split("【用户最新一句】")[-1]
            question = latest.split("改写后的检索问句：")[0].strip()
            return _Reply(f"已消解：{question}")
        if "提问质量审查器" in system:
            return _Reply("清晰")
        if "问题拆解器" in system:
            user = getattr(prompt[-1], "content", "")
            question = user.split("【用户问题】")[-1].split("子问题列表：")[0].strip()
            return _Reply(question)
        if "对话摘要器" in system:
            return _Reply("保留的历史摘要")
        if "回答汇总器" in system:
            return _Reply("汇总答案")
        return _Reply("本轮答案")


class ConfigTests(unittest.TestCase):
    def test_process_environment_has_priority(self):
        with patch.dict(os.environ, {"RAG_TEST_SETTING": "from-process"}, clear=False):
            self.assertEqual(env_value("RAG_TEST_SETTING", "fallback"), "from-process")


class ReducerTests(unittest.TestCase):
    def test_empty_update_resets_turn_lists(self):
        doc = Document(page_content="old")
        self.assertEqual(add_queries(["old"], []), [])
        self.assertEqual(add_docs([doc], []), [])
        self.assertEqual(sum_int(7, 0), 0)


class GraphRegressionTests(unittest.TestCase):
    def test_package_mode_rewrite_and_multi_turn_state_isolation(self):
        fake = _FakeLLM()

        def fake_search(query: str, k: int | None = None):
            return [
                Document(
                    page_content=f"evidence for {query}",
                    metadata={"parent_id": query, "file_name": "sample.md", "section_path": "demo"},
                )
            ]

        with patch.object(rag_graph, "_llm", return_value=fake), patch(
            "app.knowledge_base.search", side_effect=fake_search
        ):
            graph = rag_graph.build_graph_with_memory()
            config = {"configurable": {"thread_id": "phase0-regression"}}

            first = graph.invoke(
                {"messages": [HumanMessage(content="第一个专业问题是什么？")]}, config=config
            )
            second = graph.invoke(
                {"messages": [HumanMessage(content="那它的第二个结论呢？")]}, config=config
            )

        self.assertEqual(len(first["sub_answers"]), 1)
        self.assertEqual(len(second["sub_answers"]), 1)
        self.assertEqual(first["tool_calls"], 1)
        self.assertEqual(second["tool_calls"], 1)
        self.assertEqual(len(second["retrieved_docs"]), 1)
        self.assertTrue(second["queries"][-1].startswith("已消解："))
        self.assertIn("已消解：", second["sub_questions"][0])


if __name__ == "__main__":
    unittest.main()
