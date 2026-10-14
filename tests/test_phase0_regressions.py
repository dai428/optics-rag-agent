"""Phase 0 回归测试：配置优先级、包模式导入、单轮状态隔离。

这三个用例各自对应一个已修复的真实缺陷，属于「修完必须锁死」的护栏：
  1. 进程环境变量必须优先于 .env（否则 Docker / CI 传参不生效）
  2. app.rag_graph 在包模式下多轮对话不得抛 ModuleNotFoundError
  3. 第二轮不得混入第一轮的子答案与检索结果（跨轮状态污染）
"""

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
    """模拟模型返回值：只关心 .content。"""

    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    """确定性假模型，用于在不调用外部 API 的前提下测试图接线是否正确。

    按系统提示词的特征词分派行为，因此改提示词文案时这里也要同步，
    这本身是一种有价值的耦合：它保证「节点确实用了我们以为的那个提示词」。
    """

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
    """配置读取优先级。"""

    def test_process_environment_has_priority(self):
        with patch.dict(os.environ, {"RAG_TEST_SETTING": "from-process"}, clear=False):
            self.assertEqual(env_value("RAG_TEST_SETTING", "fallback"), "from-process")


class ReducerTests(unittest.TestCase):
    """reducer 的重置语义：空列表 / 0 代表「开始新一轮」。"""

    def test_empty_update_resets_turn_lists(self):
        doc = Document(page_content="old")
        self.assertEqual(add_queries(["old"], []), [])
        self.assertEqual(add_docs([doc], []), [])
        self.assertEqual(sum_int(7, 0), 0)


class GraphRegressionTests(unittest.TestCase):
    """端到端跑一次两轮对话，验证包模式可用且状态不串轮。"""

    def test_package_mode_rewrite_and_multi_turn_state_isolation(self):
        fake = _FakeLLM()

        def fake_search(query: str, k: int | None = None):
            return [
                Document(
                    page_content=f"evidence for {query}",
                    metadata={"parent_id": query, "file_name": "sample.md", "section_path": "demo"},
                )
            ]

        # ⚠️ 必须同时打补丁 retrieve 与 search：
        #    retrieve_one 现在先用 retrieve() 判断证据充分性，命中后才调 search() 回溯父块。
        #    只补 search 会让 retrieve 打到真实索引，检索结果不可控（曾导致断言拿到 5 条真实文档）。
        def fake_retrieve(query: str, k: int | None = None):
            from app.knowledge_base import RetrievalResult

            docs = fake_search(query, k)
            return RetrievalResult(
                query=query, docs=docs, top_score=0.95, abstain=False
            )

        with patch.object(rag_graph, "_llm", return_value=fake), patch(
            "app.knowledge_base.retrieve", side_effect=fake_retrieve
        ), patch("app.knowledge_base.search", side_effect=fake_search):
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
