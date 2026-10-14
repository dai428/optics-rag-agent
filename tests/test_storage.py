"""Phase 2 测试：SQLite 持久化（会话、消息、反馈、运行指标）。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.storage import Database


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self._tmp.name) / "test.db")

    def tearDown(self):
        self._tmp.cleanup()

    # ── 会话 ──────────────────────────────────────────────
    def test_schema_version_recorded(self):
        self.assertEqual(self.db.schema_version(), 1)

    def test_create_and_fetch_session(self):
        created = self.db.create_session("s1", "测试会话")
        self.assertEqual(created.id, "s1")
        self.assertEqual(created.title, "测试会话")
        self.assertEqual(created.turns, 0)

        fetched = self.db.get_session("s1")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.title, "测试会话")

    def test_touch_session_counts_first_turn(self):
        # 首次提问的轮数应为 1，而不是 0（原实现的 off-by-one）
        record = self.db.touch_session("s2", "第一个问题是什么？")
        self.assertEqual(record.turns, 1)
        self.assertEqual(record.title, "第一个问题是什么？")

        again = self.db.touch_session("s2", "第二个问题")
        self.assertEqual(again.turns, 2)
        # 标题应保持首次提问的前缀，不被后续提问覆盖
        self.assertEqual(again.title, "第一个问题是什么？")

    def test_touch_session_without_message_keeps_zero_turns(self):
        record = self.db.touch_session("empty")
        self.assertEqual(record.turns, 0)
        self.assertEqual(record.title, "新会话")

    def test_list_sessions_orders_by_recent(self):
        self.db.create_session("old")
        self.db.create_session("new")
        # 手动把 old 的活跃时间改成更早
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE sessions SET last_active = '2020-01-01T00:00:00' WHERE id = 'old'"
            )
        ids = [s.id for s in self.db.list_sessions()]
        self.assertEqual(ids[0], "new")

    def test_rename_and_delete_session(self):
        self.db.create_session("s3")
        self.assertTrue(self.db.rename_session("s3", "新标题"))
        self.assertEqual(self.db.get_session("s3").title, "新标题")
        self.assertTrue(self.db.delete_session("s3"))
        self.assertIsNone(self.db.get_session("s3"))
        self.assertFalse(self.db.delete_session("s3"))

    def test_delete_session_cascades_messages(self):
        self.db.create_session("s4")
        self.db.add_message("s4", "user", "问题")
        self.db.add_message("s4", "assistant", "回答")
        self.assertEqual(len(self.db.list_messages("s4")), 2)
        self.db.delete_session("s4")
        self.assertEqual(self.db.list_messages("s4"), [])

    def test_prune_sessions_keeps_most_recent(self):
        for i in range(5):
            self.db.create_session(f"p{i}")
            with self.db.connect() as conn:
                conn.execute(
                    "UPDATE sessions SET last_active = ? WHERE id = ?",
                    (f"2026-01-0{i + 1}T00:00:00", f"p{i}"),
                )
        removed = self.db.prune_sessions(keep=2)
        self.assertEqual(removed, 3)
        remaining = {s.id for s in self.db.list_sessions()}
        self.assertEqual(remaining, {"p3", "p4"})

    # ── 消息 ──────────────────────────────────────────────
    def test_messages_preserve_order_and_meta(self):
        self.db.create_session("m1")
        self.db.add_message("m1", "user", "第一个问题")
        self.db.add_message(
            "m1", "assistant", "第一个回答", meta={"citations": [{"index": 1}]}
        )
        messages = self.db.list_messages("m1")
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        self.assertEqual(messages[1]["meta"]["citations"][0]["index"], 1)

    # ── 反馈 ──────────────────────────────────────────────
    def test_feedback_summary_computes_positive_rate(self):
        self.db.create_session("f1")
        self.db.add_feedback("f1", "up", question="q1", answer="a1")
        self.db.add_feedback("f1", "up", question="q2", answer="a2")
        self.db.add_feedback("f1", "down", comment="数字不对", question="q3", answer="a3")

        summary = self.db.feedback_summary()
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["up"], 2)
        self.assertEqual(summary["down"], 1)
        self.assertAlmostEqual(summary["positive_rate"], 2 / 3, places=4)
        self.assertEqual(len(summary["recent"]), 3)

    def test_feedback_summary_when_empty(self):
        summary = self.db.feedback_summary()
        self.assertEqual(summary["total"], 0)
        self.assertIsNone(summary["positive_rate"])

    # ── 查询日志与指标 ────────────────────────────────────
    def test_metrics_aggregate_query_log(self):
        self.db.create_session("q1")
        self.db.log_query("q1", "问题一", retrieved=3, abstained=False, citations=3, elapsed_ms=100)
        self.db.log_query("q1", "问题二", retrieved=0, abstained=True, citations=0, elapsed_ms=200)
        self.db.log_query("q1", "问题三", error="超时", elapsed_ms=300)

        metrics = self.db.metrics()
        self.assertEqual(metrics["total_queries"], 3)
        self.assertEqual(metrics["abstained"], 1)
        self.assertAlmostEqual(metrics["abstain_rate"], 1 / 3, places=4)
        self.assertEqual(metrics["errors"], 1)
        self.assertEqual(metrics["avg_elapsed_ms"], 200.0)

    def test_recent_queries_returns_latest_first(self):
        self.db.create_session("q2")
        self.db.log_query("q2", "较早的问题")
        self.db.log_query("q2", "较晚的问题")
        recent = self.db.recent_queries(limit=5)
        self.assertEqual(recent[0]["question"], "较晚的问题")

    def test_database_persists_across_instances(self):
        path = Path(self._tmp.name) / "persist.db"
        first = Database(path)
        first.create_session("keep")
        first.add_message("keep", "user", "持久化测试")

        second = Database(path)
        self.assertIsNotNone(second.get_session("keep"))
        self.assertEqual(second.list_messages("keep")[0]["content"], "持久化测试")


if __name__ == "__main__":
    unittest.main()
