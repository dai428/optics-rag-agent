"""Phase 2 测试：HTTP 接口层（探针、会话、文档、反馈、指标、鉴权、限流）。

这些测试用 FastAPI 的 TestClient，不依赖外部模型服务：需要模型的部分用
monkeypatch 替换，从而保证在 CI 里能稳定、快速地跑完。

⚠️ 隔离要求（曾经踩过坑）
   测试绝不能写到真实语料目录。最初用 `os.environ.setdefault("CORPUS_DIR", ...)`
   试图重定向，但 settings 是模块级单例、在导入时就固化了路径，且 setdefault
   在环境变量已存在时不生效 —— 结果是上传用例把测试文件写进了 data/docs。
   现在改为在 setUp 里**直接 patch settings.docs_dir**，并显式断言它落在临时目录。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main, storage  # noqa: E402
from app.config import settings  # noqa: E402
from app.storage import Database  # noqa: E402

# 每个测试类一个独立的临时目录
_TMP_DIR = tempfile.TemporaryDirectory()
_TEST_DOCS = Path(_TMP_DIR.name) / "docs"
_TEST_DOCS.mkdir(parents=True, exist_ok=True)


class _IsolatedTestBase(unittest.TestCase):
    """把语料目录与数据库都指向临时位置的测试基类。"""

    def setUp(self):
        self._docs_patch = patch.object(settings, "docs_dir", _TEST_DOCS)
        self._docs_patch.start()
        self.addCleanup(self._docs_patch.stop)

        # 防御性断言：确认语料目录确实被重定向，避免污染真实数据
        self.assertEqual(
            Path(settings.docs_dir).resolve(),
            _TEST_DOCS.resolve(),
            "测试未成功重定向语料目录，存在污染真实数据的风险",
        )

        storage._db_singleton = Database(Path(_TMP_DIR.name) / f"{self.id()}.db")


class ApiTests(_IsolatedTestBase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    @classmethod
    def tearDownClass(cls):
        _TMP_DIR.cleanup()

    def setUp(self):
        super().setUp()
        # 每个用例清空临时语料目录，避免用例间互相看到对方上传的文件
        for path in sorted(_TEST_DOCS.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir() and path != _TEST_DOCS:
                path.rmdir()

    # ── 探针 ──────────────────────────────────────────────
    def test_health_is_lightweight(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("has_api_key", body)

    def test_ready_reports_checks(self):
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body["status"], {"ready", "degraded"})
        self.assertIn("index", body["checks"])
        self.assertIn("database", body["checks"])
        self.assertTrue(body["checks"]["database"]["ok"])

    # ── 会话 ──────────────────────────────────────────────
    def test_session_lifecycle(self):
        created = self.client.post("/api/sessions").json()
        sid = created["id"]
        self.assertEqual(created["title"], "新会话")

        listed = self.client.get("/api/sessions").json()["sessions"]
        self.assertIn(sid, [s["id"] for s in listed])

        renamed = self.client.patch(f"/api/sessions/{sid}", json={"title": "改过的标题"})
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json()["title"], "改过的标题")

        messages = self.client.get(f"/api/sessions/{sid}/messages")
        self.assertEqual(messages.status_code, 200)
        self.assertEqual(messages.json()["messages"], [])

        deleted = self.client.delete(f"/api/sessions/{sid}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.client.get(f"/api/sessions/{sid}/messages").status_code, 404)

    def test_rejects_invalid_session_id(self):
        response = self.client.get("/api/sessions/中文会话/messages")
        self.assertEqual(response.status_code, 400)

    def test_rename_missing_session_returns_404(self):
        self.assertEqual(
            self.client.patch("/api/sessions/nope", json={"title": "x"}).status_code, 404
        )

    # ── 反馈与指标 ────────────────────────────────────────
    def test_feedback_and_metrics(self):
        session = self.client.post("/api/sessions").json()
        sid = session["id"]

        recorded = self.client.post(
            "/api/feedback",
            json={
                "session_id": sid,
                "rating": "down",
                "comment": "数字不对",
                "question": "条件数是多少？",
                "answer": "1.99",
            },
        )
        self.assertEqual(recorded.status_code, 200)

        metrics = self.client.get("/api/metrics").json()
        self.assertEqual(metrics["feedback"]["total"], 1)
        self.assertEqual(metrics["feedback"]["down"], 1)
        self.assertEqual(metrics["feedback"]["recent"][0]["comment"], "数字不对")

    def test_feedback_rejects_invalid_rating(self):
        response = self.client.post(
            "/api/feedback", json={"session_id": "s", "rating": "maybe"}
        )
        self.assertEqual(response.status_code, 422)

    # ── 输入校验 ──────────────────────────────────────────
    def test_chat_rejects_oversized_message(self):
        response = self.client.post(
            "/api/chat/stream", json={"message": "x" * 9000, "session_id": "big"}
        )
        self.assertEqual(response.status_code, 422)

    def test_chat_rejects_empty_message(self):
        response = self.client.post(
            "/api/chat/stream", json={"message": "", "session_id": "empty"}
        )
        self.assertEqual(response.status_code, 422)

    def test_chat_rejects_bad_session_id(self):
        response = self.client.post(
            "/api/chat/stream", json={"message": "问题", "session_id": "非法中文"}
        )
        self.assertEqual(response.status_code, 400)

    # ── 文档上传 ──────────────────────────────────────────
    def test_upload_rejects_unsupported_extension(self):
        response = self.client.post(
            "/api/documents/upload",
            files={"file": ("bad.exe", b"MZ", "application/octet-stream")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("不支持", response.json()["detail"])

    def test_upload_rejects_empty_file(self):
        response = self.client.post(
            "/api/documents/upload", files={"file": ("empty.md", b"", "text/markdown")}
        )
        self.assertEqual(response.status_code, 400)

    def test_upload_accepts_markdown_without_reindex(self):
        response = self.client.post(
            "/api/documents/upload",
            files={"file": ("note.md", "# 标题\n\n内容".encode("utf-8"), "text/markdown")},
            data={"reindex": "false"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "uploaded")
        self.assertTrue((_TEST_DOCS / "note.md").exists())

    def test_delete_missing_document_returns_404(self):
        response = self.client.delete("/api/documents/does_not_exist.md")
        self.assertEqual(response.status_code, 404)

    # ── 访问令牌 ──────────────────────────────────────────
    def test_token_protects_write_endpoints(self):
        with patch.object(main.settings, "access_token", "secret-token"):
            # 无令牌 → 401
            denied = self.client.post(
                "/api/documents/reindex", json={"force": False}
            )
            self.assertEqual(denied.status_code, 401)

            # 错误令牌 → 401
            wrong = self.client.post(
                "/api/documents/reindex",
                json={"force": False},
                headers={"Authorization": "Bearer wrong"},
            )
            self.assertEqual(wrong.status_code, 401)

            # 正确令牌 → 放行（这里只验证鉴权通过，不实际重建索引）
            with patch.object(main, "_run_index", return_value={"ok": True, "mode": "noop"}):
                ok = self.client.post(
                    "/api/documents/reindex",
                    json={"force": False},
                    headers={"Authorization": "Bearer secret-token"},
                )
                self.assertEqual(ok.status_code, 200)

    def test_read_endpoints_stay_open_with_token(self):
        with patch.object(main.settings, "access_token", "secret-token"):
            self.assertEqual(self.client.get("/health").status_code, 200)
            self.assertEqual(self.client.get("/api/sessions").status_code, 200)


class RateLimitTests(_IsolatedTestBase):
    def setUp(self):
        super().setUp()
        self.client = TestClient(main.app)
        main._rate_buckets.clear()

    def test_rate_limit_blocks_excess_requests(self):
        with patch.object(main.settings, "rate_limit_per_minute", 2):
            with patch.object(main, "get_graph", return_value=None):
                first = self.client.post(
                    "/api/chat/stream", json={"message": "q1", "session_id": "rl"}
                )
                second = self.client.post(
                    "/api/chat/stream", json={"message": "q2", "session_id": "rl"}
                )
                third = self.client.post(
                    "/api/chat/stream", json={"message": "q3", "session_id": "rl"}
                )
        # 前两次放行（因无 Graph 返回 error 事件流），第三次被限流
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(third.status_code, 429)


if __name__ == "__main__":
    unittest.main()
