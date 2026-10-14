"""端到端验证脚本：对运行中的服务发真实请求。

用法（先启动服务）：
    python scripts/e2e_check.py http://127.0.0.1:8000

检查项：
  1. /health、/ready 探针
  2. 会话生命周期（新建 → 历史 → 重命名 → 删除）
  3. 文档上传 → 索引重建 → 提问 → 引用返回
  4. 拒答：问一个知识库里没有的问题，应明确拒答而不是编造
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    mark = "通过" if ok else "失败"
    print(f"[{mark}] {name}" + (f" —— {detail}" if detail else ""))


def request(method: str, path: str, payload=None, raw: bytes | None = None,
            content_type: str = "application/json", timeout: int = 180):
    url = f"{BASE}{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = content_type
    elif raw is not None:
        data = raw
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def stream_chat(question: str, session_id: str, timeout: int = 300) -> dict:
    """调 SSE 接口并收集事件，返回 {event: [payload...]}。"""
    url = f"{BASE}/api/chat/stream"
    data = json.dumps({"message": question, "session_id": session_id}).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    events: dict[str, list] = {}
    tokens: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        buffer = ""
        for chunk in iter(lambda: resp.read(1024), b""):
            buffer += chunk.decode("utf-8", errors="replace")
            parts = buffer.split("\n\n")
            buffer = parts.pop()
            for part in parts:
                event, payload = None, None
                for line in part.split("\n"):
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        try:
                            payload = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            payload = None
                if event:
                    events.setdefault(event, []).append(payload)
                    if event == "token" and payload:
                        tokens.append(payload.get("text", ""))
    events["_streamed_text"] = ["".join(tokens)]
    return events


def main() -> int:
    print(f"目标服务：{BASE}\n")

    # ── 1. 探针 ──
    status, body = request("GET", "/health")
    check("GET /health 返回 200", status == 200, body[:120])

    status, body = request("GET", "/ready", timeout=300)
    ready_ok = False
    if status == 200:
        try:
            data = json.loads(body)
            ready_ok = data.get("status") == "ready"
            check("GET /ready 状态为 ready", ready_ok,
                  json.dumps(data.get("checks", {}), ensure_ascii=False)[:200])
        except json.JSONDecodeError:
            check("GET /ready 返回合法 JSON", False, body[:120])
    else:
        check("GET /ready 返回 200", False, body[:200])

    # ── 2. 会话生命周期 ──
    status, body = request("POST", "/api/sessions")
    session_id = ""
    if status == 200:
        session_id = json.loads(body)["id"]
        check("POST /api/sessions 创建会话", bool(session_id), session_id)
    else:
        check("POST /api/sessions 创建会话", False, body[:200])

    if session_id:
        status, body = request("PATCH", f"/api/sessions/{session_id}",
                               {"title": "端到端验证会话"})
        check("PATCH 重命名会话", status == 200 and "端到端" in body, body[:120])

        status, body = request("GET", f"/api/sessions/{session_id}/messages")
        check("GET 会话历史（初始为空）", status == 200 and json.loads(body)["messages"] == [])

    # ── 3. 文档上传 → 索引 → 提问 → 引用 ──
    upload_name = f"e2e_{int(time.time())}.md"
    upload_body = (
        "# 端到端验证文档\n\n"
        "## 测试参数\n\n"
        "| 参数 | 数值 |\n|---|---|\n"
        "| 验证编号 | E2E-2026-1014 |\n"
        "| 条件数阈值 | 2.500 |\n"
        "| 温度点 | 25 ℃ |\n\n"
        "## 结论\n\n端到端验证文档用于确认上传、索引与检索链路可用。\n"
    ).encode("utf-8")

    boundary = "----e2eboundary"
    multipart = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{upload_name}"\r\n'
        "Content-Type: text/markdown\r\n\r\n"
    ).encode("utf-8") + upload_body + (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="subdir"\r\n\r\n\r\n'
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="reindex"\r\n\r\nfalse\r\n'
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    status, body = request(
        "POST", "/api/documents/upload", raw=multipart,
        content_type=f"multipart/form-data; boundary={boundary}", timeout=300,
    )
    check("POST 上传文档", status == 200 and upload_name in body, body[:200])

    status, body = request("POST", "/api/documents/reindex", {"force": False}, timeout=600)
    indexed = False
    if status == 200:
        report = json.loads(body)
        indexed = report.get("ok", False)
        check("POST 重建索引", indexed,
              f"模式={report.get('mode')} 文件={report.get('parsed_files')} "
              f"父块={report.get('parent_count')} 子块={report.get('child_count')}")
    else:
        check("POST 重建索引", False, body[:200])

    status, body = request("GET", "/api/documents")
    if status == 200:
        docs = json.loads(body)
        check("上传的文档出现在语料列表", upload_name in docs.get("documents", []),
              f"共 {docs.get('document_count')} 篇")

    # ── 4. 真实问答 + 引用 ──
    if session_id:
        print("\n发起真实问答（会调用模型，可能需要 30~90 秒）…")
        try:
            events = stream_chat("条件数的理论下界是多少？", session_id)
            done = (events.get("done") or [None])[0]
            if done:
                citations = done.get("citations") or []
                check("流式问答返回 done 事件", True,
                      f"耗时 {done.get('elapsed')}s，命中 {done.get('retrieved_count')} 条")
                check("返回结构化引用", len(citations) > 0, f"{len(citations)} 条")
                if citations:
                    first = citations[0]
                    check("引用包含来源与原文片段",
                          bool(first.get("source")) and bool(first.get("quote")),
                          f"{first.get('source')} | {first.get('quote', '')[:60]}")
                    check("引用带可定位的 chunk_id", bool(first.get("chunk_id")),
                          str(first.get("chunk_id")))
                check("回答非空", bool(done.get("answer")),
                      (done.get("answer") or "")[:100].replace("\n", " "))
            else:
                check("流式问答返回 done 事件", False,
                      json.dumps(events, ensure_ascii=False)[:300])
        except Exception as e:  # noqa: BLE001
            check("流式问答", False, f"{type(e).__name__}: {e}")

        # ── 5. 无证据问题的处理 ──
        # 刻意不写死「必须 abstained=True」：面对知识库里没有的问题，Agent 有两条
        # 都算正确的路径 ——
        #   (a) 检索后判定证据不足 → 明确拒答（abstained=True）
        #   (b) 判定问题本身不够具体 → 先澄清反问，既不浪费检索也不编造
        # 真正的判定标准是「没有编造答案」，而不是「必须走某一条分支」。
        # 最初的断言只认 (a)，把 (b) 误报成了失败。
        print("\n发起无证据问题（应拒答或先澄清，不得编造）…")
        try:
            events = stream_chat("请问火星探测器的钛合金焊接工艺标准是什么？", session_id)
            done = (events.get("done") or [None])[0]
            if done:
                abstained = bool(done.get("abstained"))
                answer = done.get("answer") or ""
                stages = {s.get("stage") for s in (events.get("stage") or []) if s}
                clarified = "clarify" in stages
                refused = (
                    "没有找到" in answer
                    or "没有足够" in answer
                    or "还不太明确" in answer
                    or "补充" in answer
                )
                check(
                    "无证据问题未编造答案（拒答或澄清）",
                    abstained or clarified or refused,
                    f"abstained={abstained} 澄清={clarified} "
                    f"命中={done.get('retrieved_count')} 回答={answer[:50]}",
                )
            else:
                check("无证据问题未编造答案（拒答或澄清）", False, "未收到 done 事件")
        except Exception as e:  # noqa: BLE001
            check("无证据问题未编造答案（拒答或澄清）", False, f"{type(e).__name__}: {e}")

        # ── 6. 独立会话中的拒答（无历史，问题本身也不模糊）──
        # 上面那条测的是「带历史时可能走澄清」；这里用全新会话问一个措辞具体、
        # 但知识库确实没有的问题，应当走到检索后的拒答分支。
        print("\n发起措辞具体但无证据的问题（应拒答）…")
        status, body = request("POST", "/api/sessions")
        fresh_sid = json.loads(body)["id"] if status == 200 else ""
        if fresh_sid:
            try:
                events = stream_chat(
                    "请给出钛合金 TC4 在真空钎焊工艺中的标准保温温度数值。", fresh_sid
                )
                done = (events.get("done") or [None])[0]
                if done:
                    abstained = bool(done.get("abstained"))
                    answer = done.get("answer") or ""
                    check(
                        "证据不足时明确拒答",
                        abstained or "没有" in answer,
                        f"abstained={abstained} 原因={done.get('abstain_reason', '')[:60]}",
                    )
                else:
                    check("证据不足时明确拒答", False, "未收到 done 事件")
            except Exception as e:  # noqa: BLE001
                check("证据不足时明确拒答", False, f"{type(e).__name__}: {e}")
            request("DELETE", f"/api/sessions/{fresh_sid}")

        # ── 7. 指标与反馈 ──
        status, body = request("GET", "/api/metrics")
        if status == 200:
            metrics = json.loads(body)
            check("GET /api/metrics 可用", metrics.get("total_queries", 0) >= 1,
                  f"提问 {metrics.get('total_queries')} 次，"
                  f"拒答率 {metrics.get('abstain_rate')}")

        status, body = request("POST", "/api/feedback", {
            "session_id": session_id, "rating": "up",
            "question": "条件数的理论下界是多少？", "answer": "1.732",
        })
        check("POST 提交反馈", status == 200, body[:120])

        # ── 7. 清理 ──
        status, body = request("DELETE", f"/api/sessions/{session_id}")
        check("DELETE 删除会话", status == 200, body[:120])

    status, body = request("DELETE", f"/api/documents/{upload_name}", timeout=600)
    check("DELETE 删除文档并重建索引", status == 200, body[:150])

    print("\n" + "=" * 60)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  ✗ {name}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
