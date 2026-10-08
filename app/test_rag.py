"""
test_rag.py — 光学 RAG Agent 三项测试（无需 pytest，直接 python test_rag.py）

  T1 知识库检索     —— 不需要 API Key（用本地 BGE 嵌入）
  T2 API 接口       —— 不需要 API Key（/health 与 /documents）
  T3 会话隔离 + 多轮 —— 需要 AMD_API_KEY；缺失则自动跳过

运行前环境（本机必需）：
  set HF_ENDPOINT=https://hf-mirror.com
  并清空 http(s)_proxy、设 no_proxy=*
"""
import sys
import time
from pathlib import Path


def _has_api_key() -> bool:
    env = {}
    p = Path(__file__).parent.parent / ".env"  # .env 在项目根目录
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return bool(env.get("AMD_API_KEY"))


def t1_knowledge_base():
    from .knowledge_base import corpus_info, get_retriever

    info = corpus_info()
    assert info["chunks"] > 0, "知识库为空，检查 docs/ 目录"
    hits = get_retriever(k=3).invoke("LCVR 相比传统旋转波片有什么优势？")
    assert len(hits) == 3, f"应命中 3 条，实际 {len(hits)}"
    joined = "".join(h.page_content for h in hits)
    assert ("LCVR" in joined or "液晶" in joined), "检索未命中 LCVR 相关内容"
    print(f"[T1] ✅ 知识库检索通过：{info['chunks']} 个片段，命中 {len(hits)} 条，语义相关")


def t2_api_no_key():
    from fastapi.testclient import TestClient

    from . import main

    c = TestClient(main.app)
    assert c.get("/health").json()["status"] == "ok"
    d = c.get("/documents").json()
    assert d["chunks"] > 0
    print(f"[T2] ✅ API 接口通过：/health=ok，/documents 返回 {d['chunks']} 片段、{len(d['documents'])} 文档")


def _chat(client, message: str, session_id: str, tries: int = 6):
    """带退避重试的 /chat 调用，容忍 AMD 共享模型的瞬时 429/504。"""
    last = None
    for i in range(tries):
        r = client.post("/chat", json={"message": message, "session_id": session_id})
        if r.status_code == 200:
            return r.json()
        last = r
        if r.status_code in (429, 504):
            time.sleep(5 * (i + 1))
            continue
        break
    raise AssertionError(f"/chat 失败 HTTP {last.status_code}: {last.text[:150]}")


def t3_session_isolation_and_memory():
    if not _has_api_key():
        print("[T3] ⏭️ 跳过（未配置 AMD_API_KEY）")
        return
    from fastapi.testclient import TestClient

    from . import main

    c = TestClient(main.app)

    try:
        # 会话 A：先自我介绍
        _chat(c, "你好，我叫张万森，请记住我的名字。", "sess-A")
        # 会话 A：追问 —— 应记得（多轮记忆）
        a = _chat(c, "我叫什么名字？", "sess-A")
        assert "张万森" in a["answer"], f"会话 A 未记住名字：{a['answer'][:80]}"
        print("[T3a] ✅ 多轮记忆通过：会话 A 记得用户名字")

        # 会话 B：全新会话 —— 不该知道 A 的名字（会话隔离）
        b = _chat(c, "我叫什么名字？", "sess-B")
        assert "张万森" not in b["answer"], f"会话 B 泄漏了 A 的上下文：{b['answer'][:80]}"
        print("[T3b] ✅ 会话隔离通过：会话 B 不知道 A 的名字")

        # 专业问题应触发检索
        q = _chat(c, "LCVR 是什么？相比传统旋转波片有什么优势？", "sess-A")
        assert q["used_retrieval"], "专业问题未触发知识库检索"
        print("[T3c] ✅ 工具自决检索通过：专业问题触发了 search_optics_knowledge_base")
    except AssertionError as e:
        if "429" in str(e):
            print("[T3] ⏭️ 跳过（AMD 共享模型持续限流 429，稍后重跑即可）")
            return
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("光学 RAG Agent · 测试开始")
    print("=" * 60)
    t1_knowledge_base()
    t2_api_no_key()
    t3_session_isolation_and_memory()
    print("=" * 60)
    print("测试结束")