"""
main.py — FastAPI 服务：把光学 RAG Agent 暴露成 HTTP 接口 + 托管前端页面

启动（两种皆可）：
    # 在项目根目录
    ../.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
    # 或进 app/ 目录
    ../.venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

接口分两组
─────────
【A. 基础接口】早期版本，供脚本/调试用，保持向后兼容
  GET  /health            健康检查（无需 Key）
  GET  /documents         知识库概况（无需 Key）
  POST /chat              提问（create_agent 黑盒版）
  POST /graph/chat        提问（LangGraph 显式版；含查询改写）
  POST /chat/reset        清空某会话上下文

【B. 全栈接口 /api/**】前端专用，多了「会话管理」和「流式输出」
  GET    /api/health              健康检查
  GET    /api/documents           知识库概况
  GET    /api/sessions            会话列表
  POST   /api/sessions            新建会话
  DELETE /api/sessions/{sid}      删除会话（同时清空其记忆）
  POST   /api/chat                同步问答（一次性返回完整结果）
  POST   /api/chat/stream         ★ 流式问答（SSE：阶段进度 + 逐字输出）

为什么要有流式？
    一次问答要调用多次大模型（改写 → 澄清 → 拆分 → 检索 → 生成），
    全程可能要 20~80 秒。如果等全部算完再返回，用户界面会长时间空白。
    流式把「正在改写 → 正在检索 → 正在生成」实时推给前端，并用逐字输出
    让等待过程可见 —— 这是「好用」与「难用」的分水岭。

可观测：配置 .env 里的 LANGFUSE_* 后，自动把每次链路上报 Langfuse；未配置则静默跳过。
"""

import asyncio
import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb  # 主线程预导入：规避 chromadb 在工作线程「首次导入+首次建客户端」的 Rust 绑定初始化问题（很轻，约 1.3s）

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:  # 包方式：uvicorn app.main:app（在项目根目录执行）
    from .knowledge_base import corpus_info
    from .rag_agent import build_agent, API_KEY
    from .observability import callbacks as _obs_callbacks
except ImportError:  # 扁平方式：cd app && uvicorn main:app
    from knowledge_base import corpus_info
    from rag_agent import build_agent, API_KEY
    from observability import callbacks as _obs_callbacks

app = FastAPI(title="光学 RAG Agent 服务 · 张万森", version="2.0")

_agent = None   # create_agent 懒加载
_graph = None   # LangGraph 懒加载
_SID = re.compile(r"^[A-Za-z0-9._:\-]{1,64}$")

# ══════════════════════════════════════════════════════════════════
#  会话登记表
#  说明：会话的「对话记忆」由 LangGraph 的 InMemorySaver 按 thread_id 保存，
#       但它不提供「列出所有会话」的能力。所以这里额外维护一份轻量登记表，
#       只存列表页要用的元信息（标题、时间、轮数），不存正文。
#  ⚠️ 进程重启后两者都会清空 —— 这与 InMemorySaver 的行为一致。
#     将来要持久化，把这里换成 SQLite / 云数据库即可，接口不用改。
# ══════════════════════════════════════════════════════════════════
SESSIONS: dict[str, dict[str, Any]] = {}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _touch_session(sid: str, first_message: str = "") -> dict:
    """登记或更新一个会话。首次出现时用提问前 24 字作为标题。"""
    s = SESSIONS.get(sid)
    if s is None:
        title = (first_message or "").strip()[:24] or "新会话"
        s = {
            "id": sid,
            "title": title,
            "created_at": _now(),
            "last_active": _now(),
            "turns": 0,
        }
        SESSIONS[sid] = s
    else:
        s["last_active"] = _now()
        s["turns"] += 1
    return s


def get_agent():
    global _agent
    if _agent is None:
        try:
            _agent = build_agent()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"Agent 初始化失败：{e}") from e
    return _agent


def get_graph():
    """LangGraph 版（含查询改写），懒加载。"""
    global _graph
    if _graph is None:
        try:
            try:
                from .rag_graph import build_graph_with_memory
            except ImportError:
                from rag_graph import build_graph_with_memory

            _graph = build_graph_with_memory()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"Graph 初始化失败：{e}") from e
    return _graph


def _map_llm_error(e: Exception) -> HTTPException:
    """把底层模型的报错翻译成合适的 HTTP 状态码。"""
    msg = str(e)
    low = msg.lower()
    if "429" in msg or "rate" in low or "concurrency" in low or "limit" in low:
        return HTTPException(
            status_code=429,
            detail="模型并发已达上限（AMD 共享模型限制 16 并发），请稍后重试或换用其他模型",
        )
    if "timeout" in low or "timed out" in low:
        return HTTPException(status_code=504, detail="模型响应超时，请稍后重试")
    return HTTPException(status_code=500, detail=f"调用失败：{msg[:200]}")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户问题")
    session_id: str = Field(default="default", max_length=64, description="会话 ID")


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    used_retrieval: bool
    rewritten_query: str | None = None   # 仅 /graph/chat 返回：改写后的检索问句
    retrieved_count: int = 0              # 仅 /graph/chat 返回：命中片段数


# ══════════════════════════════════════════════════════════════════
#  A. 基础接口（向后兼容，勿删——旧脚本与 test_rag.py 在用）
# ══════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "has_api_key": bool(API_KEY)}


@app.get("/documents")
def documents():
    """返回知识库语料概况（不初始化 Agent、不需要 API Key）。"""
    return corpus_info()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not _SID.match(req.session_id):
        raise HTTPException(status_code=400, detail="session_id 仅允许字母数字与 . _ : -，长度≤64")

    agent = get_agent()
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="未配置 AMD_API_KEY，/chat 不可用（/health、/documents 仍可用）",
        )

    from langchain_core.messages import HumanMessage, ToolMessage

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=req.message)]},
            config={"configurable": {"thread_id": req.session_id}},
        )
    except Exception as e:  # noqa: BLE001
        raise _map_llm_error(e) from e

    msgs = result["messages"]
    used_retrieval = any(isinstance(m, ToolMessage) for m in msgs)
    return ChatResponse(
        answer=msgs[-1].content,
        session_id=req.session_id,
        used_retrieval=used_retrieval,
    )


@app.post("/graph/chat", response_model=ChatResponse)
def graph_chat(req: ChatRequest):
    """LangGraph 显式版：先改写问句，再按规则决定是否检索，最后回答。

    相比 /chat 的可观测性更好——额外返回改写后的检索问句。
    """
    if not _SID.match(req.session_id):
        raise HTTPException(status_code=400, detail="session_id 仅允许字母数字与 . _ : -，长度≤64")

    graph = get_graph()
    if graph is None:
        raise HTTPException(status_code=503, detail="未配置 AMD_API_KEY，/graph/chat 不可用")

    from langchain_core.messages import HumanMessage

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=req.message)]},
            config={
                "configurable": {"thread_id": req.session_id},
                # Langfuse 观测（未配置时是空列表，零影响）
                "callbacks": _obs_callbacks(),
            },
        )
    except Exception as e:  # noqa: BLE001
        raise _map_llm_error(e) from e

    _touch_session(req.session_id, req.message)
    docs = result.get("retrieved_docs") or []
    queries = result.get("queries") or []
    return ChatResponse(
        answer=result["messages"][-1].content,
        session_id=req.session_id,
        used_retrieval=bool(docs),
        rewritten_query=queries[-1] if queries else None,
        retrieved_count=len(docs),
    )


@app.post("/chat/reset")
def chat_reset(session_id: str = "default"):
    """清空指定会话的记忆（让下一轮从头开始）。"""
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="未配置 AMD_API_KEY")
    cp = getattr(agent, "checkpointer", None)
    if cp is not None and hasattr(cp, "delete_thread"):
        cp.delete_thread(session_id)
    return {"status": "reset", "session_id": session_id}


# ══════════════════════════════════════════════════════════════════
#  B. 全栈接口 /api/**（前端专用）
# ══════════════════════════════════════════════════════════════════

@app.get("/api/health")
def api_health():
    return {"status": "ok", "has_api_key": bool(API_KEY), "sessions": len(SESSIONS)}


@app.get("/api/documents")
def api_documents():
    return corpus_info()


@app.get("/api/sessions")
def list_sessions():
    """会话列表，最近活跃的排前面。"""
    items = sorted(SESSIONS.values(), key=lambda s: s["last_active"], reverse=True)
    return {"sessions": items}


@app.post("/api/sessions")
def create_session():
    """新建一个空会话。返回其 id，前端拿去做 thread_id。"""
    sid = uuid.uuid4().hex[:12]
    s = _touch_session(sid)
    s["title"] = "新会话"
    return s


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    """删除会话：登记表清掉，同时把 LangGraph 里的对话记忆也删掉。

    只删登记表而不删记忆的话，下次用同一个 id 提问，模型还会记得上文 —— 那是 bug。
    """
    SESSIONS.pop(sid, None)
    graph = _graph
    if graph is not None:
        cp = getattr(graph, "checkpointer", None)
        if cp is not None and hasattr(cp, "delete_thread"):
            cp.delete_thread(sid)
    return {"status": "deleted", "session_id": sid}


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest):
    """同步问答：等图跑完一次性返回。给不支持 SSE 的调用方用。"""
    return graph_chat(req)


# ── 流式问答（SSE）────────────────────────────────────────────────
# 事件协议（前端按 event 名分发）：
#   event: stage   data: {stage, label, detail, elapsed}  某节点「已完成」
#   event: pending data: {label}                          预告「下一步在做什么」——
#                                                         updates 事件只在节点跑完时发，
#                                                         而检索+生成要几十秒，靠它填补空窗
#   event: token   data: {text}                           逐字输出
#   event: reset   data: {reason}                          作废已流出的内容 —— 长回答被上游
#                                                          掐断后重试会重流一遍，不清屏会让
#                                                          用户看到「半截 + 完整」两段重复
#   event: done    data: {answer, session_id, ...}        最终结果（权威值）
#   event: error   data: {message}                        出错
#
# ⚠️ 为什么要按节点过滤 token？
#    图里有 6 个节点都会调大模型，但只有「真正产出答案」的那几个的 token 该给用户看。
#    过滤规则（下面 ANSWER_NODES / sub_count 两个变量）：
#      · clarify（反问）、respond（汇总/闲聊）→ 总是可推
#      · retrieve_one（检索+生成）→ 仅在「问题没被拆成多个子问题」时推
#        理由：单问题路径下，最终答案就是在 retrieve_one 里生成的（respond 只透传）；
#              而多子问题时 retrieve_one 会并行跑多个分支，token 会交错串台，
#              此时应等 respond 做 reduce 后再推它的 token。
#      · rewrite / clarify_check / decompose → 内部中间产物（如"清晰"两字），不给用户看
NODE_LABELS = {
    "rewrite": "查询改写",
    "clarify_check": "澄清检查",
    "clarify": "生成反问",
    "decompose": "问题拆分",
    "retrieve_one": "资料检索",
    "respond": "生成回答",
}
ANSWER_NODES = {"clarify", "respond"}

# 「下一步是什么」的预判表。
# 为什么需要？LangGraph 的 updates 事件只在节点**跑完**时发出，
# 而单个节点（尤其检索 + 生成）实测要几十秒 —— 这段时间界面会毫无反馈。
# 图的结构是确定的，所以可以按上一个节点的结果推断下一步，提前告诉用户「正在做什么」。
_NEXT_OF = {
    "start": "查询改写",
    "rewrite": "澄清检查",
    "clarify_check": "问题拆分",     # 若需反问则改判为「生成反问」
    "clarify": "生成反问",
    "decompose": "资料检索",
    "retrieve_one": "生成回答",
    "respond": None,                 # 终点了
}


def _next_label(node: str, need_clarify: bool) -> str | None:
    if node == "clarify_check" and need_clarify:
        return "生成反问"
    return _NEXT_OF.get(node)


def _text_of(msg) -> str:
    """取出消息的纯文本（兼容 str 与多模态 list 两种 content）。"""
    c = getattr(msg, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in c
        )
    return str(c)


def _sse(event: str, data: dict) -> str:
    """拼一条 SSE 报文。ensure_ascii=False 才能让中文正常显示。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_graph(req: ChatRequest):
    """SSE 主生成器：驱动 LangGraph 的 astream，把事件转成 SSE。"""
    graph = get_graph()
    if graph is None:
        yield _sse("error", {"message": "未配置 AMD_API_KEY，无法问答"})
        return

    from langchain_core.messages import HumanMessage

    cfg = {
        "configurable": {"thread_id": req.session_id},
        "callbacks": _obs_callbacks(),   # Langfuse（未配置时为空）
    }
    inputs = {"messages": [HumanMessage(content=req.message)]}

    t0 = time.time()
    # 累积量：流式过程中自己记账，避免最后再去读一次状态
    sub_count = 1          # 子问题数，默认 1（未拆分）
    rewritten = None       # 改写后的检索问句
    doc_count = 0          # 命中资料数
    sub_questions: list[str] = []
    need_clarify = False   # 是否需要向用户反问
    final_answer = None    # 由 respond 节点的 update 给出（权威值）
    streamed: list[str] = []   # 兜底：万一没拿到 respond 的 update，用流式片段拼接

    yield _sse("stage", {"stage": "start", "label": "开始处理", "detail": req.message})
    yield _sse("pending", {"label": "查询改写"})

    try:
        async for mode, chunk in graph.astream(
            inputs, config=cfg, stream_mode=["updates", "messages", "custom"]
        ):
            # ── 0) 节点内部主动发出的信号（目前只有「重试了，请清屏」）──
            if mode == "custom":
                if isinstance(chunk, dict) and chunk.get("kind") == "reset":
                    streamed.clear()          # 后端自己的兜底缓冲同步清掉
                    yield _sse("reset", {"reason": chunk.get("reason", "")})
                continue

            # ── 1) 节点级进度 ──
            if mode == "updates":
                if not isinstance(chunk, dict):
                    continue
                for node, upd in chunk.items():
                    upd = upd if isinstance(upd, dict) else {}
                    label = NODE_LABELS.get(node, node)
                    detail = ""

                    if node == "rewrite":
                        qs = upd.get("queries") or []
                        rewritten = qs[-1] if qs else None
                        detail = rewritten or ""
                    elif node == "clarify_check":
                        need_clarify = bool(upd.get("need_clarify"))
                        detail = "问题模糊，需要先反问" if need_clarify else "问题清晰，无需反问"
                    elif node == "decompose":
                        subs = upd.get("sub_questions") or []
                        sub_count = len(subs)
                        sub_questions = list(subs)
                        detail = f"拆分为 {sub_count} 个子问题" if sub_count > 1 else "无需拆分"
                    elif node == "retrieve_one":
                        docs = upd.get("retrieved_docs") or []
                        doc_count += len(docs)
                        detail = f"累计命中 {doc_count} 条资料"
                    elif node == "respond":
                        msgs = upd.get("messages") or []
                        if msgs:
                            final_answer = _text_of(msgs[-1])
                        detail = (
                            f"汇总 {sub_count} 个子答案" if sub_count > 1
                            else "单问题，直接整理检索结果"
                        )

                    yield _sse(
                        "stage",
                        {
                            "stage": node,
                            "label": label,
                            "detail": detail,
                            "elapsed": round(time.time() - t0, 1),
                        },
                    )
                    # 预告下一步（填补节点执行期间长达数十秒的界面空窗）
                    nxt = _next_label(node, need_clarify)
                    if nxt:
                        yield _sse("pending", {"label": nxt})

            # ── 2) 逐字输出 ──
            else:
                try:
                    msg, meta = chunk
                except (TypeError, ValueError):
                    continue
                node = (meta or {}).get("langgraph_node", "")
                # 过滤规则见上方注释
                if node in ANSWER_NODES or (node == "retrieve_one" and sub_count <= 1):
                    # 只推「增量片段」（AIMessageChunk），整条 AIMessage 不推
                    if type(msg).__name__ != "AIMessageChunk":
                        continue
                    txt = _text_of(msg)
                    if txt:
                        streamed.append(txt)
                        yield _sse("token", {"text": txt})

    except asyncio.CancelledError:
        # 客户端断开（用户点了「停止」或关了页面）——静默收场
        raise
    except Exception as e:  # noqa: BLE001
        yield _sse("error", {"message": f"{type(e).__name__}: {str(e)[:200]}"})
        return

    # ── 3) 收尾 ──
    answer = final_answer or "".join(streamed) or "（未生成内容）"
    s = _touch_session(req.session_id, req.message)
    yield _sse(
        "done",
        {
            "answer": answer,
            "session_id": req.session_id,
            "title": s["title"],
            "used_retrieval": doc_count > 0,
            "rewritten_query": rewritten,
            "retrieved_count": doc_count,
            "sub_questions": sub_questions,
            "elapsed": round(time.time() - t0, 1),
        },
    )


@app.post("/api/chat/stream")
async def api_chat_stream(req: ChatRequest):
    if not _SID.match(req.session_id):
        raise HTTPException(status_code=400, detail="session_id 仅允许字母数字与 . _ : -，长度≤64")

    return StreamingResponse(
        _stream_graph(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # 让 nginx 之类的反代不要缓冲
        },
    )


# ══════════════════════════════════════════════════════════════════
#  C. 前端静态资源托管
#  构建产物在 frontend/dist（由 `npm run build` 生成）。
#  存在就挂上，不存在也不报错 —— 这样「只跑后端」的场景不受影响。
# ══════════════════════════════════════════════════════════════════
_DIST = Path(__file__).parent.parent / "frontend" / "dist"

if _DIST.exists() and (_DIST / "index.html").exists():
    if (_DIST / "assets").exists():
        app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/")
    def index_page():
        return FileResponse(_DIST / "index.html")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        """SPA 路由回退：非 /api 的未知路径一律交给前端路由处理。

        ⚠️ 必须放在最后注册，且要排除 /api 前缀 —— 否则会把接口请求也吞掉。
        """
        if full_path.startswith(("api/", "assets/")):
            raise HTTPException(status_code=404, detail="Not Found")
        # 若请求的是真实存在的静态文件（favicon 等），直接返回
        candidate = _DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")
