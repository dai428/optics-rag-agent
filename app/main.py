"""
main.py — FastAPI 服务：把科研知识库 Agent 暴露成 HTTP 接口 + 托管前端页面

启动（两种皆可）：
    # 在项目根目录
    ../.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
    # 或进 app/ 目录
    ../.venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

接口分组
────────
【A. 基础接口】向后兼容，供脚本/调试用
  GET  /health                    存活探针（不触发模型加载）
  GET  /ready                     就绪探针（检查索引与模型配置）
  GET  /documents                 知识库概况
  POST /chat                      提问（create_agent 黑盒版，对照用）
  POST /graph/chat                提问（LangGraph 显式版）
  POST /chat/reset                清空某会话记忆

【B. 全栈接口 /api/**】前端专用
  GET    /api/health              健康检查
  GET    /api/ready               就绪检查（索引状态、模型、数据库）
  GET    /api/documents           知识库概况
  GET    /api/documents/{id}      单个文档详情（含索引状态）
  POST   /api/documents/upload    上传文档（multipart）
  DELETE /api/documents/{id}      删除文档并重建索引
  POST   /api/documents/reindex   手动触发重建
  GET    /api/sessions            会话列表
  POST   /api/sessions            新建会话
  GET    /api/sessions/{sid}/messages  会话历史消息
  PATCH  /api/sessions/{sid}      重命名会话
  DELETE /api/sessions/{sid}      删除会话
  POST   /api/chat                同步问答
  POST   /api/chat/stream         流式问答（SSE）
  POST   /api/feedback            用户反馈
  GET    /api/metrics             运行指标（拒答率、耗时、引用数）

可观测：配置 .env 里的 LANGFUSE_* 后自动上报链路；未配置则静默跳过。
"""

import asyncio
import json
import re
import threading
import time
import uuid
from pathlib import Path

import chromadb  # 主线程预导入：规避 chromadb 在工作线程首次初始化 Rust 绑定的问题

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:  # 包方式：uvicorn app.main:app（在项目根目录执行）
    from .config import settings
    from .ingestion import SUPPORTED_SUFFIXES, build_index, delete_document, save_upload
    from .knowledge_base import corpus_info, index_status
    from .observability import callbacks as _obs_callbacks
    from .rag_agent import API_KEY, build_agent
    from .storage import get_database
except ImportError:  # 扁平方式：cd app && uvicorn main:app
    from config import settings
    from ingestion import SUPPORTED_SUFFIXES, build_index, delete_document, save_upload
    from knowledge_base import corpus_info, index_status
    from observability import callbacks as _obs_callbacks
    from rag_agent import API_KEY, build_agent
    from storage import get_database

app = FastAPI(
    title="科研知识库 AI Agent",
    version="3.0",
    description="面向科研文献的可溯源问答智能体：混合检索 + 重排 + 结构化引用 + 确定性工具",
)

_agent = None   # create_agent 黑盒版（对照用）
_graph = None   # LangGraph 显式版
_SID = re.compile(r"^[A-Za-z0-9._:\-]{1,64}$")

# 索引重建是重操作：用锁保证同一时刻只有一个构建在跑，避免并发写坏 staging
_index_lock = threading.Lock()

# 简易限流：session_id → 时间戳列表。单机演示够用，多实例需换成 Redis。
_rate_buckets: dict[str, list[float]] = {}
_rate_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════
#  基础设施
# ══════════════════════════════════════════════════════════════════

def db():
    return get_database()


def _validate_sid(sid: str) -> str:
    """校验会话 ID。LangGraph 的 thread_id 不能用中文（会让 invoke 挂死）。"""
    if not _SID.match(sid):
        raise HTTPException(
            status_code=400, detail="session_id 仅允许字母数字与 . _ : -，长度≤64"
        )
    return sid


def _check_rate_limit(sid: str) -> None:
    """每分钟请求数限流。RATE_LIMIT_PER_MINUTE=0 时关闭。"""
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return
    now = time.time()
    with _rate_lock:
        bucket = [t for t in _rate_buckets.get(sid, []) if now - t < 60]
        if len(bucket) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"请求过于频繁：每分钟上限 {limit} 次，请稍后重试",
            )
        bucket.append(now)
        _rate_buckets[sid] = bucket
        # 防止字典无界增长
        if len(_rate_buckets) > 1000:
            for key in [k for k, v in _rate_buckets.items() if not v][:500]:
                _rate_buckets.pop(key, None)


def _require_token(request: Request) -> None:
    """可选访问控制：配置 APP_ACCESS_TOKEN 后，写接口需要 Bearer 令牌。"""
    expected = settings.access_token
    if not expected:
        return
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if token != expected:
        raise HTTPException(status_code=401, detail="缺少或错误的访问令牌")


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
    """把底层模型的报错翻译成合适的 HTTP 状态码，并避免泄露内部细节。"""
    msg = str(e)
    low = msg.lower()
    if "429" in msg or "rate" in low or "concurrency" in low or "limit" in low:
        return HTTPException(
            status_code=429,
            detail="模型并发已达上限（上游共享模型限制），请稍后重试或换用其他模型",
        )
    if "timeout" in low or "timed out" in low:
        return HTTPException(status_code=504, detail="模型响应超时，请稍后重试")
    # 不回传完整堆栈：可能含端点、路径等内部信息
    return HTTPException(status_code=500, detail=f"模型调用失败：{type(e).__name__}")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000, description="用户问题")
    session_id: str = Field(default="default", max_length=64, description="会话 ID")


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    used_retrieval: bool
    rewritten_query: str | None = None
    retrieved_count: int = 0
    citations: list[dict] = Field(default_factory=list)
    abstained: bool = False
    abstain_reason: str = ""


class FeedbackRequest(BaseModel):
    session_id: str = Field(default="default", max_length=64)
    rating: str = Field(..., pattern="^(up|down)$")
    comment: str = Field(default="", max_length=1000)
    question: str = Field(default="", max_length=4000)
    answer: str = Field(default="", max_length=20000)


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=80)


# ══════════════════════════════════════════════════════════════════
#  A. 基础接口（向后兼容）
# ══════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """存活探针：只检查进程与配置，不加载模型、不查索引。

    为什么与 /ready 分开：存活探针会被容器编排高频调用，任何重操作（加载嵌入
    模型、扫描语料）都不该放在这里，否则探针本身就会拖慢服务。
    """
    return {"status": "ok", "has_api_key": bool(API_KEY)}


@app.get("/ready")
def ready():
    """就绪探针：确认索引可用、模型已配置、数据库可写。"""
    checks: dict[str, dict] = {}

    checks["api_key"] = {
        "ok": bool(API_KEY),
        "detail": "已配置模型 API Key" if API_KEY else "缺少 AMD_API_KEY，问答不可用",
    }

    try:
        status = index_status()
        checks["index"] = {
            "ok": status.get("status") == "ok",
            "detail": status.get("detail") or status.get("status", ""),
            "stale": status.get("stale", True),
        }
    except Exception as e:  # noqa: BLE001
        checks["index"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}

    try:
        checks["database"] = {"ok": db().schema_version() > 0, "detail": "SQLite 可用"}
    except Exception as e:  # noqa: BLE001
        checks["database"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}

    ready_ok = all(c["ok"] for c in checks.values())
    return {"status": "ready" if ready_ok else "degraded", "checks": checks}


@app.get("/documents")
def documents():
    """返回知识库语料概况（不需要 API Key）。"""
    return corpus_info()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    _validate_sid(req.session_id)
    _check_rate_limit(req.session_id)

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
    answer = msgs[-1].content
    _record_turn(req.session_id, req.message, answer, used_retrieval=used_retrieval)
    return ChatResponse(
        answer=answer,
        session_id=req.session_id,
        used_retrieval=used_retrieval,
    )


@app.post("/graph/chat", response_model=ChatResponse)
def graph_chat(req: ChatRequest):
    """LangGraph 显式版：改写 → 澄清/拆分 → 并行检索 → 汇总，返回结构化引用。"""
    _validate_sid(req.session_id)
    _check_rate_limit(req.session_id)

    graph = get_graph()
    if graph is None:
        raise HTTPException(status_code=503, detail="未配置 AMD_API_KEY，/graph/chat 不可用")

    from langchain_core.messages import HumanMessage

    started = time.time()
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=req.message)]},
            config={
                "configurable": {"thread_id": req.session_id},
                "callbacks": _obs_callbacks(),
            },
        )
    except Exception as e:  # noqa: BLE001
        raise _map_llm_error(e) from e

    docs = result.get("retrieved_docs") or []
    queries = result.get("queries") or []
    answer = result["messages"][-1].content
    citations = _citations_of(docs)
    abstained = bool(result.get("abstained"))

    _record_turn(
        req.session_id,
        req.message,
        answer,
        rewritten=queries[-1] if queries else "",
        retrieved=len(docs),
        abstained=abstained,
        citations=len(citations),
        elapsed_ms=int((time.time() - started) * 1000),
    )
    return ChatResponse(
        answer=answer,
        session_id=req.session_id,
        used_retrieval=bool(docs),
        rewritten_query=queries[-1] if queries else None,
        retrieved_count=len(docs),
        citations=citations,
        abstained=abstained,
        abstain_reason=result.get("abstain_reason", ""),
    )


@app.post("/chat/reset")
def chat_reset(session_id: str = "default"):
    """清空指定会话在旧 Agent 与主 LangGraph 中的全部记忆。"""
    _validate_sid(session_id)
    for runtime in (_agent, _graph):
        cp = getattr(runtime, "checkpointer", None) if runtime is not None else None
        if cp is not None and hasattr(cp, "delete_thread"):
            cp.delete_thread(session_id)
    db().delete_session(session_id)
    return {"status": "reset", "session_id": session_id}


# ══════════════════════════════════════════════════════════════════
#  B. 全栈接口 /api/**（前端专用）
# ══════════════════════════════════════════════════════════════════

@app.get("/api/health")
def api_health():
    return {"status": "ok", "has_api_key": bool(API_KEY), "sessions": len(db().list_sessions())}


@app.get("/api/ready")
def api_ready():
    return ready()


@app.get("/api/documents")
def api_documents():
    return corpus_info()


# ── 文档管理 ──────────────────────────────────────────────────────

@app.post("/api/documents/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    subdir: str = Form(default=""),
    reindex: bool = Form(default=True),
):
    """上传文档并（默认）立即重建索引。

    上传后立刻重建，是为了让「上传 → 提问」这个闭环不出现「文件在但搜不到」
    的困惑期。索引构建本身走 staging → 原子切换，失败不会破坏现有索引。
    """
    _require_token(request)

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上传的文件为空")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = "、".join(sorted(SUPPORTED_SUFFIXES))
        raise HTTPException(
            status_code=400, detail=f"不支持的文件类型 {suffix or '(无扩展名)'}；支持：{supported}"
        )

    safe_name = Path(file.filename or "unnamed").name
    rel_path = f"{subdir.strip('/')}/{safe_name}" if subdir.strip("/") else safe_name

    ok, message = save_upload(settings, rel_path, raw)
    if not ok:
        raise HTTPException(status_code=400, detail=message)

    response: dict = {"status": "uploaded", "rel_path": rel_path, "message": message}

    if reindex:
        report = _run_index(force=False)
        response["index"] = report
        if not report.get("ok"):
            response["status"] = "uploaded_but_index_failed"
            response["hint"] = "文件已保存，但索引未更新；请检查文件内容或手动触发重建"

    return response


@app.delete("/api/documents/{rel_path:path}")
def api_delete_document(request: Request, rel_path: str):
    """删除文档并重建索引。rel_path 使用相对语料目录的路径。"""
    _require_token(request)
    with _index_lock:
        report = delete_document(settings, rel_path)
    if not report.ok and report.mode == "delete":
        raise HTTPException(status_code=404, detail="；".join(report.warnings) or "文件不存在")
    return {"status": "deleted", "rel_path": rel_path, "index": report.to_dict()}


@app.post("/api/documents/reindex")
def api_reindex(request: Request, force: bool = Body(default=False, embed=True)):
    """手动重建索引。force=True 时忽略增量判断，全量重建。"""
    _require_token(request)
    return _run_index(force=force)


def _run_index(force: bool) -> dict:
    """在锁内执行索引构建，并让向量库单例失效以重新加载。"""
    with _index_lock:
        report = build_index(settings, force=force, incremental=not force)
        # 索引已切换，清掉单例缓存，下一次检索会读到新集合
        try:
            try:
                from . import knowledge_base as kb
            except ImportError:
                import knowledge_base as kb

            kb._vs = None
        except Exception:  # noqa: BLE001
            pass
    return report.to_dict()


# ── 会话管理 ──────────────────────────────────────────────────────

@app.get("/api/sessions")
def list_sessions():
    """会话列表，最近活跃的排前面（来自 SQLite，重启不丢）。"""
    return {"sessions": [s.to_dict() for s in db().list_sessions()]}


@app.post("/api/sessions")
def create_session():
    """新建一个空会话，返回其 id（前端拿去做 thread_id）。"""
    sid = uuid.uuid4().hex[:12]
    return db().create_session(sid, "新会话").to_dict()


@app.get("/api/sessions/{sid}/messages")
def session_messages(sid: str):
    """会话的历史消息（服务端持久化，换设备也能看到）。"""
    _validate_sid(sid)
    if db().get_session(sid) is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"session_id": sid, "messages": db().list_messages(sid)}


@app.patch("/api/sessions/{sid}")
def rename_session(sid: str, req: RenameRequest):
    _validate_sid(sid)
    if not db().rename_session(sid, req.title):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"status": "renamed", "session_id": sid, "title": req.title}


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    """删除会话：数据库记录、LangGraph 记忆、旧 Agent 记忆一起清。

    只删数据库而不删记忆的话，下次用同一个 id 提问，模型还会记得上文 —— 那是 bug。
    """
    _validate_sid(sid)
    for runtime in (_graph, _agent):
        cp = getattr(runtime, "checkpointer", None) if runtime is not None else None
        if cp is not None and hasattr(cp, "delete_thread"):
            cp.delete_thread(sid)
    db().delete_session(sid)
    return {"status": "deleted", "session_id": sid}


# ── 问答 ──────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest):
    """同步问答：等图跑完一次性返回。给不支持 SSE 的调用方用。"""
    return graph_chat(req)


@app.post("/api/feedback")
def api_feedback(request: Request, req: FeedbackRequest):
    """用户反馈。失败案例可据此进入评测集，形成改进闭环。"""
    _require_token(request)
    _validate_sid(req.session_id)
    fid = db().add_feedback(
        req.session_id,
        req.rating,
        comment=req.comment,
        question=req.question,
        answer=req.answer,
    )
    return {"status": "recorded", "feedback_id": fid}


@app.get("/api/metrics")
def api_metrics():
    """运行指标：拒答率、平均耗时、引用数、用户满意度。"""
    metrics = db().metrics()
    metrics["feedback"] = db().feedback_summary()
    metrics["recent_queries"] = db().recent_queries(limit=10)
    try:
        metrics["index"] = index_status()
    except Exception as e:  # noqa: BLE001
        metrics["index"] = {"status": "unknown", "detail": f"{type(e).__name__}: {e}"}
    return metrics


def _record_turn(
    sid: str,
    question: str,
    answer: str,
    rewritten: str = "",
    retrieved: int = 0,
    abstained: bool = False,
    citations: int = 0,
    elapsed_ms: int = 0,
    used_retrieval: bool = False,
) -> None:
    """把一轮问答写入数据库：会话、消息、查询日志。"""
    try:
        database = db()
        database.touch_session(sid, question)
        database.add_message(sid, "user", question)
        database.add_message(
            sid,
            "assistant",
            answer,
            meta={
                "rewritten_query": rewritten,
                "retrieved_count": retrieved,
                "used_retrieval": used_retrieval,
                "abstained": abstained,
                "citations": citations,
            },
        )
        database.log_query(
            sid,
            question,
            rewritten=rewritten,
            retrieved=retrieved,
            abstained=abstained,
            citations=citations,
            elapsed_ms=elapsed_ms,
        )
    except Exception as e:  # noqa: BLE001 —— 持久化失败不该让回答失败
        print(f"[warn] 会话持久化失败：{type(e).__name__}: {e}")


def _citations_of(docs) -> list[dict]:
    try:
        try:
            from .knowledge_base import build_citations
        except ImportError:
            from knowledge_base import build_citations

        return build_citations(docs)
    except Exception:  # noqa: BLE001
        return []


# ── 流式问答（SSE）────────────────────────────────────────────────
# 事件协议（前端按 event 名分发）：
#   event: stage   data: {stage, label, detail, elapsed}  某节点「已完成」
#   event: pending data: {label}                          预告「下一步在做什么」
#   event: token   data: {text}                           逐字输出
#   event: reset   data: {reason}                         作废已流出的内容
#   event: done    data: {answer, session_id, ...}        最终结果（权威值）
#   event: error   data: {message}                        出错
#
# ⚠️ 为什么 token 要按节点过滤？
#    图里有多个节点都会调大模型，但只有「真正产出答案」的那几个的 token 该给用户看。
#    · clarify（反问）、respond（汇总）→ 总是可推
#    · retrieve_one（检索+生成）→ 仅在「问题没被拆成多个子问题」时推
#      多子问题时它并行跑多个分支，token 会交错串台，必须等 respond 汇总后再推。
NODE_LABELS = {
    "begin_turn": "准备本轮",
    "rewrite": "查询改写",
    "clarify_check": "澄清检查",
    "clarify": "生成反问",
    "decompose": "问题拆分",
    "retrieve_one": "资料检索",
    "respond": "生成回答",
}
ANSWER_NODES = {"clarify", "respond"}

# 「下一步是什么」的预判表：updates 事件只在节点跑完时发出，而单节点实测要几十秒，
# 靠它填补界面空窗。
_NEXT_OF = {
    "start": "准备本轮",
    "begin_turn": "查询改写",
    "rewrite": "澄清检查",
    "clarify_check": "问题拆分",
    "clarify": "生成反问",
    "decompose": "资料检索",
    "retrieve_one": "生成回答",
    "respond": None,
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
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
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
        "callbacks": _obs_callbacks(),
    }
    inputs = {"messages": [HumanMessage(content=req.message)]}

    t0 = time.time()
    sub_count = 1
    rewritten = None
    doc_count = 0
    sub_questions: list[str] = []
    need_clarify = False
    final_answer = None
    streamed: list[str] = []
    citations: list[dict] = []
    abstained = False
    abstain_reason = ""

    yield _sse("stage", {"stage": "start", "label": "开始处理", "detail": req.message})
    yield _sse("pending", {"label": "准备本轮"})

    try:
        async for mode, chunk in graph.astream(
            inputs, config=cfg, stream_mode=["updates", "messages", "custom"]
        ):
            # ── 0) 节点内部主动发出的信号（目前只有「重试了，请清屏」）──
            if mode == "custom":
                if isinstance(chunk, dict) and chunk.get("kind") == "reset":
                    streamed.clear()
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
                        abstained = bool(upd.get("abstained"))
                        abstain_reason = upd.get("abstain_reason", "") or abstain_reason
                        detail = f"累计命中 {doc_count} 条资料"
                        if abstained:
                            detail = "未找到足够相关的证据"
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
                if node in ANSWER_NODES or (node == "retrieve_one" and sub_count <= 1):
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

    # ── 3) 收尾：读最终状态拿结构化引用 ──
    answer = final_answer or "".join(streamed) or "（未生成内容）"
    elapsed = time.time() - t0
    try:
        snapshot = graph.get_state(cfg)
        values = getattr(snapshot, "values", None) or {}
        docs = values.get("retrieved_docs") or []
        citations = _citations_of(docs)
        if not doc_count:
            doc_count = len(docs)
        abstained = bool(values.get("abstained", abstained))
        abstain_reason = values.get("abstain_reason", abstain_reason)
    except Exception as e:  # noqa: BLE001 —— 拿不到引用不该影响回答
        print(f"[warn] 读取最终状态失败：{type(e).__name__}: {e}")

    _record_turn(
        req.session_id,
        req.message,
        answer,
        rewritten=rewritten or "",
        retrieved=doc_count,
        abstained=abstained,
        citations=len(citations),
        elapsed_ms=int(elapsed * 1000),
        used_retrieval=doc_count > 0,
    )
    session = db().get_session(req.session_id)

    yield _sse(
        "done",
        {
            "answer": answer,
            "session_id": req.session_id,
            "title": session.title if session else "",
            "used_retrieval": doc_count > 0,
            "rewritten_query": rewritten,
            "retrieved_count": doc_count,
            "sub_questions": sub_questions,
            "citations": citations,
            "abstained": abstained,
            "abstain_reason": abstain_reason,
            "elapsed": round(elapsed, 1),
        },
    )


@app.post("/api/chat/stream")
async def api_chat_stream(req: ChatRequest):
    _validate_sid(req.session_id)
    _check_rate_limit(req.session_id)

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
        candidate = _DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")
