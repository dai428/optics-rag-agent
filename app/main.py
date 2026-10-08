"""
main.py — FastAPI 服务：把光学 RAG Agent 暴露成 HTTP 接口

张万森 · 光学 RAG Agent
启动：uvicorn app.main:app --host 127.0.0.1 --port 8000
接口：
  GET  /health            健康检查（无需 Key）
  GET  /documents         知识库概况（无需 Key）
  POST /chat              提问（create_agent 黑盒版；session_id 会话隔离）
  POST /graph/chat        提问（LangGraph 显式版；含查询改写）
  POST /chat/reset        清空某会话上下文

可观测：配置 .env 里的 LANGFUSE_* 后，自动把每次链路上报 Langfuse；未配置则静默跳过。
"""
import re

import chromadb  # 主线程预导入：规避 chromadb 在工作线程「首次导入+首次建客户端」的 Rust 绑定初始化问题（很轻，约 1.3s）

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:  # 包方式：uvicorn app.main:app（在项目根目录执行）
    from .knowledge_base import corpus_info
    from .rag_agent import build_agent, API_KEY
    from .observability import callbacks as _obs_callbacks
except ImportError:  # 扁平方式：cd app && uvicorn main:app
    from knowledge_base import corpus_info
    from rag_agent import build_agent, API_KEY
    from observability import callbacks as _obs_callbacks

app = FastAPI(title="光学 RAG Agent 服务 · 张万森", version="1.2")

_agent = None   # create_agent 懒加载
_graph = None   # LangGraph 懒加载
_SID = re.compile(r"^[A-Za-z0-9._:\-]{1,64}$")


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