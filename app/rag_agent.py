# D:\Desktop\rag-demo\app\rag_agent.py
"""Agent 版 RAG（create_agent + 检索工具 + 会话记忆）

与「朴素 RAG 链」的关键区别：
  朴链接：每次提问都无条件检索 → 喂给 LLM
  Agent ：把检索包成「工具」，由 LLM 自己决定要不要调用（打招呼就不检索）

生成回答：AMD 云（需 Key，读 .env）
检索嵌入：本地 BAAI/bge-small-zh-v1.5（免费）
会话记忆：InMemorySaver，按 thread_id(=session_id) 隔离

配置来源：app/config.py · 提示词来源：app/prompts.py
"""

import os

# 环境自举（同 knowledge_base.py，保证单独运行本模块时也生效）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "D:/DevCache/huggingface")
os.environ.setdefault("no_proxy", "*")
os.environ.setdefault("NO_PROXY", "*")

try:  # 包方式：app.rag_agent
    from .config import settings
    from .prompts import AGENT_SYSTEM_PROMPT, RETRIEVER_TOOL_DESCRIPTION
except ImportError:  # 扁平方式（脚本直接运行 / cd app && uvicorn main:app）
    from config import settings
    from prompts import AGENT_SYSTEM_PROMPT, RETRIEVER_TOOL_DESCRIPTION

# 兼容旧引用：main.py 等处按 API_KEY 判断服务可用性
API_KEY = settings.api_key


def build_llm(purpose: str = "answer"):
    """云端 LLM（OpenAI 兼容接口）。无 Key 返回 None。

    purpose="answer"  正式回答，用大模型 + 长超时
    purpose="rewrite" 查询改写，任务简单，可用小模型 + 短超时
    """
    if not settings.api_key:
        return None
    from langchain_openai import ChatOpenAI

    if purpose == "rewrite":
        model = settings.rewrite_model or settings.llm_model
        timeout = settings.rewrite_timeout
    else:
        model = settings.llm_model
        timeout = settings.request_timeout

    return ChatOpenAI(
        model=model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=settings.temperature,
        timeout=timeout,                    # AMD 共享模型在高并发时会排队，放宽超时
        max_retries=settings.max_retries,   # 遇 429 自动退避重试
    )


def build_agent():
    """构建 create_agent（工具自决检索 + 会话记忆）。无 Key 返回 None。"""
    from langchain.agents import create_agent
    from langchain_core.tools import create_retriever_tool
    from langgraph.checkpoint.memory import InMemorySaver

    try:  # 包方式：app.rag_agent
        from .knowledge_base import get_retriever
    except ImportError:  # 扁平方式（脚本直接运行 / cd app && uvicorn main:app）
        from knowledge_base import get_retriever

    llm = build_llm()
    if llm is None:
        return None

    retriever_tool = create_retriever_tool(
        get_retriever(),
        "search_optics_knowledge_base",
        RETRIEVER_TOOL_DESCRIPTION,
    )

    return create_agent(
        model=llm,
        tools=[retriever_tool],
        system_prompt=AGENT_SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


if __name__ == "__main__":
    import sys

    from langchain_core.messages import HumanMessage

    agent = build_agent()
    if agent is None:
        sys.exit("❌ 未配置 AMD_API_KEY，无法运行")

    cfg = {"configurable": {"thread_id": "cli-1"}}
    # 第 1 问是闲聊（不该检索），第 2/3 问是专业问题（该检索 + 记住上下文）
    for q in ["你好呀", "LCVR 相比传统旋转波片有什么优势？", "那它的定标怎么做才准？"]:
        r = agent.invoke({"messages": [HumanMessage(content=q)]}, config=cfg)
        print("=" * 60)
        print(f"❓ {q}")
        print(f"💡 {r['messages'][-1].content}")
    print("=" * 60)
