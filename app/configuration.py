# D:\Desktop\rag-demo\app\configuration.py
"""运行时配置注入。

解决的问题：图编译一次、多次调用，每次调用想用不同参数（比如这轮检索 5 条、
下轮检索 10 条），总不能重新 build 一次图。

做法：调用时传 config={"configurable": {"retrieval_k": 5}}，
      节点函数里用 Configuration.from_runnable_config(config) 取出来。
这是 LangGraph 官方推荐的「每个请求一套参数」的实现方式。
"""

from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

try:  # 包方式
    from .config import settings
    from .prompts import AGENT_SYSTEM_PROMPT, QUERY_REWRITE_SYSTEM_PROMPT
except ImportError:  # 扁平方式（cd app && uvicorn main:app）
    from config import settings
    from prompts import AGENT_SYSTEM_PROMPT, QUERY_REWRITE_SYSTEM_PROMPT


@dataclass(kw_only=True)
class Configuration:
    """一次调用可覆盖的参数。未提供的字段自动回落到全局 settings。"""

    retrieval_k: int = settings.retrieval_k
    """检索返回条数。"""

    response_system_prompt: str = AGENT_SYSTEM_PROMPT
    """回答节点用的系统提示词。"""

    query_system_prompt: str = QUERY_REWRITE_SYSTEM_PROMPT
    """查询改写节点用的系统提示词。"""

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """从 LangGraph 传入的 config 中提取本类字段；缺省值自动补齐。

        config 的结构形如：
            {"configurable": {"thread_id": "abc", "retrieval_k": 5}}
        thread_id 等 LangGraph 自己的键会被忽略，只挑我们声明的字段。
        """
        configurable = (config or {}).get("configurable", {}) or {}
        # 只保留本 dataclass 声明过的字段（含值为 None 的，表示「用默认值」）
        valid = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {
            k: v for k, v in configurable.items() if k in valid and v is not None
        }
        return cls(**kwargs)
