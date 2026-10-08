# D:\Desktop\rag-demo\app\observability.py
"""可观测性（Langfuse）—— 让每次「检索→生成」的链路可见、可回溯。

为什么需要？
-----------
  调试 RAG 最痛苦的不是「答错了」，而是「答错了不知道错在哪一环」：
    是改写把问题改歪了？还是召回没召到？还是召回了但重排把对的排后面了？还是模型答跑偏了？
  日志只能看到结果，看不到链路。Langfuse 把每一步（每个节点的输入/输出/耗时/token）
  都记下来，在网页上能像时间线一样逐帧回放。

设计原则：**零侵入 + 优雅降级**
-----------------------------
  · 业务代码不 import langfuse，只 import 本模块的 get_langfuse_handler()。
  · 没装 langfuse、或没配 LANGFUSE_* 环境变量时，返回空列表 → 完全不影响原有功能。
  · 配了之后，把 handler 塞进 graph.invoke(..., config={"callbacks": handlers}) 即生效。

Langfuse 是什么（给不熟悉的人）
-------------------------------
  一个开源的 LLM 应用可观测平台（类似「LLM 应用的 APM」）。
  自部署或云版都行。本模块用 OpenAI 兼容的方式对接，配置三个环境变量即可：
      LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
  （这三个值写在 .env 里，不要提交到 Git。）
"""

import os

# 全局缓存：handler 只初始化一次（初始化会建后台上报线程，别重复建）
_handler_cache = None
_initialized = False


def is_enabled() -> bool:
    """是否配置了 Langfuse（三个必要环境变量齐不齐）。"""
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    )


def get_langfuse_handler():
    """返回 LangChain 的 CallbackHandler；未启用时返回 None。

    用法：
        handlers = [h for h in [get_langfuse_handler()] if h]
        graph.invoke(..., config={"configurable": {...}, "callbacks": handlers})
    """
    global _handler_cache, _initialized
    if _handler_cache is not None or _initialized:
        return _handler_cache

    _initialized = True
    if not is_enabled():
        # 没配 → 静默降级，不报错。这是关键：不能因为没配观测就让整个服务起不来。
        return None

    try:
        from langfuse.langchain import CallbackHandler

        _handler_cache = CallbackHandler()
        print("[observability] Langfuse 已启用，链路将上报到 "
              f"{os.environ.get('LANGFUSE_HOST', 'https://cloud.langfuse.com')}")
    except ImportError:
        print("[observability] 未安装 langfuse，跳过观测（pip install langfuse 可启用）")
    except Exception as e:  # noqa: BLE001
        # 连不上、Key 错等都不应该拖垮主流程
        print(f"[observability] Langfuse 初始化失败，已降级为无观测：{type(e).__name__}: {e}")
    return _handler_cache


def callbacks() -> list:
    """便捷函数：直接返回可塞进 config 的 callbacks 列表（可能为空）。"""
    h = get_langfuse_handler()
    return [h] if h else []


def flush():
    """进程退出前调用，确保上报队列发完（Langfuse 是异步批量上报）。"""
    if _handler_cache is not None:
        try:
            from langfuse import Langfuse

            Langfuse().flush()
        except Exception:  # noqa: BLE001
            pass
