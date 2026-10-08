# D:\Desktop\rag-demo\app\state.py
"""图状态（State）定义。

LangGraph 的核心概念：整个流程像一条流水线（graph），
每个工位（node）读写同一个「共享白板」（state）。

本文件定义白板上要记什么：
  ── 基础 ──
  messages      ：对话历史（Human / AI / Tool 消息）
  queries       ：检索用的问句（改写后的）
  retrieved_docs：本轮检索到的文档片段（已回溯为父块）

  ── P1 新增 ──
  sub_questions ：拆分出的子问题（多子问题并行检索用）
  sub_answers   ：每个子问题的答案（map-reduce 的 map 阶段产物）
  need_clarify  ：是否需要向用户反问澄清
  clarify_questions：反问的问题列表
  history_summary：历史对话的滚动摘要（替代全量历史，控 token）
  tool_calls    ：已执行的检索/工具调用次数（预算控制）
  iterations    ：图迭代轮次（预算控制）

设计要点：字段都给了默认值，LangGraph 建图时才能推断出 schema。
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, Sequence

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def add_queries(existing: Sequence[str], new: Sequence[str]) -> list[str]:
    """归约函数（reducer）：把「新问句」追加到「已有问句」后面。

    为什么需要它？
      LangGraph 默认对同名字段是「覆盖」语义。给字段标注 Annotated[..., add_queries]
      后，节点每次返回 {"queries": [...]} 就变成「追加」而非「覆盖」。
      这与 add_messages 对消息列表的处理是同一个套路。
    """
    return list(existing) + list(new)


def add_docs(existing: Sequence[Document], new: Sequence[Document]) -> list[Document]:
    """文档累加 reducer。

    多子问题并行检索时，几个分支会各自返回一批文档；
    用这个 reducer 让它们「合并」而不是「互相覆盖」。
    并行分支的写入顺序是不确定的，所以最后统一去重。
    """
    merged = list(existing) + list(new)
    seen, out = set(), []
    for d in merged:
        key = d.metadata.get("parent_id") or hash(d.page_content)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def sum_int(existing: int, new: int) -> int:
    """整数累加 reducer。

    为什么需要？
      多子问题并行时，每个 retrieve_one 分支都会返回 {"tool_calls": N}。
      LangGraph 默认规则是「一个 step 内每个字段只能被写一次」，
      多个并行分支同时写同一个 int 字段会直接抛：
          InvalidUpdateError: At key 'tool_calls': Can receive only one value per step.
      加了这个 reducer 后，多个分支的写入会被**相加**，
      正好符合「累计消耗了多少次检索配额」的语义。
    """
    return (existing or 0) + (new or 0)


def last_int(existing: int, new: int) -> int:
    """整数取最新值 reducer（用于迭代轮次这类「只需要一个值」的字段）。"""
    return new if new is not None else existing


@dataclass(kw_only=True)
class InputState:
    """对外输入状态：外部只需要传 messages 进来。"""

    messages: Annotated[Sequence[AnyMessage], add_messages]
    """对话历史。add_messages 保证「追加 + 按 id 去重更新」，而不是整体覆盖。"""


@dataclass(kw_only=True)
class State(InputState):
    """图内部完整状态。"""

    # ── 检索 ──────────────────────────────────────────
    queries: Annotated[list[str], add_queries] = field(default_factory=list)
    """检索问句列表。首轮 = 用户原话；多轮 = 改写后的独立问句。"""

    retrieved_docs: Annotated[list[Document], add_docs] = field(default_factory=list)
    """本轮检索命中的文档片段（已回溯为父块），供回答节点拼进提示词。"""

    # ── P1-2 多子问题并行（map-reduce）───────────────
    sub_questions: list[str] = field(default_factory=list)
    """从原始问题拆出的子问题。没有子问题时为单元素列表（=原问题）。"""

    sub_answers: Annotated[list[str], add_queries] = field(default_factory=list)
    """每个子问题各自的答案（map 阶段产物），供 reduce 节点汇总。"""

    # ── P1-1 澄清反问 ────────────────────────────────
    need_clarify: bool = False
    """问题是否过于模糊、需要先向用户反问。"""

    clarify_questions: list[str] = field(default_factory=list)
    """反问给用户的问题列表。"""

    # ── P1-3 历史压缩 ────────────────────────────────
    history_summary: str = ""
    """历史对话的滚动摘要。非空时替代全量历史注入提示词。"""

    # ── P1-4 预算控制 ────────────────────────────────
    tool_calls: Annotated[int, sum_int] = 0
    """已执行的工具（检索）调用次数。

    ⚠️ 必须带 sum_int reducer：
       多子问题并行时，N 个 retrieve_one 分支会同时写这个字段。
       不带 reducer 会抛 InvalidUpdateError（一个 step 只能写一个值）。
    """

    iterations: Annotated[int, last_int] = 0
    """图迭代轮次。并行的多个分支可能各写一次，取最后一次即可。"""

    _extra: dict[str, Any] = field(default_factory=dict, repr=False)
    """预留：想加「是否兜底」等字段时往这里加。"""
