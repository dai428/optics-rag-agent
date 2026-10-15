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
  sub_answers   ：每个子问题的答案（map 阶段产物，只装有证据的）
  ungrounded_questions：证据不足、未作答的子问题
  need_clarify  ：是否需要向用户反问澄清
  clarify_questions：反问的问题列表
  history_summary：历史对话的滚动摘要（替代全量历史，控 token）
  tool_calls    ：已执行的检索/工具调用次数（预算控制）
  iterations    ：图迭代轮次（预算控制）

  ── 证据充分性（拒答）──
  abstained     ：是否因证据不足而拒答（并行分支「任一为真」）
  abstain_reason：拒答原因，多个并行分支的原因合并去重

设计要点：字段都给了默认值，LangGraph 建图时才能推断出 schema。
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, Sequence

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def add_queries(existing: Sequence[str], new: Sequence[str]) -> list[str]:
    """追加字符串列表；空列表是显式的「开始新一轮」重置信号。

    LangGraph checkpointer 会保存整个 State。若没有重置语义，上一轮的 queries 和
    sub_answers 会永久累积，后续回答就会混入旧内容。begin_turn 节点每轮首先写入
    空列表，本 reducer 据此清空单轮字段；并行分支仍可继续使用追加语义。
    """
    if not new:
        return []
    return list(existing) + list(new)


def add_docs(existing: Sequence[Document], new: Sequence[Document]) -> list[Document]:
    """累加并去重本轮文档；空列表用于在新一轮开始时清空旧命中。"""
    if not new:
        return []
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
    """累加并行分支计数；0 是新一轮开始时的显式重置信号。"""
    if new == 0:
        return 0
    return (existing or 0) + (new or 0)


def last_int(existing: int, new: int) -> int:
    """整数取最新值 reducer（用于迭代轮次这类「只需要一个值」的字段）。"""
    return new if new is not None else existing


def or_bool(existing: bool, new: bool | None) -> bool:
    """布尔求或：并行分支「任一为真即为真」。**None 是新一轮的显式重置信号**。

    ⚠️ 为什么必须带 reducer：
       多子问题并行时，N 个 retrieve_one 分支可能**同时**写 abstained。
       不带 reducer 会抛 InvalidUpdateError（一个 step 只能写一个值），
       对外表现就是「知识库里没有的多子问题提问 → HTTP 500」。

    ⚠️ 为什么重置信号用 None 而不是 False（这里返工过一版，教训值得留）：
       reducer 是被**逐个**喂值的。若拿 False 当重置，就等于「合法的否定票」
       和「新一轮开始」共用同一个值，两种语义撞车：
         update([True, False]) → or(True, False) 若按重置算 → False（拒答被抹掉）
         update([False, True]) → True
       而并行分支的写入顺序由调度决定、不确定 —— 一旦将来某个分支也显式写
       abstained=False（很自然的重构），拒答判定就会变成**非确定性 bug**：
       同样的输入，时而拒答时而不拒答。
       现在 False 只表示「我这支没拒答」，永远不会抹掉同伴的 True；
       重置只由 begin_turn 写 None 触发（它单独占一个 step，不与分支混写）。

    ⚠️ 返回值永远是 bool：字段对外契约仍是 `abstained: bool`，
       None 只是"清零"这一动作的输入编码，不会流到下游。
    """
    if new is None:
        return False
    return bool(existing) or bool(new)


_ABSTAIN_REASON_SEP = "；"


def merge_abstain_reason(existing: str, new: str) -> str:
    """合并并行分支的拒答原因：去重后按固定顺序拼接；空串是新一轮的重置信号。

    为什么要排序而不是「后写覆盖」：
       并行分支的写入顺序由调度决定、不确定。若用覆盖语义，同一问题多次请求
       可能拿到不同的原因文案；排序后输出稳定，便于测试与排查。
    """
    if not new:
        return ""
    parts = [p for p in (existing or "").split(_ABSTAIN_REASON_SEP) if p]
    for p in new.split(_ABSTAIN_REASON_SEP):
        if p and p not in parts:
            parts.append(p)
    return _ABSTAIN_REASON_SEP.join(sorted(parts))


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
    """每个子问题各自的答案（map 阶段产物），供 reduce 节点汇总。

    ⚠️ 只装「有证据支撑」的答案：判定证据不足的分支不往里写，而是写进
       ungrounded_questions。否则 reduce 会把拒答文案当素材去调模型汇总，
       产出「看似有结论、实际无证据」的内容 —— 正是本项目的核心红线。
    """

    ungrounded_questions: Annotated[list[str], add_queries] = field(default_factory=list)
    """证据不足、未作答的子问题。用于向用户如实交代「哪几问没找到资料」。"""

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

    # ── 证据充分性（拒答）────────────────────────────
    abstained: Annotated[bool, or_bool] = False
    """是否因证据不足而拒答。

    ⚠️ 必须带 or_bool reducer：多子问题并行时 N 个 retrieve_one 分支会同时写
       这个字段，不带 reducer 会抛 InvalidUpdateError —— 库外的多子问题提问
       会直接 500（2026-10-15 实测复现）。语义「任一分支判定证据不足即为真」。
    """

    abstain_reason: Annotated[str, merge_abstain_reason] = ""
    """拒答原因，用于向用户解释「为什么没答」而不是含糊其辞。

    ⚠️ 同样必须带 reducer（与 abstained 成对）：并行分支各写各的原因，
       合并时去重排序，保证同一问题输出稳定。begin_turn 写空串重置本轮。"""

    _extra: dict[str, Any] = field(default_factory=dict, repr=False)
    """预留：想加「是否兜底」等字段时往这里加。"""
