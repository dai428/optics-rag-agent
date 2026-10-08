# D:\Desktop\rag-demo\app\rag_graph.py
"""显式 LangGraph 版 RAG（Agentic RAG：改写 → 澄清 → 拆分 → 并行检索 → 汇总）

流程图
------
                          ┌─(闲聊)───→ respond → __end__
  __start__ → rewrite ────┼─(需澄清)─→ clarify → __end__（把问题抛回给用户）
                          └─(正常)───→ decompose ─┬─→ retrieve_one ─┐
                                                  └─→ retrieve_one ─┴→ respond → __end__
                              （Send 动态并行，子问题数决定分支数）

四个能力（对应优化计划 P1）
--------------------------
  P1-1 澄清反问 ：clarify —— 问题太模糊时先反问，不瞎猜
  P1-2 多子问题 ：decompose + Send 并行 + respond（map-reduce）
  P1-3 历史压缩 ：summarize_history —— 长对话滚动摘要
  P1-4 预算控制 ：MAX_TOOL_CALLS / MAX_ITERATIONS 硬上限

与 rag_agent.py（create_agent 黑盒版）的区别：
  create_agent ：框架帮你把「思考→调工具→回答」的循环藏起来，只见配置不见流程
  rag_graph    ：流程图自己画，每个节点干什么、走到哪一步，全部可见可改

配置来源：app/config.py · 提示词来源：app/prompts.py
"""

import os

# 环境自举（与其他模块一致，保证单独运行本模块时也生效）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "D:/DevCache/huggingface")
os.environ.setdefault("no_proxy", "*")
os.environ.setdefault("NO_PROXY", "*")

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

try:  # 包方式：app.rag_graph
    from .configuration import Configuration
    from .config import settings
    from .prompts import (
        AGGREGATE_SYSTEM_PROMPT,
        AGGREGATE_USER_TEMPLATE,
        CLARIFY_SYSTEM_PROMPT,
        CLARIFY_USER_TEMPLATE,
        DECOMPOSE_SYSTEM_PROMPT,
        DECOMPOSE_USER_TEMPLATE,
        SUB_ANSWER_SYSTEM_PROMPT,
        SUB_ANSWER_USER_TEMPLATE,
        SUMMARIZE_SYSTEM_PROMPT,
        SUMMARIZE_USER_TEMPLATE,
    )
    from .state import State
except ImportError:  # 扁平方式：cd app && python rag_graph.py
    from configuration import Configuration
    from config import settings
    from prompts import (
        AGGREGATE_SYSTEM_PROMPT,
        AGGREGATE_USER_TEMPLATE,
        CLARIFY_SYSTEM_PROMPT,
        CLARIFY_USER_TEMPLATE,
        DECOMPOSE_SYSTEM_PROMPT,
        DECOMPOSE_USER_TEMPLATE,
        SUB_ANSWER_SYSTEM_PROMPT,
        SUB_ANSWER_USER_TEMPLATE,
        SUMMARIZE_SYSTEM_PROMPT,
        SUMMARIZE_USER_TEMPLATE,
    )
    from state import State

# ── 预算控制（P1-4）────────────────────────────────────────────────
# 为什么需要？
#   Agent 可能陷入「检索 → 觉得资料不够 → 再检索」的死循环，一路烧 token。
#   加硬上限是工程上唯一可靠的办法 —— 不能指望提示词让模型「自觉」。
MAX_TOOL_CALLS = 8        # 单轮最多检索/工具调用次数
MAX_ITERATIONS = 10       # 单轮最多图迭代次数
MAX_SUB_QUESTIONS = 3     # 一个复合问题最多拆几个子问题
# 历史摘要触发阈值：对话超过这么多条消息，就把旧的部分压缩成摘要
SUMMARIZE_TRIGGER = 8

# ── 轻量规则：判断这一句是否需要检索 ────────────────────────────────
# 为什么不全交给模型判断？
#   一次模型调用约 2~5 秒。用 10 行规则先挡掉闲聊，能省下一次调用，
#   且规则是确定性的——不会出现「模型今天心情不好不想检索」这种抖动。
_CHITCHAT = (
    "你好", "您好", "hi", "hello", "在吗", "谢谢", "感谢",
    "你是谁", "你叫什么", "再见", "拜拜", "test", "测试",
)


def _is_chitchat(text: str) -> bool:
    """极短句或纯客套 → 判为闲聊，跳过检索。"""
    t = text.strip().lower()
    if len(t) <= 4:  # 「你好呀」「在么」「hi」这类
        return True
    return any(kw in t for kw in _CHITCHAT) and len(t) <= 12


def _history_text(messages, limit: int = 6) -> str:
    """把最近几轮对话拼成纯文本，供改写的提示词使用。

    为什么要截断？对话越长，拼进提示词的 token 越多、越贵越慢，
    且很久之前的对话对当前这轮的意义很低。保留最近 limit 条足够。
    """
    lines = []
    for m in messages[-limit:]:
        if isinstance(m, HumanMessage):
            lines.append(f"用户：{m.content}")
        elif isinstance(m, AIMessage) and m.content:
            lines.append(f"助手：{m.content}")
    return "\n".join(lines) if lines else "（无历史对话）"


def _build_context(docs, max_chars_per_doc: int = 3000) -> str:
    """把检索到的文档拼成提示词里的「资料」段。

    每条都带来源文件名与章节路径 —— 这是 prompt 规则 3（标注出处）的前提。
    单条截断 max_chars_per_doc：父块最长 4000 字，多条拼起来会撑爆上下文，
    留 3000 字/条既保信息量又控总量。
    """
    if not docs:
        return "（本轮无需检索，或未检索到资料）"
    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("file_name") or os.path.basename(d.metadata.get("source", "未知来源"))
        sec = d.metadata.get("section_path", "")
        score = d.metadata.get("rerank_score")
        head = f"[资料 {i}] 来源：{src}"
        if sec:
            head += f" ｜ 章节：{sec}"
        if score is not None:
            head += f" ｜ 相关度：{score}"
        content = d.page_content
        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc] + "\n…（片段过长已截断）"
        blocks.append(f"{head}\n{content}")
    return "\n\n".join(blocks)


def _llm(purpose: str = "answer"):
    """统一拿 LLM 实例（延迟导入，避免循环依赖）。"""
    try:
        from .rag_agent import build_llm
    except ImportError:
        from rag_agent import build_llm
    return build_llm(purpose=purpose)


# ══════════════════════════════════════════════════════════════════
#  节点函数：签名统一为 (state, config) -> dict（返回要更新的字段）
# ══════════════════════════════════════════════════════════════════

def rewrite_query(state: State, config: RunnableConfig) -> dict:
    """节点 1：查询改写 + 指代消解 + 历史滚动摘要（P1-3）。

    首轮对话 → 直接用用户原话；多轮 → 让模型把省略句补全成独立问句。
    """
    cfg = Configuration.from_runnable_config(config)
    question = state.messages[-1].content
    updates: dict = {"iterations": state.iterations + 1}

    # ── P1-3 历史滚动摘要：消息数超过阈值就压缩旧历史 ──
    if len(state.messages) > SUMMARIZE_TRIGGER and _llm("rewrite") is not None:
        older = state.messages[:-2]  # 留最后两条（上一轮问答）不压
        if older:
            summary = _summarize(state.history_summary, older, config)
            if summary:
                updates["history_summary"] = summary

    # 首轮：无历史可参考，原话即问句
    history = _history_text(state.messages[:-1])
    if history == "（无历史对话）":
        updates["queries"] = [question]
        return updates

    llm = _llm("rewrite")
    if llm is None:
        updates["queries"] = [question]
        return updates

    from prompts import QUERY_REWRITE_SYSTEM_PROMPT, QUERY_REWRITE_USER_TEMPLATE

    prompt = [
        SystemMessage(content=cfg.query_system_prompt),
        HumanMessage(content=QUERY_REWRITE_USER_TEMPLATE.format(history=history, question=question)),
    ]
    try:
        rewritten = llm.invoke(prompt, config=config).content.strip()
    except Exception:  # noqa: BLE001 —— 改写失败不能让整条流程挂掉
        rewritten = question

    # 兜底：模型偶尔会啰嗦或返回空串
    if not rewritten or len(rewritten) > 200:
        rewritten = question
    updates["queries"] = [rewritten]
    return updates


def _summarize(existing: str, new_turns, config) -> str:
    """把「已有摘要 + 新增对话」合并成新摘要。失败返回空串（保持原摘要不变）。"""
    llm = _llm("rewrite")
    if llm is None:
        return ""
    turns = []
    for m in new_turns:
        if isinstance(m, HumanMessage):
            turns.append(f"用户：{m.content}")
        elif isinstance(m, AIMessage) and m.content:
            turns.append(f"助手：{m.content[:300]}")  # 助手回答长，截断
    if not turns:
        return ""
    prompt = [
        SystemMessage(content=SUMMARIZE_SYSTEM_PROMPT),
        HumanMessage(
            content=SUMMARIZE_USER_TEMPLATE.format(
                existing_summary=existing or "（无）", new_turns="\n".join(turns)
            )
        ),
    ]
    try:
        return llm.invoke(prompt, config=config).content.strip()[:600]
    except Exception:  # noqa: BLE001
        return ""


def clarify_check(state: State, config: RunnableConfig) -> dict:
    """节点 2：澄清反问检查（P1-1）。

    问题模糊时不在瞎猜中浪费检索，而是把问题抛回给用户。
    这是「真 Agent」与「问答机」的重要分界 —— 前者会承认自己不确定。
    """
    question = state.messages[-1].content

    # 闲聊不问 ｜ 问题够长说明信息量足，不问
    if _is_chitchat(question) or len(question) >= 30:
        return {"need_clarify": False}

    llm = _llm("rewrite")
    if llm is None:
        return {"need_clarify": False}

    history = _history_text(state.messages[:-1], limit=4)
    prompt = [
        SystemMessage(content=CLARIFY_SYSTEM_PROMPT),
        HumanMessage(content=CLARIFY_USER_TEMPLATE.format(history=history, question=question)),
    ]
    try:
        raw = llm.invoke(prompt, config=config).content.strip()
    except Exception:  # noqa: BLE001 —— 检查失败一律按清晰处理（不影响主流程）
        return {"need_clarify": False}

    # 严格按约定的输出格式解析：首行必须是「清晰」或「不清晰」
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    first = lines[0] if lines else ""
    if "不清晰" not in first:
        return {"need_clarify": False}

    # 剩余行就是反问句。过滤掉模型偶尔多写的标签行与解释行
    _BAD_PREFIX = ("原因", "可能", "候选", "方向", "说明", "注意", "示例", "标签")
    questions = []
    for ln in lines[1:]:
        s = ln.strip().lstrip("0123456789.、)）-•* ").strip().strip("「」\"'“”")
        if len(s) < 6:
            continue
        if any(s.startswith(p) or s.startswith(p + "：") for p in _BAD_PREFIX):
            continue
        if "清晰" in s or "判断" in s:
            continue
        questions.append(s)
    if not questions:
        return {"need_clarify": False}
    return {"need_clarify": True, "clarify_questions": questions[:3]}


def route_after_rewrite(state: State) -> Literal["clarify", "decompose", "respond"]:
    """条件边：改写之后决定走哪条路。

    - 闲聊   → 直接 respond（不检索）
    - 需澄清 → clarify（把问题抛回用户）
    - 正常   → decompose（拆分 + 并行检索）
    """
    question = state.messages[-1].content
    if _is_chitchat(question):
        return "respond"
    if state.need_clarify:
        return "clarify"
    return "decompose"


def clarify(state: State, config: RunnableConfig) -> dict:
    """节点 3：把反问抛回给用户（不检索、不回答）。

    P1-4 预算控制：这里显式停手，避免为一个模糊问题消耗多次检索和生成。
    """
    qs = state.clarify_questions or ["请再具体说明一下你想了解的内容？"]
    lines = ["你的问题还不太明确，我需要你补充一下信息："]
    for i, q in enumerate(qs, 1):
        lines.append(f"{i}. {q}")
    lines.append("\n补充后我再基于知识库为你查找。")
    return {"messages": [AIMessage(content="\n".join(lines))]}


def decompose(state: State, config: RunnableConfig) -> dict:
    """节点 4：把复合问题拆成子问题（P1-2 的 map 前置）。

    不需要拆时返回「原问题」单元素列表 —— 这样后续并行节点的代码
    不用写两套分支逻辑（统一按列表处理）。
    """
    question = state.messages[-1].content
    llm = _llm("rewrite")
    if llm is None:
        return {"sub_questions": [question]}

    prompt = [
        SystemMessage(content=DECOMPOSE_SYSTEM_PROMPT),
        HumanMessage(content=DECOMPOSE_USER_TEMPLATE.format(question=question)),
    ]
    try:
        raw = llm.invoke(prompt, config=config).content.strip()
    except Exception:  # noqa: BLE001
        return {"sub_questions": [question]}

    subs = []
    for ln in raw.splitlines():
        s = ln.strip().lstrip("0123456789.、)）-•* ").strip()
        s = s.strip("「」\"'“”")
        if len(s) >= 4:
            subs.append(s)
    # 兜底：模型没拆出东西 / 拆得比原问题还碎 → 用原问题
    if not subs:
        subs = [question]
    return {"sub_questions": subs[:MAX_SUB_QUESTIONS]}


def route_to_retrieve(state: State):
    """条件边（Send 分发）：为每个子问题各发一个 retrieve_one 分支。

    LangGraph 的 Send API：返回 [Send("节点名", {局部状态}), ...]
    就会**并行**触发多个同名节点，各自拿到独立的一份状态。
    这是实现 map-reduce 并行的标准做法 —— 比串行 for 循环快 N 倍
    （N = 子问题数），且天然隔离（一个子问题检索失败不影响其他）。
    """
    from langgraph.types import Send

    subs = state.sub_questions or [state.messages[-1].content]
    return [
        Send("retrieve_one", {"sub_questions": [s], "queries": [s]})
        for s in subs[:MAX_SUB_QUESTIONS]
    ]


def retrieve_one(state, config: RunnableConfig) -> dict:
    """节点 5（并行执行，map 阶段）：为单个子问题检索并生成答案。

    ⚠️ 关键坑：本节点的 state 是 **dict 而不是 State 对象**。
       因为它是被 `Send` 触发的 —— 我们传给 Send 的第二个参数
       (`{"sub_questions": [s], "queries": [s]}`) 会作为该分支的**局部状态**，
       而局部状态就是普通 dict，LangGraph 不会把它实例化成 State dataclass。
       所以这里所有取值都要用 `.get()` 而非属性访问。

    为什么在这里就把答案生成掉，而不是等所有检索完再统一生成？
      因为每个子问题需要**自己那批资料**。如果只汇总资料，最后生成时
      上下文里混着 3 组资料，模型容易串台（答 A 问题引用 B 资料）。
      各子问题独立「检索 + 回答」，再由 reduce 节点统一整合，更干净。
    """
    # 兼容 dict（Send 局部状态）与 State（若被其它路径调用）
    def _get(key, default=None):
        if isinstance(state, dict):
            return state.get(key, default)
        return getattr(state, key, default)

    subs = _get("sub_questions") or []
    msgs = _get("messages") or []
    question = subs[0] if subs else (msgs[-1].content if msgs else "")
    tool_calls = _get("tool_calls", 0) or 0

    try:
        from .knowledge_base import search
    except ImportError:
        from knowledge_base import search

    # P1-4 预算控制：已达上限就不再检索（返回提示，让模型别瞎编）
    if tool_calls >= MAX_TOOL_CALLS:
        return {"sub_answers": [f"（{question}）已达检索调用上限，本轮未再检索。"]}

    try:
        docs = search(question, k=settings.rerank_top_n)
    except Exception as e:  # noqa: BLE001 —— 单个子问题失败不拖垮整轮
        return {"sub_answers": [f"（{question}）检索失败：{e}"]}

    context = _build_context(docs)
    llm = _llm("answer")
    if llm is None:
        return {
            "sub_answers": [f"（{question}）未配置 API Key，无法生成回答。"],
            "tool_calls": tool_calls + 1,
        }

    prompt = [
        SystemMessage(content=SUB_ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=SUB_ANSWER_USER_TEMPLATE.format(context=context, question=question)),
    ]
    try:
        ans = llm.invoke(prompt, config=config).content.strip()
    except Exception as e:  # noqa: BLE001
        ans = f"（{question}）生成失败：{str(e)[:120]}"
    # ⚠️ 必须把 docs 一并写回 state：
    #    retrieved_docs 是对外可观测的字段（接口返回「命中片段数」、RAGAS 评测取上下文），
    #    只写 sub_answers 会让上游完全看不到「这轮到底检索到了什么」。
    #    多分支并行时靠 state.add_docs reducer 自动累加去重。
    return {"sub_answers": [ans], "retrieved_docs": docs, "tool_calls": tool_calls + 1}


def respond(state: State, config: RunnableConfig) -> dict:
    """节点 6（reduce 阶段）：汇总子答案 → 最终回答。

    三种情况：
      1. 闲聊（无子答案）→ 直接聊
      2. 单子问题答案 → 直接作为最终回答（省一次模型调用）
      3. 多子问题 → 调模型做 reduce 汇总
    """
    cfg = Configuration.from_runnable_config(config)
    question = state.messages[-1].content
    answers = [a for a in (state.sub_answers or []) if a]

    # ── 情况 1：闲聊 ──
    if _is_chitchat(question):
        llm = _llm("answer")
        if llm is None:
            return {"messages": [AIMessage(content="❌ 未配置 AMD_API_KEY，无法生成回答。")]}
        reply = llm.invoke(
            [SystemMessage(content=cfg.response_system_prompt), *state.messages], config=config
        )
        return {"messages": [reply]}

    # ── 情况 2：单子问题，答案即终答 ──
    if len(answers) == 1:
        return {"messages": [AIMessage(content=answers[0])]}

    if not answers:
        return {"messages": [AIMessage(content="知识库中未找到相关答案。")]}

    # ── 情况 3：多子问题，reduce 汇总 ──
    llm = _llm("answer")
    if llm is None:
        body = "\n\n".join(f"【{i}】{a}" for i, a in enumerate(answers, 1))
        return {"messages": [AIMessage(content=body)]}

    answers_text = "\n\n".join(f"- {a}" for a in answers)
    prompt = [
        SystemMessage(content=AGGREGATE_SYSTEM_PROMPT),
        HumanMessage(content=AGGREGATE_USER_TEMPLATE.format(question=question, answers=answers_text)),
    ]
    try:
        reply = llm.invoke(prompt, config=config)
    except Exception as e:  # noqa: BLE001
        body = "\n\n".join(f"【{i}】{a}" for i, a in enumerate(answers, 1))
        reply = AIMessage(content=f"（汇总失败：{str(e)[:100]}）\n\n{body}")
    return {"messages": [reply]}


# ══════════════════════════════════════════════════════════════════
#  组装图
# ══════════════════════════════════════════════════════════════════

def build_graph(checkpointer=None):
    """构建并编译 LangGraph 图。

    checkpointer：会话记忆存储。传 InMemorySaver() 后，
    同一 thread_id 的多轮调用会自动带上历史 messages（这就是「记住上下文」的原理）。

    ⚠️ 踩坑记录（2026-10-08）：thread_id 不要用中文/空格等非 ASCII 字符。
       实测把中文问题直接拼成 thread_id（如 f"par-{中文问题}"）后，
       graph.invoke() 会**挂住不返回**（stream 模式正常、invoke 卡死），
       而同一张图换成纯 ASCII 的 thread_id 就在 77 秒内跑完。
       → 约定：所有 thread_id 用「英文前缀 + 序号」，如 "session-12"、"cli-graph"。
    """
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(State, input_schema=State)

    builder.add_node("rewrite", rewrite_query)
    builder.add_node("clarify_check", clarify_check)
    builder.add_node("clarify", clarify)
    builder.add_node("decompose", decompose)
    builder.add_node("retrieve_one", retrieve_one)   # 并行分支节点
    builder.add_node("respond", respond)

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "clarify_check")
    # 条件边：澄清检查后三选一
    builder.add_conditional_edges(
        "clarify_check",
        route_after_rewrite,
        {"clarify": "clarify", "decompose": "decompose", "respond": "respond"},
    )
    builder.add_edge("clarify", END)
    # 条件边：Send 动态并行 —— 每个子问题一个 retrieve_one 分支
    builder.add_conditional_edges("decompose", route_to_retrieve, ["retrieve_one"])
    # 所有并行分支汇合到 respond（LangGraph 会自动等全部分支完成）
    builder.add_edge("retrieve_one", "respond")
    builder.add_edge("respond", END)

    graph = builder.compile(checkpointer=checkpointer)
    graph.name = "OpticsAgenticRAGGraph"
    return graph


def build_graph_with_memory():
    """带会话记忆的图（服务用）。"""
    from langgraph.checkpoint.memory import InMemorySaver

    return build_graph(checkpointer=InMemorySaver())


if __name__ == "__main__":
    graph = build_graph_with_memory()
    cfg = {"configurable": {"thread_id": "cli-graph"}, "recursion_limit": MAX_ITERATIONS + 40}

    tests = [
        "你好呀",
        "评估网格用了多少个波长点和温度点？总共多少点？",
        "那它的随机种子是多少？",
        "对比一下 DE 加权路线和 NSGA-II 路线的优缺点",
    ]
    for q in tests:
        r = graph.invoke({"messages": [HumanMessage(content=q)]}, config=cfg)
        print("=" * 70)
        print(f"❓ {q}")
        print(f"🔎 检索问句：{r.get('queries', [])[-1:]}")
        print(f"🧩 子问题  ：{r.get('sub_questions')}")
        print(f"❓ 需澄清  ：{r.get('need_clarify')} {r.get('clarify_questions') or ''}")
        print(f"📊 预算    ：tool_calls={r.get('tool_calls')} iterations={r.get('iterations')}")
        print(f"📝 摘要    ：{(r.get('history_summary') or '（无）')[:80]}")
        print(f"💡 {r['messages'][-1].content[:300]}")
    print("=" * 70)
