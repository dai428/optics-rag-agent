"""
L2 练习脚手架 — 你在这三个函数里填空

张万森 · L2 LLM API 与 Prompt 工程
────────────────────────────────────────────────────────
参考实现见 raw_api_demo.py（可跑，已验证）。
本文件是留给你自己写的版本——**不要抄，抄了就白学了**。

跑法：
    cd D:/Desktop/rag-demo/learn/L2_api
    ../../.venv/Scripts/python.exe l2_exercise.py
────────────────────────────────────────────────────────
"""
import json
from pathlib import Path

import requests

_ENV_PATH = Path(__file__).parent.parent.parent / ".env"


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


ENV = load_env(_ENV_PATH)
API_KEY = ENV["AMD_API_KEY"]
BASE_URL = ENV.get("AMD_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1")
MODEL = ENV.get("AMD_LLM_MODEL", "Qwen3.8-27B")


# ══════════════════════════════════════════════════════════════════
# 任务 2：多轮对话（自己写）
# ══════════════════════════════════════════════════════════════════
def chat_multi(questions: list[str], system: str = "") -> list[str]:
    """
    依次提问，每次把之前的对话历史都带上。

    实现要点：
      1. 建一个 messages 列表，system 提示词先进去（如果有）
      2. 循环每个问题：
           messages.append({"role": "user", "content": q})
           发请求 → 拿回答
           messages.append({"role": "assistant", "content": 回答})   ← 关键：历史要回填
      3. 最后打印完整的 messages，看清「记忆」的本质

    返回：所有回答组成的列表
    """
    # TODO: 你来写
    raise NotImplementedError("任务 2 未完成")


# ══════════════════════════════════════════════════════════════════
# 任务 3：用光学语料做「带上下文的提问」
# ══════════════════════════════════════════════════════════════════
def ask_with_context(material: str, question: str) -> str:
    """
    把材料 + 问题拼进 prompt，让模型基于材料回答。

    实现要点：
      1. 写一个四段式 system prompt（角色/任务/约束/输出格式）
         关键是给出「退路」：材料中没有的信息，要求模型明确说「材料中未包含」
      2. 把 material 和 question 拼成 user 消息
      3. 发请求，返回回答

    这个函数就是 L3「手写 RAG」的雏形——区别只是材料由程序自动检索，而非手挑。
    """
    # TODO: 你来写
    raise NotImplementedError("任务 3 未完成")


def load_material(
    filename: str = "研究本体详细报告_2026-09-18.md",
    max_chars: int = 800,
) -> str:
    """从你的光学语料里读一段材料（默认截前 800 字）。"""
    docs = Path(__file__).parent.parent.parent / "data" / "docs"
    path = docs / "实验报告" / filename
    if not path.exists():
        raise FileNotFoundError(f"语料不存在：{path}")
    return path.read_text(encoding="utf-8")[:max_chars]


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"模型：{MODEL}\n" + "=" * 60)

    # 任务 2 验证：这三问有上下文依赖，第二问必须靠记忆才能答
    print("\n【任务 2 · 多轮对话】")
    answers = chat_multi(
        [
            "什么是 LCVR？",
            "它相比传统旋转波片有什么优势？",   # ← 靠记忆才知道「它」指 LCVR
            "那它的定标难点在哪？",             # ← 继续靠记忆
        ],
        system="你是光学专业助手，回答简洁，每问不超过三句话。",
    )
    for i, a in enumerate(answers, 1):
        print(f"\n第 {i} 轮回答：{a}")

    # 任务 3 验证：基于你自己的材料提问
    print("\n" + "=" * 60)
    print("\n【任务 3 · 基于光学材料提问】")
    material = load_material()
    print(f"材料前 200 字：\n{material[:200]}...\n")
    print("问题 1（材料里应该有）：本研究的核心目标是什么？")
    print(ask_with_context(material, "本研究的核心目标是什么？"))
    print("\n问题 2（材料里应该没有，测幻觉）：这个研究的经费是多少？")
    print(ask_with_context(material, "这个研究的经费是多少？"))
