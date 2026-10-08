"""
raw_api_demo.py — L2 参考实现：不依赖任何框架，直接用 requests 调 LLM API

张万森 · L2 LLM API 与 Prompt 工程
────────────────────────────────────────────────────────
这个文件是「标准答案」，先读它，再自己写你的版本。

它做了一件 LangChain 帮你做的事（但你看得见每一步）：
    手写 HTTP 请求 → 理解 OpenAI 兼容接口的报文格式

运行：
    cd D:/Desktop/rag-demo/learn/L2_api
    ../../.venv/Scripts/python.exe raw_api_demo.py
────────────────────────────────────────────────────────
"""
import json
from pathlib import Path

import requests

# ── 1. 读配置（不用 python-dotenv，手写一遍让你看清它做了什么）────────
_ENV_PATH = Path(__file__).parent.parent.parent / ".env"


def load_env(path: Path) -> dict[str, str]:
    """把 .env 读成一个字典。键值对，跳过注释和空行。"""
    env: dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件：{path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


ENV = load_env(_ENV_PATH)
API_KEY: str = ENV["AMD_API_KEY"]
BASE_URL: str = ENV.get("AMD_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1")
MODEL: str = ENV.get("AMD_LLM_MODEL", "Qwen3.8-27B")


# ── 2. 核心：一次 API 调用 ──────────────────────────────────────────
def chat_once(prompt: str, system: str = "", temperature: float = 0.1) -> str:
    """
    发一次请求，拿一次回答。

    参数：
        prompt      用户问题
        system      系统提示词（设定角色和行为规则）
        temperature 温度，0=最确定，1=最发散

    返回：
        模型回答的纯文本
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=120,
    )
    resp.raise_for_status()          # HTTP 4xx/5xx → 直接抛异常
    data = resp.json()

    # ── 3. 报文结构：答案藏在很深的嵌套里 ──
    # {
    #   "choices": [
    #       {"message": {"role": "assistant", "content": "回答文本"}, ...}
    #   ],
    #   "usage": {"prompt_tokens": 38, "completion_tokens": 120, "total_tokens": 158}
    # }
    return data["choices"][0]["message"]["content"]


def chat_once_with_usage(prompt: str, system: str = "") -> tuple[str, dict]:
    """同上，但把 token 用量也返回 —— 让你看见「钱花在哪」。"""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": MODEL, "messages": messages, "temperature": 0.1},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"], data.get("usage", {})


# ── 4. 演示 ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"模型：{MODEL}")
    print(f"地址：{BASE_URL}")
    print("=" * 60)

    # 演示一：同一个问题，两次温度对比
    question = "用一句话解释什么是斯托克斯参量。"

    for temp in (0.1, 1.0):
        print(f"\n【temperature = {temp}】")
        print(chat_once(question, system="你是光学专业助手，回答简洁准确。", temperature=temp))

    # 演示二：看 token 用量
    print("\n" + "=" * 60)
    answer, usage = chat_once_with_usage("用一句话解释什么是 LCVR。")
    print(f"回答：{answer}")
    print(f"用量：{usage}")
