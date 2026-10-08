# D:\Desktop\rag-demo\app\fix_deps.py
"""一键修复依赖兼容问题（新环境 clone 后跑一次即可）。

背景
----
本项目的依赖组合有一处已知冲突：
  · langchain-community==0.4.2 是 sunset 版，移除了 chat_models/vertexai.py
  · ragas（评测库）在其 ragas/llms/base.py 顶部无条件 import 这个模块
  → 结果：`import ragas` 直接 ModuleNotFoundError

本脚本做两件事：
  1. 给 langchain_community.chat_models 补一个 vertexai 垫片（shim）；
  2. 顺带做一次自检，打印关键依赖版本，方便确认环境正常。

用法：
    cd D:/Desktop/rag-demo/app
    ../.venv/Scripts/python.exe fix_deps.py
"""

import importlib.util
import os
import sys
from pathlib import Path

try:
    from ._shims import SHIM_SOURCE
except ImportError:
    from _shims import SHIM_SOURCE


def apply_vertexai_shim() -> bool:
    """把垫片写进 site-packages。返回是否做了写入（已存在则跳过）。"""
    spec = importlib.util.find_spec("langchain_community")
    if spec is None or not spec.submodule_search_locations:
        print("❌ 找不到 langchain_community，请先装依赖")
        return False

    pkg = Path(spec.submodule_search_locations[0])
    target = pkg / "chat_models" / "vertexai.py"

    if target.exists():
        print(f"✅ 垫片已存在，跳过：{target}")
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SHIM_SOURCE, encoding="utf-8")
    print(f"✅ 已写入垫片：{target}")
    return True


def self_check():
    """打印关键依赖版本 + 尝试导入 ragas。"""
    print("\n── 依赖自检 ──")
    checks = [
        ("langchain", "langchain"),
        ("langchain_core", "langchain_core"),
        ("langchain_community", "langchain_community"),
        ("langchain_openai", "langchain_openai"),
        ("langgraph", "langgraph"),
        ("chromadb", "chromadb"),
        ("ragas", "ragas"),
        ("openai", "openai"),
    ]
    for name, mod in checks:
        try:
            m = __import__(mod)
            print(f"  {name:22s} {getattr(m, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:22s} ❌ {type(e).__name__}: {e}")

    print("\n── ragas 导入测试 ──")
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: F401
        from ragas.metrics import faithfulness  # noqa: F401

        print("  ✅ ragas 导入成功")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ ragas 导入失败：{type(e).__name__}: {e}")


if __name__ == "__main__":
    apply_vertexai_shim()
    self_check()
