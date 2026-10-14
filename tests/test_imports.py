"""导入契约测试：所有模块必须能在包模式下导入。

为什么需要这个文件
------------------
项目支持两种运行方式（`python -m app.xxx` 与 `cd app && python xxx.py`），
每个模块都要同时兼容，这很容易漏。实际已经因此踩过三次坑：

  1. rag_graph.py 里 `from prompts import ...` → 包模式下第二轮对话 ModuleNotFoundError
  2. eval_rag.py 里 `from .ingestion... import` → 扁平模式下 ImportError
  3. eval_gen.py 里裸 `import config` → 包模式下 ModuleNotFoundError

这类缺陷只在「换一种启动方式」时才暴露，手工测试极易漏掉，而一旦漏掉就是
「面试官换个命令启动就崩」。所以用测试把契约锁死：**逐个导入，一个都不能少**。
"""

from __future__ import annotations

import importlib
import unittest

# 全部需要能被导入的模块（新增模块时请同步补上）
MODULES = [
    "app",
    "app.config",
    "app.chunker",
    "app.embeddings",
    "app.storage",
    "app.state",
    "app.prompts",
    "app.configuration",
    "app.observability",
    "app.knowledge_base",
    "app.rag_agent",
    "app.rag_graph",
    "app.main",
    "app.eval_rag",
    "app.eval_gen",
    "app.verify_p1",
    "app.tools",
    "app.tools.optics",
    "app.ingestion",
    "app.ingestion.loaders",
    "app.ingestion.manifest",
    "app.ingestion.indexer",
]


class ImportContractTests(unittest.TestCase):
    def test_all_modules_import_in_package_mode(self):
        failures: list[str] = []
        for name in MODULES:
            try:
                importlib.import_module(name)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name} → {type(exc).__name__}: {exc}")

        self.assertEqual(
            failures,
            [],
            "以下模块在包模式下导入失败（检查是否漏了 try/except ImportError 双模式分支）：\n  "
            + "\n  ".join(failures),
        )

    def test_no_module_level_bare_local_imports(self):
        """源码里不应存在**模块级**的裸本地导入。

        裸导入（`import config`）只允许出现在 try/except 的 fallback 分支里。
        这里检查的是「模块顶层且不在 try 块内」的裸导入 —— 那种写法在包模式下必崩。
        """
        import ast
        from pathlib import Path

        local_modules = {
            "config", "chunker", "embeddings", "storage", "state", "prompts",
            "configuration", "observability", "knowledge_base", "rag_agent",
            "rag_graph", "main", "eval_rag", "eval_gen", "verify_p1",
            "ingestion", "tools",
        }
        app_dir = Path(__file__).resolve().parent.parent / "app"
        offenders: list[str] = []

        for path in sorted(app_dir.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            # 收集所有 try 块内的 import 行号（这些是允许的 fallback）
            in_try: set[int] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    for child in ast.walk(node):
                        if isinstance(child, (ast.Import, ast.ImportFrom)):
                            in_try.add(child.lineno)

            for node in tree.body:  # 只看模块顶层
                if node.lineno in in_try:
                    continue
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in local_modules:
                            offenders.append(
                                f"{path.relative_to(app_dir.parent)}:{node.lineno} "
                                f"import {alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and (node.module or "").split(".")[0] in local_modules:
                        offenders.append(
                            f"{path.relative_to(app_dir.parent)}:{node.lineno} "
                            f"from {node.module} import ..."
                        )

        self.assertEqual(
            offenders,
            [],
            "以下裸导入位于模块顶层（不是 try 的 fallback 分支），包模式下会崩：\n  "
            + "\n  ".join(offenders),
        )


class EvalSetFallbackTests(unittest.TestCase):
    """评测集回退：公开克隆里没有私有评测集，必须能自动回退。"""

    def test_eval_gen_falls_back_to_sample_set(self):
        from app import config
        from app.eval_gen import _eval_set_path

        path = _eval_set_path()
        self.assertTrue(
            path.exists(),
            f"评测集路径不存在：{path}（回退逻辑失效，公开克隆后无法跑生成评测）",
        )
        self.assertIn(path, (config.settings.eval_set_path, config.settings.sample_eval_set_path))

    def test_eval_rag_falls_back_to_sample_set(self):
        from app import config
        from app.eval_rag import _eval_file

        path = _eval_file(sample=False)
        self.assertTrue(path.exists(), f"评测集路径不存在：{path}")
        self.assertIn(path, (config.settings.eval_set_path, config.settings.sample_eval_set_path))

    def test_sample_eval_set_is_tracked_public_asset(self):
        """公开示例评测集必须存在于仓库中（它是面试官复现评测的唯一入口）。"""
        from app.config import settings

        self.assertTrue(
            settings.sample_eval_set_path.exists(),
            "缺少 data/sample_eval_set.json —— 公开仓库将无法复现评测",
        )


if __name__ == "__main__":
    unittest.main()
