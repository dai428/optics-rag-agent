"""
corpus_stats.py — L1 练习：统计语料目录

张万森 · L1 Python 工程能力训练
────────────────────────────────────────────────────────
任务：扫描 data/docs 下的 .md/.txt/.csv，按子目录统计文件数与总字节数。

这是「知识库构建」的第一步——你以后手写 RAG 时，第一件事就是数清语料。

⚠️ 这个文件**能正常跑出结果**，但里面有 3 处工程缺陷。
   能跑 ≠ 写对。你的任务不是运行它，而是读它、找出问题、说出为什么。

   提示：3 处问题分别涉及
     ① 目录不存在时会怎样
     ② 顶层文件会不会被统计到
     ③ 文件读不动时会怎样
────────────────────────────────────────────────────────
"""
import os
from pathlib import Path

DOCS_DIR = Path("D:/Desktop/rag-demo/data/docs")


def scan_corpus(docs_dir):
    stats = {}
    for pattern in ("**/*.md", "**/*.txt", "**/*.csv"):
        for f in sorted(docs_dir.glob(pattern)):
            group = f.parent.name
            if group not in stats:
                stats[group] = {"files": 0, "bytes": 0}
            stats[group]["files"] += 1
            stats[group]["bytes"] += os.path.getsize(f)
    return stats


def main():
    stats = scan_corpus(DOCS_DIR)
    total_files = 0
    total_bytes = 0
    for name, info in stats.items():
        print(f"{name:12s} {info['files']:3d} 个文件  {info['bytes']:8d} 字节")
        total_files += info["files"]
        total_bytes += info["bytes"]
    print(f"{'合计':12s} {total_files:3d} 个文件  {total_bytes:8d} 字节")


if __name__ == "__main__":
    main()
