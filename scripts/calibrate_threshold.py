"""标定拒答阈值：测量「相关问题」与「无关问题」的重排分数分布。

为什么需要标定而不是拍一个数
----------------------------
拒答阈值直接决定两个错误率的平衡：
  · 阈值太高 → 该答的也拒答（漏答，用户体验差）
  · 阈值太低 → 不该答的硬答（幻觉，科研场景不可接受）

所以必须用真实数据看两类问题的分数分布，把阈值放在两者之间的空隙里。
"""
from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

# 允许从项目根目录直接运行本脚本（scripts/ 不在包的搜索路径里）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("PYTHONUTF8", "1")

from app.knowledge_base import retrieve  # noqa: E402

# 应能答出的问题（来自公开示例语料）
RELEVANT = [
    "测量矩阵条件数 κ2 的理论下界是多少？",
    "LCVR 相比传统机械旋转波片有什么优势？",
    "温度从 25℃ 升到 40℃ 条件数恶化了多少？",
    "Stokes 参量里的 S3 表示什么？",
    "延迟量随温度变化的经验模型是什么形式？",
    "示例实验用的光源波长和测量次数 N 是多少？",
    "偏振度 DOP 的计算公式是什么？",
    "LCVR 的主要缺点是什么？",
]

# 语料里完全没有的问题
IRRELEVANT = [
    "请问火星探测器的钛合金焊接工艺标准是什么？",
    "如何用 Python 写一个快速排序算法？",
    "红楼梦的作者是谁，成书于哪一年？",
    "红烧肉的做法需要放多少冰糖？",
    "2026 年世界杯的举办城市有哪些？",
    "请解释一下 TCP 三次握手的过程。",
    "贝多芬第九交响曲首演于哪一年？",
    "如何在家养一只柯基犬？",
]


def scores(questions: list[str]) -> list[float]:
    out = []
    for q in questions:
        result = retrieve(q, k=5)
        top = result.top_score
        out.append(top if top is not None else float("-inf"))
        print(f"  {top:>8.3f}  {q}")
    return out


def main() -> None:
    print("=== 相关问题（应能作答）===")
    relevant = scores(RELEVANT)
    print("\n=== 无关问题（应拒答）===")
    irrelevant = scores(IRRELEVANT)

    print("\n=== 分布 ===")
    print(f"相关问题：min={min(relevant):.3f}  中位={statistics.median(relevant):.3f}  "
          f"max={max(relevant):.3f}")
    print(f"无关问题：min={min(irrelevant):.3f}  中位={statistics.median(irrelevant):.3f}  "
          f"max={max(irrelevant):.3f}")

    gap_low, gap_high = max(irrelevant), min(relevant)
    print(f"\n两类之间的空隙：[{gap_low:.3f}, {gap_high:.3f}]")

    if gap_low < gap_high:
        mid = (gap_low + gap_high) / 2
        print(f"建议阈值：{mid:.2f}（居中于空隙）")
        print(f"  用该阈值：相关题全部通过={all(s > mid for s in relevant)}，"
              f"无关题全部拒答={all(s < mid for s in irrelevant)}")
        print("\n⚠️ 请把该值写入 .env 的 ABSTAIN_THRESHOLD，并确认量纲：")
        print("   bge-reranker-base 输出 sigmoid 概率（0~1）；")
        print("   若换成输出 raw logit 的重排模型，此阈值必须重新标定。")
    else:
        print("⚠️ 两类分数重叠，单一阈值无法完全分开。")
        print(f"   重叠区间的无关题最高分 {gap_low:.3f}，相关问题最低分 {gap_high:.3f}")
        print("   建议改用「阈值 + 关键词命中」双条件，或换更强的重排模型。")


if __name__ == "__main__":
    main()
