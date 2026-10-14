"""Agent 工具集：让智能体从「只会检索」变成「会算、会核验」。

设计原则
--------
1. **纯函数、无副作用、无网络**：工具只做确定性计算，可被单元测试完整覆盖。
   这是与「让大模型自己算」最本质的区别 —— 大模型的算术不可靠，工具可靠。
2. **输入输出都是 JSON 友好的结构**：便于直接回填到 Agent 的 ToolMessage。
3. **失败要返回结构化错误，而不是抛异常**：Agent 需要知道「为什么算不了」，
   才能向用户解释或换一种方式提问。

三个工具
--------
  optics_condition_number  ：由分析向量组装测量矩阵 A，算条件数 κ₂ 与理论下界比较
  optics_stokes            ：由光强测量与仪器矩阵反解 Stokes 向量，算 DOP/偏振角
  verify_numbers           ：核验答案中的数字是否真的出现在检索证据里（防幻觉）
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

# 4 次测量下测量矩阵条件数的理论下界（正四面体构型）
CONDITION_NUMBER_LOWER_BOUND = math.sqrt(3.0)


# ══════════════════════════════════════════════════════════════════
#  工具 1：测量矩阵条件数
# ══════════════════════════════════════════════════════════════════

def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """按行归一化。分析向量的长度不影响条件数，但会影响可读性。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def condition_number(analysis_vectors: list[list[float]]) -> dict:
    """计算测量矩阵的条件数 κ₂ = σ_max / σ_min。

    参数
    ----
    analysis_vectors : N 个分析状态的 Stokes 向量（每个 4 维），N ≥ 4

    返回
    ----
    {
      "ok": bool,
      "condition_number": float,
      "singular_values": [...],
      "lower_bound": 1.732,
      "gap_to_optimal": float,     # 相对理论下界的差距
      "error": str | None
    }
    """
    try:
        matrix = np.asarray(analysis_vectors, dtype=float)
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": f"分析向量不是合法数值矩阵：{exc}"}

    if matrix.ndim != 2:
        return {"ok": False, "error": "analysis_vectors 必须是二维数组（N × 4）"}
    if matrix.shape[1] != 4:
        return {
            "ok": False,
            "error": f"每个分析向量必须是 4 维 Stokes 向量，当前为 {matrix.shape[1]} 维",
        }
    if matrix.shape[0] < 4:
        return {
            "ok": False,
            "error": f"测量次数 N={matrix.shape[0]} 少于 4，无法唯一反解 Stokes 向量",
        }

    normalized = _normalize_rows(matrix)
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    sigma_min = float(singular_values[-1])
    if sigma_min <= 0:
        return {"ok": False, "error": "测量矩阵奇异（最小奇异值为 0），无法反解"}

    kappa = float(singular_values[0] / sigma_min)
    return {
        "ok": True,
        "condition_number": round(kappa, 4),
        "singular_values": [round(float(s), 4) for s in singular_values],
        "lower_bound": round(CONDITION_NUMBER_LOWER_BOUND, 4),
        "gap_to_optimal": round(kappa / CONDITION_NUMBER_LOWER_BOUND - 1.0, 4),
        "measurement_count": int(matrix.shape[0]),
        "note": (
            "条件数越接近 √3≈1.732 越好；该值直接线性放大相对测量误差。"
        ),
    }


def ideal_tetrahedron_vectors() -> list[list[float]]:
    """正四面体构型的 4 个分析向量（条件数达到理论下界 √3）。

    用于向用户演示「什么是最优构型」，也是单元测试的基准输入。
    """
    a = 1.0 / math.sqrt(3.0)
    return [
        [1.0, a, a, a],
        [1.0, a, -a, -a],
        [1.0, -a, a, -a],
        [1.0, -a, -a, a],
    ]


# ══════════════════════════════════════════════════════════════════
#  工具 2：Stokes 重建
# ══════════════════════════════════════════════════════════════════

def reconstruct_stokes(intensities: list[float], instrument_matrix: list[list[float]]) -> dict:
    """由光强测量反解 Stokes 向量，并给出 DOP 与偏振角。"""
    try:
        i_vec = np.asarray(intensities, dtype=float)
        matrix = np.asarray(instrument_matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": f"输入不是合法数值：{exc}"}

    if matrix.ndim != 2 or matrix.shape[1] != 4:
        return {"ok": False, "error": "instrument_matrix 必须是 N × 4 矩阵"}
    if i_vec.ndim != 1 or i_vec.shape[0] != matrix.shape[0]:
        return {
            "ok": False,
            "error": f"光强向量长度 {i_vec.size} 与测量矩阵行数 {matrix.shape[0]} 不一致",
        }

    try:
        stokes, residuals, rank, _ = np.linalg.lstsq(matrix, i_vec, rcond=None)
    except np.linalg.LinAlgError as exc:
        return {"ok": False, "error": f"最小二乘求解失败：{exc}"}

    s0, s1, s2, s3 = (float(v) for v in stokes)
    if s0 <= 0:
        return {
            "ok": False,
            "error": f"解出的 S₀={s0:.4f} 非正，输入光强或仪器矩阵可能有误",
        }

    dop = math.sqrt(s1 * s1 + s2 * s2 + s3 * s3) / s0
    # 偏振方位角（线偏振部分的方位，0~180°）
    angle = math.degrees(math.atan2(s2, s1)) / 2.0 % 180.0
    ellipticity = math.degrees(math.asin(max(-1.0, min(1.0, s3 / s0)))) / 2.0

    residual_norm = float(np.linalg.norm(residuals)) if residuals.size else 0.0
    return {
        "ok": True,
        "stokes": [round(v, 6) for v in stokes],
        "dop": round(dop, 6),
        "azimuth_deg": round(angle, 4),
        "ellipticity_deg": round(ellipticity, 4),
        "matrix_rank": int(rank),
        "residual_norm": round(residual_norm, 8),
        "note": (
            "DOP>1 说明测量噪声较大或仪器矩阵标定有偏；"
            "residual_norm 是拟合残差，可用来判断数据一致性。"
        ),
    }


# ══════════════════════════════════════════════════════════════════
#  工具 3：数字核验（防幻觉）
# ══════════════════════════════════════════════════════════════════

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class NumberCheck:
    value: str
    supported: bool
    where: str


def _normalize_number(token: str) -> str:
    """归一化数字：去掉多余的 0 与正负号，便于比较 1.870 与 1.87。"""
    try:
        value = float(token)
    except ValueError:
        return token
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text if text else "0"


def verify_numbers(answer: str, evidence: list[str], min_length: int = 2) -> dict:
    """核验答案里的数字是否出现在检索证据中。

    这是最廉价、也最有效的幻觉检测：科研场景中模型编造数字是最致命的错误，
    而数字核验不需要调用任何模型，纯字符串比对即可给出确定性结论。

    参数
    ----
    answer   : 模型生成的回答
    evidence : 检索到的原文片段列表
    min_length : 只核验长度不小于该值的数字（默认 2，过滤「1」「2」这类序号）

    返回
    ----
    {"ok": True, "checked": n, "unsupported": [...], "supported_ratio": float}
    """
    if not answer:
        return {"ok": False, "error": "answer 为空"}
    if not evidence:
        return {
            "ok": False,
            "error": "没有提供证据片段，无法核验",
        }

    haystack = set()
    for chunk in evidence:
        for token in NUMBER_RE.findall(chunk or ""):
            if len(token.lstrip("-")) >= min_length:
                haystack.add(_normalize_number(token))

    checks: list[NumberCheck] = []
    for token in NUMBER_RE.findall(answer):
        if len(token.lstrip("-")) < min_length:
            continue
        normalized = _normalize_number(token)
        supported = normalized in haystack
        checks.append(
            NumberCheck(
                value=token,
                supported=supported,
                where="证据中已找到" if supported else "证据中未出现",
            )
        )

    if not checks:
        return {
            "ok": True,
            "checked": 0,
            "unsupported": [],
            "supported_ratio": 1.0,
            "note": "回答中没有需要核验的数字",
        }

    unsupported = [c.value for c in checks if not c.supported]
    supported_count = len(checks) - len(unsupported)
    return {
        "ok": True,
        "checked": len(checks),
        "supported": supported_count,
        "unsupported": unsupported,
        "supported_ratio": round(supported_count / len(checks), 4),
        "verdict": (
            "全部数字均可在证据中找到"
            if not unsupported
            else f"有 {len(unsupported)} 个数字未在证据中出现，需人工复核"
        ),
    }


# ══════════════════════════════════════════════════════════════════
#  LangChain 工具封装（供 Agent 调用）
# ══════════════════════════════════════════════════════════════════

TOOL_DESCRIPTIONS = {
    "optics_condition_number": (
        "计算偏振测量矩阵的条件数 κ₂ 并与理论下界 √3≈1.732 比较。"
        "当用户询问「某组测量配置的条件数是多少」「这个构型好不好」时调用。"
        "输入是若干 4 维 Stokes 分析向量。"
    ),
    "optics_stokes": (
        "由实测光强和仪器矩阵反解 Stokes 向量，输出 DOP 与偏振方位角。"
        "当用户给出光强读数并要求计算偏振态时调用。"
    ),
    "verify_numbers": (
        "核验一段回答中的所有数字是否真的出现在给定的证据原文里。"
        "在给出包含具体数值的结论前调用，用于自查是否编造数字。"
    ),
}


def build_langchain_tools():
    """构建 LangChain 工具列表。仅在需要工具调用时导入 langchain。"""
    import json

    from langchain_core.tools import tool

    @tool
    def optics_condition_number(analysis_vectors: list[list[float]]) -> str:
        """计算测量矩阵条件数 κ₂（输入为 N×4 的 Stokes 分析向量）。"""
        return json.dumps(condition_number(analysis_vectors), ensure_ascii=False)

    @tool
    def optics_stokes(intensities: list[float], instrument_matrix: list[list[float]]) -> str:
        """由光强与仪器矩阵反解 Stokes 向量、DOP 与偏振方位角。"""
        return json.dumps(reconstruct_stokes(intensities, instrument_matrix), ensure_ascii=False)

    @tool
    def verify_numbers(answer: str, evidence: list[str]) -> str:
        """核验回答中的数字是否都出现在证据原文中。"""
        return json.dumps(verify_numbers(answer, evidence), ensure_ascii=False)

    return [optics_condition_number, optics_stokes, verify_numbers]
