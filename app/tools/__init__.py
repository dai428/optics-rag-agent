"""Agent 工具集：确定性计算与核验，用于抑制幻觉、支撑可验证结论。"""

from .optics import (
    CONDITION_NUMBER_LOWER_BOUND,
    TOOL_DESCRIPTIONS,
    build_langchain_tools,
    condition_number,
    ideal_tetrahedron_vectors,
    reconstruct_stokes,
    verify_numbers,
)

__all__ = [
    "CONDITION_NUMBER_LOWER_BOUND",
    "TOOL_DESCRIPTIONS",
    "build_langchain_tools",
    "condition_number",
    "ideal_tetrahedron_vectors",
    "reconstruct_stokes",
    "verify_numbers",
]
