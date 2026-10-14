"""Phase 1 测试：确定性 Agent 工具（光学计算 + 数字核验）。

这些工具的价值就在于「可被完整测试」——大模型的算术不可靠，
而工具是纯函数，能给出确定性的、可复现的结果。
"""

from __future__ import annotations

import unittest

from app.tools.optics import (
    CONDITION_NUMBER_LOWER_BOUND,
    condition_number,
    ideal_tetrahedron_vectors,
    reconstruct_stokes,
    verify_numbers,
)


class ConditionNumberTests(unittest.TestCase):
    def test_ideal_tetrahedron_hits_lower_bound(self):
        result = condition_number(ideal_tetrahedron_vectors())
        self.assertTrue(result["ok"])
        # 正四面体构型的条件数应等于理论下界 √3
        self.assertAlmostEqual(result["condition_number"], CONDITION_NUMBER_LOWER_BOUND, places=3)
        self.assertAlmostEqual(result["gap_to_optimal"], 0.0, places=3)

    def test_degenerate_matrix_is_rejected(self):
        # 4 个完全相同的分析向量 → 矩阵秩亏，无法反解
        result = condition_number([[1, 0, 0, 0]] * 4)
        self.assertFalse(result["ok"])
        self.assertIn("奇异", result["error"])

    def test_rejects_wrong_dimension_and_too_few(self):
        self.assertFalse(condition_number([[1, 0, 0]] * 4)["ok"])
        self.assertFalse(condition_number([[1, 0, 0, 0]] * 3)["ok"])

    def test_worse_geometry_has_larger_condition_number(self):
        good = condition_number(ideal_tetrahedron_vectors())["condition_number"]
        # 让一个分析向量退化，条件数必然变大
        bad_vectors = ideal_tetrahedron_vectors()
        bad_vectors[3] = bad_vectors[0]
        bad = condition_number(bad_vectors)
        self.assertTrue(bad["ok"])
        self.assertGreater(bad["condition_number"], good)


class StokesTests(unittest.TestCase):
    def test_recovers_known_state(self):
        matrix = ideal_tetrahedron_vectors()
        import numpy as np

        target = np.array([1.0, 0.6, 0.0, 0.8])  # DOP = 1 的完全偏振态
        intensities = (np.asarray(matrix) @ target).tolist()

        result = reconstruct_stokes(intensities, matrix)
        self.assertTrue(result["ok"])
        for expected, actual in zip(target.tolist(), result["stokes"]):
            self.assertAlmostEqual(expected, actual, places=6)
        self.assertAlmostEqual(result["dop"], 1.0, places=6)

    def test_rejects_mismatched_lengths(self):
        result = reconstruct_stokes([1.0, 2.0], ideal_tetrahedron_vectors())
        self.assertFalse(result["ok"])
        self.assertIn("不一致", result["error"])

    def test_rejects_bad_matrix_shape(self):
        result = reconstruct_stokes([1.0, 2.0, 3.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertFalse(result["ok"])


class VerifyNumbersTests(unittest.TestCase):
    def test_detects_fabricated_number(self):
        answer = "条件数为 1.87，恶化约 14.4%，温度系数是 9.99。"
        evidence = ["25 ℃ 时条件数 1.87，40 ℃ 时恶化 14.4%。"]
        result = verify_numbers(answer, evidence)
        self.assertTrue(result["ok"])
        self.assertIn("9.99", result["unsupported"])
        self.assertNotIn("1.87", result["unsupported"])
        self.assertLess(result["supported_ratio"], 1.0)

    def test_all_numbers_supported(self):
        answer = "下界是 1.732，测量次数 N=6。"
        evidence = ["条件数理论下界 √3 ≈ 1.732，推荐测量次数 6。"]
        result = verify_numbers(answer, evidence)
        self.assertEqual(result["unsupported"], [])
        self.assertEqual(result["supported_ratio"], 1.0)

    def test_ignores_short_numbers(self):
        # 单个数字（序号、枚举）不应被当作需要核验的数值
        result = verify_numbers("第 1 点、第 2 点", ["证据文本"])
        self.assertEqual(result["checked"], 0)

    def test_requires_evidence(self):
        self.assertFalse(verify_numbers("答案 1.23", [])["ok"])
        self.assertFalse(verify_numbers("", ["证据"])["ok"])

    def test_trailing_zero_normalization(self):
        # 1.870 与 1.87 是同一个数，不应误报为编造
        result = verify_numbers("条件数 1.870", ["条件数为 1.87"])
        self.assertEqual(result["unsupported"], [])


if __name__ == "__main__":
    unittest.main()
