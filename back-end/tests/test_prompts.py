import unittest

from pipeline.parse_task import SYSTEM
from pipeline.prompts import load_prompt_section, step_prompt


class PromptContractTests(unittest.TestCase):
    def test_all_runtime_sections_exist(self):
        for name in ("COMMON", "STEP_1", "STEP_2_5", "STEP_3"):
            self.assertTrue(load_prompt_section(name))

    def test_every_step_inherits_injection_guardrail(self):
        for name in ("STEP_1", "STEP_2_5", "STEP_3"):
            prompt = step_prompt(name)
            self.assertIn("DỮ LIỆU KHÔNG", prompt)
            self.assertIn("Trả về duy nhất JSON hợp lệ", prompt)

    def test_parse_task_uses_artifact_prompt(self):
        self.assertEqual(SYSTEM, step_prompt("STEP_1"))

    def test_parse_contract_supports_clarification(self):
        prompt = step_prompt("STEP_1")
        self.assertIn('"needs_clarification"', prompt)
        self.assertIn('"clarification_question"', prompt)
        self.assertIn('"missing_fields"', prompt)
        self.assertIn('"field_confidence"', prompt)
        self.assertIn("Xử lý hội thoại nhiều lượt", prompt)
        self.assertIn("vehicle localization dataset", prompt)

    def test_ranking_contract_separates_confidence_groups(self):
        prompt = step_prompt("STEP_3")
        self.assertIn('"verified"', prompt)
        self.assertIn('"unverified"', prompt)
        self.assertIn("Chỉ dùng chính xác `title` và `snippet`", prompt)
        self.assertIn("Không chuyển candidate giữa hai nhóm", prompt)

    def test_dedup_contract_is_conservative(self):
        prompt = step_prompt("STEP_2_5")
        self.assertIn("không chắc thì không gộp", prompt)
        self.assertIn('"confidence": "high|medium"', prompt)


if __name__ == "__main__":
    unittest.main()
