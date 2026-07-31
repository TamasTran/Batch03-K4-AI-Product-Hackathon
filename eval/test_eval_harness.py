import unittest

from tests.run_eval import (
    GOLDEN_PATH,
    REQUIRED_AXES,
    _intent_signature,
    _load_goldens,
    _multi_turn_input,
)


class EvalHarnessTests(unittest.TestCase):
    def test_golden_suite_has_exactly_50_cases_and_all_axes(self):
        rows = _load_goldens(GOLDEN_PATH)
        self.assertEqual(len(rows), 50)
        covered = {
            axis
            for row in rows
            for axis in row.get("axis", [])
        }
        self.assertTrue(REQUIRED_AXES <= covered)

    def test_paraphrase_group_has_three_vehicle_phrasings(self):
        rows = _load_goldens(GOLDEN_PATH)
        grouped = [
            row for row in rows
            if row.get("intent_group") == "vehicle_detection"
        ]
        self.assertEqual(len(grouped), 3)

    def test_multi_turn_transcript_preserves_all_three_parts(self):
        transcript = _multi_turn_input(
            "tôi cần data",
            "Bạn cần tác vụ gì?",
            "phân loại spam",
        )
        self.assertIn("Yêu cầu ban đầu: tôi cần data", transcript)
        self.assertIn("Assistant hỏi làm rõ: Bạn cần tác vụ gì?", transcript)
        self.assertIn("User mới nhất: phân loại spam", transcript)

    def test_intent_signature_compares_semantic_core_fields(self):
        first = {
            "task_type": "object detection",
            "modality": "image",
            "domain": "computer vision",
            "subject": "vehicles",
        }
        second = dict(first)
        self.assertEqual(_intent_signature(first), _intent_signature(second))


if __name__ == "__main__":
    unittest.main()
