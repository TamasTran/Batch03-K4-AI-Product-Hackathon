"""Manual live-LLM regression for the semantic intent contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
load_dotenv(ROOT_DIR / ".env")
load_dotenv(BACKEND_DIR / ".env")

from api import _analysis_input, _clarification_transcript
from pipeline.llm import get_llm_config, llm_label
from pipeline.parse_task import parse_task


CASES = [
    "Tìm dataset cho bài toán phát hiện ô tô.",
    "Tìm xe trong ảnh",
    "Định vị phương tiện",
    "Khoanh vùng xe trên đường",
    "vehicle localization dataset",
    "Tôi cần dữ liệu để train model",
]


def summary(text: str, intent: dict, parse_mode: str) -> dict:
    return {
        "input": text,
        "parse_mode": parse_mode,
        "task_type": intent.get("task_type"),
        "modality": intent.get("modality"),
        "domain": intent.get("domain"),
        "field_confidence": intent.get("field_confidence"),
        "needs_clarification": intent.get("needs_clarification"),
        "missing_fields": intent.get("missing_fields"),
    }


def main() -> int:
    if get_llm_config() is None:
        print("LIVE_TEST_SKIPPED: chưa cấu hình LLM.")
        return 2

    print(f"LIVE_MODEL: {llm_label()}")
    rows: list[dict] = []
    for text in CASES:
        intent, parse_mode = parse_task(text)
        row = summary(text, intent, parse_mode)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if not parse_mode.startswith("LLM"):
            print(
                "LIVE_TEST_BLOCKED: provider không trả kết quả LLM; "
                "parse_task đã chuyển sang heuristic nên không thể đánh giá semantic prompt."
            )
            return 2

    first_ambiguous = rows[-1]
    first_intent, _ = parse_task(CASES[-1])
    context = _clarification_transcript(
        CASES[-1],
        None,
        first_intent.get("clarification_question"),
    )
    multi_turn_input = _analysis_input("nhận diện ảnh", context)
    second_intent, second_mode = parse_task(multi_turn_input)
    multi_turn = summary(
        "multi-turn: Tôi cần dữ liệu để train model → nhận diện ảnh",
        second_intent,
        second_mode,
    )
    print(json.dumps(multi_turn, ensure_ascii=False))

    semantic_rows = rows[:5]
    canonical = {
        (row["task_type"], row["modality"], row["domain"])
        for row in semantic_rows
    }
    failures = []
    if len(canonical) != 1:
        failures.append(f"semantic variants diverged: {sorted(canonical)}")
    if any(row["needs_clarification"] for row in semantic_rows):
        failures.append("a clear vehicle-localization prompt requested clarification")
    if not first_ambiguous["needs_clarification"]:
        failures.append("the intentionally ambiguous prompt was accepted")
    if multi_turn["needs_clarification"]:
        failures.append("the clarified multi-turn prompt asked again")

    if failures:
        print("LIVE_TEST_FAILED:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("LIVE_TEST_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
