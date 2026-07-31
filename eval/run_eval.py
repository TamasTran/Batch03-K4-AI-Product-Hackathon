from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(ROOT_DIR / ".env")
load_dotenv(BACKEND_DIR / ".env")

from agent import SearchAgent  # noqa: E402
from pipeline.constraints import evaluate_constraints  # noqa: E402


GOLDEN_PATH = Path(__file__).with_name("golden_queries.json")
DEFAULT_CHECKPOINT = Path(__file__).with_name(".eval_checkpoint.json")
REQUIRED_FIELDS = {
    "query",
}
REQUIRED_AXES = {
    "subject_task_overlap",
    "subject_no_overlap",
    "clarity",
    "ambiguity",
    "paraphrase",
    "large_candidate_pool",
    "small_candidate_pool",
    "multi_turn",
    "encoding",
    "complex_constraints",
}


def _normalized(value: Any) -> str:
    return str(value or "").casefold().replace("_", "-")


def _candidate_title(candidate: dict[str, Any]) -> str:
    raw = candidate.get("raw_metadata") or {}
    raw_nested = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    return str(
        candidate.get("title")
        or candidate.get("name")
        or raw.get("title")
        or raw_nested.get("title")
        or candidate.get("id")
        or ""
    )


def _load_goldens(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Golden file must contain a JSON array")
    if len(rows) != 50:
        raise ValueError(f"Expected exactly 50 golden queries, found {len(rows)}")
    for index, row in enumerate(rows, start=1):
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(f"Golden #{index} is missing fields: {sorted(missing)}")
        mode = row.get("mode", "search")
        if mode not in {"search", "clarification", "multi_turn"}:
            raise ValueError(f"Golden #{index} has unsupported mode={mode!r}")
        if mode == "search":
            for field in (
                "expected_subject_keywords",
                "must_not_match_if_only",
                "known_bad_examples",
                "known_good_examples_pattern",
            ):
                if not isinstance(row.get(field), list) or not row[field]:
                    raise ValueError(
                        f"Golden #{index} field {field} must be non-empty"
                    )
        if mode == "multi_turn" and not row.get("follow_up"):
            raise ValueError(f"Golden #{index} multi_turn needs follow_up")
    covered_axes = {
        axis
        for row in rows
        for axis in row.get("axis", [])
    }
    missing_axes = REQUIRED_AXES - covered_axes
    if missing_axes:
        raise ValueError(
            f"Golden suite is missing required axes: {sorted(missing_axes)}"
        )
    return rows


def _synthetic_task_only_candidate(case: dict[str, Any]) -> dict[str, Any]:
    task_text = " ".join(case["must_not_match_if_only"])
    return {
        "id": f"synthetic/{task_text.replace(' ', '-')}",
        "title": f"Generic {task_text} benchmark",
        "description": f"Dataset for {task_text}. No subject-specific metadata.",
        "tags": task_text.split(),
        "source": "Synthetic eval",
        "url": "https://example.invalid/eval",
        "confidence": "verified",
    }


def _assert_case(
    case: dict[str, Any],
    intent: dict[str, Any],
    verified: list[dict[str, Any]],
    *,
    pool_count: int | None = None,
) -> list[str]:
    failures: list[str] = []
    mode = case.get("mode", "search")
    if mode == "clarification":
        return failures

    subject_value = _normalized(intent.get("subject"))
    expected_subjects = [_normalized(x) for x in case["expected_subject_keywords"]]
    if not any(keyword in subject_value for keyword in expected_subjects):
        failures.append(
            "intent subject mismatch: "
            f"got {intent.get('subject')!r}, expected one of {expected_subjects}"
        )

    probe = evaluate_constraints(intent, _synthetic_task_only_candidate(case))
    if probe["constraint_status"] == "matched":
        failures.append(
            "OR-matching regression: task-only synthetic candidate became matched"
        )
    if probe.get("constraint_subject_matched") is not False:
        failures.append(
            "OR-matching regression: task-only synthetic candidate reports "
            f"constraint_subject_matched={probe.get('constraint_subject_matched')!r}"
        )

    verified_refs = [
        _normalized(f"{row.get('id', '')} {_candidate_title(row)}")
        for row in verified
    ]
    for bad in case["known_bad_examples"]:
        bad_norm = _normalized(bad)
        hits = [ref for ref in verified_refs if bad_norm in ref]
        if hits:
            failures.append(
                f"known bad candidate reached verified: {bad!r} -> {hits[:3]}"
            )

    good_patterns = [_normalized(x) for x in case["known_good_examples_pattern"]]
    good_hits = [
        _candidate_title(row)
        for row, ref in zip(verified, verified_refs)
        if any(pattern in ref for pattern in good_patterns)
    ]
    if not good_hits:
        failures.append(
            "no verified title/id contains a known-good pattern: "
            f"{case['known_good_examples_pattern']}"
        )

    for field, expected in case.get("expected_intent", {}).items():
        actual = _normalized(intent.get(field))
        if isinstance(expected, list):
            if not any(_normalized(value) in actual for value in expected):
                failures.append(
                    f"intent {field} mismatch: got {intent.get(field)!r}, "
                    f"expected one of {expected}"
                )
        elif _normalized(expected) != actual:
            failures.append(
                f"intent {field} mismatch: got {intent.get(field)!r}, "
                f"expected {expected!r}"
            )

    pool = case.get("pool_expectation") or {}
    if pool and pool_count is None:
        failures.append("candidate pool count unavailable")
    if pool_count is not None:
        if "min" in pool and pool_count < int(pool["min"]):
            failures.append(
                f"candidate pool too small: {pool_count} < {pool['min']}"
            )
        if "max" in pool and pool_count > int(pool["max"]):
            failures.append(
                f"candidate pool too large: {pool_count} > {pool['max']}"
            )

    for probe in case.get("constraint_probes", []):
        evaluated = evaluate_constraints(intent, probe["candidate"])
        expected_status = probe["expected_status"]
        if evaluated.get("constraint_status") != expected_status:
            failures.append(
                f"constraint probe {probe.get('name', probe['candidate']['id'])}: "
                f"status={evaluated.get('constraint_status')!r}, "
                f"expected={expected_status!r}; "
                f"notes={evaluated.get('constraint_notes')}"
            )
    return failures


def _rank_pool_count(run: Any) -> int | None:
    for event in run.tool_events:
        if event.tool == "rank_datasets":
            value = event.args.get("candidate_count")
            return int(value) if isinstance(value, int) else None
    return None


def _multi_turn_input(
    original: str,
    assistant_question: str | None,
    follow_up: str,
) -> str:
    return (
        f"Yêu cầu ban đầu: {original}\n"
        f"Assistant hỏi làm rõ: {assistant_question or ''}\n"
        f"User mới nhất: {follow_up}"
    )


def _intent_signature(intent: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        _normalized(intent.get(field))
        for field in ("task_type", "modality", "domain", "subject")
    )


def _credentials() -> dict[str, str]:
    return {
        "username": os.getenv("KAGGLE_USERNAME", ""),
        "key": os.getenv("KAGGLE_KEY", ""),
        "serpapi_api_key": os.getenv("SERPAPI_API_KEY", ""),
        "bing_search_api_key": os.getenv("BING_SEARCH_API_KEY", ""),
        "google_cse_api_key": os.getenv("GOOGLE_CSE_API_KEY", ""),
        "google_cse_id": os.getenv("GOOGLE_CSE_ID", ""),
    }


def _load_checkpoint(path: Path, resume: bool) -> dict[str, Any]:
    if not resume or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_checkpoint(path: Path, results: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the 50-query golden eval against the full dataset pipeline."
    )
    parser.add_argument("--goldens", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="With --resume, keep passed cases and execute failed cases again.",
    )
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--count", type=int)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["Hugging Face", "OpenML", "Zenodo"],
    )
    parser.add_argument("--no-web-fallback", action="store_true")
    args = parser.parse_args()

    cases = _load_goldens(args.goldens)
    start = max(0, args.start - 1)
    stop = len(cases) if args.count is None else min(len(cases), start + args.count)
    selected = list(enumerate(cases[start:stop], start=start + 1))
    results = _load_checkpoint(args.checkpoint, args.resume)
    passed = 0
    failed = 0
    started_at = time.time()
    group_baselines: dict[str, tuple[str, ...]] = {}

    for index, case in selected:
        key = str(index)
        if (
            args.resume
            and key in results
            and not (args.rerun_failed and not results[key].get("passed"))
        ):
            prior = results[key]
            status = "PASS" if prior.get("passed") else "FAIL"
            print(f"[{index:02d}/50] {status} (checkpoint) {case['query']}")
            passed += int(bool(prior.get("passed")))
            failed += int(not prior.get("passed"))
            continue

        print(f"[{index:02d}/50] RUN  {case['query']}", flush=True)
        agent = SearchAgent(
            enabled_sources=args.sources,
            limit=args.limit,
            credentials=_credentials(),
            web_fallback_enabled=not args.no_web_fallback,
        )
        try:
            first_run = agent.run(case["query"])
            if case.get("mode") == "multi_turn":
                failures = []
                if first_run.status != "clarification_required":
                    failures.append(
                        "multi-turn first turn should require clarification, "
                        f"got {first_run.status}"
                    )
                agent = SearchAgent(
                    enabled_sources=args.sources,
                    limit=args.limit,
                    credentials=_credentials(),
                    web_fallback_enabled=not args.no_web_fallback,
                )
                run = agent.run(_multi_turn_input(
                    case["query"],
                    first_run.clarification_question,
                    case["follow_up"],
                ))
                failures.extend(_assert_case(
                    case,
                    run.intent,
                    run.verified,
                    pool_count=_rank_pool_count(run),
                ))
                if run.status == "clarification_required":
                    failures.append(
                        "multi-turn follow-up repeated clarification instead "
                        "of continuing"
                    )
            else:
                run = first_run
                failures = _assert_case(
                    case,
                    run.intent,
                    run.verified,
                    pool_count=_rank_pool_count(run),
                )

            expected_status = case.get(
                "expected_status",
                "clarification_required"
                if case.get("mode") == "clarification"
                else "answered",
            )
            if run.status != expected_status:
                failures.append(
                    f"pipeline status={run.status}, expected={expected_status}, "
                    f"missing_fields={run.missing_fields}"
                )
            expected_missing = set(case.get("expected_missing_fields", []))
            if expected_missing and not expected_missing <= set(run.missing_fields):
                failures.append(
                    f"missing_fields={run.missing_fields}, expected at least "
                    f"{sorted(expected_missing)}"
                )

            group = case.get("intent_group")
            if group and run.status == "answered":
                signature = _intent_signature(run.intent)
                baseline = group_baselines.setdefault(group, signature)
                if signature != baseline:
                    failures.append(
                        f"paraphrase intent mismatch for group {group}: "
                        f"got={signature}, baseline={baseline}"
                    )
            record = {
                "query": case["query"],
                "passed": not failures,
                "failures": failures,
                "status": run.status,
                "parse_mode": run.parse_mode,
                "rank_mode": run.rank_mode,
                "warnings": run.errors,
                "pool_count": _rank_pool_count(run),
                "first_turn_status": (
                    first_run.status
                    if case.get("mode") == "multi_turn"
                    else None
                ),
                "intent": run.intent,
                "verified": [
                    {
                        "id": row.get("id"),
                        "title": _candidate_title(row),
                        "score": row.get("total_score"),
                        "constraint_status": row.get("constraint_status"),
                        "needs_review": row.get("needs_review", False),
                        "review_warning": row.get("review_warning"),
                    }
                    for row in run.verified
                ],
            }
        except Exception as exc:
            record = {
                "query": case["query"],
                "passed": False,
                "failures": [f"{type(exc).__name__}: {exc}"],
            }

        results[key] = record
        _save_checkpoint(args.checkpoint, results)
        if record["passed"]:
            passed += 1
            print(
                f"[{index:02d}/50] PASS verified={len(record.get('verified', []))}",
                flush=True,
            )
        else:
            failed += 1
            print(f"[{index:02d}/50] FAIL", flush=True)
            for reason in record["failures"]:
                print(f"           - {reason}", flush=True)

    elapsed = time.time() - started_at
    print(
        f"\nSUMMARY: {passed} passed, {failed} failed, "
        f"{len(selected)} evaluated in {elapsed:.1f}s"
    )
    print(f"Checkpoint: {args.checkpoint.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
