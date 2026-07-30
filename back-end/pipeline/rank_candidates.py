import json
import logging
import re
from difflib import SequenceMatcher
from functools import lru_cache

from .llm import (
    LLM_TEMPERATURE,
    call_json,
    call_text_with_metadata,
    extract_json,
    get_client,
    llm_label,
)
from .prompts import step_prompt


logger = logging.getLogger(__name__)

WEIGHTS = {
    "task_match": 0.35,
    "domain_fit": 0.30,
    "label_overlap": 0.20,
    "size_adequacy": 0.15,
}
UNVERIFIED_WARNING = (
    "Kết quả web search này chưa được xác minh đầy đủ; "
    "hãy tự kiểm tra link trước khi sử dụng."
)
MIN_VERIFIED_SCORE = 3.0
MIN_UNVERIFIED_SCORE = 3.0
RANK_BATCH_SIZE = 8
RANK_MAX_OUTPUT_TOKENS = 4000

CORE_KEYWORD_SYSTEM = """Bạn trích xuất từ khóa cốt lõi để kiểm tra metadata dataset.
Chỉ trả JSON object: {"core_keywords": ["keyword"]}.
Liệt kê 3-5 từ/cụm từ ngắn gồm khái niệm subject và đồng nghĩa phổ biến.
Loại bỏ từ chung chỉ mô tả task, modality hoặc loại dataset.
Ví dụ:
- subject="human detection", task_type="object detection"
  -> ["human", "person", "people", "pedestrian"]
- subject="vehicles", task_type="image classification"
  -> ["vehicle", "car", "truck", "automobile"]
- subject="sentiment analysis on product reviews", task_type="text classification"
  -> ["sentiment", "review", "opinion", "rating"]
Áp dụng cùng quy tắc cho mọi domain, kể cả domain chưa có trong ví dụ."""


class RankingResponseTruncatedError(ValueError):
    def __init__(
        self,
        expected_count: int,
        observed_count: int,
        detail: str,
        usage: dict | None = None,
    ):
        self.expected_count = expected_count
        self.observed_count = observed_count
        self.usage = usage or {}
        super().__init__(
            "LLM ranking response truncated, "
            f"got {observed_count}/{expected_count} objects: {detail}"
        )


def _reasoning_style(text: str) -> str:
    tokens = re.findall(r"[a-z0-9\u00c0-\u024f]+", str(text).casefold())
    return " ".join(tokens)


def _reasoning_echoes_constraint_notes(
    result: dict,
    candidates: list[dict],
) -> bool:
    """Detect a uniformly templated batch that appears derived from auto hints."""
    rows = result.get("verified", []) + result.get("unverified", [])
    reasonings = [
        _reasoning_style(row.get("reasoning", ""))
        for row in rows
        if row.get("reasoning")
    ]
    if len(reasonings) < 2 or len(reasonings) != len(rows):
        return False

    reference = reasonings[0]
    uniformly_styled = all(
        SequenceMatcher(None, reference, reasoning).ratio() >= 0.72
        for reasoning in reasonings[1:]
    )
    if not uniformly_styled:
        return False

    notes_by_id = {
        candidate["id"]: _reasoning_style(
            " ".join(candidate.get("constraint_notes", []))
        )
        for candidate in candidates
    }
    note_like = 0
    for row, reasoning in zip(rows, reasonings):
        note = notes_by_id.get(row.get("id"), "")
        if note and SequenceMatcher(None, reasoning, note).ratio() >= 0.48:
            note_like += 1
    return note_like == len(rows)


def _normalize_keywords(values: list) -> list[str]:
    output = []
    for value in values:
        keyword = " ".join(
            re.findall(r"[a-z0-9\u00c0-\u024f]+", str(value).casefold())
        )
        if len(keyword) <= 2:
            continue
        if keyword not in output:
            output.append(keyword)
    return output[:5]


@lru_cache(maxsize=256)
def _llm_core_keywords_cached(subject: str, task_type: str) -> tuple[str, ...]:
    result = call_json(
        CORE_KEYWORD_SYSTEM,
        json.dumps(
            {"subject": subject, "task_type": task_type},
            ensure_ascii=False,
        ),
        180,
    )
    if not isinstance(result, dict) or not isinstance(
        result.get("core_keywords"), list
    ):
        raise ValueError("LLM core keyword response không đúng schema")
    keywords = _normalize_keywords(result["core_keywords"])
    if not keywords:
        raise ValueError("LLM không trả core keyword hợp lệ")
    return tuple(keywords)


def _fallback_core_keywords(subject: str, task_type: str) -> list[str]:
    """Domain-agnostic fallback: subtract task tokens from subject tokens."""
    subject_tokens = re.findall(r"[a-z0-9\u00c0-\u024f]+", subject.casefold())
    task_tokens = set(
        re.findall(r"[a-z0-9\u00c0-\u024f]+", task_type.casefold())
    )
    generic = {"general", "specific", "data", "dataset", "domain", "task"}
    return _normalize_keywords([
        token
        for token in subject_tokens
        if token not in task_tokens and token not in generic
    ])


def _resolve_core_keywords(intent: dict) -> tuple[list[str], str]:
    subject = str(intent.get("subject") or "").strip()
    task_type = str(intent.get("task_type") or "").strip()
    if not subject or subject.casefold() in {"general", "any", "unknown"}:
        return [], "not_applicable"
    if get_client():
        try:
            return list(
                _llm_core_keywords_cached(subject.casefold(), task_type.casefold())
            ), "llm_cached"
        except Exception as exc:
            logger.warning(
                "Core keyword extraction LLM failed; using generic token "
                "fallback: %s",
                exc,
            )
    return _fallback_core_keywords(subject, task_type), "generic_fallback"


def _core_keywords(intent: dict) -> list[str]:
    return _resolve_core_keywords(intent)[0]


def _candidate_metadata_text(candidate: dict) -> str:
    raw = candidate.get("raw_metadata") or {}
    raw_nested = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    return " ".join([
        str(candidate.get("id") or ""),
        str(candidate.get("title") or ""),
        str(candidate.get("description") or ""),
        " ".join(str(tag) for tag in candidate.get("tags", [])),
        str(raw.get("title") or ""),
        str(raw_nested.get("title") or ""),
    ]).casefold()


def _metadata_has_core_keyword(candidate: dict, cores: list[str]) -> bool:
    metadata_tokens = set(
        re.findall(r"[a-z0-9\u00c0-\u024f]+", _candidate_metadata_text(candidate))
    )
    normalized_tokens = metadata_tokens | {
        token[:-1]
        for token in metadata_tokens
        if token.endswith("s") and len(token) > 4
    }
    for core in cores:
        core_tokens = set(core.split())
        if core in _candidate_metadata_text(candidate):
            return True
        if core_tokens and core_tokens <= normalized_tokens:
            return True
        if any(
            abs(len(core_token) - len(token)) <= 2
            and SequenceMatcher(None, core_token, token).ratio() >= 0.84
            for core_token in core_tokens
            for token in normalized_tokens
        ):
            return True
    return False


def _apply_core_keyword_guard(
    intent: dict,
    candidates: list[dict],
) -> tuple[list[dict], list[str], list[str], str]:
    """Flag high-scoring rows whose metadata lacks the subject core."""
    cores, keyword_source = _resolve_core_keywords(intent)
    if not cores:
        return candidates, [], [], keyword_source
    guarded = []
    flagged_ids = []
    primary_core = cores[0]
    for candidate in candidates:
        row = dict(candidate)
        if (
            float(row.get("total_score") or 0) >= MIN_VERIFIED_SCORE
            and not _metadata_has_core_keyword(row, cores)
        ):
            warning = (
                f"⚠️ Core keyword '{primary_core}' không tìm thấy trong metadata "
                "— cần tự kiểm tra thêm trước khi dùng."
            )
            row["constraint_status"] = "needs_review"
            row["needs_review"] = True
            row["review_warning"] = warning
            row["core_keyword"] = primary_core
            notes = list(row.get("constraint_notes", []))
            if warning not in notes:
                notes.append(warning)
            row["constraint_notes"] = notes
            flagged_ids.append(str(row.get("id")))
            logger.warning(
                "Core keyword guard flagged candidate %s: %s",
                row.get("id"),
                warning,
            )
        guarded.append(row)
    return guarded, flagged_ids, cores, keyword_source


def score_total(row: dict) -> float:
    return round(sum(float(row[k]) * weight for k, weight in WEIGHTS.items()), 2)


def score_unverified(row: dict) -> float:
    return round((float(row["task_match"]) + float(row["domain_fit"])) / 2, 2)


def _tokens(text: str) -> set[str]:
    return {x for x in re.findall(r"[a-z0-9]+", text.lower()) if len(x) > 2}


def _wanted_tokens(intent: dict) -> set[str]:
    return _tokens(
        " ".join(intent["search_keywords_en"])
        + " "
        + intent["task_type"]
        + " "
        + intent["domain"]
    )


def _heuristic_verified(intent: dict, candidates: list[dict]) -> list[dict]:
    wanted = _wanted_tokens(intent)
    output = []
    for item in candidates:
        haystack = " ".join(
            [item["id"], item.get("description", ""), " ".join(item.get("tags", []))]
        )
        overlap = len(wanted & _tokens(haystack))
        base = min(5, 2 + overlap)
        constraint_status = item.get("constraint_status", "unknown")
        if constraint_status == "matched":
            base = min(5, base + 1)
        elif constraint_status == "mismatch":
            base = min(base, 2)
        downloads = item.get("downloads")
        size = 4 if isinstance(downloads, int) and downloads >= 100 else 3
        license_value = str(item.get("license") or "").lower()
        restricted = any(
            x in license_value for x in ("non-commercial", "research", "other")
        )
        row = {
            "id": item["id"],
            "task_match": base,
            "domain_fit": base,
            "label_overlap": max(1, min(5, base - (0 if intent["needs_labels"] else 1))),
            "size_adequacy": size,
            "access_type": "restricted" if restricted else "open",
            "reasoning": (
                "Mức khớp ước tính từ tên, mô tả, tag và ràng buộc: "
                + " ".join(item.get("constraint_notes", []))
            ).strip(),
        }
        row["total_score"] = score_total(row)
        output.append(row)
    return output


def _heuristic_unverified(intent: dict, candidates: list[dict]) -> list[dict]:
    wanted = _wanted_tokens(intent)
    output = []
    for item in candidates:
        evidence = f"{item.get('title', '')} {item.get('snippet', '')}"
        overlap = len(wanted & _tokens(evidence))
        base = min(5, 1 + overlap)
        if item.get("constraint_status") == "mismatch":
            base = min(base, 2)
        row = {
            "id": item["id"],
            "task_match": base,
            "domain_fit": base,
            "reasoning": (
                "Mức khớp sơ bộ chỉ dựa trên title/snippet. " + UNVERIFIED_WARNING
            ),
        }
        row["preliminary_score"] = score_unverified(row)
        output.append(row)
    return output


def _llm_rank_batch(
    intent: dict,
    verified: list[dict],
    unverified: list[dict],
) -> dict:
    verified_compact = [{
        "id": x["id"],
        "source": x["source"],
        "downloads": x.get("downloads"),
        "tags": x.get("tags", [])[:12],
        "license": x.get("license"),
        "description": x.get("description", "")[:500],
        "sample_count": x.get("sample_count"),
        "features_text": x.get("features_text", "")[:1500],
        "constraint_status": x.get("constraint_status", "unknown"),
        "constraint_score": x.get("constraint_score"),
        "constraint_notes": x.get("constraint_notes", []),
        "constraint_task_matched": x.get("constraint_task_matched"),
        "constraint_subject_matched": x.get("constraint_subject_matched"),
        "confidence": "verified",
    } for x in verified]
    unverified_compact = [{
        "id": x["id"],
        "source": x["source"],
        "title": x.get("title", ""),
        "snippet": x.get("snippet", "")[:500],
        "url": x["url"],
        "constraint_status": x.get("constraint_status", "unknown"),
        "constraint_notes": x.get("constraint_notes", []),
        "constraint_task_matched": x.get("constraint_task_matched"),
        "constraint_subject_matched": x.get("constraint_subject_matched"),
        "confidence": "unverified",
    } for x in unverified]
    expected_count = len(verified) + len(unverified)
    try:
        raw_text, usage = call_text_with_metadata(
            step_prompt("STEP_3"),
            json.dumps({
                "intent": intent,
                "verified_candidates": verified_compact,
                "unverified_candidates": unverified_compact,
            }, ensure_ascii=False),
            RANK_MAX_OUTPUT_TOKENS,
        )
        result = extract_json(raw_text)
    except json.JSONDecodeError as exc:
        observed_count = len(re.findall(r'"id"\s*:', exc.doc or ""))
        raise RankingResponseTruncatedError(
            expected_count,
            observed_count,
            f"{exc.msg} at character {exc.pos}",
            usage if "usage" in locals() else {},
        ) from exc
    if not isinstance(result, dict):
        raise ValueError("LLM phải trả JSON object gồm verified và unverified")
    result.setdefault("verified", [])
    result.setdefault("unverified", [])
    allowed_verified = {x["id"] for x in verified}
    allowed_unverified = {x["id"] for x in unverified}
    all_allowed = allowed_verified | allowed_unverified
    returned_ids = {
        row.get("id") for row in result["verified"] + result["unverified"]
    }
    if returned_ids - all_allowed:
        raise ValueError("LLM trả candidate ngoài danh sách")
    # Drop wrong-lane objects instead of routing them: their score schema follows
    # the emitted lane and is therefore unsafe for the authoritative input lane.
    result["verified"] = [
        row for row in result["verified"] if row.get("id") in allowed_verified
    ]
    result["unverified"] = [
        row for row in result["unverified"] if row.get("id") in allowed_unverified
    ]
    result["_usage"] = usage
    return result


def _llm_rank(intent: dict, verified: list[dict], unverified: list[dict]) -> dict:
    tagged_candidates = [
        ("verified", candidate) for candidate in verified
    ] + [
        ("unverified", candidate) for candidate in unverified
    ]
    merged = {"verified": [], "unverified": []}
    total_batches = max(
        1,
        (len(tagged_candidates) + RANK_BATCH_SIZE - 1) // RANK_BATCH_SIZE,
    )
    batch_sizes = [
        len(tagged_candidates[offset:offset + RANK_BATCH_SIZE])
        for offset in range(0, len(tagged_candidates), RANK_BATCH_SIZE)
    ]
    logger.info(
        "LLM ranking batch plan: candidates=%d, batches=%d, "
        "configured_batch_size=%d, actual_batch_sizes=%s, strategy=fixed, "
        "temperature=%s",
        len(tagged_candidates),
        total_batches if tagged_candidates else 0,
        RANK_BATCH_SIZE,
        batch_sizes,
        LLM_TEMPERATURE,
    )
    merged["_batching"] = {
        "strategy": "fixed",
        "configured_batch_size": RANK_BATCH_SIZE,
        "batch_count": total_batches if tagged_candidates else 0,
        "actual_batch_sizes": batch_sizes,
        "max_output_tokens_per_batch": RANK_MAX_OUTPUT_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "batches": [],
    }

    for batch_index, offset in enumerate(
        range(0, len(tagged_candidates), RANK_BATCH_SIZE),
        start=1,
    ):
        batch = tagged_candidates[offset:offset + RANK_BATCH_SIZE]
        batch_verified = [
            candidate for lane, candidate in batch if lane == "verified"
        ]
        batch_unverified = [
            candidate for lane, candidate in batch if lane == "unverified"
        ]
        try:
            result = _llm_rank_batch(intent, batch_verified, batch_unverified)
        except RankingResponseTruncatedError as exc:
            completion_tokens = exc.usage.get("completion_tokens")
            finish_reason = exc.usage.get("finish_reason")
            merged["_batching"]["batches"].append({
                "batch_index": batch_index,
                "batch_size": len(batch),
                "completion_tokens": completion_tokens,
                "token_cap": RANK_MAX_OUTPUT_TOKENS,
                "token_utilization": (
                    round(completion_tokens / RANK_MAX_OUTPUT_TOKENS, 4)
                    if isinstance(completion_tokens, int)
                    else None
                ),
                "finish_reason": finish_reason,
                "truncated_json": True,
            })
            logger.error(
                "LLM ranking response truncated, got %d/%d objects "
                "(batch %d/%d, completion_tokens=%s, token_cap=%d, "
                "finish_reason=%s); candidates in this batch will use heuristic",
                exc.observed_count,
                exc.expected_count,
                batch_index,
                total_batches,
                completion_tokens,
                RANK_MAX_OUTPUT_TOKENS,
                finish_reason,
            )
            continue
        except Exception as exc:
            logger.error(
                "LLM ranking batch %d/%d failed validation: %s; "
                "candidates in this batch will use heuristic",
                batch_index,
                total_batches,
                exc,
            )
            continue
        if _reasoning_echoes_constraint_notes(
            result,
            batch_verified + batch_unverified,
        ):
            logger.warning(
                "rank reasoning trông như đang echo constraint_notes, "
                "có thể LLM đang không tự đánh giá độc lập "
                "(batch %d/%d, candidates=%d)",
                batch_index,
                total_batches,
                len(batch),
            )
        usage = result.pop("_usage", {})
        completion_tokens = usage.get("completion_tokens")
        finish_reason = usage.get("finish_reason")
        batch_usage = {
            "batch_index": batch_index,
            "batch_size": len(batch),
            "completion_tokens": completion_tokens,
            "token_cap": RANK_MAX_OUTPUT_TOKENS,
            "token_utilization": (
                round(completion_tokens / RANK_MAX_OUTPUT_TOKENS, 4)
                if isinstance(completion_tokens, int)
                else None
            ),
            "finish_reason": finish_reason,
            "truncated_json": False,
        }
        merged["_batching"]["batches"].append(batch_usage)
        logger.info(
            "LLM ranking batch usage: batch=%d/%d, batch_size=%d, "
            "completion_tokens=%s, token_cap=%d, utilization=%s, finish_reason=%s",
            batch_index,
            total_batches,
            len(batch),
            completion_tokens,
            RANK_MAX_OUTPUT_TOKENS,
            batch_usage["token_utilization"],
            finish_reason,
        )
        merged["verified"].extend(result["verified"])
        merged["unverified"].extend(result["unverified"])
    return merged


def rank_candidates(
    intent: dict,
    candidates: list[dict],
    *,
    include_diagnostics: bool = False,
) -> tuple[list[dict], str] | tuple[list[dict], str, dict]:
    verified = [x for x in candidates if x.get("confidence", "verified") == "verified"]
    unverified = [x for x in candidates if x.get("confidence") == "unverified"]
    mode = "Heuristic (chưa cấu hình LLM)"
    llm_scored_verified = 0
    llm_scored_unverified = 0
    heuristic_fallback_verified = 0
    heuristic_fallback_unverified = 0
    llm_verified_ids: set[str] = set()
    llm_unverified_ids: set[str] = set()
    batching_diagnostics = {
        "strategy": "not_run",
        "configured_batch_size": RANK_BATCH_SIZE,
        "batch_count": 0,
        "actual_batch_sizes": [],
        "max_output_tokens_per_batch": RANK_MAX_OUTPUT_TOKENS,
        "temperature": LLM_TEMPERATURE,
    }

    if get_client():
        try:
            ranked_groups = _llm_rank(intent, verified, unverified)
            batching_diagnostics = ranked_groups.pop(
                "_batching",
                batching_diagnostics,
            )
            ranked_verified = ranked_groups["verified"]
            ranked_unverified = ranked_groups["unverified"]
            llm_scored_verified = len(ranked_verified)
            llm_scored_unverified = len(ranked_unverified)
            returned_verified_ids = {
                row.get("id") for row in ranked_verified if row.get("id")
            }
            returned_unverified_ids = {
                row.get("id") for row in ranked_unverified if row.get("id")
            }
            llm_verified_ids = set(returned_verified_ids)
            llm_unverified_ids = set(returned_unverified_ids)
            missing_verified = [
                candidate
                for candidate in verified
                if candidate["id"] not in returned_verified_ids
            ]
            missing_unverified = [
                candidate
                for candidate in unverified
                if candidate["id"] not in returned_unverified_ids
            ]
            missing_count = len(missing_verified) + len(missing_unverified)
            total_count = len(verified) + len(unverified)
            if missing_count:
                heuristic_fallback_verified = len(missing_verified)
                heuristic_fallback_unverified = len(missing_unverified)
                logger.warning(
                    "LLM missing scores for %d/%d candidates "
                    "(verified=%d, unverified=%d); falling back to heuristic "
                    "for missing candidates",
                    missing_count,
                    total_count,
                    len(missing_verified),
                    len(missing_unverified),
                )
                ranked_verified.extend(
                    _heuristic_verified(intent, missing_verified)
                )
                ranked_unverified.extend(
                    _heuristic_unverified(intent, missing_unverified)
                )
            mode = f"LLM ({llm_label()})"
        except Exception as exc:
            ranked_verified = _heuristic_verified(intent, verified)
            ranked_unverified = _heuristic_unverified(intent, unverified)
            heuristic_fallback_verified = len(verified)
            heuristic_fallback_unverified = len(unverified)
            mode = f"Heuristic (LLM lỗi: {exc})"
    else:
        ranked_verified = _heuristic_verified(intent, verified)
        ranked_unverified = _heuristic_unverified(intent, unverified)
        heuristic_fallback_verified = len(verified)
        heuristic_fallback_unverified = len(unverified)

    verified_by_id = {x["id"]: x for x in ranked_verified}
    unverified_by_id = {x["id"]: x for x in ranked_unverified}
    merged_verified = []
    merged_unverified = []

    for candidate in verified:
        score = verified_by_id.get(candidate["id"])
        if not score:
            continue
        if candidate.get("constraint_subject_matched") is False:
            score = {
                **score,
                "task_match": min(int(score["task_match"]), 2),
                "domain_fit": min(int(score["domain_fit"]), 2),
            }
        row = {**candidate, **score, "confidence": "verified"}
        row["total_score"] = score_total(row)
        merged_verified.append(row)

    for candidate in unverified:
        score = unverified_by_id.get(candidate["id"])
        if not score:
            continue
        if candidate.get("constraint_subject_matched") is False:
            score = {
                **score,
                "task_match": min(int(score["task_match"]), 2),
                "domain_fit": min(int(score["domain_fit"]), 2),
            }
        reasoning = score.get("reasoning", "")
        if "chưa được xác minh" not in reasoning.lower():
            reasoning = f"{reasoning.rstrip()} {UNVERIFIED_WARNING}".strip()
        row = {
            **candidate,
            "task_match": score["task_match"],
            "domain_fit": score["domain_fit"],
            "reasoning": reasoning,
            "preliminary_score": score_unverified(score),
            "confidence": "unverified",
        }
        # Deliberately do not add label_overlap, size_adequacy, access_type,
        # license, downloads, or likes to web-search candidates.
        merged_unverified.append(row)

    merged_verified.sort(
        key=lambda x: (x["access_type"] == "open", x["total_score"]), reverse=True
    )
    merged_unverified.sort(key=lambda x: x["preliminary_score"], reverse=True)
    scored_verified = [
        {
            **row,
            "scoring_source": (
                "llm" if row["id"] in llm_verified_ids else "heuristic"
            ),
        }
        for row in merged_verified
    ]
    scored_unverified = [
        {
            **row,
            "scoring_source": (
                "llm" if row["id"] in llm_unverified_ids else "heuristic"
            ),
        }
        for row in merged_unverified
    ]
    excluded_by_threshold: list[dict] = []
    kept_verified = []
    kept_unverified = []

    for row in merged_verified:
        reasons = []
        if row["total_score"] < MIN_VERIFIED_SCORE:
            reasons.append(
                f"total_score={row['total_score']} < {MIN_VERIFIED_SCORE}"
            )
        if row.get("task_match", 0) < 3:
            reasons.append(f"task_match={row.get('task_match', 0)} < 3")
        if reasons:
            excluded_by_threshold.append({
                "id": row["id"],
                "confidence": "verified",
                "total_score": row["total_score"],
                "task_match": row.get("task_match"),
                "domain_fit": row.get("domain_fit"),
                "label_overlap": row.get("label_overlap"),
                "size_adequacy": row.get("size_adequacy"),
                "reasoning": row.get("reasoning", ""),
                "exclusion_reasons": reasons,
            })
        else:
            kept_verified.append(row)

    for row in merged_unverified:
        reasons = []
        if row["preliminary_score"] < MIN_UNVERIFIED_SCORE:
            reasons.append(
                f"preliminary_score={row['preliminary_score']} < "
                f"{MIN_UNVERIFIED_SCORE}"
            )
        if row.get("task_match", 0) < 3:
            reasons.append(f"task_match={row.get('task_match', 0)} < 3")
        if reasons:
            excluded_by_threshold.append({
                "id": row["id"],
                "confidence": "unverified",
                "preliminary_score": row["preliminary_score"],
                "task_match": row.get("task_match"),
                "domain_fit": row.get("domain_fit"),
                "reasoning": row.get("reasoning", ""),
                "exclusion_reasons": reasons,
            })
        else:
            kept_unverified.append(row)

    if excluded_by_threshold:
        logger.warning(
            "Ranking threshold excluded %d/%d scored candidates",
            len(excluded_by_threshold),
            len(scored_verified) + len(scored_unverified),
        )

    (
        kept_verified,
        core_guard_flagged_ids,
        resolved_core_keywords,
        core_keyword_source,
    ) = _apply_core_keyword_guard(
        intent,
        kept_verified,
    )
    ranked = kept_verified + kept_unverified
    diagnostics = {
        "input_candidate_count": len(candidates),
        "input_verified_count": len(verified),
        "input_unverified_count": len(unverified),
        "llm_scored_count": llm_scored_verified + llm_scored_unverified,
        "llm_scored_verified": llm_scored_verified,
        "llm_scored_unverified": llm_scored_unverified,
        "heuristic_fallback_count": (
            heuristic_fallback_verified + heuristic_fallback_unverified
        ),
        "heuristic_fallback_verified": heuristic_fallback_verified,
        "heuristic_fallback_unverified": heuristic_fallback_unverified,
        "llm_batching": batching_diagnostics,
        "scored_before_threshold": {
            "verified": scored_verified,
            "unverified": scored_unverified,
        },
        "threshold_excluded_count": len(excluded_by_threshold),
        "excluded_by_threshold": excluded_by_threshold,
        "ranked_after_threshold_count": len(ranked),
        "core_keyword_guard": {
            "keywords": resolved_core_keywords,
            "keyword_source": core_keyword_source,
            "flagged_count": len(core_guard_flagged_ids),
            "flagged_ids": core_guard_flagged_ids,
        },
    }
    if include_diagnostics:
        return ranked, mode, diagnostics
    return ranked, mode
