from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Any


LANGUAGE_ALIASES = {
    "vietnamese": {"vi", "vie", "vietnamese", "tiếng việt"},
    "english": {"en", "eng", "english"},
    "multilingual": {"multilingual", "multi-lingual"},
}

TASK_TYPE_KEYWORDS = {
    "classification",
    "classify",
    "detection",
    "detect",
    "detector",
    "recognition",
    "recognize",
    "segmentation",
    "segment",
    "localization",
    "tracking",
    "track",
    "prediction",
    "predict",
    "analysis",
}

GENERIC_SUBJECTS = {
    "",
    "any",
    "general",
    "generic",
    "other",
    "unknown",
    "data",
    "dataset",
}

SUBJECT_ALIASES = {
    "human": {
        "human", "person", "persons", "people", "pedestrian", "pedestrians",
        "face", "faces", "crowd", "crowds",
    },
    "person": {
        "human", "person", "persons", "people", "pedestrian", "pedestrians",
        "face", "faces", "crowd", "crowds",
    },
    "people": {
        "human", "person", "persons", "people", "pedestrian", "pedestrians",
        "face", "faces", "crowd", "crowds",
    },
    "pedestrian": {
        "human", "person", "persons", "people", "pedestrian", "pedestrians",
        "face", "faces", "crowd", "crowds",
    },
    "vehicle": {
        "vehicle", "vehicles", "car", "cars", "automobile", "automobiles",
        "truck", "trucks", "bus", "buses", "motorcycle", "motorcycles",
    },
    "vehicles": {
        "vehicle", "vehicles", "car", "cars", "automobile", "automobiles",
        "truck", "trucks", "bus", "buses", "motorcycle", "motorcycles",
    },
}


def _text(candidate: dict[str, Any]) -> str:
    return " ".join([
        str(candidate.get("id") or ""),
        str(candidate.get("title") or ""),
        str(candidate.get("description") or ""),
        str(candidate.get("snippet") or ""),
        " ".join(str(tag) for tag in candidate.get("tags", [])),
        str(candidate.get("features_text") or ""),
    ]).lower()


def _subject_text(candidate: dict[str, Any]) -> str:
    """Build subject evidence without trusting boilerplate label tables alone."""
    description = str(candidate.get("description") or "")
    description = re.sub(
        r"dataset\s+labels.*?(?:number\s+of\s+images|how\s+to\s+use)",
        " ",
        description,
        flags=re.IGNORECASE | re.DOTALL,
    )
    raw_metadata = candidate.get("raw_metadata") or {}
    raw_title = raw_metadata.get("title") or (
        (raw_metadata.get("metadata") or {}).get("title")
        if isinstance(raw_metadata.get("metadata"), dict)
        else ""
    )
    return " ".join([
        str(candidate.get("id") or ""),
        str(candidate.get("title") or ""),
        str(raw_title or ""),
        description[:700],
        str(candidate.get("snippet") or "")[:700],
        " ".join(str(tag) for tag in candidate.get("tags", [])),
        str(candidate.get("features_text") or ""),
    ]).lower()


def _subject_identity_text(candidate: dict[str, Any]) -> str:
    raw_metadata = candidate.get("raw_metadata") or {}
    raw_title = raw_metadata.get("title") or (
        (raw_metadata.get("metadata") or {}).get("title")
        if isinstance(raw_metadata.get("metadata"), dict)
        else ""
    )
    return " ".join([
        str(candidate.get("id") or ""),
        str(candidate.get("title") or ""),
        str(raw_title or ""),
        " ".join(str(tag) for tag in candidate.get("tags", [])),
    ]).lower()


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def _task_type_keywords(intent: dict[str, Any]) -> set[str]:
    return _tokens(str(intent.get("task_type") or "")) & TASK_TYPE_KEYWORDS


def _subject_keyword_groups(intent: dict[str, Any]) -> list[set[str]]:
    """Independent subject terms that must EACH have evidence.

    Each group is a set of interchangeable synonyms (OR within the group,
    e.g. person/pedestrian/face). Distinct groups are independent qualifiers
    of a compound subject (AND across groups, e.g. "bank" AND "churn") — a
    dataset about healthcare churn must not count as evidence for "bank
    churn" just because it shares the word "churn".
    """
    subject = str(
        intent.get("subject") or intent.get("preferred_domain") or "general"
    ).strip().lower()
    if subject in GENERIC_SUBJECTS:
        return set()
    base_tokens = _tokens(subject) - TASK_TYPE_KEYWORDS
    groups: list[set[str]] = []
    for token in base_tokens:
        aliases = SUBJECT_ALIASES.get(token)
        if aliases:
            groups.append(aliases)
        elif len(token) > 2:
            groups.append({token})
    return groups


def _subject_keywords(intent: dict[str, Any]) -> set[str]:
    """Return content-bearing subject terms, excluding generic task words."""
    keywords: set[str] = set()
    for group in _subject_keyword_groups(intent):
        keywords.update(group)
    return keywords


def evaluate_constraints(intent: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    evidence = _text(candidate)
    evidence_tokens = _tokens(evidence)
    subject_evidence_tokens = _tokens(_subject_text(candidate))
    notes: list[str] = []
    mismatch = False
    matched = 0
    checked = 0
    task_keywords = _task_type_keywords(intent)
    subject_groups = _subject_keyword_groups(intent)
    subject_keywords = _subject_keywords(intent)
    task_matches = task_keywords & evidence_tokens
    group_hits = [group & subject_evidence_tokens for group in subject_groups]
    subject_matches = (
        set().union(*group_hits) if group_hits and all(group_hits) else set()
    )
    if "human" in subject_keywords:
        subject_text = _subject_text(candidate)
        identity_matches = subject_keywords & _tokens(_subject_identity_text(candidate))
        proximity = re.search(
            r"\b(?:human|person|persons|people|pedestrian|pedestrians|face|faces|"
            r"crowd|crowds)\b.{0,45}\b(?:detect\w*|track\w*|locali[sz]\w*|"
            r"recogn\w*|activity|presence|fall\w*)\b|"
            r"\b(?:detect\w*|track\w*|locali[sz]\w*|recogn\w*)\b.{0,45}"
            r"\b(?:human|person|persons|people|pedestrian|pedestrians|face|faces|"
            r"crowd|crowds)\b",
            subject_text,
        )
        if not identity_matches and not proximity:
            subject_matches = set()
        nonhuman_sense = re.search(
            r"\bhuman(?:[- ](?:generated|authored|written|vein|skin|tissue|"
            r"organ|genome|protein|cell|cells)|\s+vs\.?\s+ai[- ]generated)\b",
            subject_text,
        )
        specific_human_object = subject_matches - {"human"}
        if nonhuman_sense and not specific_human_object:
            subject_matches = set()
    subject_required = bool(subject_keywords)

    if task_keywords:
        checked += 1
        if task_matches:
            matched += 1
            notes.append(
                f"Khớp loại tác vụ: {', '.join(sorted(task_matches))}."
            )
        else:
            notes.append(
                "Chưa thấy bằng chứng rõ về loại tác vụ "
                f"{intent.get('task_type')}."
            )

    required_language = intent.get("required_language") or intent.get("language") or "any"
    if required_language != "any":
        checked += 1
        aliases = LANGUAGE_ALIASES.get(required_language, {required_language})
        language_tags = {
            str(tag).lower().split(":", 1)[-1]
            for tag in candidate.get("tags", [])
            if str(tag).lower().startswith("language:")
        }
        if language_tags and language_tags & aliases:
            matched += 1
            notes.append(f"Khớp ngôn ngữ {required_language}.")
        elif language_tags:
            mismatch = True
            notes.append(f"Metadata ngôn ngữ không khớp {required_language}.")
        elif any(
            re.search(rf"\b{re.escape(alias)}\b", evidence)
            for alias in aliases
            if len(alias) > 2
        ):
            matched += 1
            notes.append(f"Khớp ngôn ngữ {required_language}.")
        else:
            notes.append(f"Chưa đủ metadata để xác nhận ngôn ngữ {required_language}.")

    subject = intent.get("subject") or intent.get("preferred_domain") or "general"
    if subject_required:
        checked += 1
        if subject_matches:
            matched += 1
            notes.append(
                "Có bằng chứng về subject "
                f"{subject}: {', '.join(sorted(subject_matches))}."
            )
        else:
            notes.append(f"Chưa thấy bằng chứng rõ cho subject {subject}.")

    required_labels = [str(label).lower() for label in intent.get("required_labels", [])]
    if required_labels:
        checked += 1
        label_matches = [label for label in required_labels if label in evidence]
        if label_matches:
            matched += 1
            notes.append(f"Metadata nhắc tới label: {', '.join(label_matches)}.")
        else:
            notes.append("Chưa xác nhận được label schema từ metadata.")

    minimum_samples = intent.get("minimum_samples")
    if isinstance(minimum_samples, int) and minimum_samples > 0:
        checked += 1
        sample_count = candidate.get("sample_count")
        if isinstance(sample_count, int):
            if sample_count >= minimum_samples:
                matched += 1
                notes.append(f"Đủ quy mô tối thiểu: {sample_count} mẫu.")
            else:
                mismatch = True
                notes.append(f"Chỉ có {sample_count} mẫu, dưới mức {minimum_samples}.")
        else:
            notes.append("Chưa có sample count để xác nhận quy mô.")

    if mismatch:
        status = "mismatch"
    elif subject_required and not subject_matches and task_matches:
        status = "partial"
    elif checked and matched == checked:
        status = "matched"
    else:
        status = "unknown"
    score = 5.0 if not checked else round(1 + 4 * matched / checked, 2)
    if mismatch:
        score = min(score, 1.5)
    return {
        **candidate,
        "constraint_status": status,
        "constraint_score": score,
        "constraint_notes": notes,
        "constraint_task_keywords": sorted(task_keywords),
        "constraint_subject_keywords": sorted(subject_keywords),
        "constraint_task_matched": bool(task_matches),
        "constraint_subject_matched": bool(subject_matches),
    }


def prepare_candidates(
    intent: dict[str, Any],
    candidates: list[dict[str, Any]],
    max_candidates: int = 60,
) -> list[dict[str, Any]]:
    """Evaluate constraints and build a source-balanced candidate pool."""
    evaluated = [evaluate_constraints(intent, candidate) for candidate in candidates]
    priority = {"matched": 0, "unknown": 1, "partial": 2, "mismatch": 3}
    buckets: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for candidate in sorted(
        evaluated,
        key=lambda item: (
            priority[item["constraint_status"]],
            -item["constraint_score"],
            -(item.get("downloads") or 0),
        ),
    ):
        buckets[candidate["source"]].append(candidate)

    selected: list[dict[str, Any]] = []
    source_names = list(buckets)
    while source_names and len(selected) < max_candidates:
        remaining = []
        for source in source_names:
            if buckets[source] and len(selected) < max_candidates:
                selected.append(buckets[source].popleft())
            if buckets[source]:
                remaining.append(source)
        source_names = remaining
    return selected
