from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any

from .llm import call_json, get_client
from .prompts import step_prompt


AUTO_MERGE_THRESHOLD = 0.85
AMBIGUOUS_THRESHOLD = 0.50
GENERIC_TOKENS = {
    "data",
    "dataset",
    "datasets",
    "database",
    "benchmark",
    "corpus",
}
ORGANIZATION_PREFIXES = {"ms", "microsoft"}
DISTINCTIVE_MODIFIERS = {"tiny", "mini", "small", "subset", "reduced"}


def normalize_dataset_name(name: str) -> str:
    value = unicodedata.normalize("NFKD", str(name or ""))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.lower().replace("_", " ").replace("-", " ")
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"[/\\]", " ", value)
    value = re.sub(r"\bv(?:ersion)?\s*\d+(?:\.\d+)*\b", " ", value)
    value = re.sub(r"\b(?:19|20)\d{2}\b", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    tokens = [token for token in value.split() if token not in GENERIC_TOKENS]
    while len(tokens) > 1 and tokens[0] in ORGANIZATION_PREFIXES:
        tokens.pop(0)
    return " ".join(tokens)


def similarity_score(name_a: str, name_b: str) -> float:
    left = normalize_dataset_name(name_a)
    right = normalize_dataset_name(name_b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    left_tokens, right_tokens = set(left.split()), set(right.split())
    if (left_tokens & DISTINCTIVE_MODIFIERS) != (right_tokens & DISTINCTIVE_MODIFIERS):
        return 0.49
    left_numbers = {token for token in left_tokens if token.isdigit()}
    right_numbers = {token for token in right_tokens if token.isdigit()}
    if left_numbers != right_numbers and (left_numbers or right_numbers):
        return 0.49
    sequence = SequenceMatcher(None, left, right).ratio()
    token_jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return round(max(sequence, token_jaccard), 4)


def candidate_name(candidate: dict[str, Any]) -> str:
    if candidate.get("title"):
        return str(candidate["title"])
    if candidate.get("name"):
        return str(candidate["name"])
    dataset_id = str(candidate.get("id") or "")
    return dataset_id.rstrip("/").rsplit("/", 1)[-1]


def _candidate_key(index: int, candidate: dict[str, Any]) -> str:
    return f"candidate_{index}:{candidate.get('source', '')}:{candidate.get('id', '')}"


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def _llm_ambiguous_groups(
    candidates: list[dict[str, Any]],
    ambiguous_pairs: list[tuple[int, int, float]],
    intent: dict[str, Any],
) -> list[dict[str, Any]]:
    involved = sorted({index for left, right, _ in ambiguous_pairs for index in (left, right)})
    rows = [{
        "member_id": _candidate_key(index, candidates[index]),
        "name": candidate_name(candidates[index]),
        "source": candidates[index].get("source"),
        "url": candidates[index].get("url"),
        "description": (
            candidates[index].get("description")
            or candidates[index].get("snippet")
            or ""
        )[:500],
        "confidence": candidates[index].get("confidence", "verified"),
    } for index in involved]
    result = call_json(
        step_prompt("STEP_2_5"),
        json.dumps(
            {
                "intent": {
                    "domain": intent.get("domain"),
                    "task_type": intent.get("task_type"),
                },
                "ambiguous_candidates": rows,
            },
            ensure_ascii=False,
        ),
        1800,
    )
    if not isinstance(result, list):
        raise ValueError("Dedup LLM phải trả JSON array")
    return result


def _choose_representative(members: list[dict[str, Any]]) -> dict[str, Any]:
    def quality(item: dict[str, Any]) -> tuple[int, int]:
        verified = item.get("confidence", "verified") == "verified"
        metadata_fields = ("license", "downloads", "likes", "description", "tags")
        completeness = sum(item.get(field) not in (None, "", []) for field in metadata_fields)
        return int(verified), completeness

    return max(members, key=quality)


def _merge_group(members: list[dict[str, Any]]) -> dict[str, Any]:
    representative = _choose_representative(members)
    merged = dict(representative)
    verified_members = [
        item for item in members if item.get("confidence", "verified") == "verified"
    ]
    merged["confidence"] = "verified" if verified_members else "unverified"
    merged["sources"] = [
        {"source": item["source"], "url": item["url"]}
        for item in members
        if item.get("source") and item.get("url")
    ]
    merged["duplicate_count"] = len(members)
    if merged["confidence"] == "verified":
        verified_representative = _choose_representative(verified_members)
        for field in (
            "id", "url", "source", "downloads", "likes", "tags",
            "license", "description", "raw_metadata", "doi",
        ):
            if field in verified_representative:
                merged[field] = verified_representative[field]
    else:
        allowed = {
            "id", "url", "source", "title", "snippet", "confidence",
            "sources", "duplicate_count", "possible_duplicate_of",
        }
        merged = {key: value for key, value in merged.items() if key in allowed}
    return merged


def deduplicate_candidates(
    candidates: list[dict[str, Any]],
    intent: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Deduplicate candidates within one query using conservative two-tier matching."""
    if len(candidates) < 2:
        return [
            {
                **candidate,
                "sources": candidate.get("sources") or [{
                    "source": candidate["source"], "url": candidate["url"]
                }],
                "duplicate_count": 1,
            }
            for candidate in candidates
        ]

    intent = intent or {}
    union_find = _UnionFind(len(candidates))
    ambiguous_pairs: list[tuple[int, int, float]] = []
    for left, right in combinations(range(len(candidates)), 2):
        score = similarity_score(
            candidate_name(candidates[left]),
            candidate_name(candidates[right]),
        )
        if score >= AUTO_MERGE_THRESHOLD:
            union_find.union(left, right)
        elif score >= AMBIGUOUS_THRESHOLD:
            ambiguous_pairs.append((left, right, score))

    medium_pairs: list[tuple[int, int]] = []
    if ambiguous_pairs and get_client():
        try:
            decisions = _llm_ambiguous_groups(candidates, ambiguous_pairs, intent)
            key_to_index = {
                _candidate_key(index, candidate): index
                for index, candidate in enumerate(candidates)
            }
            allowed_edges = {
                frozenset((left, right)) for left, right, _ in ambiguous_pairs
            }
            for decision in decisions:
                member_ids = decision.get("member_ids") or []
                indexes = [key_to_index[key] for key in member_ids if key in key_to_index]
                if len(indexes) < 2:
                    continue
                pairs = list(combinations(indexes, 2))
                if not all(frozenset(pair) in allowed_edges for pair in pairs):
                    continue
                if decision.get("confidence") == "high":
                    for left, right in pairs:
                        union_find.union(left, right)
                elif decision.get("confidence") == "medium":
                    medium_pairs.extend(pairs)
        except Exception:
            # Conservative boundary: an LLM/provider failure never causes a merge.
            medium_pairs = []

    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        groups.setdefault(union_find.find(index), []).append(index)

    merged = [_merge_group([candidates[index] for index in indexes]) for indexes in groups.values()]
    original_to_merged: dict[int, int] = {}
    for merged_index, indexes in enumerate(groups.values()):
        for original_index in indexes:
            original_to_merged[original_index] = merged_index

    for left, right in medium_pairs:
        merged_left = original_to_merged[left]
        merged_right = original_to_merged[right]
        if merged_left == merged_right:
            continue
        left_name = candidate_name(merged[merged_left])
        right_name = candidate_name(merged[merged_right])
        merged[merged_left].setdefault("possible_duplicate_of", []).append(right_name)
        merged[merged_right].setdefault("possible_duplicate_of", []).append(left_name)

    return merged
