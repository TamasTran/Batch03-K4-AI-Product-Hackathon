from __future__ import annotations

from typing import Any

from pipeline.parse_task import parse_task
from pipeline.rank_candidates import rank_candidates
from pipeline.deduplicate import deduplicate_candidates
from pipeline.constraints import prepare_candidates
from sources import SOURCE_REGISTRY
from sources.web_fallback import search_web_fallback
from sources.enrich import enrich_candidates


def analyze_task(text: str) -> tuple[dict[str, Any], str]:
    return parse_task(text)


def search_registry(
    source_name: str,
    keyword: str,
    limit: int = 12,
    username: str = "",
    key: str = "",
    **credentials: Any,
) -> list[dict[str, Any]]:
    if source_name not in SOURCE_REGISTRY:
        raise ValueError(f"Nguồn không được hỗ trợ: {source_name}")
    return SOURCE_REGISTRY[source_name](
        keyword, limit=limit, username=username, key=key, **credentials
    )


def verify_candidates(
    candidates: list[dict[str, Any]], limit: int | None = None
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for item in candidates:
        if item.get("id") and item.get("url") and item.get("source"):
            normalized = dict(item)
            normalized.setdefault("confidence", "verified")
            url_key = normalized["url"].rstrip("/").lower()
            current = unique.get(url_key)
            # A verified API result always wins over a duplicate web-search URL.
            if current is None or normalized["confidence"] == "verified":
                unique[url_key] = normalized
    verified = [x for x in unique.values() if x["confidence"] == "verified"]
    unverified = [x for x in unique.values() if x["confidence"] == "unverified"]
    if limit and limit > 0:
        verified = verified[:limit]
        unverified = unverified[:limit]
    return verified + unverified


def rank_datasets(
    intent: dict[str, Any], candidates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    return rank_candidates(intent, candidates, include_diagnostics=True)


TOOL_FUNCTIONS = {
    "analyze_task": analyze_task,
    "search_registry": search_registry,
    "search_web_fallback": search_web_fallback,
    "verify_candidates": verify_candidates,
    "prepare_candidates": prepare_candidates,
    "enrich_candidates": enrich_candidates,
    "deduplicate_candidates": deduplicate_candidates,
    "rank_datasets": rank_datasets,
}
