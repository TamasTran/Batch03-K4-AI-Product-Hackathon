from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import quote

import requests


def _sample_count(card: dict[str, Any]) -> int | None:
    dataset_info = card.get("dataset_info") or {}
    if isinstance(dataset_info, list):
        infos = dataset_info
    else:
        infos = [dataset_info]
    total = 0
    found = False
    for info in infos:
        splits = info.get("splits") or []
        if isinstance(splits, dict):
            splits = list(splits.values())
        for split in splits:
            count = split.get("num_examples") if isinstance(split, dict) else None
            if isinstance(count, int):
                total += count
                found = True
    return total if found else None


def _enrich_huggingface(candidate: dict[str, Any], timeout: int) -> dict[str, Any]:
    dataset_id = quote(candidate["id"], safe="/")
    response = requests.get(
        f"https://huggingface.co/api/datasets/{dataset_id}",
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    card = payload.get("cardData") or {}
    tags = list(dict.fromkeys([
        *candidate.get("tags", []),
        *payload.get("tags", []),
        *[f"language:{value}" for value in card.get("language", [])],
        *card.get("task_categories", []),
    ]))
    description = (
        card.get("pretty_name")
        or payload.get("description")
        or candidate.get("description")
        or ""
    )
    metadata_text = json.dumps(card, ensure_ascii=False, default=str)
    return {
        **candidate,
        "tags": tags,
        "description": description,
        "sample_count": _sample_count(card),
        "features_text": metadata_text[:6000],
        "metadata_enriched": True,
    }


def enrich_candidates(
    candidates: list[dict[str, Any]],
    max_candidates: int = 40,
    timeout: int = 12,
) -> list[dict[str, Any]]:
    """Fetch detail metadata where a stable detail API is available."""
    output = [dict(candidate) for candidate in candidates]
    jobs: dict[Any, int] = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        for index, candidate in enumerate(output[:max_candidates]):
            if candidate.get("source") == "Hugging Face":
                jobs[executor.submit(_enrich_huggingface, candidate, timeout)] = index
        for future in as_completed(jobs):
            index = jobs[future]
            try:
                output[index] = future.result()
            except Exception as exc:
                output[index]["metadata_enriched"] = False
                output[index]["enrichment_error"] = type(exc).__name__
    return output
