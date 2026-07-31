from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import quote

import requests


def _as_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    return value if isinstance(value, list) else [value]


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
        *[f"language:{value}" for value in _as_list(card.get("language"))],
        *_as_list(card.get("task_categories")),
    ]))
    description = (
        payload.get("description")
        or candidate.get("description")
        or ""
    )
    title = (
        card.get("pretty_name")
        or payload.get("name")
        or candidate.get("title")
        or str(candidate["id"]).rstrip("/").rsplit("/", 1)[-1]
    )
    metadata_text = json.dumps(card, ensure_ascii=False, default=str)
    return {
        **candidate,
        "title": title,
        "tags": tags,
        "description": description,
        "sample_count": _sample_count(card),
        "features_text": metadata_text[:6000],
        "metadata_enriched": True,
    }


def _enrich_zenodo(candidate: dict[str, Any], timeout: int) -> dict[str, Any]:
    response = requests.get(
        f"https://zenodo.org/api/records/{quote(str(candidate['id']), safe='')}",
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    metadata = payload.get("metadata") or {}
    title = metadata.get("title") or candidate.get("title")
    description = metadata.get("description") or candidate.get("description") or ""
    if not title:
        raise ValueError("Zenodo metadata did not contain a title")
    return {
        **candidate,
        "title": title,
        "description": description,
        "tags": metadata.get("keywords") or candidate.get("tags") or [],
        "raw_metadata": payload,
        "metadata_enriched": True,
    }


def _title_from_existing_metadata(candidate: dict[str, Any]) -> str | None:
    title = candidate.get("title") or candidate.get("name")
    raw = candidate.get("raw_metadata") or {}
    metadata = raw.get("metadata") or {}
    card = raw.get("cardData") or {}
    return (
        title
        or raw.get("name")
        or raw.get("title")
        or metadata.get("title")
        or card.get("pretty_name")
    )


def enrich_candidates(
    candidates: list[dict[str, Any]],
    max_candidates: int = 40,
    timeout: int = 12,
) -> list[dict[str, Any]]:
    """Fetch detail metadata where a stable detail API is available."""
    output = [
        {**candidate, "title": _title_from_existing_metadata(candidate)}
        for candidate in candidates
    ]
    jobs: dict[Any, int] = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        for index, candidate in enumerate(output[:max_candidates]):
            if candidate.get("source") == "Hugging Face":
                jobs[executor.submit(_enrich_huggingface, candidate, timeout)] = index
            elif candidate.get("source") == "Zenodo" and not candidate.get("title"):
                jobs[executor.submit(_enrich_zenodo, candidate, timeout)] = index
        for future in as_completed(jobs):
            index = jobs[future]
            try:
                output[index] = future.result()
            except Exception as exc:
                output[index]["metadata_enriched"] = False
                output[index]["enrichment_error"] = f"{type(exc).__name__}: {exc}"
    return output
