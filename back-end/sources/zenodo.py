from typing import Any

import requests


def search_zenodo_datasets(
    keyword: str, limit: int = 12, timeout: int = 12, **_: Any
) -> list[dict]:
    response = requests.get(
        "https://zenodo.org/api/records/",
        params={
            "q": f'({keyword}) AND resource_type.type:dataset',
            "size": min(limit, 25),
            "sort": "bestmatch",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    items = (payload.get("hits") or {}).get("hits") or []
    results = []
    for item in items[:limit]:
        metadata = item.get("metadata") or {}
        record_id = item.get("id")
        title = metadata.get("title")
        if record_id is None or not title:
            continue
        links = item.get("links") or {}
        license_value = metadata.get("license")
        if isinstance(license_value, dict):
            license_value = license_value.get("id") or license_value.get("title")
        description = metadata.get("description") or ""
        results.append({
            "id": str(record_id),
            "url": links.get("html") or f"https://zenodo.org/records/{record_id}",
            "source": "Zenodo",
            "downloads": (item.get("stats") or {}).get("downloads"),
            "likes": None,
            "tags": metadata.get("keywords") or [],
            "license": license_value,
            "description": description,
            "doi": metadata.get("doi") or item.get("doi"),
            "confidence": "verified",
            "raw_metadata": item,
        })
    return results
