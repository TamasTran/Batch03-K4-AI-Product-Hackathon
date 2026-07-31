from typing import Any
from urllib.parse import quote
from functools import lru_cache
import re

import requests


@lru_cache(maxsize=1)
def _openml_catalog(timeout: int) -> list[dict]:
    response = requests.get(
        "https://www.openml.org/api/v1/json/data/list/status/active/limit/2000",
        timeout=timeout,
    )
    response.raise_for_status()
    return (response.json().get("data") or {}).get("dataset") or []


def search_openml_datasets(
    keyword: str, limit: int = 12, timeout: int = 12, **_: Any
) -> list[dict]:
    # OpenML's v1 list endpoint accepts a data_name path filter and returns
    # dataset metadata under data.dataset.
    encoded = quote(keyword.strip(), safe="")
    response = requests.get(
        f"https://www.openml.org/api/v1/json/data/list/data_name/{encoded}/limit/{limit}",
        timeout=timeout,
    )
    if response.status_code == 412:
        # OpenML's data_name filter is exact and uses 412 for "No results".
        # Fall back to a cached API catalog and perform transparent name matching.
        wanted = {
            token for token in re.findall(r"[a-z0-9]+", keyword.lower()) if len(token) > 2
        }
        catalog = _openml_catalog(timeout)
        items = [
            item for item in catalog
            if wanted & set(re.findall(r"[a-z0-9]+", str(item.get("name", "")).lower()))
        ][:limit]
    else:
        response.raise_for_status()
        payload = response.json()
        items = (payload.get("data") or {}).get("dataset") or []
    results = []
    for item in items[:limit]:
        dataset_id = item.get("did")
        name = item.get("name")
        if dataset_id is None or not name:
            continue
        results.append({
            "id": str(dataset_id),
            "title": name,
            "url": f"https://www.openml.org/d/{dataset_id}",
            "source": "OpenML",
            "downloads": item.get("NumberOfDownloads"),
            "likes": None,
            "tags": ["tabular", "openml"],
            "license": item.get("licence"),
            "description": name,
            "confidence": "verified",
            "raw_metadata": item,
        })
    return results
