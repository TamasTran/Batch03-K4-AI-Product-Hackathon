from typing import Any
import requests


def search_kaggle_datasets(
    keyword: str, limit: int = 12, timeout: int = 12,
    username: str = "", key: str = "", **_: Any
) -> list[dict]:
    if not username or not key:
        raise ValueError("Thiếu KAGGLE_USERNAME hoặc KAGGLE_KEY")
    response = requests.get(
        "https://www.kaggle.com/api/v1/datasets/list",
        params={"search": keyword, "page": 1, "pageSize": limit, "sortBy": "hottest"},
        auth=(username, key),
        timeout=timeout,
    )
    response.raise_for_status()
    results = []
    for item in response.json()[:limit]:
        ref = item.get("ref")
        if not ref:
            continue
        results.append({
            "id": ref,
            "url": item.get("url") or f"https://www.kaggle.com/datasets/{ref}",
            "source": "Kaggle",
            "downloads": item.get("downloadCount"),
            "likes": item.get("voteCount"),
            "tags": [tag.get("name", "") for tag in item.get("tags", []) if tag.get("name")],
            "license": item.get("licenseName"),
            "description": item.get("subtitle") or item.get("description") or "",
            "confidence": "verified",
            "raw_metadata": item,
        })
    return results
