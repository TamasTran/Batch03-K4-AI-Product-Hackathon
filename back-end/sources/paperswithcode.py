from typing import Any
import requests


def search_pwc_datasets(keyword: str, limit: int = 12, timeout: int = 12, **_: Any) -> list[dict]:
    response = requests.get(
        "https://paperswithcode.com/api/v1/datasets/",
        params={"q": keyword, "page_size": limit},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    results = []
    for item in payload.get("results", []):
        dataset_id = item.get("id") or item.get("name")
        if not dataset_id:
            continue
        url = item.get("url")
        if url and url.startswith("/"):
            url = "https://paperswithcode.com" + url
        results.append({
            "id": str(dataset_id),
            "url": url or f"https://paperswithcode.com/dataset/{dataset_id}",
            "source": "Papers with Code",
            "downloads": None,
            "likes": None,
            "tags": [],
            "license": None,
            "description": item.get("description") or item.get("name") or "",
            "confidence": "verified",
            "raw_metadata": item,
        })
    return results
