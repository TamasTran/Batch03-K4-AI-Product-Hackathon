from typing import Any
import requests


def search_hf_datasets(keyword: str, limit: int = 12, timeout: int = 12, **_: Any) -> list[dict]:
    response = requests.get(
        "https://huggingface.co/api/datasets",
        params={"search": keyword, "limit": limit, "sort": "downloads", "direction": -1},
        timeout=timeout,
    )
    response.raise_for_status()
    results = []
    for item in response.json():
        dataset_id = item.get("id")
        if not dataset_id:
            continue
        card = item.get("cardData") or {}
        # The Hub list API commonly omits a separate display-name field.  The
        # repository basename is still the dataset's real name (and avoids
        # passing the full owner/id identifier through as a fake title).
        title = (
            card.get("pretty_name")
            or item.get("name")
            or dataset_id.rstrip("/").rsplit("/", 1)[-1]
        )
        results.append({
            "id": dataset_id,
            "title": title,
            "url": f"https://huggingface.co/datasets/{dataset_id}",
            "source": "Hugging Face",
            "downloads": item.get("downloads"),
            "likes": item.get("likes"),
            "tags": item.get("tags") or [],
            "license": card.get("license"),
            "description": item.get("description") or "",
            "confidence": "verified",
            "raw_metadata": item,
        })
    return results
