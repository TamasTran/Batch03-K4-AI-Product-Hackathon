from __future__ import annotations

from typing import Any

import requests


DEFAULT_WEB_DOMAINS = [
    "zenodo.org",
    "data.gov.vn",
    "academictorrents.com",
    "physionet.org",
    "shapenet.org",
    "data.world",
]


def _provider(credentials: dict[str, str]) -> str:
    if credentials.get("serpapi_api_key"):
        return "serpapi"
    if credentials.get("bing_search_api_key"):
        return "bing"
    if credentials.get("google_cse_api_key") and credentials.get("google_cse_id"):
        return "google"
    raise ValueError(
        "Chưa cấu hình web search key. Dùng SERPAPI_API_KEY, "
        "BING_SEARCH_API_KEY, hoặc GOOGLE_CSE_API_KEY + GOOGLE_CSE_ID."
    )


def _search(query: str, limit: int, timeout: int, credentials: dict[str, str]) -> list[dict]:
    provider = _provider(credentials)
    if provider == "serpapi":
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": query,
                "num": min(limit, 10),
                "api_key": credentials["serpapi_api_key"],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return [
            {"title": row.get("title"), "url": row.get("link"), "snippet": row.get("snippet") or ""}
            for row in response.json().get("organic_results", [])
        ]
    if provider == "bing":
        response = requests.get(
            "https://api.bing.microsoft.com/v7.0/search",
            params={"q": query, "count": min(limit, 50), "responseFilter": "Webpages"},
            headers={"Ocp-Apim-Subscription-Key": credentials["bing_search_api_key"]},
            timeout=timeout,
        )
        response.raise_for_status()
        return [
            {"title": row.get("name"), "url": row.get("url"), "snippet": row.get("snippet") or ""}
            for row in (response.json().get("webPages") or {}).get("value", [])
        ]
    response = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={
            "q": query,
            "num": min(limit, 10),
            "key": credentials["google_cse_api_key"],
            "cx": credentials["google_cse_id"],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return [
        {"title": row.get("title"), "url": row.get("link"), "snippet": row.get("snippet") or ""}
        for row in response.json().get("items", [])
    ]


def search_web_fallback(
    keywords: list[str],
    domains: list[str] | None = None,
    limit: int = 5,
    timeout: int = 12,
    **credentials: Any,
) -> list[dict]:
    """Search fixed domains and return snippet-only, explicitly unverified results."""
    domains = domains or DEFAULT_WEB_DOMAINS
    unique: dict[str, dict] = {}
    for keyword in keywords:
        for domain in domains:
            query = f"site:{domain} {keyword}"
            for row in _search(query, limit, timeout, credentials):
                url = row.get("url")
                title = row.get("title")
                if not url or not title:
                    continue
                unique[url] = {
                    "id": url,
                    "url": url,
                    "source": f"Web search · {domain}",
                    "title": title,
                    "snippet": row.get("snippet") or "",
                    "confidence": "unverified",
                }
                if len(unique) >= limit:
                    return list(unique.values())
    return list(unique.values())
