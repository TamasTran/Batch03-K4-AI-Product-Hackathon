from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable


LOG_DIR = Path(__file__).resolve().parent / "logs"
SENSITIVE_KEY_PARTS = ("api_key", "password", "secret", "authorization")
SENSITIVE_TOKEN_KEYS = {
    "token",
    "access_token",
    "refresh_token",
    "id_token",
}


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).lower()
    return (
        any(part in normalized for part in SENSITIVE_KEY_PARTS)
        or normalized in SENSITIVE_TOKEN_KEYS
        or normalized.endswith("_access_token")
    )


def _redact_and_normalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): (
                "***"
                if _is_sensitive_key(key) and item
                else _redact_and_normalize(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_redact_and_normalize(item) for item in value]
    if hasattr(value, "model_dump"):
        return _redact_and_normalize(value.model_dump())
    if hasattr(value, "__dict__"):
        return _redact_and_normalize(vars(value))
    return str(value)


def serialize_tool_events(events: Iterable[Any]) -> list[dict[str, Any]]:
    return [
        {
            "tool": event.tool,
            "args": _redact_and_normalize(event.args),
            "status": event.status,
            "error": event.error,
            "result": _redact_and_normalize(event.result),
        }
        for event in events
    ]


def persist_run(
    *,
    request_id: str,
    request: dict[str, Any],
    status: str,
    tool_events: Iterable[Any],
    final_result: dict[str, Any] | None = None,
    error: str | None = None,
) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    target = LOG_DIR / f"run_{request_id}.json"
    temporary = LOG_DIR / f".run_{request_id}.tmp"
    payload = {
        "request_id": request_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "request": _redact_and_normalize(request),
        "error": error,
        "tool_events": serialize_tool_events(tool_events),
        "final_result": _redact_and_normalize(final_result),
    }
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
