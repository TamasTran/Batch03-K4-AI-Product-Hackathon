from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests


LLM_TEMPERATURE = 0


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str = ""


def _value(name: str) -> str:
    return os.getenv(name, "").strip()


def infer_provider(model: str) -> str:
    value = model.lower()
    if _value("LLM_PROVIDER"):
        explicit = _value("LLM_PROVIDER").lower().replace("-", "_")
        aliases = {
            "google": "gemini",
            "google_gemini": "gemini",
            "custom": "openai_compatible",
            "openai_compat": "openai_compatible",
        }
        return aliases.get(explicit, explicit)
    if _value("LLM_BASE_URL") and _value("LLM_API_KEY"):
        return "openai_compatible"
    if _value("OPENROUTER_API_KEY") and "/" in value:
        return "openrouter"
    if value.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
        return "openai"
    if value.startswith("claude"):
        return "anthropic"
    if value.startswith("gemini"):
        return "gemini"
    if "/" in value:
        return "openrouter"
    return "openai"


def _default_model() -> str:
    if _value("LLM_MODEL"):
        return _value("LLM_MODEL")
    # Backward compatibility with the original project.
    if _value("ANTHROPIC_MODEL"):
        return _value("ANTHROPIC_MODEL")
    if _value("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    if _value("ANTHROPIC_API_KEY"):
        return "claude-sonnet-4-6"
    if _value("GEMINI_API_KEY") or _value("GOOGLE_API_KEY"):
        return "gemini-2.5-flash"
    if _value("OPENROUTER_API_KEY"):
        return "openai/gpt-4o-mini"
    if _value("LLM_API_KEY"):
        return "gpt-4o-mini"
    return ""


def get_llm_config() -> LLMConfig | None:
    model = _default_model()
    if not model:
        return None
    provider = infer_provider(model)
    provider_keys = {
        "openai": _value("OPENAI_API_KEY"),
        "anthropic": _value("ANTHROPIC_API_KEY"),
        "gemini": _value("GEMINI_API_KEY") or _value("GOOGLE_API_KEY"),
        "openrouter": _value("OPENROUTER_API_KEY"),
        "openai_compatible": _value("LLM_API_KEY"),
    }
    api_key = provider_keys.get(provider, "") or _value("LLM_API_KEY")
    if not api_key:
        return None
    base_url = _value("LLM_BASE_URL")
    return LLMConfig(provider=provider, model=model, api_key=api_key, base_url=base_url)


def get_client() -> LLMConfig | None:
    """Backward-compatible availability check used by the pipeline."""
    return get_llm_config()


def llm_label() -> str:
    config = get_llm_config()
    return f"{config.provider}:{config.model}" if config else "heuristic"


def extract_json(text: str) -> Any:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    return json.loads(cleaned)


def _raise_provider_error(response: requests.Response, provider: str) -> None:
    if not response.ok:
        raise RuntimeError(f"{provider} API trả HTTP {response.status_code}")


def _anthropic_text(config: LLMConfig, system: str, user: str, max_tokens: int) -> str:
    text, _ = _anthropic_response(config, system, user, max_tokens)
    return text


def _anthropic_response(
    config: LLMConfig, system: str, user: str, max_tokens: int
) -> tuple[str, dict[str, Any]]:
    from anthropic import Anthropic

    client = Anthropic(api_key=config.api_key)
    message = client.messages.create(
        model=config.model,
        max_tokens=max_tokens,
        temperature=LLM_TEMPERATURE,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(
        block.text
        for block in message.content
        if getattr(block, "type", "") == "text"
    )
    return text, {
        "completion_tokens": getattr(message.usage, "output_tokens", None),
        "finish_reason": getattr(message, "stop_reason", None),
    }


def _openai_compatible_text(
    config: LLMConfig, system: str, user: str, max_tokens: int
) -> str:
    text, _ = _openai_compatible_response(config, system, user, max_tokens)
    return text


def _openai_compatible_response(
    config: LLMConfig, system: str, user: str, max_tokens: int
) -> tuple[str, dict[str, Any]]:
    if config.provider == "openrouter":
        base_url = config.base_url or "https://openrouter.ai/api"
    elif config.provider == "openai":
        base_url = config.base_url or "https://api.openai.com"
    else:
        base_url = config.base_url
    if not base_url:
        raise RuntimeError("LLM_BASE_URL là bắt buộc với provider openai_compatible")
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    if config.provider == "openrouter":
        headers["HTTP-Referer"] = _value("OPENROUTER_SITE_URL") or "http://localhost"
        headers["X-Title"] = _value("OPENROUTER_APP_NAME") or "DataScout AI"
    response = requests.post(
        f"{endpoint}/chat/completions",
        headers=headers,
        json={
            "model": config.model,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=90,
    )
    _raise_provider_error(response, config.provider)
    payload = response.json()
    choice = payload["choices"][0]
    content = choice["message"]["content"]
    if isinstance(content, list):
        text = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    else:
        text = str(content)
    usage = payload.get("usage") or {}
    return text, {
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
    }


def _gemini_text(config: LLMConfig, system: str, user: str, max_tokens: int) -> str:
    text, _ = _gemini_response(config, system, user, max_tokens)
    return text


def _gemini_response(
    config: LLMConfig, system: str, user: str, max_tokens: int
) -> tuple[str, dict[str, Any]]:
    model = quote(config.model, safe="-_.")
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": config.api_key, "Content-Type": "application/json"},
        json={
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": LLM_TEMPERATURE,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        },
        timeout=90,
    )
    _raise_provider_error(response, "gemini")
    payload = response.json()
    candidate = payload["candidates"][0]
    parts = candidate["content"]["parts"]
    text = "".join(part.get("text", "") for part in parts)
    usage = payload.get("usageMetadata") or {}
    return text, {
        "completion_tokens": usage.get("candidatesTokenCount"),
        "finish_reason": candidate.get("finishReason"),
    }


def call_text_with_metadata(
    system: str, user: str, max_tokens: int = 1800
) -> tuple[str, dict[str, Any]]:
    config = get_llm_config()
    if config is None:
        raise RuntimeError(
            "Chưa cấu hình LLM. Hãy đặt LLM_MODEL và API key tương ứng trong .env."
        )
    if config.provider == "anthropic":
        return _anthropic_response(config, system, user, max_tokens)
    if config.provider == "gemini":
        return _gemini_response(config, system, user, max_tokens)
    if config.provider in {"openai", "openrouter", "openai_compatible"}:
        return _openai_compatible_response(config, system, user, max_tokens)
    raise ValueError(f"LLM_PROVIDER không được hỗ trợ: {config.provider}")


def call_json(system: str, user: str, max_tokens: int = 1800) -> Any:
    config = get_llm_config()
    if config is None:
        raise RuntimeError(
            "Chưa cấu hình LLM. Hãy đặt LLM_MODEL và API key tương ứng trong .env."
        )
    if config.provider == "anthropic":
        text = _anthropic_text(config, system, user, max_tokens)
    elif config.provider == "gemini":
        text = _gemini_text(config, system, user, max_tokens)
    elif config.provider in {"openai", "openrouter", "openai_compatible"}:
        text = _openai_compatible_text(config, system, user, max_tokens)
    else:
        raise ValueError(f"LLM_PROVIDER không được hỗ trợ: {config.provider}")
    return extract_json(text)
