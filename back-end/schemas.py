from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ConfigResponse(BaseModel):
    llm_enabled: bool
    llm_provider: str | None = None
    llm_model: str | None = None
    kaggle_configured: bool
    web_search_configured: bool
    available_sources: list[str]


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    clarification_context: str | None = Field(default=None, max_length=4000)
    enabled_sources: list[str] = Field(
        default_factory=lambda: ["Hugging Face", "OpenML", "Zenodo"]
    )
    web_fallback_enabled: bool = True
    web_domains: list[str] | None = None
    limit: int = Field(default=15, ge=5, le=30)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Mô tả đề tài không được để trống.")
        return value


class ToolEventResponse(BaseModel):
    tool: str
    args: dict[str, Any]
    status: str
    error: str | None = None


class SearchResponse(BaseModel):
    status: str
    intent: dict[str, Any]
    verified: list[dict[str, Any]] = Field(default_factory=list)
    unverified: list[dict[str, Any]] = Field(default_factory=list)
    guidance: dict[str, Any] = Field(default_factory=dict)
    parse_mode: str
    rank_mode: str
    errors: list[str] = Field(default_factory=list)
    tool_events: list[ToolEventResponse] = Field(default_factory=list)
    clarification_question: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    clarification_context: str | None = None
