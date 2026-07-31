from __future__ import annotations

import os
from pathlib import Path
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from agent import SearchAgent, has_search_source
from audit_log import persist_run
from pipeline.llm import get_llm_config
from schemas import ConfigResponse, HealthResponse, SearchRequest, SearchResponse, ToolEventResponse


ROOT_DIR = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)
load_dotenv(ROOT_DIR / ".env")
load_dotenv(Path(__file__).with_name(".env"))

SOURCE_NAMES = {
    "Hugging Face",
    "Kaggle",
    "OpenML",
    "Zenodo",
}
SOURCE_ORDER = (
    "Hugging Face",
    "Kaggle",
    "OpenML",
    "Zenodo",
)
EXPECTED_CLIENT_VERSION = os.getenv("EXPECTED_CLIENT_VERSION", "1.1.0")
MAX_CLIENT_BUILD_AGE_DAYS = int(os.getenv("MAX_CLIENT_BUILD_AGE_DAYS", "30"))


def _log_client_version(payload: SearchRequest, request_id: str) -> None:
    if not payload.client_version or not payload.client_build_hash:
        logger.warning(
            "Request %s đến từ client không có version/build hash; "
            "có thể đang test nhầm code cũ",
            request_id,
        )
    elif payload.client_version != EXPECTED_CLIENT_VERSION:
        logger.warning(
            "Request %s đến từ client version cũ/không khớp "
            "(client=%s, expected=%s, build=%s); có thể đang test nhầm code cũ",
            request_id,
            payload.client_version,
            EXPECTED_CLIENT_VERSION,
            payload.client_build_hash,
        )
    if payload.client_built_at:
        try:
            built_at = datetime.fromisoformat(
                payload.client_built_at.replace("Z", "+00:00")
            )
            age = datetime.now(timezone.utc) - built_at.astimezone(timezone.utc)
            if age.days > MAX_CLIENT_BUILD_AGE_DAYS:
                logger.warning(
                    "Request %s đến từ client build %d ngày tuổi "
                    "(threshold=%d); có thể đang test nhầm code cũ",
                    request_id,
                    age.days,
                    MAX_CLIENT_BUILD_AGE_DAYS,
                )
        except ValueError:
            logger.warning(
                "Request %s có client_built_at không hợp lệ: %r",
                request_id,
                payload.client_built_at,
            )


def _effective_sources(requested: list[str], request_id: str) -> list[str]:
    requested_set = set(requested)
    effective = [source for source in SOURCE_ORDER if source in requested_set]
    logger.info(
        "Request %s dùng source selection=%s",
        request_id,
        effective,
    )
    return effective


def _analysis_input(query: str, clarification_context: str | None) -> str:
    if not clarification_context:
        return query
    return (
        "Lịch sử hội thoại:\n"
        f"{clarification_context}\n"
        f"User mới nhất: {query}"
    )


def _clarification_transcript(
    query: str,
    clarification_context: str | None,
    clarification_question: str | None,
) -> str:
    if clarification_context:
        transcript = f"{clarification_context}\nUser: {query}"
    else:
        transcript = f"User ban đầu: {query}"
    return (
        f"{transcript}\n"
        f"Assistant hỏi làm rõ: {clarification_question or 'Bạn có thể mô tả rõ hơn yêu cầu không?'}"
    )


def _cors_origins() -> list[str]:
    configured = os.getenv("CORS_ORIGINS", "")
    if configured.strip():
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


app = FastAPI(
    title="DataScout AI API",
    version="1.1.0",
    description="REST API cho agent tìm kiếm và xếp hạng dataset đa nguồn.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    # Static servers and Vite may move to 3001/5174 when their preferred port
    # is occupied. Keep local development functional without opening CORS to
    # arbitrary network hosts.
    allow_origin_regex=r"^https?://(?:localhost|127\.0\.0\.1)(?::\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    expose_headers=["X-Request-ID"],
)


@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="datascout-api", version="1.1.0")


@app.get("/api/v1/config", response_model=ConfigResponse, tags=["system"])
def config() -> ConfigResponse:
    llm = get_llm_config()
    return ConfigResponse(
        backend_version="1.1.0",
        expected_client_version=EXPECTED_CLIENT_VERSION,
        llm_enabled=llm is not None,
        llm_provider=llm.provider if llm else None,
        llm_model=llm.model if llm else None,
        kaggle_configured=bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")),
        web_search_configured=bool(
            os.getenv("SERPAPI_API_KEY")
            or os.getenv("BING_SEARCH_API_KEY")
            or (os.getenv("GOOGLE_CSE_API_KEY") and os.getenv("GOOGLE_CSE_ID"))
        ),
        available_sources=sorted(SOURCE_NAMES),
    )


@app.post("/api/v1/search", response_model=SearchResponse, tags=["search"])
def search(payload: SearchRequest, response: Response) -> SearchResponse:
    request_id = uuid4().hex
    response.headers["X-Request-ID"] = request_id
    _log_client_version(payload, request_id)
    unknown_sources = set(payload.enabled_sources) - SOURCE_NAMES
    if unknown_sources:
        raise HTTPException(
            status_code=422,
            detail=f"Nguồn không hợp lệ: {', '.join(sorted(unknown_sources))}",
        )
    if not has_search_source(payload.enabled_sources, payload.web_fallback_enabled):
        raise HTTPException(status_code=422, detail="Hãy bật ít nhất một nguồn dữ liệu.")

    effective_query = _analysis_input(payload.query, payload.clarification_context)
    effective_sources = _effective_sources(payload.enabled_sources, request_id)
    logger.info(
        "Request %s client_version=%s build_hash=%s "
        "web_fallback_enabled=%s requested_sources=%s effective_sources=%s",
        request_id,
        payload.client_version,
        payload.client_build_hash,
        payload.web_fallback_enabled,
        payload.enabled_sources,
        effective_sources,
    )
    audit_request = {
        **payload.model_dump(),
        "requested_sources": list(payload.enabled_sources),
        "effective_sources": effective_sources,
        "effective_web_fallback_enabled": payload.web_fallback_enabled,
    }

    agent = SearchAgent(
        enabled_sources=effective_sources,
        limit=payload.limit,
        credentials={
            "username": os.getenv("KAGGLE_USERNAME", ""),
            "key": os.getenv("KAGGLE_KEY", ""),
            "serpapi_api_key": os.getenv("SERPAPI_API_KEY", ""),
            "bing_search_api_key": os.getenv("BING_SEARCH_API_KEY", ""),
            "google_cse_api_key": os.getenv("GOOGLE_CSE_API_KEY", ""),
            "google_cse_id": os.getenv("GOOGLE_CSE_ID", ""),
        },
        web_fallback_enabled=payload.web_fallback_enabled,
        web_domains=payload.web_domains,
    )
    try:
        run = agent.run(effective_query)
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        try:
            path = persist_run(
                request_id=request_id,
                request=audit_request,
                status="error",
                tool_events=agent.events,
                error=error_message,
            )
            logger.error(
                "Persisted failed agent run %s to %s", request_id, path
            )
        except Exception:
            logger.exception("Could not persist failed agent run %s", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline không thể hoàn tất: {error_message}",
        ) from exc

    verified = [item for item in run.ranked if item.get("confidence", "verified") == "verified"]
    unverified = [item for item in run.ranked if item.get("confidence") == "unverified"]
    response_body = SearchResponse(
        status=run.status,
        intent=run.intent,
        verified=verified,
        unverified=unverified,
        guidance=run.guidance,
        parse_mode=run.parse_mode,
        rank_mode=run.rank_mode,
        errors=run.errors,
        clarification_question=run.clarification_question,
        missing_fields=run.missing_fields,
        clarification_context=(
            _clarification_transcript(
                payload.query,
                payload.clarification_context,
                run.clarification_question,
            )
            if run.status == "clarification_required"
            else None
        ),
        tool_events=[
            ToolEventResponse(
                tool=event.tool,
                args=event.args,
                status=event.status,
                error=event.error,
                result=event.result,
            )
            for event in run.tool_events
        ],
    )
    try:
        path = persist_run(
            request_id=request_id,
            request=audit_request,
            status=run.status,
            tool_events=run.tool_events,
            final_result={
                "intent": run.intent,
                "verified": verified,
                "unverified": unverified,
                "guidance": run.guidance,
                "errors": run.errors,
                "parse_mode": run.parse_mode,
                "rank_mode": run.rank_mode,
            },
        )
        logger.info("Persisted agent run %s to %s", request_id, path)
    except Exception:
        logger.exception("Could not persist agent run %s", request_id)
    return response_body
