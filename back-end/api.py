from __future__ import annotations

import os
from pathlib import Path
import logging
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
    "Papers with Code",
    "OpenML",
    "Zenodo",
}


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
    version="1.0.0",
    description="REST API cho agent tìm kiếm và xếp hạng dataset đa nguồn.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    expose_headers=["X-Request-ID"],
)


@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="datascout-api", version="1.0.0")


@app.get("/api/v1/config", response_model=ConfigResponse, tags=["system"])
def config() -> ConfigResponse:
    llm = get_llm_config()
    return ConfigResponse(
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
    unknown_sources = set(payload.enabled_sources) - SOURCE_NAMES
    if unknown_sources:
        raise HTTPException(
            status_code=422,
            detail=f"Nguồn không hợp lệ: {', '.join(sorted(unknown_sources))}",
        )
    if not has_search_source(payload.enabled_sources, payload.web_fallback_enabled):
        raise HTTPException(status_code=422, detail="Hãy bật ít nhất một nguồn dữ liệu.")

    effective_query = _analysis_input(payload.query, payload.clarification_context)

    agent = SearchAgent(
        enabled_sources=payload.enabled_sources,
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
                request=payload.model_dump(),
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
            )
            for event in run.tool_events
        ],
    )
    try:
        path = persist_run(
            request_id=request_id,
            request=payload.model_dump(),
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
