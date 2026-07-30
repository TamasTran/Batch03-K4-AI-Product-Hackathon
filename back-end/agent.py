from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Any, Callable

from pipeline.fallback_suggestions import build_guidance
from tools import TOOL_FUNCTIONS


logger = logging.getLogger(__name__)


def _candidate_debug_ref(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": candidate.get("id"),
        "title": (
            candidate.get("title")
            or candidate.get("name")
            or candidate.get("id")
        ),
        "subject": candidate.get("subject"),
        "source": candidate.get("source"),
        "confidence": candidate.get("confidence", "verified"),
    }


def has_search_source(
    enabled_sources: dict[str, bool] | list[str],
    web_fallback_enabled: bool,
) -> bool:
    registry_enabled = (
        any(enabled_sources.values())
        if isinstance(enabled_sources, dict)
        else bool(enabled_sources)
    )
    return registry_enabled or web_fallback_enabled


@dataclass
class ToolEvent:
    tool: str
    args: dict[str, Any]
    status: str
    result: Any = None
    error: str | None = None


@dataclass
class AgentRun:
    status: str
    intent: dict[str, Any]
    ranked: list[dict[str, Any]]
    guidance: dict[str, Any]
    parse_mode: str
    rank_mode: str
    errors: list[str] = field(default_factory=list)
    tool_events: list[ToolEvent] = field(default_factory=list)
    clarification_question: str | None = None
    missing_fields: list[str] = field(default_factory=list)

    @property
    def verified(self) -> list[dict[str, Any]]:
        return [x for x in self.ranked if x.get("confidence", "verified") == "verified"]

    @property
    def unverified(self) -> list[dict[str, Any]]:
        return [x for x in self.ranked if x.get("confidence") == "unverified"]


class SearchAgent:
    """Deterministic multi-step dataset agent with an inspectable tool trace."""

    def __init__(
        self,
        *,
        enabled_sources: list[str],
        limit: int = 15,
        credentials: dict[str, str] | None = None,
        web_fallback_enabled: bool = True,
        web_domains: list[str] | None = None,
    ) -> None:
        self.enabled_sources = enabled_sources
        self.limit = limit
        self.credentials = credentials or {}
        self.web_fallback_enabled = web_fallback_enabled
        self.web_domains = web_domains
        self.events: list[ToolEvent] = []

    def _call(self, name: str, args: dict[str, Any], func: Callable[..., Any] | None = None) -> Any:
        implementation = func or TOOL_FUNCTIONS[name]
        safe_args = {
            key: ("***" if any(part in key.lower() for part in ("key", "token", "password")) and value else value)
            for key, value in args.items()
            if key != "candidates"
        }
        if "candidates" in args:
            safe_args["candidate_count"] = len(args["candidates"])
            safe_args["candidate_inputs"] = [
                _candidate_debug_ref(candidate)
                for candidate in args["candidates"]
            ]
        event = ToolEvent(tool=name, args=safe_args, status="running")
        self.events.append(event)
        try:
            event.result = implementation(**args)
            event.status = "success"
            return event.result
        except Exception as exc:
            event.status = "error"
            event.error = f"{type(exc).__name__}: {exc}"
            raise

    def run(self, query: str) -> AgentRun:
        self.events = []
        intent, parse_mode = self._call("analyze_task", {"text": query})
        if intent.get("needs_clarification"):
            return AgentRun(
                status="clarification_required",
                intent=intent,
                ranked=[],
                guidance={"alternatives": [], "registries": []},
                parse_mode=parse_mode,
                rank_mode="not_run",
                tool_events=self.events,
                clarification_question=intent.get("clarification_question"),
                missing_fields=list(intent.get("missing_fields") or []),
            )
        errors: list[str] = []
        per_query = max(5, min(12, self.limit))

        def verified_branch() -> tuple[list[dict[str, Any]], list[str]]:
            found: list[dict[str, Any]] = []
            branch_errors: list[str] = []
            keywords = list(dict.fromkeys(intent["search_keywords_en"]))[:4]
            jobs = [
                (source_name, keyword)
                for source_name in self.enabled_sources
                for keyword in keywords
            ]

            def search_one(job: tuple[str, str]) -> tuple[str, str, list[dict[str, Any]], Exception | None]:
                source_name, keyword = job
                args = {
                    "source_name": source_name,
                    "keyword": keyword,
                    "limit": per_query,
                    **self.credentials,
                }
                try:
                    return source_name, keyword, self._call("search_registry", args), None
                except Exception as exc:
                    return source_name, keyword, [], exc

            # Registry calls are independent. Running them concurrently keeps one
            # slow provider from multiplying latency by source × keyword count.
            with ThreadPoolExecutor(max_workers=min(8, max(1, len(jobs)))) as registry_pool:
                for source_name, keyword, rows, error in registry_pool.map(search_one, jobs):
                    if error is not None:
                        branch_errors.append(f"{source_name} · “{keyword}”: {error}")
                    else:
                        found.extend({**row, "confidence": "verified"} for row in rows)
            return found, branch_errors

        def web_branch() -> tuple[list[dict[str, Any]], list[str]]:
            if not self.web_fallback_enabled:
                return [], []
            args = {
                "keywords": intent["search_keywords_en"],
                "domains": self.web_domains,
                "limit": self.limit,
                **self.credentials,
            }
            try:
                rows = self._call("search_web_fallback", args)
                return [{**row, "confidence": "unverified"} for row in rows], []
            except Exception as exc:
                return [], [f"Web fallback: {exc}"]

        # Hybrid Step 2: API adapters and catch-all web search are independent.
        with ThreadPoolExecutor(max_workers=2) as executor:
            verified_future = executor.submit(verified_branch)
            web_future = executor.submit(web_branch)
            verified_found, verified_errors = verified_future.result()
            web_found, web_errors = web_future.result()
        errors.extend(verified_errors)
        errors.extend(web_errors)

        candidates = self._call(
            "verify_candidates",
            {"candidates": verified_found + web_found, "limit": None},
        )
        pool_limit = max(40, self.limit * 4)
        candidates = self._call(
            "prepare_candidates",
            {"intent": intent, "candidates": candidates, "max_candidates": pool_limit},
        )
        candidates = self._call(
            "enrich_candidates",
            {"candidates": candidates, "max_candidates": pool_limit},
        )
        candidates = self._call(
            "prepare_candidates",
            {"intent": intent, "candidates": candidates, "max_candidates": pool_limit},
        )
        candidates = self._call(
            "deduplicate_candidates",
            {"candidates": candidates, "intent": intent},
        )
        rank_result = self._call(
            "rank_datasets", {"intent": intent, "candidates": candidates}
        )
        if len(rank_result) == 3:
            ranked, rank_mode, rank_diagnostics = rank_result
        else:
            ranked, rank_mode = rank_result
            rank_diagnostics = {
                "input_candidate_count": len(candidates),
                "diagnostics_unavailable": True,
            }
        rank_event = next(
            event for event in reversed(self.events)
            if event.tool == "rank_datasets"
        )
        rank_event.result = {
            "verified": rank_diagnostics.get(
                "scored_before_threshold", {}
            ).get("verified", []),
            "unverified": rank_diagnostics.get(
                "scored_before_threshold", {}
            ).get("unverified", []),
            "ranked_after_threshold": ranked,
            "rank_mode": rank_mode,
            "diagnostics": rank_diagnostics,
        }

        mismatch_excluded = [
            {
                "id": item.get("id"),
                "title": item.get("title") or item.get("id"),
                "constraint_status": item.get("constraint_status"),
                "constraint_notes": item.get("constraint_notes", []),
                "reason": "constraint_status=mismatch",
            }
            for item in ranked
            if item.get("constraint_status", "unknown") == "mismatch"
        ]
        ranked_before_mismatch = len(ranked)
        ranked = [
            item for item in ranked
            if item.get("constraint_status", "unknown") != "mismatch"
        ]
        logger.info(
            "Constraint mismatch filter excluded %d/%d ranked candidates",
            len(mismatch_excluded),
            ranked_before_mismatch,
        )
        rank_diagnostics["constraint_mismatch_input_count"] = ranked_before_mismatch
        rank_diagnostics["constraint_mismatch_excluded_count"] = len(
            mismatch_excluded
        )
        rank_diagnostics["excluded_by_constraint_mismatch"] = mismatch_excluded
        rank_diagnostics["ranked_after_mismatch_count"] = len(ranked)

        guidance = self._call(
            "build_fallback_guidance",
            {"intent": intent, "ranked": ranked},
            build_guidance,
        )
        verified_before_limit = [
            item for item in ranked
            if item.get("confidence", "verified") == "verified"
        ]
        unverified_before_limit = [
            item for item in ranked if item.get("confidence") == "unverified"
        ]
        verified_ranked = (
            verified_before_limit[:self.limit] if verified_before_limit else []
        )
        unverified_ranked = (
            unverified_before_limit[:self.limit] if unverified_before_limit else []
        )
        excluded_by_limit = [
            {
                "id": item.get("id"),
                "title": item.get("title") or item.get("id"),
                "confidence": item.get("confidence", "verified"),
                "reason": f"lane limit={self.limit}",
            }
            for item in (
                verified_before_limit[self.limit:]
                + unverified_before_limit[self.limit:]
            )
        ]
        logger.info(
            "Lane limit excluded %d candidates "
            "(verified_before=%d, unverified_before=%d, limit=%d)",
            len(excluded_by_limit),
            len(verified_before_limit),
            len(unverified_before_limit),
            self.limit,
        )
        rank_diagnostics["limit"] = self.limit
        rank_diagnostics["limit_excluded_count"] = len(excluded_by_limit)
        rank_diagnostics["excluded_by_limit"] = excluded_by_limit
        rank_diagnostics["final_verified_count"] = len(verified_ranked)
        rank_diagnostics["final_unverified_count"] = len(unverified_ranked)
        return AgentRun(
            status="answered",
            intent=intent,
            ranked=verified_ranked + unverified_ranked,
            guidance=guidance,
            parse_mode=parse_mode,
            rank_mode=rank_mode,
            errors=errors,
            tool_events=self.events,
        )
