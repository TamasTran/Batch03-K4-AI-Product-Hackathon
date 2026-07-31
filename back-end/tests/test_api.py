import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app
from agent import AgentRun, ToolEvent


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.persist_patcher = patch("api.persist_run")
        self.persist_run = self.persist_patcher.start()
        self.addCleanup(self.persist_patcher.stop)
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    @patch("api.SearchAgent.run")
    def test_search_returns_request_id_and_persists_full_tool_results(self, run):
        run.return_value = AgentRun(
            status="answered",
            intent={"task_type": "object detection"},
            ranked=[],
            guidance={},
            parse_mode="test",
            rank_mode="test",
            tool_events=[
                ToolEvent(
                    tool="rank_datasets",
                    args={"candidate_count": 1},
                    status="success",
                    result={
                        "diagnostics": {
                            "scored_before_threshold": {
                                "verified": [{
                                    "id": "owner/dataset",
                                    "total_score": 2.5,
                                    "reasoning": "Điểm thấp.",
                                }],
                                "unverified": [],
                            }
                        }
                    },
                )
            ],
        )
        response = self.client.post(
            "/api/v1/search",
            json={
                "query": "Tìm dataset phát hiện xe",
                "enabled_sources": ["Hugging Face"],
                "web_fallback_enabled": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["X-Request-ID"])
        persisted = self.persist_run.call_args.kwargs
        self.assertEqual(
            persisted["request_id"],
            response.headers["X-Request-ID"],
        )
        self.assertEqual(
            persisted["tool_events"][0].result["diagnostics"]
            ["scored_before_threshold"]["verified"][0]["id"],
            "owner/dataset",
        )
        self.assertEqual(
            response.json()["tool_events"][0]["result"]["diagnostics"]
            ["scored_before_threshold"]["verified"][0]["id"],
            "owner/dataset",
        )

    def test_rejects_no_source(self):
        response = self.client.post(
            "/api/v1/search",
            json={
                "query": "Vietnamese sentiment dataset",
                "enabled_sources": [],
                "web_fallback_enabled": False,
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("ít nhất một nguồn", response.json()["detail"])

    @patch("api.SearchAgent.run")
    def test_clarification_response_contract(self, run):
        run.return_value = AgentRun(
            status="clarification_required",
            intent={
                "task_type": "machine learning",
                "needs_clarification": True,
            },
            ranked=[],
            guidance={"alternatives": [], "registries": []},
            parse_mode="heuristic",
            rank_mode="not_run",
            clarification_question="Bạn muốn dùng dữ liệu cho tác vụ nào?",
            missing_fields=["task_type"],
        )
        response = self.client.post(
            "/api/v1/search",
            json={
                "query": "Tìm cho tôi một dataset",
                "enabled_sources": ["Hugging Face"],
                "web_fallback_enabled": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "clarification_required")
        self.assertEqual(body["verified"], [])
        self.assertEqual(body["rank_mode"], "not_run")
        self.assertEqual(body["missing_fields"], ["task_type"])
        self.assertIn(
            "User ban đầu: Tìm cho tôi một dataset",
            body["clarification_context"],
        )
        self.assertIn(
            "Assistant hỏi làm rõ: Bạn muốn dùng dữ liệu cho tác vụ nào?",
            body["clarification_context"],
        )

    @patch("api.SearchAgent.run")
    def test_follow_up_is_combined_with_clarification_context(self, run):
        run.return_value = AgentRun(
            status="clarification_required",
            intent={"needs_clarification": True},
            ranked=[],
            guidance={},
            parse_mode="heuristic",
            rank_mode="not_run",
            clarification_question="Bạn cần ngôn ngữ nào?",
        )
        self.client.post(
            "/api/v1/search",
            json={
                "query": "Phân loại ảnh",
                "clarification_context": "Tìm cho tôi một dataset",
                "enabled_sources": ["Hugging Face"],
                "web_fallback_enabled": False,
            },
        )
        effective_query = run.call_args.args[0]
        self.assertIn("Tìm cho tôi một dataset", effective_query)
        self.assertIn("Phân loại ảnh", effective_query)
        self.assertIn("Lịch sử hội thoại:", effective_query)

    @patch("api.SearchAgent.run")
    def test_follow_up_carries_original_question_assistant_question_and_answer(self, run):
        run.return_value = AgentRun(
            status="answered",
            intent={"needs_clarification": False},
            ranked=[],
            guidance={},
            parse_mode="test",
            rank_mode="test",
        )
        context = (
            "User ban đầu: Tôi cần dữ liệu để train model\n"
            "Assistant hỏi làm rõ: Bạn muốn mô hình thực hiện tác vụ gì?"
        )
        self.client.post(
            "/api/v1/search",
            json={
                "query": "nhận diện ảnh",
                "clarification_context": context,
                "enabled_sources": ["Hugging Face"],
                "web_fallback_enabled": False,
            },
        )
        effective_query = run.call_args.args[0]
        self.assertIn("Tôi cần dữ liệu để train model", effective_query)
        self.assertIn("Bạn muốn mô hình thực hiện tác vụ gì?", effective_query)
        self.assertIn("User mới nhất: nhận diện ảnh", effective_query)

    @patch("api.SearchAgent.run")
    def test_search_contract_separates_confidence_lanes(self, run):
        run.return_value = AgentRun(
            status="answered",
            intent={"task_type": "sentiment classification"},
            ranked=[
                {
                    "id": "verified/result",
                    "url": "https://example.test/verified",
                    "confidence": "verified",
                },
                {
                    "id": "web/result",
                    "url": "https://example.test/web",
                    "confidence": "unverified",
                },
            ],
            guidance={"alternatives": [], "registries": []},
            parse_mode="heuristic",
            rank_mode="heuristic",
            tool_events=[ToolEvent(tool="analyze_task", args={}, status="success")],
        )
        response = self.client.post(
            "/api/v1/search",
            json={
                "query": "Vietnamese sentiment dataset",
                "enabled_sources": ["Hugging Face"],
                "web_fallback_enabled": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["verified"]), 1)
        self.assertEqual(len(body["unverified"]), 1)
        self.assertEqual(body["tool_events"][0]["tool"], "analyze_task")


    @patch("api.SearchAgent")
    def test_partial_sources_are_respected_and_explicit_web_false_is_preserved(
        self, agent_class
    ):
        agent_class.return_value.run.return_value = AgentRun(
            status="answered",
            intent={},
            ranked=[],
            guidance={},
            parse_mode="test",
            rank_mode="test",
        )
        response = self.client.post(
            "/api/v1/search",
            json={
                "query": "Vietnamese sentiment dataset",
                "client_version": "1.1.0",
                "client_build_hash": "abc123",
                "client_built_at": "2026-07-31T00:00:00Z",
                "enabled_sources": ["Hugging Face", "OpenML"],
                "web_fallback_enabled": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            agent_class.call_args.kwargs["enabled_sources"],
            ["Hugging Face", "OpenML"],
        )
        self.assertFalse(agent_class.call_args.kwargs["web_fallback_enabled"])

    @patch("api.SearchAgent")
    def test_web_only_search_does_not_silently_enable_registries(
        self, agent_class
    ):
        agent_class.return_value.run.return_value = AgentRun(
            status="answered",
            intent={},
            ranked=[],
            guidance={},
            parse_mode="test",
            rank_mode="test",
        )
        response = self.client.post(
            "/api/v1/search",
            json={
                "query": "Vietnamese sentiment dataset",
                "enabled_sources": [],
                "web_fallback_enabled": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(agent_class.call_args.kwargs["enabled_sources"], [])

    def test_cors_accepts_local_fallback_port(self):
        response = self.client.options(
            "/api/v1/search",
            headers={
                "Origin": "http://127.0.0.1:3001",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "http://127.0.0.1:3001",
        )


if __name__ == "__main__":
    unittest.main()
