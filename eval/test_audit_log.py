import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import ToolEvent
from audit_log import persist_run


class AuditLogTests(unittest.TestCase):
    def test_persist_run_writes_full_event_result_and_redacts_secrets(self):
        event = ToolEvent(
            tool="rank_datasets",
            args={
                "candidate_count": 1,
                "candidate_inputs": [{
                    "id": "owner/dataset",
                    "title": "Dataset title",
                    "subject": "vehicles",
                }],
                "api_key": "***",
            },
            status="success",
            result={
                "diagnostics": {
                    "scored_before_threshold": {
                        "verified": [{
                            "id": "owner/dataset",
                            "total_score": 2.7,
                            "task_match": 2,
                            "domain_fit": 3,
                            "label_overlap": 3,
                            "size_adequacy": 3,
                            "reasoning": "Khớp một phần.",
                        }],
                        "unverified": [],
                    }
                }
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch("audit_log.LOG_DIR", Path(directory)):
                path = persist_run(
                    request_id="test-request",
                    request={"query": "vehicle dataset", "api_key": "secret"},
                    status="answered",
                    tool_events=[event],
                    final_result={"verified": [], "unverified": []},
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["request_id"], "test-request")
        self.assertEqual(payload["request"]["api_key"], "***")
        saved_event = payload["tool_events"][0]
        self.assertEqual(
            saved_event["args"]["candidate_inputs"][0]["title"],
            "Dataset title",
        )
        scored = saved_event["result"]["diagnostics"][
            "scored_before_threshold"
        ]["verified"][0]
        self.assertEqual(scored["id"], "owner/dataset")
        self.assertEqual(scored["reasoning"], "Khớp một phần.")

    def test_token_usage_metrics_are_not_mistaken_for_secrets(self):
        event = ToolEvent(
            tool="rank_datasets",
            args={},
            status="success",
            result={
                "completion_tokens": 1234,
                "token_cap": 4000,
                "access_token": "secret",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch("audit_log.LOG_DIR", Path(directory)):
                path = persist_run(
                    request_id="token-metrics",
                    request={"query": "test"},
                    status="answered",
                    tool_events=[event],
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["tool_events"][0]["result"]
        self.assertEqual(result["completion_tokens"], 1234)
        self.assertEqual(result["token_cap"], 4000)
        self.assertEqual(result["access_token"], "***")


if __name__ == "__main__":
    unittest.main()
