import json
import unittest
from unittest.mock import patch
from agent import SearchAgent, _redact_tool_error, has_search_source
from pipeline.parse_task import apply_confidence_policy, parse_task, _heuristic
from pipeline.rank_candidates import (
    RANK_BATCH_SIZE,
    RANK_MAX_OUTPUT_TOKENS,
    RankingResponseTruncatedError,
    _llm_rank,
    _llm_rank_batch,
    _apply_core_keyword_guard,
    _core_keywords,
    _llm_core_keywords_cached,
    _resolve_core_keywords,
    _reasoning_echoes_constraint_notes,
    rank_candidates,
)
from pipeline.fallback_suggestions import build_guidance
from sources.huggingface import search_hf_datasets
from sources.kaggle import search_kaggle_datasets
from sources.paperswithcode import search_pwc_datasets
from tools import verify_candidates
from sources.web_fallback import search_web_fallback
from pipeline.deduplicate import (
    deduplicate_candidates,
    normalize_dataset_name,
    similarity_score,
)
from pipeline.constraints import evaluate_constraints, prepare_candidates
from sources.enrich import enrich_candidates


SAMPLE = {
    "id": "owner/verified", "url": "https://example.test/verified", "source": "Hugging Face",
    "downloads": 500, "likes": 10, "tags": ["text-classification", "vietnamese"],
    "license": "apache-2.0", "description": "Vietnamese sentiment classification", "raw_metadata": {},
}


class PipelineTests(unittest.TestCase):
    def test_provider_error_redacts_plain_and_url_encoded_credentials(self):
        secret = "abc+/=secret"
        error = RuntimeError(
            "failed https://provider.test?q=x&api_key=abc%2B%2F%3Dsecret"
        )
        message = _redact_tool_error(
            error,
            {"serpapi_api_key": secret, "keyword": "safe"},
        )
        self.assertNotIn(secret, message)
        self.assertNotIn("abc%2B%2F%3Dsecret", message)
        self.assertIn("***", message)

    @patch("sources.kaggle.requests.get")
    def test_kaggle_candidates_include_rankable_title(self, get):
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.return_value = [{
            "ref": "owner/helmet-detection",
            "title": "Construction Helmet Detection",
            "subtitle": "Annotated construction images",
            "url": "https://www.kaggle.com/datasets/owner/helmet-detection",
            "tags": [],
        }]
        rows = search_kaggle_datasets(
            "construction helmet",
            username="owner",
            key="secret",
        )
        self.assertEqual(rows[0]["title"], "Construction Helmet Detection")

    def test_general_subject_is_not_treated_as_positive_subject_evidence(self):
        intent = {
            "task_type": "image classification",
            "subject": "general",
            "preferred_domain": "general",
            "required_language": "any",
            "required_labels": [],
            "minimum_samples": None,
        }
        row = evaluate_constraints(
            intent,
            {
                "id": "generic/general-image-classification",
                "title": "General Image Classification Dataset",
                "description": "A general benchmark for image classification.",
                "tags": ["image", "classification"],
                "source": "Synthetic",
            },
        )
        self.assertEqual(row["constraint_subject_keywords"], [])
        self.assertFalse(row["constraint_subject_matched"])

    @patch("sources.paperswithcode.requests.get")
    def test_retired_paperswithcode_api_has_clear_error(self, get):
        get.return_value.raise_for_status.return_value = None
        get.return_value.headers = {"content-type": "text/html; charset=utf-8"}
        get.return_value.url = "https://huggingface.co/papers/trending"
        with self.assertRaisesRegex(
            RuntimeError, "Papers with Code dataset API không còn khả dụng"
        ):
            search_pwc_datasets("helmet dataset")

    def test_heuristic_human_detection_keeps_subject_and_can_search(self):
        with patch("pipeline.parse_task.get_client", return_value=None):
            intent, mode = parse_task("human object detection dataset")
        self.assertIn("heuristic", mode.lower())
        self.assertEqual(intent["task_type"], "object detection")
        self.assertEqual(intent["modality"], "image")
        self.assertEqual(intent["subject"], "human")
        self.assertFalse(intent["needs_clarification"])
        self.assertTrue(any("human" in keyword for keyword in intent["search_keywords_en"]))

    def test_heuristic_construction_helmet_keeps_specific_subject_and_keywords(self):
        with patch("pipeline.parse_task.get_client", return_value=None):
            intent, _ = parse_task(
                "I need an image dataset for construction helmet object detection"
            )
        self.assertEqual(intent["subject"], "construction helmet")
        self.assertEqual(intent["domain"], "computer vision")
        self.assertTrue(
            any("construction helmet" in kw for kw in intent["search_keywords_en"]),
            f"Expected 'construction helmet' in at least one keyword: {intent['search_keywords_en']}",
        )
        self.assertTrue(
            any("object detection" in kw and "construction helmet" in kw
                for kw in intent["search_keywords_en"]),
            f"Expected keyword combining 'object detection' and 'construction helmet': {intent['search_keywords_en']}",
        )

    def test_heuristic_vietnamese_pedestrian_detection_can_search(self):
        with patch("pipeline.parse_task.get_client", return_value=None):
            intent, _ = parse_task("tìm dữ liệu object detection dành cho người đi bộ")
        self.assertEqual(intent["subject"], "pedestrian")
        self.assertFalse(intent["needs_clarification"])

    def test_heuristic_text_image_classification_keeps_text_subject(self):
        with patch("pipeline.parse_task.get_client", return_value=None):
            intent, _ = parse_task("image classification dataset for text images")
        self.assertEqual(intent["task_type"], "image classification")
        self.assertEqual(intent["subject"], "text")
        self.assertFalse(intent["needs_clarification"])

    def test_generic_prompt_requires_clarification(self):
        intent = _heuristic("Tìm cho tôi một dataset tốt")
        self.assertTrue(intent["needs_clarification"])
        self.assertIn("task_type", intent["missing_fields"])
        self.assertLess(intent["intent_confidence"], 0.5)
        self.assertTrue(intent["clarification_question"])

    def test_specific_prompt_can_continue_to_search(self):
        intent = _heuristic(
            "Tìm dataset tiếng Việt có nhãn để phân tích cảm xúc bình luận sản phẩm"
        )
        self.assertFalse(intent["needs_clarification"])
        self.assertEqual(intent["missing_fields"], [])
        self.assertGreaterEqual(intent["intent_confidence"], 0.9)

    def test_image_classification_is_not_misread_as_sentiment(self):
        intent = _heuristic("Tìm dataset để phân loại ảnh")
        self.assertEqual(intent["task_type"], "image classification")
        self.assertEqual(intent["modality"], "image")
        self.assertFalse(intent["needs_clarification"])

    def test_image_recognition_follow_up_can_continue_to_search(self):
        intent = _heuristic(
            "Yêu cầu ban đầu: Tìm cho tôi một dataset tốt\n"
            "Thông tin bổ sung của người dùng: nhận diện ảnh"
        )
        self.assertEqual(intent["task_type"], "image recognition")
        self.assertEqual(intent["domain"], "computer vision")
        self.assertEqual(intent["modality"], "image")
        self.assertFalse(intent["needs_clarification"])

    def test_vehicle_detection_prompt_can_continue_to_search(self):
        intent = _heuristic("tìm dataset cho bài toán phát hiện ô tô")
        self.assertEqual(intent["task_type"], "object detection")
        self.assertEqual(intent["domain"], "computer vision")
        self.assertEqual(intent["modality"], "image")
        self.assertFalse(intent["needs_clarification"])

    def test_broad_classification_prompt_asks_for_modality(self):
        intent = _heuristic("Tôi muốn phân loại dữ liệu")
        self.assertEqual(intent["task_type"], "classification")
        self.assertTrue(intent["needs_clarification"])
        self.assertIn("modality", intent["missing_fields"])

    def test_domain_only_prompt_does_not_allow_inferred_task(self):
        intent = _heuristic("Tìm cho tôi một dataset y tế")
        self.assertEqual(intent["domain"], "biomedical")
        self.assertTrue(intent["needs_clarification"])
        self.assertIn("task_type", intent["missing_fields"])

    def test_confidence_gate_accepts_semantic_llm_result_without_keyword_validation(self):
        inferred = {
            "task_type": "image classification",
            "domain": "computer vision",
            "modality": "image",
            "subject": "animals",
            "required_language": "any",
            "required_labels": [],
            "needs_labels": True,
            "field_confidence": {
                "task_type": 0.94,
                "modality": 0.92,
                "domain": 0.88,
                "required_language": 0.3,
                "required_labels": 0.3,
            },
        }
        intent = apply_confidence_policy(inferred)
        self.assertFalse(intent["needs_clarification"])
        self.assertNotIn("task_type", intent["missing_fields"])
        self.assertNotIn("modality", intent["missing_fields"])

    def test_confidence_gate_requires_clarification_for_low_core_confidence(self):
        intent = apply_confidence_policy({
            "task_type": "machine learning",
            "domain": "general",
            "modality": "any",
            "required_language": "any",
            "required_labels": [],
            "needs_labels": False,
            "field_confidence": {
                "task_type": 0.2,
                "modality": 0.2,
                "domain": 0.2,
                "required_language": 0.3,
                "required_labels": 0.3,
            },
        })
        self.assertTrue(intent["needs_clarification"])
        self.assertEqual(
            intent["missing_fields"],
            ["task_type", "modality", "domain"],
        )

    @patch("pipeline.parse_task.get_client", return_value=object())
    @patch("pipeline.parse_task.call_json")
    def test_semantic_vehicle_phrasings_converge_without_whitelist(self, call_json, _):
        call_json.return_value = {
            "task_type": "object detection",
            "domain": "computer vision",
            "modality": "image",
            "language": "any",
            "subject": "vehicles",
            "required_language": "any",
            "required_labels": [],
            "preferred_domain": "computer vision",
            "minimum_samples": None,
            "hard_constraints": [],
            "search_keywords_en": ["vehicle object detection dataset"],
            "needs_labels": True,
            "is_narrow_domain": False,
            "is_sensitive_domain": False,
            "sensitive_reason": "",
            "field_confidence": {
                "task_type": 0.95,
                "modality": 0.94,
                "domain": 0.9,
                "required_language": 0.3,
                "required_labels": 0.3,
            },
            "clarification_question": "",
        }
        variants = [
            "Tìm dataset cho bài toán phát hiện ô tô.",
            "Tìm xe trong ảnh",
            "Định vị phương tiện",
            "Khoanh vùng xe trên đường",
            "vehicle localization dataset",
        ]
        results = [parse_task(text)[0] for text in variants]
        self.assertTrue(all(row["task_type"] == "object detection" for row in results))
        self.assertTrue(all(row["modality"] == "image" for row in results))
        self.assertTrue(all(row["domain"] == "computer vision" for row in results))
        self.assertTrue(all(not row["needs_clarification"] for row in results))

    @patch("pipeline.parse_task.get_client", return_value=object())
    @patch("pipeline.parse_task.call_json")
    def test_multi_turn_llm_uses_full_history_and_does_not_repeat_question(
        self, call_json, _
    ):
        call_json.return_value = {
            "task_type": "image recognition",
            "domain": "computer vision",
            "modality": "image",
            "language": "any",
            "subject": "general",
            "required_language": "any",
            "required_labels": [],
            "preferred_domain": "computer vision",
            "minimum_samples": None,
            "hard_constraints": [],
            "search_keywords_en": ["image recognition dataset"],
            "needs_labels": True,
            "is_narrow_domain": False,
            "is_sensitive_domain": False,
            "sensitive_reason": "",
            "field_confidence": {
                "task_type": 0.93,
                "modality": 0.96,
                "domain": 0.9,
                "required_language": 0.3,
                "required_labels": 0.3,
            },
            "clarification_question": "",
        }
        transcript = (
            "Lịch sử hội thoại:\n"
            "User ban đầu: Tôi cần dữ liệu để train model\n"
            "Assistant hỏi làm rõ: Bạn muốn mô hình thực hiện tác vụ gì và xử lý loại dữ liệu nào?\n"
            "User mới nhất: nhận diện ảnh"
        )
        intent, parse_mode = parse_task(transcript)
        self.assertTrue(parse_mode.startswith("LLM"))
        self.assertEqual(intent["task_type"], "image recognition")
        self.assertEqual(intent["modality"], "image")
        self.assertFalse(intent["needs_clarification"])
        self.assertIn("Assistant hỏi làm rõ", call_json.call_args.args[1])
        self.assertIn("User mới nhất: nhận diện ảnh", call_json.call_args.args[1])

    def test_search_agent_stops_before_registry_when_intent_is_ambiguous(self):
        ambiguous = {
            "task_type": "machine learning",
            "domain": "general",
            "modality": "any",
            "search_keywords_en": ["machine learning"],
            "needs_clarification": True,
            "missing_fields": ["task_type", "modality", "domain"],
            "clarification_question": "Bạn muốn làm tác vụ nào?",
        }
        fake_tools = {"analyze_task": lambda text: (ambiguous, "test")}
        with patch.dict("agent.TOOL_FUNCTIONS", fake_tools, clear=True):
            run = SearchAgent(
                enabled_sources=["Hugging Face"],
                web_fallback_enabled=True,
            ).run("Tìm dataset")
        self.assertEqual(run.status, "clarification_required")
        self.assertEqual(run.ranked, [])
        self.assertEqual([event.tool for event in run.tool_events], ["analyze_task"])

    def test_catch_all_web_search_counts_as_a_search_source(self):
        registries = {
            "Hugging Face": False,
            "Kaggle": False,
            "Papers with Code": False,
        }
        self.assertTrue(has_search_source(registries, web_fallback_enabled=True))
        self.assertFalse(has_search_source(registries, web_fallback_enabled=False))

    @patch("pipeline.parse_task.get_client", return_value=None)
    def test_sensitive_parse(self, _):
        intent, _ = parse_task("phát hiện khuôn mặt giả mạo")
        self.assertTrue(intent["is_sensitive_domain"])

    @patch("pipeline.parse_task.get_client", return_value=None)
    def test_sentiment_intent_extracts_specific_constraints(self, _):
        intent, _ = parse_task(
            "Tìm dataset tiếng Việt có nhãn để phân tích cảm xúc bình luận sản phẩm"
        )
        self.assertEqual(intent["required_language"], "vietnamese")
        self.assertEqual(intent["subject"], "sentiment")
        self.assertIn("positive", intent["required_labels"])
        self.assertIn("vietnamese", intent["search_keywords_en"][0])

    @patch("pipeline.rank_candidates.get_client", return_value=None)
    def test_rank_only_keeps_verified_candidates(self, _):
        intent, _ = parse_task("Vietnamese sentiment classification")
        ranked, _ = rank_candidates(intent, [SAMPLE])
        self.assertEqual([x["id"] for x in ranked], ["owner/verified"])
        self.assertTrue(all(x["url"] for x in ranked))

    @patch("pipeline.rank_candidates.llm_label", return_value="test-model")
    @patch("pipeline.rank_candidates._llm_rank")
    @patch("pipeline.rank_candidates.get_client", return_value=object())
    def test_rank_falls_back_per_candidate_when_llm_omits_scores(
        self, _, llm_rank, __
    ):
        omitted = {
            **SAMPLE,
            "id": "owner/omitted",
            "url": "https://example.test/omitted",
            "description": "Vietnamese sentiment classification dataset",
        }
        llm_rank.return_value = {
            "verified": [{
                "id": SAMPLE["id"],
                "task_match": 5,
                "domain_fit": 5,
                "label_overlap": 4,
                "size_adequacy": 4,
                "access_type": "open",
                "reasoning": "Khớp trực tiếp.",
            }],
            "unverified": [],
        }
        intent = {
            "search_keywords_en": ["vietnamese sentiment classification"],
            "task_type": "sentiment classification",
            "domain": "NLP",
            "needs_labels": True,
        }
        with self.assertLogs("pipeline.rank_candidates", level="WARNING") as logs:
            ranked, mode = rank_candidates(intent, [SAMPLE, omitted])
        self.assertEqual(
            {row["id"] for row in ranked},
            {SAMPLE["id"], omitted["id"]},
        )
        self.assertEqual(mode, "LLM (test-model)")
        self.assertIn("1/2 candidates", "\n".join(logs.output))
        self.assertIn("verified=1, unverified=0", "\n".join(logs.output))

    @patch("pipeline.rank_candidates.llm_label", return_value="test-model")
    @patch("pipeline.rank_candidates._llm_rank")
    @patch("pipeline.rank_candidates.get_client", return_value=object())
    def test_rank_falls_back_entirely_when_llm_returns_both_lanes_empty(
        self, _, llm_rank, __
    ):
        llm_rank.return_value = {"verified": [], "unverified": []}
        intent = {
            "search_keywords_en": ["vietnamese sentiment classification"],
            "task_type": "sentiment classification",
            "domain": "NLP",
            "needs_labels": True,
        }
        ranked, mode, diagnostics = rank_candidates(
            intent,
            [SAMPLE],
            include_diagnostics=True,
        )
        self.assertEqual([row["id"] for row in ranked], [SAMPLE["id"]])
        self.assertIn("verified và unverified đều rỗng", mode)
        self.assertEqual(diagnostics["llm_scored_count"], 0)
        self.assertEqual(diagnostics["heuristic_fallback_count"], 1)
        self.assertEqual(
            diagnostics["scored_before_threshold"]["verified"][0]
            ["scoring_source"],
            "heuristic",
        )

    @patch("pipeline.rank_candidates._llm_rank_batch")
    def test_llm_ranking_batches_and_merges_all_candidates(self, rank_batch):
        candidates = [
            {
                **SAMPLE,
                "id": f"owner/dataset-{index}",
                "url": f"https://example.test/dataset-{index}",
            }
            for index in range(RANK_BATCH_SIZE * 2 + 1)
        ]

        def score_batch(_, verified, unverified):
            return {
                "verified": [{
                    "id": row["id"],
                    "task_match": 4,
                    "domain_fit": 4,
                    "label_overlap": 3,
                    "size_adequacy": 3,
                    "access_type": "open",
                    "reasoning": "Khớp metadata đầu vào.",
                } for row in verified],
                "unverified": [{
                    "id": row["id"],
                    "task_match": 4,
                    "domain_fit": 4,
                    "reasoning": "Khớp title và snippet.",
                } for row in unverified],
            }

        rank_batch.side_effect = score_batch
        result = _llm_rank(
            {"task_type": "classification", "domain": "vision"},
            candidates,
            [],
        )
        self.assertEqual(rank_batch.call_count, 3)
        self.assertEqual(len(result["verified"]), len(candidates))
        self.assertEqual(result["_batching"]["strategy"], "fixed")
        self.assertEqual(
            result["_batching"]["actual_batch_sizes"],
            [RANK_BATCH_SIZE, RANK_BATCH_SIZE, 1],
        )
        self.assertEqual(result["_batching"]["temperature"], 0)

    @patch("pipeline.rank_candidates.call_text_with_metadata")
    def test_llm_rank_batch_sends_verified_title_to_ranker(self, call):
        call.return_value = (
            json.dumps({
                "verified": [{
                    "id": SAMPLE["id"],
                    "task_match": 4,
                    "domain_fit": 4,
                    "label_overlap": 3,
                    "size_adequacy": 3,
                    "access_type": "open",
                    "reasoning": "Khớp.",
                }],
                "unverified": [],
            }),
            {},
        )
        _llm_rank_batch(
            {"task_type": "sentiment", "subject": "sentiment"},
            [{**SAMPLE, "title": "Vietnamese Sentiment Reviews"}],
            [],
        )
        payload = json.loads(call.call_args.args[1])
        self.assertEqual(
            payload["verified_candidates"][0]["title"],
            "Vietnamese Sentiment Reviews",
        )

    @patch("pipeline.rank_candidates.call_text_with_metadata")
    def test_llm_rank_batch_drops_verified_row_with_incomplete_scores(self, call):
        call.return_value = (
            json.dumps({
                "verified": [{
                    "id": SAMPLE["id"],
                    "task_match": 5,
                    "domain_fit": 5,
                    "reasoning": "Thiếu các trục điểm.",
                }],
                "unverified": [],
            }),
            {},
        )
        result = _llm_rank_batch(
            {"task_type": "sentiment", "subject": "sentiment"},
            [SAMPLE],
            [],
        )
        self.assertEqual(result["verified"], [])

    @patch("pipeline.rank_candidates.call_text_with_metadata")
    def test_llm_rank_batch_raises_specific_truncation_error(self, call_text):
        truncated = (
            '{"verified":[{"id":"owner/one"},{"id":"owner/two"}],'
            '"unverified":['
        )
        call_text.return_value = (
            truncated,
            {"completion_tokens": 3998, "finish_reason": "length"},
        )
        candidates = [
            {
                **SAMPLE,
                "id": f"owner/{index}",
                "url": f"https://example.test/{index}",
            }
            for index in range(5)
        ]
        with self.assertRaises(RankingResponseTruncatedError) as caught:
            _llm_rank_batch(
                {"task_type": "classification", "domain": "vision"},
                candidates,
                [],
            )
        self.assertEqual(caught.exception.expected_count, 5)
        self.assertEqual(caught.exception.observed_count, 2)
        self.assertEqual(caught.exception.usage["completion_tokens"], 3998)
        self.assertEqual(
            call_text.call_args.args[2],
            RANK_MAX_OUTPUT_TOKENS,
        )

    @patch("pipeline.rank_candidates.call_text_with_metadata")
    def test_llm_rank_batch_drops_wrong_lane_object(self, call_text):
        verified = [{**SAMPLE, "id": "owner/verified"}]
        call_text.return_value = (
            '{"verified":[{"id":"owner/verified","task_match":4,'
            '"domain_fit":4,"label_overlap":3,"size_adequacy":3,'
            '"access_type":"open","reasoning":"match"}],'
            '"unverified":[{"id":"owner/verified","task_match":4,'
            '"domain_fit":4,"reasoning":"wrong lane"}]}',
            {"completion_tokens": 100, "finish_reason": "stop"},
        )
        result = _llm_rank_batch({}, verified, [])
        self.assertEqual(len(result["verified"]), 1)
        self.assertEqual(result["unverified"], [])

    def test_uniform_reasoning_that_echoes_constraint_notes_is_detected(self):
        candidates = [
            {
                **SAMPLE,
                "id": f"owner/{index}",
                "constraint_notes": [
                    "Khớp loại tác vụ detection.",
                    "Có bằng chứng về subject human.",
                ],
            }
            for index in range(3)
        ]
        result = {
            "verified": [
                {
                    "id": row["id"],
                    "reasoning": (
                        "Khớp loại tác vụ detection và có bằng chứng "
                        "về subject human."
                    ),
                }
                for row in candidates
            ],
            "unverified": [],
        }
        self.assertTrue(
            _reasoning_echoes_constraint_notes(result, candidates)
        )

    def test_independent_reasoning_styles_do_not_trigger_echo_warning(self):
        candidates = [
            {
                **SAMPLE,
                "id": "owner/human",
                "constraint_notes": ["Có bằng chứng về subject human."],
            },
            {
                **SAMPLE,
                "id": "owner/road",
                "constraint_notes": ["Có bằng chứng về subject road."],
            },
        ]
        result = {
            "verified": [
                {
                    "id": "owner/human",
                    "reasoning": "Ảnh camera có bounding box cho pedestrian.",
                },
                {
                    "id": "owner/road",
                    "reasoning": "Mask pixel mô tả mặt đường trong cảnh đô thị.",
                },
            ],
            "unverified": [],
        }
        self.assertFalse(
            _reasoning_echoes_constraint_notes(result, candidates)
        )

    @patch("pipeline.rank_candidates._llm_rank_batch")
    def test_llm_rank_logs_echo_warning_without_changing_scores(self, rank_batch):
        candidates = [
            {
                **SAMPLE,
                "id": f"owner/{index}",
                "constraint_notes": [
                    "Khớp loại tác vụ detection.",
                    "Có bằng chứng về subject human.",
                ],
            }
            for index in range(2)
        ]
        rank_batch.return_value = {
            "verified": [
                {
                    "id": row["id"],
                    "task_match": 4,
                    "domain_fit": 4,
                    "label_overlap": 3,
                    "size_adequacy": 3,
                    "access_type": "open",
                    "reasoning": (
                        "Khớp loại tác vụ detection và có bằng chứng "
                        "về subject human."
                    ),
                }
                for row in candidates
            ],
            "unverified": [],
        }
        with self.assertLogs(
            "pipeline.rank_candidates", level="WARNING"
        ) as logs:
            result = _llm_rank({}, candidates, [])
        self.assertEqual(len(result["verified"]), 2)
        self.assertIn(
            "echo constraint_notes",
            "\n".join(logs.output),
        )

    @patch(
        "pipeline.rank_candidates._resolve_core_keywords",
        return_value=(
            ["human", "person", "people", "pedestrian"],
            "llm_cached",
        ),
    )
    def test_core_keyword_guard_flags_high_score_without_subject_metadata(self, _):
        intent = {
            "subject": "human detection",
            "task_type": "object detection",
        }
        candidate = {
            **SAMPLE,
            "id": "owner/forklift-object-detection",
            "title": "Forklift Object Detection",
            "description": "Annotated industrial vehicles.",
            "tags": ["object-detection", "forklift"],
            "total_score": 4.2,
            "constraint_status": "matched",
        }
        with self.assertLogs(
            "pipeline.rank_candidates", level="WARNING"
        ) as logs:
            guarded, flagged, keywords, source = _apply_core_keyword_guard(
                intent, [candidate]
            )
        self.assertEqual(flagged, [candidate["id"]])
        self.assertEqual(
            keywords, ["human", "person", "people", "pedestrian"]
        )
        self.assertEqual(source, "llm_cached")
        self.assertEqual(guarded[0]["constraint_status"], "needs_review")
        self.assertTrue(guarded[0]["needs_review"])
        self.assertIn("Core keyword 'human'", guarded[0]["review_warning"])
        self.assertIn("Core keyword guard flagged", "\n".join(logs.output))
        self.assertEqual(guarded[0]["total_score"], 4.2)

    @patch(
        "pipeline.rank_candidates._resolve_core_keywords",
        return_value=(
            ["human", "person", "people", "pedestrian"],
            "llm_cached",
        ),
    )
    def test_core_keyword_guard_accepts_human_alias_in_metadata(self, _):
        intent = {
            "subject": "human detection",
            "task_type": "object detection",
        }
        candidates = [
            {
                **SAMPLE,
                "id": "owner/person-detection",
                "title": "Person Detection Dataset",
                "description": "",
                "tags": ["pedestrian"],
                "total_score": 4.0,
            },
            {
                **SAMPLE,
                "id": "owner/people-counting",
                "title": "People Counting",
                "description": "",
                "tags": [],
                "total_score": 3.5,
            },
        ]
        guarded, flagged, _, _ = _apply_core_keyword_guard(intent, candidates)
        self.assertEqual(flagged, [])
        self.assertTrue(all("review_warning" not in row for row in guarded))

    def test_core_keyword_guard_does_not_flag_below_threshold(self):
        candidate = {
            **SAMPLE,
            "id": "owner/unrelated",
            "title": "Forklift Detection",
            "total_score": 2.99,
        }
        guarded, flagged, _, _ = _apply_core_keyword_guard(
            {
                "subject": "human detection",
                "task_type": "object detection",
            },
            [candidate],
        )
        self.assertEqual(flagged, [])
        self.assertNotIn("constraint_status", guarded[0])
        self.assertNotIn("review_warning", guarded[0])

    @patch("pipeline.rank_candidates.get_client", return_value=None)
    def test_core_keyword_fallback_subtracts_task_tokens_without_domain_map(self, _):
        self.assertEqual(
            _core_keywords({
                "subject": "rare coral bleaching segmentation",
                "task_type": "semantic segmentation",
            }),
            ["rare", "coral", "bleaching"],
        )

    @patch("pipeline.rank_candidates.get_client", return_value=object())
    @patch("pipeline.rank_candidates.call_json")
    def test_core_keyword_llm_result_is_cached_by_intent(self, call_json_mock, _):
        _llm_core_keywords_cached.cache_clear()
        call_json_mock.return_value = {
            "core_keywords": ["vehicle", "car", "truck", "automobile"]
        }
        intent = {
            "subject": "vehicles",
            "task_type": "image classification",
        }
        first, first_source = _resolve_core_keywords(intent)
        second, second_source = _resolve_core_keywords(intent)
        self.assertEqual(first, ["vehicle", "car", "truck", "automobile"])
        self.assertEqual(second, first)
        self.assertEqual(first_source, "llm_cached")
        self.assertEqual(second_source, "llm_cached")
        self.assertEqual(call_json_mock.call_count, 1)

    @patch("pipeline.rank_candidates.get_client", return_value=object())
    @patch("pipeline.rank_candidates.call_json")
    def test_core_keyword_llm_failure_uses_generic_fallback(self, call_json_mock, _):
        _llm_core_keywords_cached.cache_clear()
        call_json_mock.side_effect = RuntimeError("provider unavailable")
        with self.assertLogs(
            "pipeline.rank_candidates", level="WARNING"
        ) as logs:
            keywords, source = _resolve_core_keywords({
                "subject": "deep sea plankton classification",
                "task_type": "image classification",
            })
        self.assertEqual(keywords, ["deep", "sea", "plankton"])
        self.assertEqual(source, "generic_fallback")
        self.assertIn("using generic token fallback", "\n".join(logs.output))

    @patch("pipeline.rank_candidates._llm_rank_batch")
    def test_truncated_batch_is_logged_and_left_for_per_candidate_fallback(
        self, rank_batch
    ):
        candidates = [
            {
                **SAMPLE,
                "id": f"owner/dataset-{index}",
                "url": f"https://example.test/dataset-{index}",
            }
            for index in range(RANK_BATCH_SIZE + 1)
        ]
        rank_batch.side_effect = [
            RankingResponseTruncatedError(RANK_BATCH_SIZE, 3, "cut off"),
            {
                "verified": [{
                    "id": candidates[-1]["id"],
                    "task_match": 4,
                    "domain_fit": 4,
                    "label_overlap": 3,
                    "size_adequacy": 3,
                    "access_type": "open",
                    "reasoning": "Khớp metadata đầu vào.",
                }],
                "unverified": [],
            },
        ]
        with self.assertLogs("pipeline.rank_candidates", level="ERROR") as logs:
            result = _llm_rank(
                {"task_type": "classification", "domain": "vision"},
                candidates,
                [],
            )
        self.assertEqual(len(result["verified"]), 1)
        self.assertIn(
            f"got 3/{RANK_BATCH_SIZE} objects",
            "\n".join(logs.output),
        )

    def test_fallback_when_no_good_open_result(self):
        guidance = build_guidance({"is_narrow_domain": True, "domain": "3D"}, [])
        self.assertTrue(guidance["alternatives"])
        self.assertIn("chưa verify", guidance["registries"][0])

    def test_unverified_result_cannot_satisfy_verified_fallback(self):
        guidance = build_guidance(
            {"is_narrow_domain": False, "domain": "general"},
            [{"confidence": "unverified", "preliminary_score": 5}],
        )
        self.assertFalse(guidance["good_open_found"])
        self.assertTrue(guidance["alternatives"])

    def test_cross_source_coco_is_merged_and_verified_metadata_wins(self):
        verified = {
            **SAMPLE,
            "id": "detection-datasets/coco",
            "url": "https://huggingface.co/datasets/detection-datasets/coco",
            "source": "Hugging Face",
            "license": "apache-2.0",
            "confidence": "verified",
        }
        web = {
            "id": "https://example.test/ms-coco-2017",
            "url": "https://example.test/ms-coco-2017",
            "source": "Web search · example.test",
            "title": "MS COCO Dataset 2017",
            "snippet": "MS COCO images.",
            "confidence": "unverified",
        }
        with patch("pipeline.deduplicate.get_client", return_value=None):
            rows = deduplicate_candidates([verified, web])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["confidence"], "verified")
        self.assertEqual(rows[0]["url"], verified["url"])
        self.assertEqual(rows[0]["license"], "apache-2.0")
        self.assertEqual(len(rows[0]["sources"]), 2)

    def test_imagenet_and_tiny_imagenet_are_not_auto_merged(self):
        imagenet = {
            **SAMPLE,
            "id": "ImageNet",
            "url": "https://example.test/imagenet",
            "source": "Source A",
            "confidence": "verified",
        }
        tiny = {
            **SAMPLE,
            "id": "Tiny-ImageNet",
            "url": "https://example.test/tiny-imagenet",
            "source": "Source B",
            "confidence": "verified",
        }
        with patch("pipeline.deduplicate.get_client", return_value=None):
            rows = deduplicate_candidates([imagenet, tiny])
        self.assertEqual(len(rows), 2)
        self.assertLess(similarity_score("ImageNet", "Tiny-ImageNet"), 0.50)

    def test_different_numeric_dataset_variants_are_not_merged(self):
        self.assertLess(similarity_score("CIFAR-10", "CIFAR-100"), 0.50)

    def test_dataset_name_normalization_removes_versions_and_years(self):
        self.assertEqual(normalize_dataset_name("MS COCO-dataset-v2 2017"), "coco")

    @patch("pipeline.deduplicate.get_client", return_value=object())
    @patch("pipeline.deduplicate._llm_ambiguous_groups")
    def test_medium_llm_duplicate_is_flagged_but_not_merged(self, decide, _):
        first = {
            **SAMPLE,
            "id": "coco-images",
            "source": "Source A",
            "url": "https://example.test/coco-images",
            "confidence": "verified",
        }
        second = {
            **SAMPLE,
            "id": "coco-annotations",
            "source": "Source B",
            "url": "https://example.test/coco-annotations",
            "confidence": "verified",
        }
        decide.return_value = [{
            "group_representative_name": "COCO",
            "member_ids": [
                "candidate_0:Source A:coco-images",
                "candidate_1:Source B:coco-annotations",
            ],
            "confidence": "medium",
        }]
        rows = deduplicate_candidates([first, second])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.get("possible_duplicate_of") for row in rows))

    @patch("sources.huggingface.requests.get")
    def test_hf_schema_uses_api_response(self, get):
        get.return_value.json.return_value = [{
            "id": "real/api-result", "downloads": 4, "likes": 2,
            "tags": ["audio"], "cardData": {"license": "mit"},
        }]
        get.return_value.raise_for_status.return_value = None
        result = search_hf_datasets("audio")
        self.assertEqual(result[0]["id"], "real/api-result")
        self.assertEqual(result[0]["license"], "mit")
        self.assertTrue(result[0]["url"])

    def test_verify_candidates_deduplicates_and_rejects_missing_url(self):
        duplicate = {**SAMPLE, "downloads": 999}
        invalid = {**SAMPLE, "id": "missing/url", "url": ""}
        result = verify_candidates([SAMPLE, duplicate, invalid])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["downloads"], 999)

    def test_verify_does_not_cut_candidates_before_ranking(self):
        candidates = [
            {**SAMPLE, "id": f"owner/dataset-{index}", "url": f"https://example.test/{index}"}
            for index in range(25)
        ]
        self.assertEqual(len(verify_candidates(candidates, limit=None)), 25)

    def test_language_constraint_demotes_definite_mismatch(self):
        intent = {
            "required_language": "vietnamese",
            "subject": "product reviews",
            "required_labels": ["positive", "negative"],
        }
        vietnamese = {
            **SAMPLE,
            "id": "owner/vietnamese-reviews",
            "tags": ["language:vi", "sentiment"],
            "description": "Product reviews with positive and negative labels",
        }
        english = {
            **SAMPLE,
            "id": "owner/english-reviews",
            "tags": ["language:en", "sentiment"],
            "description": "Product reviews with positive and negative labels",
        }
        good = evaluate_constraints(intent, vietnamese)
        bad = evaluate_constraints(intent, english)
        self.assertEqual(good["constraint_status"], "matched")
        self.assertEqual(bad["constraint_status"], "mismatch")
        prepared = prepare_candidates(intent, [english, vietnamese])
        self.assertEqual(prepared[0]["id"], vietnamese["id"])

    def test_human_detection_requires_subject_evidence(self):
        intent = {
            "task_type": "object detection",
            "subject": "human detection",
            "required_language": "any",
            "required_labels": [],
        }
        unrelated = {
            **SAMPLE,
            "id": "owner/forklift-object-detection",
            "description": (
                "Dataset Labels\n['forklift', 'person']\n"
                "Number of Images\n{'train': 295}\nHow to Use"
            ),
            "tags": ["object-detection", "forklift"],
        }
        result = evaluate_constraints(intent, unrelated)
        self.assertEqual(result["constraint_status"], "partial")
        self.assertTrue(result["constraint_task_matched"])
        self.assertFalse(result["constraint_subject_matched"])
        self.assertIn("human", result["constraint_subject_keywords"])
        self.assertTrue(
            any("subject human detection" in note for note in result["constraint_notes"])
        )

    def test_human_subject_aliases_match_people_and_pedestrians(self):
        intent = {
            "task_type": "object detection",
            "subject": "human detection",
            "required_language": "any",
            "required_labels": [],
        }
        for word in ("human", "person", "people", "pedestrian"):
            with self.subTest(word=word):
                candidate = {
                    **SAMPLE,
                    "id": f"owner/{word}",
                    "description": f"Object detection images annotated for {word}",
                    "tags": ["object-detection"],
                }
                result = evaluate_constraints(intent, candidate)
                self.assertEqual(result["constraint_status"], "matched")
                self.assertTrue(result["constraint_subject_matched"])

    @patch("pipeline.rank_candidates._llm_rank")
    @patch("pipeline.rank_candidates.get_client", return_value=object())
    @patch("pipeline.rank_candidates.llm_label", return_value="test-model")
    def test_task_only_partial_match_is_capped_below_threshold(
        self, _, __, llm_rank
    ):
        intent = {
            "search_keywords_en": ["human object detection"],
            "task_type": "object detection",
            "domain": "computer vision",
            "subject": "human detection",
            "needs_labels": True,
        }
        candidate = evaluate_constraints(intent, {
            **SAMPLE,
            "id": "owner/license-plate-detection",
            "description": "License plate object detection dataset",
            "tags": ["object-detection", "license-plate"],
        })
        llm_rank.return_value = {
            "verified": [{
                "id": candidate["id"],
                "task_match": 5,
                "domain_fit": 5,
                "label_overlap": 5,
                "size_adequacy": 5,
                "access_type": "open",
                "reasoning": "Copied the generic detection constraint note.",
            }],
            "unverified": [],
        }
        ranked, _, diagnostics = rank_candidates(
            intent, [candidate], include_diagnostics=True
        )
        self.assertEqual(ranked, [])
        excluded = diagnostics["excluded_by_threshold"][0]
        self.assertEqual(excluded["task_match"], 2)
        self.assertEqual(excluded["domain_fit"], 2)

    @patch("sources.enrich.requests.get")
    def test_huggingface_enrichment_adds_language_labels_and_size(self, get):
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.return_value = {
            "tags": ["text-classification"],
            "cardData": {
                "language": ["vi"],
                "task_categories": ["text-classification"],
                "dataset_info": {
                    "features": [{"name": "label", "dtype": {"names": ["negative", "positive"]}}],
                    "splits": [{"name": "train", "num_examples": 1000}],
                },
            },
        }
        rows = enrich_candidates([SAMPLE], max_candidates=1)
        self.assertTrue(rows[0]["metadata_enriched"])
        self.assertIn("language:vi", rows[0]["tags"])
        self.assertEqual(rows[0]["sample_count"], 1000)
        self.assertIn("positive", rows[0]["features_text"])

    @patch("sources.enrich.requests.get")
    def test_huggingface_enrichment_normalizes_scalar_card_fields(self, get):
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.return_value = {
            "id": "owner/data",
            "tags": [],
            "cardData": {
                "language": "en",
                "task_categories": "text-classification",
            },
        }
        rows = enrich_candidates([{
            **SAMPLE,
            "id": "owner/data",
            "title": "Data",
        }])
        self.assertIn("language:en", rows[0]["tags"])
        self.assertIn("text-classification", rows[0]["tags"])
        self.assertNotIn("language:e", rows[0]["tags"])

    def test_enrichment_recovers_zenodo_title_from_search_metadata(self):
        candidate = {
            "id": "12345",
            "url": "https://zenodo.org/records/12345",
            "source": "Zenodo",
            "raw_metadata": {
                "metadata": {"title": "Construction Helmet Image Dataset"}
            },
        }
        rows = enrich_candidates([candidate], max_candidates=1)
        self.assertEqual(rows[0]["title"], "Construction Helmet Image Dataset")

    @patch("sources.web_fallback.requests.get")
    def test_web_fallback_is_hardcoded_unverified_and_snippet_only(self, get):
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.return_value = {
            "organic_results": [{
                "title": "Dataset page",
                "link": "https://data.example/dataset",
                "snippet": "A dataset search result.",
            }]
        }
        rows = search_web_fallback(
            ["sentiment dataset"],
            ["data.example"],
            limit=1,
            serpapi_api_key="test",
        )
        self.assertEqual(rows[0]["confidence"], "unverified")
        self.assertEqual(
            set(rows[0]),
            {"id", "url", "source", "title", "snippet", "confidence"},
        )

    @patch("pipeline.rank_candidates.get_client", return_value=None)
    def test_unverified_ranking_uses_only_two_scores(self, _):
        intent, _ = parse_task("Vietnamese sentiment classification")
        candidate = {
            "id": "https://example.test/web",
            "url": "https://example.test/web",
            "source": "Web search · example.test",
            "title": "Vietnamese sentiment dataset",
            "snippet": "Dataset for sentiment classification.",
            "confidence": "unverified",
        }
        ranked, _ = rank_candidates(intent, [candidate])
        row = ranked[0]
        self.assertEqual(row["confidence"], "unverified")
        self.assertIn("preliminary_score", row)
        self.assertNotIn("label_overlap", row)
        self.assertNotIn("size_adequacy", row)
        self.assertNotIn("license", row)
        self.assertIn("chưa được xác minh", row["reasoning"])

    @patch("pipeline.rank_candidates.get_client", return_value=None)
    def test_irrelevant_candidate_is_removed_by_relevance_threshold(self, _):
        intent, _ = parse_task("Vietnamese sentiment classification")
        unrelated = {
            **SAMPLE,
            "id": "owner/weather-numbers",
            "url": "https://example.test/weather",
            "description": "Hourly temperature and rainfall measurements",
            "tags": ["weather", "tabular"],
            "constraint_status": "unknown",
        }
        ranked, _ = rank_candidates(intent, [unrelated])
        self.assertEqual(ranked, [])

    @patch("pipeline.rank_candidates.get_client", return_value=None)
    def test_rank_diagnostics_keep_full_scores_for_threshold_exclusions(self, _):
        intent = {
            "search_keywords_en": ["car classification"],
            "task_type": "car classification",
            "domain": "computer vision",
            "needs_labels": True,
        }
        unrelated = {
            **SAMPLE,
            "id": "owner/continual-simulator",
            "url": "https://example.test/continual-simulator",
            "description": "Endless Continual Learning Simulator",
            "tags": ["continual-learning"],
        }
        ranked, _, diagnostics = rank_candidates(
            intent,
            [unrelated],
            include_diagnostics=True,
        )
        self.assertEqual(ranked, [])
        self.assertEqual(diagnostics["input_candidate_count"], 1)
        self.assertEqual(diagnostics["threshold_excluded_count"], 1)
        scored = diagnostics["scored_before_threshold"]["verified"][0]
        self.assertEqual(scored["id"], unrelated["id"])
        self.assertEqual(scored["scoring_source"], "heuristic")
        self.assertIn("total_score", scored)
        self.assertIn("task_match", scored)
        self.assertIn("domain_fit", scored)
        self.assertIn("label_overlap", scored)
        self.assertIn("size_adequacy", scored)
        self.assertIn("reasoning", scored)
        excluded = diagnostics["excluded_by_threshold"][0]
        self.assertEqual(excluded["id"], unrelated["id"])
        self.assertTrue(excluded["exclusion_reasons"])

    def test_search_agent_preserves_pipeline_and_redacts_key(self):
        intent = {
            "search_keywords_en": ["vietnamese sentiment"],
            "task_type": "classification",
            "domain": "NLP",
            "needs_labels": True,
            "is_narrow_domain": False,
        }
        fake_tools = {
            "analyze_task": lambda text: (intent, "test"),
            "search_registry": lambda **kwargs: [{
                **SAMPLE, "title": "Vietnamese Sentiment Dataset",
            }],
            "verify_candidates": verify_candidates,
            "prepare_candidates": lambda **kwargs: kwargs["candidates"],
            "enrich_candidates": lambda **kwargs: kwargs["candidates"],
            "deduplicate_candidates": lambda **kwargs: kwargs["candidates"],
            "rank_datasets": lambda **kwargs: ([{
                **SAMPLE, "access_type": "open", "total_score": 4.0,
            }], "test"),
        }
        with patch.dict("agent.TOOL_FUNCTIONS", fake_tools, clear=True):
            run = SearchAgent(
                enabled_sources=["Kaggle"],
                credentials={"username": "demo", "key": "secret"},
                web_fallback_enabled=False,
            ).run("query")
        self.assertEqual(run.ranked[0]["id"], SAMPLE["id"])
        search_event = next(x for x in run.tool_events if x.tool == "search_registry")
        self.assertEqual(search_event.args["key"], "***")
        self.assertEqual(search_event.args["keyword"], "vietnamese sentiment")
        self.assertEqual(search_event.args["candidate_count"], 1)
        self.assertTrue(all(x.status == "success" for x in run.tool_events))

    def test_search_agent_excludes_definite_constraint_mismatch(self):
        intent = {
            "search_keywords_en": ["vietnamese sentiment"],
            "task_type": "sentiment classification",
            "domain": "NLP",
            "needs_labels": True,
            "is_narrow_domain": False,
        }
        good = {
            **SAMPLE,
            "id": "owner/vi",
            "title": "Vietnamese Sentiment Dataset",
            "constraint_status": "matched",
        }
        bad = {
            **SAMPLE,
            "id": "owner/en",
            "title": "English Sentiment Dataset",
            "url": "https://example.test/en",
            "constraint_status": "mismatch",
        }
        fake_tools = {
            "analyze_task": lambda text: (intent, "test"),
            "search_registry": lambda **kwargs: [good, bad],
            "verify_candidates": verify_candidates,
            "prepare_candidates": lambda **kwargs: kwargs["candidates"],
            "enrich_candidates": lambda **kwargs: kwargs["candidates"],
            "deduplicate_candidates": lambda **kwargs: kwargs["candidates"],
            "rank_datasets": lambda **kwargs: ([{
                **row,
                "confidence": "verified",
                "access_type": "open",
                "total_score": 4.0,
            } for row in kwargs["candidates"]], "test"),
        }
        with patch.dict("agent.TOOL_FUNCTIONS", fake_tools, clear=True):
            run = SearchAgent(
                enabled_sources=["Hugging Face"],
                web_fallback_enabled=False,
            ).run("query")
        self.assertEqual([row["id"] for row in run.ranked], ["owner/vi"])


if __name__ == "__main__":
    unittest.main()
