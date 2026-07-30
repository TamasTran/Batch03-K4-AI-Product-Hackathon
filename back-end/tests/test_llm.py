import os
import unittest
from unittest.mock import patch

from pipeline.llm import (
    LLM_TEMPERATURE,
    call_json,
    get_llm_config,
    infer_provider,
    _gemini_text,
)


class LLMRouterTests(unittest.TestCase):
    def test_llm_temperature_is_deterministic(self):
        self.assertEqual(LLM_TEMPERATURE, 0)

    def test_model_name_infers_provider(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(infer_provider("gpt-4o-mini"), "openai")
            self.assertEqual(infer_provider("claude-sonnet-4-6"), "anthropic")
            self.assertEqual(infer_provider("gemini-2.5-flash"), "gemini")
            self.assertEqual(infer_provider("openai/gpt-4o-mini"), "openrouter")

    def test_gpt_model_selects_openai_key(self):
        with patch.dict(os.environ, {
            "LLM_MODEL": "gpt-4o-mini",
            "OPENAI_API_KEY": "test-openai",
        }, clear=True):
            config = get_llm_config()
        self.assertIsNotNone(config)
        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.api_key, "test-openai")

    def test_gemini_model_selects_gemini_key(self):
        with patch.dict(os.environ, {
            "LLM_MODEL": "gemini-2.5-flash",
            "GEMINI_API_KEY": "test-gemini",
        }, clear=True):
            config = get_llm_config()
        self.assertIsNotNone(config)
        self.assertEqual(config.provider, "gemini")

    def test_explicit_compatible_provider_overrides_model_inference(self):
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai_compatible",
            "LLM_MODEL": "local-model",
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "https://llm.example/api/v1",
        }, clear=True):
            config = get_llm_config()
        self.assertIsNotNone(config)
        self.assertEqual(config.provider, "openai_compatible")
        self.assertEqual(config.base_url, "https://llm.example/api/v1")

    def test_missing_matching_key_disables_llm(self):
        with patch.dict(os.environ, {
            "LLM_MODEL": "gpt-4o-mini",
            "ANTHROPIC_API_KEY": "wrong-provider-key",
        }, clear=True):
            self.assertIsNone(get_llm_config())

    @patch("pipeline.llm._openai_compatible_text", return_value='```json\n{"ok": true}\n```')
    def test_call_json_routes_and_parses_openai_response(self, complete):
        with patch.dict(os.environ, {
            "LLM_MODEL": "gpt-4o-mini",
            "OPENAI_API_KEY": "test-openai",
        }, clear=True):
            result = call_json("system", "user")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(complete.call_args.args[0].provider, "openai")

    @patch("pipeline.llm.requests.post")
    def test_gemini_key_is_sent_in_header_not_url(self, post):
        post.return_value.ok = True
        post.return_value.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]
        }
        with patch.dict(os.environ, {
            "LLM_MODEL": "gemini-2.5-flash",
            "GEMINI_API_KEY": "secret-gemini",
        }, clear=True):
            config = get_llm_config()
            text = _gemini_text(config, "system", "user", 100)
        self.assertEqual(text, '{"ok": true}')
        kwargs = post.call_args.kwargs
        self.assertNotIn("params", kwargs)
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "secret-gemini")
        self.assertNotIn("secret-gemini", post.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
