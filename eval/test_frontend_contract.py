from html.parser import HTMLParser
from pathlib import Path
import re
import unittest

from api import SOURCE_NAMES

ROOT = Path(__file__).resolve().parents[2]
HTML_PATH = ROOT / "front-end" / "index.html"
JS_PATH = ROOT / "front-end" / "app.js"


class _ButtonParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.buttons: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "button":
            self.buttons.append(dict(attrs))


class FrontendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = HTML_PATH.read_text(encoding="utf-8")
        cls.js = JS_PATH.read_text(encoding="utf-8")
        cls.parser = _ButtonParser()
        cls.parser.feed(cls.html)

    def test_every_non_submit_button_has_a_declared_action(self) -> None:
        inert = []
        for attrs in self.parser.buttons:
            if attrs.get("type") == "submit" or attrs.get("value") == "close":
                continue
            if any(key in attrs for key in ("id", "data-prompt", "data-view")):
                continue
            # The dialog close icon submits its method=dialog parent form.
            if attrs.get("aria-label") == "Đóng":
                continue
            inert.append(attrs)
        self.assertEqual(inert, [], f"Buttons without an action contract: {inert}")

    def test_all_interactive_ids_are_wired_in_javascript(self) -> None:
        interactive_ids = {
            attrs["id"]
            for attrs in self.parser.buttons
            if attrs.get("id") and attrs.get("type") != "submit"
        }
        missing = sorted(
            element_id
            for element_id in interactive_ids
            if f'#{element_id}' not in self.js
        )
        self.assertEqual(missing, [])

    def test_dynamic_html_is_escaped_and_urls_are_protocol_checked(self) -> None:
        self.assertIn('replace(/[&<>"\']', self.js)
        self.assertIn('["http:", "https:"].includes(url.protocol)', self.js)
        self.assertNotRegex(self.js, re.compile(r'target="_blank"(?! rel=)'))

    def test_history_is_real_persisted_state_not_static_samples(self) -> None:
        self.assertIn("localStorage.setItem(HISTORY_KEY", self.js)
        self.assertIn("data-history-id", self.js)
        self.assertNotIn("Vietnamese sentiment data</span>", self.html)

    def test_search_sends_client_identity_for_version_diagnostics(self) -> None:
        self.assertIn("client_version: CLIENT_VERSION", self.js)
        self.assertIn("client_build_hash: CLIENT_BUILD_HASH", self.js)

    def test_frontend_only_offers_sources_supported_by_backend(self) -> None:
        offered = set(re.findall(r'name="source"\s+value="([^"]+)"', self.html))
        self.assertLessEqual(offered, SOURCE_NAMES)

    def test_unconfigured_optional_sources_are_disabled_from_api_config(self) -> None:
        self.assertIn("config.available_sources", self.js)
        self.assertIn("!config.web_search_configured", self.js)
        self.assertIn("elements.webFallback.disabled = true", self.js)


if __name__ == "__main__":
    unittest.main()
