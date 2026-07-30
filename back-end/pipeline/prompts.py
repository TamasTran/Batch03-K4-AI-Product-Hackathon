from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


PROMPT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "system_prompt.md"


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def load_prompt_section(name: str) -> str:
    marker = re.escape(name.upper())
    match = re.search(
        rf"<!-- SECTION:{marker} -->(.*?)<!-- ENDSECTION -->",
        _prompt_text(),
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"Không tìm thấy prompt section: {name}")
    return match.group(1).strip()


def step_prompt(name: str) -> str:
    return f"{load_prompt_section('COMMON')}\n\n{load_prompt_section(name)}"
