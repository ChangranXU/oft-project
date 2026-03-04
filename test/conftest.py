from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FakeChatTokenizer:
    """Minimal tokenizer double for chat-template based unit tests."""

    def __init__(self, prompt_ids: list[int], full_ids: list[int]) -> None:
        self._prompt_ids = prompt_ids
        self._full_ids = full_ids

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> dict[str, list[int]]:
        del messages, tokenize
        if add_generation_prompt:
            return {"input_ids": list(self._prompt_ids)}
        return {"input_ids": list(self._full_ids)}


@pytest.fixture
def fake_chat_tokenizer_factory():
    def _factory(prompt_ids: list[int], full_ids: list[int]) -> Any:
        return FakeChatTokenizer(prompt_ids=prompt_ids, full_ids=full_ids)

    return _factory
