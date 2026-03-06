from __future__ import annotations

import pytest

from utils.template import (
    IGNORE_INDEX,
    _extract_input_ids,
    _infer_truncation_lengths,
    normalize_user_prompt,
    tokenize_sft_example,
)


def test_extract_input_ids_from_mapping_single_item_batch() -> None:
    assert _extract_input_ids({"input_ids": [[1, 2, 3]]}) == [1, 2, 3]


def test_extract_input_ids_rejects_multi_sequence_batch() -> None:
    with pytest.raises(ValueError):
        _extract_input_ids({"input_ids": [[1], [2]]})


@pytest.mark.parametrize(
    ("source_len", "target_len", "cutoff_len", "expected"),
    [
        (3000, 2000, 1000, (600, 400)),
        (100, 1000, 1000, (100, 900)),
    ],
)
def test_infer_truncation_lengths(source_len: int, target_len: int, cutoff_len: int, expected: tuple[int, int]) -> None:
    assert _infer_truncation_lengths(source_len, target_len, cutoff_len) == expected


def test_normalize_user_prompt_with_no_input_token() -> None:
    assert normalize_user_prompt("instruction", "none") == "instruction"


def test_normalize_user_prompt_with_input_text() -> None:
    assert normalize_user_prompt("instruction", "x=1") == "instruction\nx=1"


def test_tokenize_sft_example_masks_prompt_tokens(fake_chat_tokenizer_factory) -> None:
    tokenizer = fake_chat_tokenizer_factory(prompt_ids=[10, 11, 12], full_ids=[10, 11, 12, 20, 21])
    tokenized = tokenize_sft_example(
        tokenizer=tokenizer,
        user_prompt="hello",
        assistant_response="world",
        cutoff_len=128,
    )
    assert tokenized["labels"] == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 20, 21]
    assert tokenized["attention_mask"] == [1, 1, 1, 1, 1]


def test_tokenize_sft_example_drops_examples_longer_than_cutoff(fake_chat_tokenizer_factory) -> None:
    tokenizer = fake_chat_tokenizer_factory(
        prompt_ids=[1, 2, 3, 4, 5, 6],
        full_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    )
    tokenized = tokenize_sft_example(
        tokenizer=tokenizer,
        user_prompt="prompt",
        assistant_response="answer",
        cutoff_len=8,
    )
    assert tokenized is None


def test_tokenize_sft_example_requires_positive_cutoff(fake_chat_tokenizer_factory) -> None:
    tokenizer = fake_chat_tokenizer_factory(prompt_ids=[1], full_ids=[1, 2])
    with pytest.raises(ValueError):
        tokenize_sft_example(
            tokenizer=tokenizer,
            user_prompt="prompt",
            assistant_response="answer",
            cutoff_len=0,
        )
