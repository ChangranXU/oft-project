from __future__ import annotations

from typing import Any


IGNORE_INDEX = -100
NO_INPUT_TOKENS = {"<noinput>", "< noinput >", "none", "null", "n/a"}


def _extract_input_ids(tokenized: Any) -> list[int]:
    """Normalize tokenizer outputs across transformers versions."""
    if isinstance(tokenized, dict):
        if "input_ids" not in tokenized:
            raise ValueError("Chat template output dict is missing `input_ids`.")
        input_ids = tokenized["input_ids"]
    else:
        input_ids = tokenized

    if not isinstance(input_ids, list):
        raise TypeError(f"`input_ids` should be list[int], got {type(input_ids)}.")
    return [int(token_id) for token_id in input_ids]


def _infer_truncation_lengths(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    """Allocate token budget between prompt (source) and response (target)."""
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


def normalize_user_prompt(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text and input_text.lower() not in NO_INPUT_TOKENS:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def build_chat_messages(user_prompt: str, assistant_response: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]


def build_prompt_messages(user_prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": user_prompt}]


def tokenize_sft_example(
    tokenizer: Any,
    user_prompt: str,
    assistant_response: str,
    cutoff_len: int,
) -> dict[str, list[int]]:
    if cutoff_len <= 0:
        raise ValueError("`cutoff_len` must be a positive integer.")

    prompt_messages = build_prompt_messages(user_prompt)
    full_messages = build_chat_messages(user_prompt, assistant_response)

    prompt_ids = _extract_input_ids(
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full_ids = _extract_input_ids(
        tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )

    if len(prompt_ids) > len(full_ids):
        raise ValueError(
            "Prompt token ids are longer than full conversation token ids. "
            "Please verify tokenizer chat template compatibility."
        )

    if full_ids[: len(prompt_ids)] == prompt_ids:
        response_start = len(prompt_ids)
    else:
        # Fallback to common-prefix boundary to avoid silently masking the
        # whole sample when tokenizer chat templates differ slightly.
        response_start = 0
        common_bound = min(len(prompt_ids), len(full_ids))
        while response_start < common_bound and prompt_ids[response_start] == full_ids[response_start]:
            response_start += 1
        if response_start == 0:
            raise ValueError(
                "Unable to locate response boundary from tokenizer chat template outputs. "
                "Please check the selected model/template combination."
            )

    source_len = response_start
    target_len = len(full_ids) - response_start

    if len(full_ids) > cutoff_len:
        source_ids = full_ids[:source_len]
        target_ids = full_ids[response_start:]
        new_source_len, new_target_len = _infer_truncation_lengths(source_len, target_len, cutoff_len)
        full_ids = source_ids[:new_source_len] + target_ids[:new_target_len]
        response_start = new_source_len

    labels = [IGNORE_INDEX] * response_start + full_ids[response_start:]

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }

