from __future__ import annotations

import ast
import os
import re
import warnings
from statistics import mean
from typing import Any

import evaluate
from codebleu import calc_codebleu


os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
_CODE_EVAL = evaluate.load("code_eval")


def extract_python_code(text: str) -> str:
    text = text.strip()
    if "```python" in text:
        body = text.split("```python", maxsplit=1)[1]
        return body.split("```", maxsplit=1)[0].strip()
    if "```" in text:
        body = text.split("```", maxsplit=1)[1]
        return body.split("```", maxsplit=1)[0].strip()
    return text


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def parse_pass_k(values: str | list[int] | None) -> list[int]:
    if values is None:
        return [1]
    if isinstance(values, list):
        parsed = sorted(set(int(v) for v in values if int(v) > 0))
    else:
        tokens = [int(v.strip()) for v in values.split(",") if v.strip()]
        parsed = sorted(set(v for v in tokens if v > 0))
    return parsed or [1]


def ensure_humaneval_candidate(prompt: str, completion: str) -> str:
    prompt_text = prompt.rstrip()
    code = extract_python_code(completion)
    code = code.strip()
    if not code:
        return prompt_text
    if code.startswith(prompt_text):
        return code
    return f"{prompt_text}\n{code}".rstrip()


def build_humaneval_test_case(test: str, entry_point: str) -> str:
    test_code = str(test).rstrip()
    function_name = str(entry_point).strip()
    if not function_name:
        return test_code
    if re.search(rf"check\s*\(\s*{re.escape(function_name)}\s*\)", test_code):
        return test_code
    return f"{test_code}\n\ncheck({function_name})"


def build_mbpp_prompt(example: dict[str, Any]) -> str:
    prompt = str(example["prompt"]).strip()
    code = str(example.get("code", "")).strip()
    signature = None
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") and stripped.endswith(":"):
            signature = stripped[4:-1].strip()
            break
    if signature:
        return f"{prompt}\nPython function: {signature}"
    return prompt


def compute_codegen_metrics(
    all_predictions: list[list[str]],
    first_predictions: list[str],
    codebleu_references: list[str],
    pass_references: list[str],
    pass_k_values: list[int],
) -> dict[str, Any]:
    syntax_rate = mean(is_valid_python(code) for code in first_predictions) if first_predictions else 0.0

    raw_pass = _CODE_EVAL.compute(
        predictions=all_predictions,
        references=pass_references,
        k=pass_k_values,
    )
    pass_scores = raw_pass[0] if isinstance(raw_pass, tuple) else raw_pass
    pass_at_k: dict[str, float] = {}
    for k in pass_k_values:
        key = f"pass@{k}"
        value = pass_scores.get(key, 0.0)
        if hasattr(value, "item"):
            value = value.item()
        pass_at_k[key] = float(value)

    try:
        codebleu = calc_codebleu(
            references=codebleu_references,
            predictions=first_predictions,
            lang="python",
        )["codebleu"]
    except Exception as exc:  # pragma: no cover - environment-dependent dependency mismatch
        warnings.warn(
            "CodeBLEU computation failed; setting CodeBLEU to 0.0. "
            "Install compatible versions: "
            "`tree-sitter==0.22.3` and `tree-sitter-python==0.21.0`. "
            f"Original error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        codebleu = 0.0

    return {
        "syntax_rate": float(syntax_rate),
        "pass_at_k": pass_at_k,
        "codebleu": float(codebleu),
    }


def summarize_metrics(name: str, metrics: dict[str, Any]) -> str:
    pass_part = ", ".join(f"{k}: {v:.4f}" for k, v in metrics["pass_at_k"].items())
    return (
        f"{name} -> SyntaxRate: {metrics['syntax_rate']:.4f}, "
        f"{pass_part}, CodeBLEU: {metrics['codebleu']:.4f}"
    )


def extract_assistant_answer(decoded_text: str) -> str:
    if "assistant\n" in decoded_text:
        return decoded_text.split("assistant\n")[-1].strip()
    return decoded_text.strip()


def try_extract_function_name(code: str) -> str | None:
    match = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", code)
    if match:
        return match.group(1)
    return None

