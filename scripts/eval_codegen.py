#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.eval_utils import (  # noqa: E402
    build_humaneval_test_case,
    build_mbpp_prompt,
    compute_codegen_metrics,
    ensure_humaneval_candidate,
    extract_assistant_answer,
    extract_python_code,
    parse_pass_k,
    summarize_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on HumanEval/MBPP.")
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID or local model/adapter path.",
    )
    parser.add_argument(
        "--benchmark",
        choices=["all", "humaneval", "mbpp"],
        default="all",
        help="Which benchmark to run.",
    )
    parser.add_argument(
        "--pass_k",
        default="1",
        help="Comma-separated k values for pass@k, e.g. 1 or 1,5,10.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy decoding).",
    )
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save JSON evaluation results.",
    )
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    return parser.parse_args()


def _load_model_and_tokenizer(model_path_or_id: str, trust_remote_code: bool):
    model_path = Path(model_path_or_id).expanduser()
    adapter_config = model_path / "adapter_config.json"

    if adapter_config.exists():
        adapter_cfg = json.loads(adapter_config.read_text(encoding="utf-8"))
        base_model_name = adapter_cfg.get("base_model_name_or_path")
        if not base_model_name:
            raise ValueError(f"`base_model_name_or_path` missing in {adapter_config}")

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=trust_remote_code,
            )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(base_model, str(model_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path_or_id,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_id,
            trust_remote_code=trust_remote_code,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def _generate_candidates(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    num_candidates: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[str]:
    messages = [{"role": "user", "content": prompt.strip()}]
    chat_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # transformers>=5 returns a BatchEncoding by default, while older versions
    # may return a bare tensor. Support both for stable eval behavior.
    model_inputs: dict[str, torch.Tensor]
    if isinstance(chat_inputs, dict) or hasattr(chat_inputs, "keys"):
        if hasattr(chat_inputs, "to"):
            chat_inputs = chat_inputs.to(device)
        input_ids = chat_inputs["input_ids"]
        attention_mask = chat_inputs.get("attention_mask")
        model_inputs = {"input_ids": input_ids}
        if isinstance(attention_mask, torch.Tensor):
            model_inputs["attention_mask"] = attention_mask
    else:
        if not isinstance(chat_inputs, torch.Tensor):
            raise TypeError(f"Unexpected chat template output type: {type(chat_inputs)}")
        model_inputs = {"input_ids": chat_inputs.to(device)}

    do_sample = num_candidates > 1 or temperature > 0
    if temperature <= 0:
        temperature = 1.0

    with torch.inference_mode():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_candidates,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = model_inputs["input_ids"].shape[-1]
    candidates: list[str] = []
    for output in outputs:
        gen_tokens = output[prompt_len:]
        decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        candidates.append(extract_python_code(extract_assistant_answer(decoded)))
    return candidates


def evaluate_humaneval(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    pass_k: list[int],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    dataset = load_dataset("openai/openai_humaneval", split="test")
    candidate_count = max(pass_k)

    all_predictions: list[list[str]] = []
    first_predictions: list[str] = []
    codebleu_references: list[str] = []
    pass_references: list[str] = []

    for row in tqdm(dataset, desc="HumanEval"):
        prompt = str(row["prompt"]).strip()
        candidates = _generate_candidates(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            num_candidates=candidate_count,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        humaneval_candidates = [ensure_humaneval_candidate(prompt, c) for c in candidates]
        all_predictions.append(humaneval_candidates)
        first_predictions.append(humaneval_candidates[0])
        codebleu_references.append(prompt + str(row["canonical_solution"]))
        pass_references.append(
            build_humaneval_test_case(
                test=str(row["test"]),
                entry_point=str(row["entry_point"]),
            )
        )

    return compute_codegen_metrics(
        all_predictions=all_predictions,
        first_predictions=first_predictions,
        codebleu_references=codebleu_references,
        pass_references=pass_references,
        pass_k_values=pass_k,
    )


def evaluate_mbpp(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    pass_k: list[int],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    # NOTE: datasets>=3 no longer supports remote dataset scripts (e.g. mbpp.py
    # in Muennighoff/mbpp). Use the canonical parquet-backed `mbpp` dataset and
    # pin to the sanitized config + test split.
    dataset = load_dataset("mbpp", "sanitized", split="test")
    candidate_count = max(pass_k)

    all_predictions: list[list[str]] = []
    first_predictions: list[str] = []
    codebleu_references: list[str] = []
    pass_references: list[str] = []

    for row in tqdm(dataset, desc="MBPP"):
        prompt = build_mbpp_prompt(row)
        candidates = _generate_candidates(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            num_candidates=candidate_count,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        all_predictions.append(candidates)
        first_predictions.append(candidates[0])
        codebleu_references.append(str(row["code"]))
        test_setup_parts: list[str] = []
        test_setup = str(row.get("test_setup_code", "")).strip()
        if test_setup:
            test_setup_parts.append(test_setup)
        for import_stmt in row.get("test_imports", []):
            import_line = str(import_stmt).strip()
            if import_line:
                test_setup_parts.append(import_line)
        test_body = "\n".join(row["test_list"]).strip()
        if test_setup_parts:
            pass_references.append("\n".join(test_setup_parts + [test_body]))
        else:
            pass_references.append(test_body)

    return compute_codegen_metrics(
        all_predictions=all_predictions,
        first_predictions=first_predictions,
        codebleu_references=codebleu_references,
        pass_references=pass_references,
        pass_k_values=pass_k,
    )


def save_results(results: dict[str, Any], output_dir: Path, model_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "__").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{safe_name}_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    pass_k = parse_pass_k(args.pass_k)
    model, tokenizer, device = _load_model_and_tokenizer(args.model, args.trust_remote_code)

    results: dict[str, Any] = {
        "model": args.model,
        "benchmark": args.benchmark,
        "pass_k": pass_k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "generated_at": datetime.now().isoformat(),
        "metrics": {},
    }

    if args.benchmark in {"all", "humaneval"}:
        humaneval_metrics = evaluate_humaneval(
            model=model,
            tokenizer=tokenizer,
            device=device,
            pass_k=pass_k,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        results["metrics"]["humaneval"] = humaneval_metrics
        print(summarize_metrics("HumanEval", humaneval_metrics))

    if args.benchmark in {"all", "mbpp"}:
        mbpp_metrics = evaluate_mbpp(
            model=model,
            tokenizer=tokenizer,
            device=device,
            pass_k=pass_k,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        results["metrics"]["mbpp"] = mbpp_metrics
        print(summarize_metrics("MBPP", mbpp_metrics))

    out_path = save_results(
        results=results,
        output_dir=(PROJECT_ROOT / args.output_dir).resolve(),
        model_name=args.model,
    )
    print(f"Saved evaluation results to: {out_path}")


if __name__ == "__main__":
    main()

