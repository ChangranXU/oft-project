from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset

from .args import DataArguments
from .template import IGNORE_INDEX, normalize_user_prompt, tokenize_sft_example


def _load_dataset_registry(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    registry_path = dataset_dir / "dataset_info.json"
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Dataset registry not found: {registry_path}. "
            "Please create data/dataset_info.json."
        )
    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)
    if not isinstance(registry, dict):
        raise ValueError("dataset_info.json must contain a top-level dictionary.")
    return registry


def _pick_field(example: dict[str, Any], names: list[str], default: str = "") -> str:
    for name in names:
        if name in example and example[name] is not None:
            return str(example[name])
    return default


def _build_examples_mapper(spec: dict[str, Any]):
    columns = spec.get("columns", {})
    instruction_field = columns.get("instruction", "instruction")
    input_field = columns.get("input", columns.get("query", "input"))
    output_field = columns.get("output", "output")

    def _mapper(example: dict[str, Any]) -> dict[str, str]:
        instruction = _pick_field(example, [instruction_field, "instruction", "prompt", "question"])
        input_text = _pick_field(example, [input_field, "input", "query"])
        output = _pick_field(example, [output_field, "output", "response", "answer", "completion"])
        user_prompt = normalize_user_prompt(instruction=instruction, input_text=input_text)
        return {
            "user_prompt": user_prompt,
            "assistant_response": output.strip(),
        }

    return _mapper


def _tokenize_batch(batch: dict[str, list[Any]], tokenizer: Any, cutoff_len: int) -> dict[str, list[list[int]]]:
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    attention_masks: list[list[int]] = []
    for user_prompt, assistant_response in zip(batch["user_prompt"], batch["assistant_response"]):
        tokenized = tokenize_sft_example(
            tokenizer=tokenizer,
            user_prompt=str(user_prompt),
            assistant_response=str(assistant_response),
            cutoff_len=cutoff_len,
        )
        input_ids.append(tokenized["input_ids"])
        labels.append(tokenized["labels"])
        attention_masks.append(tokenized["attention_mask"])
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_masks}


def _has_supervision(labels: list[int]) -> bool:
    return any(label != IGNORE_INDEX for label in labels)


def load_and_prepare_sft_datasets(
    data_args: DataArguments,
    tokenizer: Any,
    project_root: Path,
) -> tuple[Dataset, Dataset | None]:
    dataset_dir = (project_root / data_args.dataset_dir).resolve()
    registry = _load_dataset_registry(dataset_dir)
    workers = data_args.preprocessing_num_workers

    tokenized_datasets: list[Dataset] = []
    for dataset_name in data_args.dataset_list:
        if dataset_name not in registry:
            known = ", ".join(sorted(registry.keys()))
            raise KeyError(f"Dataset `{dataset_name}` not found in dataset_info.json. Known datasets: {known}")

        spec = registry[dataset_name]
        file_name = spec.get("file_name")
        if not file_name:
            raise ValueError(f"Dataset `{dataset_name}` is missing `file_name` in dataset_info.json.")

        data_file = (dataset_dir / file_name).resolve()
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_file}")

        dataset = load_dataset("json", data_files=str(data_file), split="train")
        if data_args.max_samples is not None:
            max_samples = min(data_args.max_samples, len(dataset))
            dataset = dataset.select(range(max_samples))

        mapper = _build_examples_mapper(spec)
        dataset = dataset.map(
            mapper,
            remove_columns=dataset.column_names,
            num_proc=workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Formatting {dataset_name}",
        )
        dataset = dataset.filter(
            lambda row: bool(str(row["user_prompt"]).strip()) and bool(str(row["assistant_response"]).strip()),
            num_proc=workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Filtering empty rows {dataset_name}",
        )
        dataset = dataset.map(
            lambda batch: _tokenize_batch(batch, tokenizer, data_args.cutoff_len),
            batched=True,
            num_proc=None,
            remove_columns=dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Tokenizing {dataset_name}",
        )
        dataset = dataset.filter(
            lambda row: _has_supervision(row["labels"]),
            num_proc=workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Filtering unsupervised rows {dataset_name}",
        )
        tokenized_datasets.append(dataset)

    if not tokenized_datasets:
        raise ValueError("No datasets loaded. Please check `dataset` and dataset files.")

    if len(tokenized_datasets) == 1:
        merged = tokenized_datasets[0]
    else:
        merged = concatenate_datasets(tokenized_datasets)

    merged = merged.shuffle(seed=data_args.seed)
    if data_args.val_size <= 0:
        return merged, None

    if 0 < data_args.val_size < 1:
        split = merged.train_test_split(test_size=data_args.val_size, seed=data_args.seed)
        return split["train"], split["test"]

    val_count = int(data_args.val_size)
    if val_count <= 0 or val_count >= len(merged):
        raise ValueError(f"val_size={data_args.val_size} is invalid for dataset size {len(merged)}.")

    split = merged.train_test_split(test_size=val_count, seed=data_args.seed)
    return split["train"], split["test"]

