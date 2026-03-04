from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset

import utils.data as data_module
from utils.args import DataArguments
from utils.data import _build_examples_mapper, _load_dataset_registry, load_and_prepare_sft_datasets
from utils.template import IGNORE_INDEX


def _write_dataset_registry(dataset_dir: Path, content: object) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset_info.json").write_text(json.dumps(content), encoding="utf-8")


def test_load_dataset_registry_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_dataset_registry(tmp_path / "data")


def test_load_dataset_registry_requires_dictionary(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data"
    _write_dataset_registry(dataset_dir, ["not", "a", "dict"])
    with pytest.raises(ValueError):
        _load_dataset_registry(dataset_dir)


def test_build_examples_mapper_uses_custom_columns() -> None:
    mapper = _build_examples_mapper(
        {
            "columns": {
                "instruction": "task",
                "query": "context",
                "output": "answer_col",
            }
        }
    )
    mapped = mapper({"task": "solve", "context": "x=1", "answer_col": "  done  "})
    assert mapped["user_prompt"] == "solve\nx=1"
    assert mapped["assistant_response"] == "done"


def test_build_examples_mapper_omits_no_input_tokens() -> None:
    mapper = _build_examples_mapper({"columns": {"instruction": "instruction", "input": "input", "output": "output"}})
    mapped = mapper({"instruction": "explain", "input": "none", "output": "ok"})
    assert mapped["user_prompt"] == "explain"


def test_load_and_prepare_dataset_missing_name_raises(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets"
    _write_dataset_registry(
        dataset_dir,
        {
            "toy": {
                "file_name": "toy.json",
                "columns": {"instruction": "instruction", "input": "input", "output": "output"},
            }
        },
    )
    (dataset_dir / "toy.json").write_text("[]", encoding="utf-8")

    args = DataArguments(dataset="missing", dataset_dir="datasets")
    with pytest.raises(KeyError):
        load_and_prepare_sft_datasets(data_args=args, tokenizer=object(), project_root=tmp_path)


def test_load_and_prepare_filters_and_splits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets"
    _write_dataset_registry(
        dataset_dir,
        {
            "toy": {
                "file_name": "toy.json",
                "columns": {"instruction": "instruction", "input": "input", "output": "output"},
            }
        },
    )
    (dataset_dir / "toy.json").write_text("[]", encoding="utf-8")

    rows = [
        {"instruction": "sum two nums", "input": "a,b", "output": "ans1"},
        {"instruction": "   ", "input": "", "output": "ans2"},
        {"instruction": "unsupervised", "input": "", "output": "ans3"},
        {"instruction": "echo", "input": "none", "output": "ans4"},
    ]
    seen_prompts: list[str] = []

    def fake_load_dataset(*args, **kwargs):
        assert args[0] == "json"
        assert kwargs["split"] == "train"
        return Dataset.from_list(rows)

    def fake_tokenize_sft_example(tokenizer, user_prompt, assistant_response, cutoff_len):
        del tokenizer, cutoff_len
        seen_prompts.append(user_prompt)
        if assistant_response == "ans3":
            labels = [IGNORE_INDEX, IGNORE_INDEX]
            input_ids = [7, 8]
        else:
            labels = [IGNORE_INDEX, 42, 43]
            input_ids = [1, 2, 3]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    monkeypatch.setattr(data_module, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(data_module, "tokenize_sft_example", fake_tokenize_sft_example)

    args = DataArguments(
        dataset="toy",
        dataset_dir="datasets",
        cutoff_len=128,
        overwrite_cache=True,
        preprocessing_num_workers=None,
        val_size=0.5,
        seed=13,
    )
    train_dataset, eval_dataset = load_and_prepare_sft_datasets(
        data_args=args,
        tokenizer=object(),
        project_root=tmp_path,
    )

    assert eval_dataset is not None
    assert len(train_dataset) == 1
    assert len(eval_dataset) == 1
    assert len(train_dataset) + len(eval_dataset) == 2

    assert "sum two nums\na,b" in seen_prompts
    assert "echo" in seen_prompts

    for dataset in (train_dataset, eval_dataset):
        record = dataset[0]
        assert len(record["input_ids"]) == len(record["labels"]) == len(record["attention_mask"])
        assert any(label != IGNORE_INDEX for label in record["labels"])
