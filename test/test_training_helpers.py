from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import train_sft
from utils.args import build_app_config


class FakeTrainingArguments:
    __dataclass_fields__ = {
        "output_dir": None,
        "overwrite_output_dir": None,
        "logging_steps": None,
        "save_strategy": None,
        "save_steps": None,
        "per_device_train_batch_size": None,
        "per_device_eval_batch_size": None,
        "gradient_accumulation_steps": None,
        "learning_rate": None,
        "num_train_epochs": None,
        "lr_scheduler_type": None,
        "bf16": None,
        "fp16": None,
        "gradient_checkpointing": None,
        "eval_steps": None,
        "report_to": None,
        "run_name": None,
        "remove_unused_columns": None,
        "dataloader_pin_memory": None,
        "seed": None,
        "do_train": None,
        "do_eval": None,
        "evaluation_strategy": None,
        "warmup_steps": None,
        "warmup_ratio": None,
    }

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _build_cfg(**overrides):
    raw = {
        "model_name_or_path": "dummy/model",
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "oft",
        "oft_rank": 8,
        "oft_block_size": 0,
        "dataset": "toy",
        "dataset_dir": "data",
        "cutoff_len": 128,
        "output_dir": "trained_models/test",
        "overwrite_output_dir": False,
        "report_to": "none",
        "num_train_epochs": 2.0,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5,
        "warmup_ratio": 0.25,
        "warmup_steps": None,
        "eval_strategy": "steps",
        "eval_steps": 10,
    }
    raw.update(overrides)
    return build_app_config(raw)


def test_build_training_args_disables_eval_when_no_eval_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_sft, "TrainingArguments", FakeTrainingArguments)
    cfg = _build_cfg(eval_strategy="steps")
    args = train_sft.build_training_args(
        cfg=cfg,
        output_dir="out",
        report_to=["none"],
        has_eval_dataset=False,
        train_dataset_size=100,
    )
    assert args.do_eval is False
    assert args.evaluation_strategy == "no"
    assert args.report_to == []


def test_build_training_args_computes_warmup_steps_from_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_sft, "TrainingArguments", FakeTrainingArguments)
    cfg = _build_cfg(warmup_steps=None, warmup_ratio=0.25, num_train_epochs=2.0)
    args = train_sft.build_training_args(
        cfg=cfg,
        output_dir="out",
        report_to=["none"],
        has_eval_dataset=True,
        train_dataset_size=64,
    )
    assert args.warmup_steps == 4


def test_build_app_config_rejects_deepspeed_override() -> None:
    with pytest.raises(ValueError, match="DeepSpeed is no longer supported"):
        _build_cfg(deepspeed="deepspeed/ds_z2_config.json")


def test_prepare_output_dir_raises_for_non_empty_without_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        train_sft.prepare_output_dir(
            output_dir=str(output_dir),
            overwrite_output_dir=False,
            do_train=True,
            resume_from_checkpoint=None,
        )


def test_prepare_output_dir_overwrite_cleans_existing_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    nested_dir = output_dir / "nested"
    nested_dir.mkdir(parents=True)
    (output_dir / "stale.txt").write_text("x", encoding="utf-8")
    (nested_dir / "nested.txt").write_text("y", encoding="utf-8")

    train_sft.prepare_output_dir(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        do_train=True,
        resume_from_checkpoint=None,
    )

    assert output_dir.exists()
    assert list(output_dir.iterdir()) == []


def test_align_model_special_tokens_updates_config_and_generation_config() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(eos_token_id=2, bos_token_id=3, pad_token_id=4),
        generation_config=SimpleNamespace(eos_token_id=[9], bos_token_id=1, pad_token_id=None),
    )
    tokenizer = SimpleNamespace(eos_token_id=5, bos_token_id=6, pad_token_id=7)

    updated = train_sft.align_model_special_tokens(model, tokenizer)

    assert updated == {"eos_token_id": 5, "bos_token_id": 6, "pad_token_id": 7}
    assert model.config.eos_token_id == 5
    assert model.config.bos_token_id == 6
    assert model.config.pad_token_id == 7
    assert model.generation_config.eos_token_id == [5, 9]
    assert model.generation_config.bos_token_id == 6
    assert model.generation_config.pad_token_id == 7
