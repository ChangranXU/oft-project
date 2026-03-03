#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.args import AppConfig, build_app_config  # noqa: E402
from utils.config import dump_resolved_config, load_yaml_with_overrides  # noqa: E402
from utils.data import load_and_prepare_sft_datasets  # noqa: E402
from utils.peft_oft import OFTApplyArguments, apply_oft_adapter  # noqa: E402
from utils.swanlab_utils import prepare_swanlab  # noqa: E402
from utils.template import IGNORE_INDEX  # noqa: E402


def parse_cli() -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(description="Train OFT-SFT model with YAML config.")
    parser.add_argument(
        "config",
        nargs="?",
        default="config/qwen2_5_oft_sft.yaml",
        help="Path to YAML config file.",
    )
    known, unknown = parser.parse_known_args()
    return Path(known.config), unknown


def resolve_path(path_str: str, project_root: Path) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((project_root / path).resolve())


def build_training_args(
    cfg: AppConfig,
    output_dir: str,
    report_to: list[str],
    has_eval_dataset: bool,
) -> TrainingArguments:
    has_eval = has_eval_dataset and cfg.train.eval_strategy is not None and cfg.train.eval_strategy != "no"
    evaluation_strategy = cfg.train.eval_strategy if has_eval else "no"
    report_to_value: list[str] | str = [] if report_to == ["none"] else report_to

    deepspeed_path = cfg.train.deepspeed
    if deepspeed_path:
        deepspeed_path = resolve_path(deepspeed_path, PROJECT_ROOT)

    kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        overwrite_output_dir=cfg.output.overwrite_output_dir,
        logging_steps=cfg.output.logging_steps,
        save_strategy=cfg.output.save_strategy,
        save_steps=cfg.output.save_steps,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.num_train_epochs,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        bf16=cfg.train.bf16,
        fp16=cfg.train.fp16,
        ddp_timeout=cfg.train.ddp_timeout,
        deepspeed=deepspeed_path,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        eval_steps=cfg.train.eval_steps,
        report_to=report_to_value,
        run_name=cfg.output.run_name,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        seed=cfg.data.seed,
        do_train=cfg.method.do_train,
        do_eval=has_eval,
    )

    training_fields = TrainingArguments.__dataclass_fields__
    if "evaluation_strategy" in training_fields:
        kwargs["evaluation_strategy"] = evaluation_strategy
    else:
        kwargs["eval_strategy"] = evaluation_strategy

    # Remove None values to keep TrainingArguments clean.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return TrainingArguments(**kwargs)


def main() -> None:
    config_path, overrides = parse_cli()
    raw_cfg = load_yaml_with_overrides(config_path, overrides)
    cfg = build_app_config(raw_cfg)
    set_seed(cfg.data.seed)

    output_dir = resolve_path(cfg.output.output_dir, PROJECT_ROOT)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name_or_path,
        use_fast=cfg.model.use_fast_tokenizer,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    attn_impl = cfg.model.attn_implementation
    if cfg.model.flash_attention_2 and not attn_impl:
        attn_impl = "flash_attention_2"
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name_or_path, **model_kwargs)

    train_dataset, eval_dataset = load_and_prepare_sft_datasets(
        data_args=cfg.data,
        tokenizer=tokenizer,
        project_root=PROJECT_ROOT,
    )

    oft_args = OFTApplyArguments(
        oft_rank=cfg.method.oft_rank,
        oft_block_size=cfg.method.oft_block_size,
        oft_target=cfg.method.oft_target_list,
        module_dropout=cfg.method.module_dropout,
    )
    model = apply_oft_adapter(model, oft_args)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    report_to = cfg.output.report_to_list
    report_to, swanlab_callbacks = prepare_swanlab(
        report_to=report_to,
        swanlab_args=cfg.swanlab,
        flat_config=cfg.raw_config,
    )

    training_args = build_training_args(
        cfg=cfg,
        output_dir=output_dir,
        report_to=report_to,
        has_eval_dataset=eval_dataset is not None,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=8 if (cfg.train.bf16 or cfg.train.fp16) else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=swanlab_callbacks,
    )

    if cfg.method.do_train:
        train_result = trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        dump_resolved_config(cfg.raw_config, Path(output_dir) / "resolved_config.yaml")
        if not cfg.output.save_only_model:
            trainer.save_state()

    if eval_dataset is not None and training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    print(f"Training finished. Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()

