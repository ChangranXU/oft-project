from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal


def _split_csv(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def _to_report_to(value: str | list[str] | None) -> list[str]:
    items = _split_csv(value)
    return items or ["none"]


@dataclass
class ModelArguments:
    model_name_or_path: str
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = True
    flash_attention_2: bool = False
    attn_implementation: str | None = None


@dataclass
class MethodArguments:
    stage: Literal["sft"] = "sft"
    do_train: bool = True
    finetuning_type: Literal["oft"] = "oft"
    oft_rank: int = 0
    oft_block_size: int = 32
    oft_target: str | list[str] = "all"
    module_dropout: float = 0.0

    @property
    def oft_target_list(self) -> list[str]:
        return _split_csv(self.oft_target)


@dataclass
class DataArguments:
    dataset: str | list[str] | None = None
    dataset_dir: str = "data"
    template: str = "qwen"
    cutoff_len: int = 4096
    overwrite_cache: bool = False
    preprocessing_num_workers: int | None = None
    max_samples: int | None = None
    val_size: float = 0.0
    seed: int = 42

    @property
    def dataset_list(self) -> list[str]:
        return _split_csv(self.dataset)


@dataclass
class OutputArguments:
    output_dir: str = "trained_models/qwen-3b-oft"
    logging_steps: int = 10
    save_strategy: Literal["no", "steps", "epoch"] = "epoch"
    save_steps: int = 500
    overwrite_output_dir: bool = False
    save_only_model: bool = False
    report_to: str | list[str] = "none"
    run_name: str | None = None
    plot_loss: bool = False

    @property
    def report_to_list(self) -> list[str]:
        return _to_report_to(self.report_to)


@dataclass
class TrainArguments:
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5.0e-6
    num_train_epochs: float = 3.0
    lr_scheduler_type: str = "cosine"
    warmup_steps: int | None = None
    warmup_ratio: float = 0.0
    bf16: bool = False
    fp16: bool = False
    ddp_timeout: int = 1800
    resume_from_checkpoint: str | None = None
    gradient_checkpointing: bool = False
    deepspeed: str | None = None
    per_device_eval_batch_size: int = 1
    eval_strategy: str | None = None
    eval_steps: int | None = None


@dataclass
class SwanLabArguments:
    swanlab_project: str = "oft-project"
    swanlab_workspace: str | None = None
    swanlab_run_name: str | None = None
    swanlab_mode: Literal["cloud", "local"] = "cloud"
    swanlab_api_key: str | None = None
    swanlab_logdir: str | None = None
    swanlab_lark_webhook_url: str | None = None
    swanlab_lark_secret: str | None = None


@dataclass
class AppConfig:
    model: ModelArguments
    method: MethodArguments
    data: DataArguments
    output: OutputArguments
    train: TrainArguments
    swanlab: SwanLabArguments
    raw_config: dict[str, Any]


def _from_dict(cls: type[Any], values: dict[str, Any]) -> Any:
    cls_fields = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in values.items() if k in cls_fields}
    return cls(**kwargs)


def build_app_config(raw_cfg: dict[str, Any]) -> AppConfig:
    model = _from_dict(ModelArguments, raw_cfg)
    method = _from_dict(MethodArguments, raw_cfg)
    data = _from_dict(DataArguments, raw_cfg)
    output = _from_dict(OutputArguments, raw_cfg)
    train = _from_dict(TrainArguments, raw_cfg)
    swanlab = _from_dict(SwanLabArguments, raw_cfg)

    # Compatibility aliases inspired by LlamaFactory conventions.
    if "flash_attn" in raw_cfg and "flash_attention_2" not in raw_cfg:
        value = str(raw_cfg["flash_attn"]).lower()
        model.flash_attention_2 = value in {"true", "1", "flash_attention_2"}

    if "evaluation_strategy" in raw_cfg and train.eval_strategy is None:
        train.eval_strategy = raw_cfg["evaluation_strategy"]

    if method.stage != "sft":
        raise ValueError("This project currently supports stage=sft only.")
    if method.finetuning_type != "oft":
        raise ValueError("This project currently supports finetuning_type=oft only.")
    if method.oft_rank < 0:
        raise ValueError("`oft_rank` must be >= 0.")
    if method.oft_block_size < 0:
        raise ValueError("`oft_block_size` must be >= 0.")
    if method.oft_rank == 0 and method.oft_block_size == 0:
        raise ValueError("Either `oft_rank` or `oft_block_size` must be non-zero.")
    if method.oft_rank != 0 and method.oft_block_size != 0:
        raise ValueError(
            "Specify only one of `oft_rank` or `oft_block_size` (set the other to 0) to satisfy PEFT OFT constraints."
        )
    if not (0.0 <= method.module_dropout < 1.0):
        raise ValueError("`module_dropout` must be in [0.0, 1.0).")
    if data.cutoff_len <= 0:
        raise ValueError("`cutoff_len` must be a positive integer.")
    if data.val_size < 0:
        raise ValueError("`val_size` must be >= 0.")
    if not data.dataset_list:
        raise ValueError("Please provide `dataset` in config, e.g. CodeAlpaca-20k,Code-Evol-Instruct-OSS.")

    return AppConfig(
        model=model,
        method=method,
        data=data,
        output=output,
        train=train,
        swanlab=swanlab,
        raw_config=raw_cfg,
    )

