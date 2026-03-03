from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from peft import OFTConfig, TaskType, get_peft_model


FORBIDDEN_MODULE_PATTERNS = ("lm_head", "output_layer")


@dataclass
class OFTApplyArguments:
    oft_rank: int
    oft_block_size: int
    oft_target: list[str]
    module_dropout: float


def find_all_linear_modules(model: Any) -> list[str]:
    """Find linear module suffixes, excluding output heads."""
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if any(pattern in module_name for pattern in FORBIDDEN_MODULE_PATTERNS):
            continue

        is_linear = isinstance(module, torch.nn.Linear)
        is_conv1d_like = module.__class__.__name__ == "Conv1D"
        if is_linear or is_conv1d_like:
            names.add(module_name.split(".")[-1])
    return sorted(names)


def resolve_oft_targets(model: Any, raw_targets: list[str]) -> list[str]:
    if len(raw_targets) == 1 and raw_targets[0] == "all":
        targets = find_all_linear_modules(model)
        if not targets:
            raise ValueError("No linear modules found for OFT target=all.")
        return targets
    if not raw_targets:
        raise ValueError("`oft_target` resolved to empty list.")
    return raw_targets


def apply_oft_adapter(model: Any, args: OFTApplyArguments) -> Any:
    target_modules = resolve_oft_targets(model, args.oft_target)
    peft_config = OFTConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.oft_rank,
        oft_block_size=args.oft_block_size,
        target_modules=target_modules,
        module_dropout=args.module_dropout,
    )
    return get_peft_model(model, peft_config)

