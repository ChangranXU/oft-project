from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from peft import OFTConfig, TaskType, get_peft_model


FORBIDDEN_MODULE_PATTERNS = ("lm_head", "output_layer")


@dataclass
class OFTApplyArguments:
    oft_rank: int
    oft_block_size: int
    oft_target: list[str]
    module_dropout: float


def find_all_linear_modules(model: Any) -> list[str]:
    """Find linear-like module full names, excluding output heads.

    We follow LlamaFactory-style discovery by checking class names instead of
    strict `isinstance(nn.Linear)` so quantized linear wrappers (e.g. bitsandbytes
    `Linear8bitLt`) are also discovered.

    Important: we return the *suffix* module names (e.g. "q_proj") rather than
    full dotted paths (e.g. "model.layers.0.self_attn.q_proj"). PEFT matches
    targets by exact-or-suffix logic, and using suffixes is the standard and
    most robust way to apply adapters across all layers.
    """
    forbidden_patterns = set(FORBIDDEN_MODULE_PATTERNS)
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type == "internlm2":
        forbidden_patterns.add("output")

    names: set[str] = set()
    for module_name, module in model.named_modules():
        if any(pattern in module_name for pattern in forbidden_patterns):
            continue

        class_name = module.__class__.__name__
        is_linear_like = "Linear" in class_name and "Embedding" not in class_name
        if is_linear_like:
            names.add(module_name.split(".")[-1])
    return sorted(names)


def resolve_oft_targets(model: Any, raw_targets: list[str]) -> list[str]:
    cleaned_targets: list[str] = []
    for target in raw_targets:
        value = str(target).strip()
        if value and value not in cleaned_targets:
            cleaned_targets.append(value)

    if len(cleaned_targets) == 1 and cleaned_targets[0] in {"all", "all-linear"}:
        targets = find_all_linear_modules(model)
        if not targets:
            raise ValueError("No linear modules found for OFT target=all.")
        return targets
    if not cleaned_targets:
        raise ValueError("`oft_target` resolved to empty list.")
    return cleaned_targets


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

