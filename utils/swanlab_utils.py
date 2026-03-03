from __future__ import annotations

from typing import Any

from transformers import TrainerCallback

from .args import SwanLabArguments


class SwanLabConfigCallback(TrainerCallback):
    """Log run configuration to SwanLab once training starts."""

    def __init__(self, flat_config: dict[str, Any]) -> None:
        self.flat_config = flat_config

    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        if not state.is_world_process_zero:
            return
        import swanlab  # type: ignore

        swanlab.config.update(self.flat_config)


def prepare_swanlab(
    report_to: list[str],
    swanlab_args: SwanLabArguments,
    flat_config: dict[str, Any],
) -> tuple[list[str], list[TrainerCallback]]:
    """Build SwanLab callbacks and remove `swanlab` from TrainingArguments.report_to."""
    if "swanlab" not in report_to:
        return report_to, []

    try:
        import swanlab  # type: ignore
        from swanlab.integration.transformers import SwanLabCallback  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SwanLab logging requested but `swanlab` is not installed. "
            "Install dependencies from requirements.txt."
        ) from exc

    if swanlab_args.swanlab_api_key:
        swanlab.login(api_key=swanlab_args.swanlab_api_key)

    callback = SwanLabCallback(
        project=swanlab_args.swanlab_project,
        workspace=swanlab_args.swanlab_workspace,
        experiment_name=swanlab_args.swanlab_run_name,
        mode=swanlab_args.swanlab_mode,
        logdir=swanlab_args.swanlab_logdir,
        config={"Framework": "oft-project"},
        tags=["OFT", "SFT"],
    )
    report_to_filtered = [item for item in report_to if item != "swanlab"]
    if not report_to_filtered:
        report_to_filtered = ["none"]

    callbacks: list[TrainerCallback] = [callback, SwanLabConfigCallback(flat_config=flat_config)]
    return report_to_filtered, callbacks

