from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_yaml_with_overrides(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config and merge CLI overrides like key=value."""
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    base_cfg = OmegaConf.load(cfg_path)
    override_cfg = OmegaConf.from_cli(overrides or [])
    merged = OmegaConf.merge(base_cfg, override_cfg)
    container = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(container, dict):
        raise ValueError("Config must resolve to a dictionary.")
    return container


def dump_resolved_config(config: dict[str, Any], output_path: str | Path) -> None:
    """Save resolved config to YAML for reproducibility."""
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = OmegaConf.to_yaml(OmegaConf.create(config), resolve=True)
    out_path.write_text(yaml_text, encoding="utf-8")

