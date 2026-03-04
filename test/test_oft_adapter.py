from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("peft")

import utils.peft_oft as oft_module
from utils.peft_oft import OFTApplyArguments, apply_oft_adapter, find_all_linear_modules, resolve_oft_targets


def _make_instance(class_name: str):
    cls = type(class_name, (), {})
    return cls()


class DummyModel:
    def __init__(self, modules: list[tuple[str, object]], model_type: str = "qwen") -> None:
        self._modules = modules
        self.config = SimpleNamespace(model_type=model_type)

    def named_modules(self):
        return list(self._modules)


def test_find_all_linear_modules_returns_unique_suffixes() -> None:
    model = DummyModel(
        modules=[
            ("model.layers.0.self_attn.q_proj", _make_instance("Linear8bitLt")),
            ("model.layers.1.self_attn.q_proj", _make_instance("Linear")),
            ("model.layers.0.mlp.down_proj", _make_instance("Linear")),
            ("lm_head", _make_instance("Linear")),
            ("model.output_layer", _make_instance("Linear")),
            ("model.embed_tokens", _make_instance("Embedding")),
        ]
    )
    assert find_all_linear_modules(model) == ["down_proj", "q_proj"]


def test_find_all_linear_modules_excludes_output_for_internlm2() -> None:
    model = DummyModel(
        modules=[
            ("model.layers.0.output", _make_instance("Linear")),
            ("model.layers.0.self_attn.v_proj", _make_instance("Linear")),
        ],
        model_type="internlm2",
    )
    assert find_all_linear_modules(model) == ["v_proj"]


def test_resolve_oft_targets_trims_and_deduplicates() -> None:
    model = DummyModel(modules=[])
    assert resolve_oft_targets(model, [" q_proj ", "q_proj", "  ", "v_proj"]) == ["q_proj", "v_proj"]


def test_resolve_oft_targets_all_expands_discovered_modules() -> None:
    model = DummyModel(modules=[("model.layers.0.self_attn.k_proj", _make_instance("Linear"))])
    assert resolve_oft_targets(model, ["all"]) == ["k_proj"]


def test_resolve_oft_targets_all_raises_when_no_linear_modules() -> None:
    model = DummyModel(modules=[("model.embed_tokens", _make_instance("Embedding"))])
    with pytest.raises(ValueError):
        resolve_oft_targets(model, ["all"])


def test_apply_oft_adapter_builds_config_and_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeOFTConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_get_peft_model(model, peft_config):
        captured["model"] = model
        captured["peft_config"] = peft_config
        return "wrapped-model"

    monkeypatch.setattr(oft_module, "OFTConfig", FakeOFTConfig)
    monkeypatch.setattr(oft_module, "get_peft_model", fake_get_peft_model)

    model = DummyModel(modules=[("model.layers.0.self_attn.q_proj", _make_instance("Linear"))])
    args = OFTApplyArguments(
        oft_rank=16,
        oft_block_size=0,
        oft_target=[" q_proj ", "q_proj"],
        module_dropout=0.2,
    )
    result = apply_oft_adapter(model, args)

    assert result == "wrapped-model"
    config = captured["peft_config"]
    assert isinstance(config, FakeOFTConfig)
    assert config.kwargs["task_type"] == oft_module.TaskType.CAUSAL_LM
    assert config.kwargs["inference_mode"] is False
    assert config.kwargs["r"] == 16
    assert config.kwargs["oft_block_size"] == 0
    assert config.kwargs["target_modules"] == ["q_proj"]
    assert config.kwargs["module_dropout"] == 0.2
