from types import SimpleNamespace

import torch.nn as nn

from xfuser.model_executor.models.transformers.base_transformer import (
    xFuserTransformerBaseWrapper,
)


class _DummyWrapper(xFuserTransformerBaseWrapper):
    def forward(self, *args, **kwargs):
        raise NotImplementedError


def _transformer():
    transformer = nn.Module()
    transformer.blocks = nn.ModuleList([nn.Linear(1, 1) for _ in range(5)])
    return transformer


def _patch_parallel_state(monkeypatch, split):
    state = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pp_config=SimpleNamespace(attn_layer_num_for_pp=split)
        )
    )
    monkeypatch.setattr(
        "xfuser.model_executor.models.transformers.base_transformer."
        "get_runtime_state",
        lambda: state,
    )
    monkeypatch.setattr(
        "xfuser.model_executor.models.transformers.base_transformer."
        "get_pipeline_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "xfuser.model_executor.models.transformers.base_transformer."
        "get_pipeline_parallel_world_size",
        lambda: 2,
    )


def test_missing_split_warns_and_uses_naive_equal_block_fallback(monkeypatch):
    _patch_parallel_state(monkeypatch, None)
    warnings = []
    monkeypatch.setattr(
        "xfuser.model_executor.models.transformers.base_transformer.logger.warning",
        lambda message, *args: warnings.append(message % args),
    )
    wrapper = object.__new__(_DummyWrapper)

    result = wrapper._split_transformer_blocks(_transformer(), ["blocks"])

    assert len(result.blocks) == 3
    assert any("naive equal-block fallback" in message for message in warnings)


def test_explicit_split_remains_authoritative(monkeypatch):
    _patch_parallel_state(monkeypatch, [2, 3])
    warnings = []
    monkeypatch.setattr(
        "xfuser.model_executor.models.transformers.base_transformer.logger.warning",
        lambda message, *args: warnings.append(message % args),
    )
    wrapper = object.__new__(_DummyWrapper)

    result = wrapper._split_transformer_blocks(_transformer(), ["blocks"])

    assert len(result.blocks) == 2
    assert not warnings
