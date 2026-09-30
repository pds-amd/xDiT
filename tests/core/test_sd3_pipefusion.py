"""Behavior-level SD3 PipeFusion configuration tests."""

from types import SimpleNamespace

import pytest

from xfuser.model_executor.models.runner_models import base_model
from xfuser.model_executor.models.runner_models.stable_diffusion import (
    xFuserStableDiffusionModel,
)


def test_sd3_rejects_unimplemented_stage_local_replicated_load(monkeypatch):
    model = object.__new__(xFuserStableDiffusionModel)
    config = SimpleNamespace(
        pipefusion_parallel_degree=2,
        memory_efficient_replicated_load=True,
    )
    monkeypatch.setattr(
        base_model.xFuserModel,
        "_validate_config",
        lambda *_args: None,
    )

    with pytest.raises(ValueError, match="does not support"):
        model._validate_config(config)


def test_sd3_allows_regular_pipefusion_load(monkeypatch):
    model = object.__new__(xFuserStableDiffusionModel)
    config = SimpleNamespace(
        pipefusion_parallel_degree=2,
        memory_efficient_replicated_load=False,
    )
    monkeypatch.setattr(
        base_model.xFuserModel,
        "_validate_config",
        lambda *_args: None,
    )

    model._validate_config(config)
