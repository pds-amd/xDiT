import ast
from pathlib import Path

import pytest
import torch

import xfuser.model_executor.pipelines.base_pipeline as base_pipeline
from xfuser.model_executor.pipelines.base_pipeline import (
    xFuserPipelineBaseWrapper,
)
from xfuser.model_executor.pipefusion import (
    PipeFusionStageOutputCache,
    normalize_pipefusion_scm_mask,
    pipefusion_async_computation_mask,
)

ROOT = Path(__file__).resolve().parents[2]


def test_normalize_scm_isolates_cache_steps_and_computes_edges():
    assert normalize_pipefusion_scm_mask(
        (0, 0, 0, 1, 0, 0, 0, 1),
        8,
    ) == (1, 0, 1, 1, 0, 1, 0, 1)


def test_async_schedule_slices_warmup_and_forces_fresh_predecessor():
    pipeline = type(
        "_Pipeline",
        (),
        {"_xdit_pipefusion_scm_mask": (1, 0, 1, 0, 1)},
    )()

    assert pipefusion_async_computation_mask(pipeline, 5, 1) == (1, 1, 0, 1)


def test_pipeline_without_scm_uses_all_compute_schedule():
    assert pipefusion_async_computation_mask(object(), 5, 2) == (1, 1, 1)


def test_stage_output_cache_reuses_arbitrary_patch_payload():
    cache = PipeFusionStageOutputCache((1, 0), num_patches=2)
    calls = []

    first = cache.resolve(0, 1, lambda: ("stage", calls.append(True)))
    second = cache.resolve(1, 1, lambda: pytest.fail("must not recompute"))

    assert first == second == ("stage", None)
    assert calls == [True]


def test_stage_output_cache_rejects_missing_predecessor():
    cache = PipeFusionStageOutputCache((0,), num_patches=1)

    with pytest.raises(RuntimeError, match="no computed predecessor"):
        cache.resolve(0, 0, lambda: None)


def test_base_async_init_queues_per_patch_side_channels(monkeypatch):
    class _Pipeline(xFuserPipelineBaseWrapper):
        def __call__(self):
            raise NotImplementedError

    class _State:
        num_pipeline_patch = 2
        pp_patches_height = [1, 1]

        def set_patched_mode(self, patch_mode):
            self.patch_mode = patch_mode

    class _Group:
        def __init__(self):
            self.tasks = []

        def add_pipeline_recv_task(self, idx, name="latent"):
            self.tasks.append((name, idx))

    state = _State()
    group = _Group()
    monkeypatch.setattr(base_pipeline, "get_runtime_state", lambda: state)
    monkeypatch.setattr(base_pipeline, "get_pp_group", lambda: group)
    monkeypatch.setattr(base_pipeline, "is_pipeline_first_stage", lambda: False)
    monkeypatch.setattr(base_pipeline, "is_pipeline_last_stage", lambda: False)
    pipeline = object.__new__(_Pipeline)

    pipeline._init_async_pipeline(
        num_timesteps=1,
        latents=torch.zeros(1),
        num_pipeline_warmup_steps=0,
        per_patch_recv_segments=("encoder_hidden_states",),
    )

    assert group.tasks == [
        ("encoder_hidden_states", 0),
        ("latent", 0),
        ("encoder_hidden_states", 1),
        ("latent", 1),
    ]


@pytest.mark.parametrize(
    ("relative_path", "class_name"),
    [
        ("pipeline_flux.py", "xFuserFluxPipeline"),
        ("pipeline_flux2.py", "xFuserFlux2PipelineBase"),
        ("pipeline_stable_diffusion_3.py", "xFuserStableDiffusion3Pipeline"),
        ("pipeline_qwen_image.py", "xFuserQwenImagePipelineBase"),
        ("pipeline_z_image.py", "xFuserZImagePipeline"),
        ("pipeline_wan_pipefusion.py", "xFuserWanTI2VPipeFusionPipeline"),
    ],
)
def test_rdna4_async_pipelines_use_shared_driver_and_stage_output_cache(
    relative_path,
    class_name,
):
    path = ROOT / "xfuser/model_executor/pipelines" / relative_path
    module = ast.parse(path.read_text())
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    async_method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_async_pipeline"
    )
    async_calls = {
        node.func.attr
        for node in ast.walk(async_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    class_calls = {
        node.func.attr
        for node in ast.walk(class_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    async_functions = {
        node.func.id
        for node in ast.walk(async_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
    }
    assert "_pipefusion_async_computation_mask" in class_calls
    assert "_pipefusion_stage_output_cache" in async_calls
    assert "PipeFusionAsyncDriver" in async_functions
    assert "PipeFusionAsyncCallbacks" in async_functions
