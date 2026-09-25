"""Dependency-light checks for Z-Image PipeFusion wiring."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "xfuser/model_executor/models/runner_models/z_image.py"
TRANSFORMER = (
    ROOT / "xfuser/model_executor/models/transformers/transformer_z_image.py"
)
PIPELINE = ROOT / "xfuser/model_executor/pipelines/pipeline_z_image.py"
RUNTIME_STATE = ROOT / "xfuser/core/distributed/runtime_state.py"


def _classes(path):
    return {
        node.name: node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef)
    }


def _method(node, name):
    return next(
        child
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        and child.name == name
    )


def _source(node):
    return ast.unparse(node)


def test_z_image_runners_enable_pipefusion_and_stage_local_loading():
    classes = _classes(RUNNER)
    for name in ("xFuserZImageModel", "xFuserZImageTurboModel"):
        runner = classes[name]
        capabilities = next(
            node.value
            for node in runner.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "capabilities"
                for target in node.targets
            )
        )
        capability_values = {
            keyword.arg: keyword.value.value
            for keyword in capabilities.keywords
            if isinstance(keyword.value, ast.Constant)
        }
        assert capability_values["pipefusion_parallel_degree"] is True
        assert capability_values["use_parallel_vae"] is True
        if name == "xFuserZImageTurboModel":
            assert "supports_step_caching" not in capability_values

        load_source = _source(_method(runner, "_load_model"))
        assert "plan_pipefusion_components" in load_source
        assert "ZImageTransformer2DModel" in load_source
        assert "mark_pipeline_stage_blockwise" in load_source

        validation_source = _source(_method(runner, "_validate_args"))
        assert "ulysses_degree" in validation_source
        assert "ring_degree" in validation_source


def test_z_image_transformer_slices_all_phases_and_caches_spatial_kv():
    classes = _classes(TRANSFORMER)
    wrapper = classes["xFuserZImagePipeFusionTransformerWrapper"]
    assert ast.literal_eval(
        next(
            node.value
            for node in wrapper.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "transformer_blocks_name"
                for target in node.targets
            )
        )
    ) == ["noise_refiner", "context_refiner", "layers"]

    forward_source = _source(_method(wrapper, "forward"))
    assert "is_pipeline_first_stage()" in forward_source
    assert "is_pipeline_last_stage()" in forward_source
    assert "patch_start_height" in forward_source
    assert "_xfuser_checkpoint_fqn" in _source(
        _method(wrapper, "_starts_at_zero")
    )

    processor = classes["xFuserZImagePipeFusionAttnProcessor"]
    processor_source = _source(_method(processor, "__call__"))
    assert "update_and_get_kv_cache" in processor_source
    assert "_xfuser_image_tokens" in processor_source
    assert "caption_key" in processor_source


def test_z_image_pipeline_has_patch_schedule_cfg_and_parallel_vae():
    pipeline = _classes(PIPELINE)["xFuserZImagePipeline"]
    call_source = _source(_method(pipeline, "__call__"))
    assert "_sync_pipeline" in call_source
    assert "_async_pipeline" in call_source
    assert "get_classifier_free_guidance_world_size" in call_source
    assert "gather_latents_for_vae" in call_source
    assert "gather_broadcast_latents" in call_source

    async_source = _source(_method(pipeline, "_async_pipeline"))
    assert "pp_patches_start_end_idx_global" in async_source
    assert "PipeFusionPatchLayout.from_runtime_state" in async_source
    assert "PipeFusionTransport" in async_source
    assert "state._reset_recv_buffer()" in _source(
        _method(pipeline, "_align_patch_metadata")
    )

    cfg_source = _source(_method(pipeline, "_combine_cfg"))
    assert "get_cfg_group().all_gather" in cfg_source
    assert "guidance_scale" in cfg_source


def test_z_image_dbcache_declares_every_stage_block_group():
    source = RUNNER.read_text()
    for block_name in ("noise_refiner", "context_refiner", "layers"):
        assert f'("{block_name}", "Pattern_3")' in source


def test_z_image_runtime_uses_scalar_patch_size():
    source = RUNTIME_STATE.read_text()
    assert 'removeprefix("xFuser")' in source
    assert 'pipeline_name.startswith("ZImage")' in source
    assert "all_patch_size[0]" in source
