"""Dependency-light checks for Wan2.2 TI2V PipeFusion wiring."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "xfuser/model_executor/models/runner_models/wan.py"
TRANSFORMER = ROOT / "xfuser/model_executor/models/transformers/transformer_wan.py"
PIPELINE = ROOT / "xfuser/model_executor/pipelines/pipeline_wan_pipefusion.py"
RUNTIME = ROOT / "xfuser/core/distributed/runtime_state.py"


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
        if isinstance(child, ast.FunctionDef) and child.name == name
    )


def _source(node):
    return ast.unparse(node)


def test_ti2v_runner_enables_pp_cfg_and_stage_local_loading():
    runner = _classes(RUNNER)["xFuserWan22TI2VModel"]
    capabilities = next(
        node.value
        for node in runner.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "capabilities"
            for target in node.targets
        )
    )
    values = {
        keyword.arg: keyword.value.value
        for keyword in capabilities.keywords
        if isinstance(keyword.value, ast.Constant)
    }
    assert values["pipefusion_parallel_degree"] is True
    assert values["use_cfg_parallel"] is True
    assert values["use_parallel_vae"] is True

    load = _source(_method(runner, "_load_model"))
    assert "plan_pipefusion_components" in load
    assert "WanTransformer3DModel" in load
    assert "mark_pipeline_stage_blockwise" in load

    validation = _source(_method(runner, "_validate_config"))
    assert "config.task != 'i2v'" in validation
    assert "config.ulysses_degree > 1" in validation
    assert "config.ring_degree > 1" in validation


def test_wan_transformer_has_stage_slicing_global_rope_and_stale_kv():
    classes = _classes(TRANSFORMER)
    wrapper = classes["xFuserWanPipeFusionTransformerWrapper"]
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
    ) == ["blocks"]
    forward = _source(_method(wrapper, "forward"))
    assert "is_pipeline_first_stage()" in forward
    assert "is_pipeline_last_stage()" in forward
    assert "pipeline_hidden_states" in forward
    rope = _source(_method(wrapper, "_rotary_emb"))
    assert "patch_start_height" in rope
    assert "row:row+pph" in rope.replace(" ", "")

    processor = classes["xFuserWanPipeFusionAttnProcessor"]
    processor_source = _source(_method(processor, "__call__"))
    assert "update_and_get_kv_cache" in processor_source
    assert "slice_dim=1" in processor_source


def test_wan_pipeline_keeps_temporal_extent_and_supports_pp_cfg_vae():
    pipeline = _classes(PIPELINE)["xFuserWanTI2VPipeFusionPipeline"]
    async_source = _source(_method(pipeline, "_async_pipeline"))
    assert "split_sizes=heights" in async_source
    assert "split_dim=3" in async_source
    assert "PipeFusionPatchLayout.from_runtime_state" in async_source
    assert "condition_patches = layout.split(condition)" in async_source
    assert "mask_patches = layout.split(first_frame_mask)" in async_source
    assert "PipeFusionTransport" in async_source

    backbone = _source(_method(pipeline, "_backbone_forward"))
    assert "get_cfg_group().all_gather" in backbone
    assert "cache_context(cache_key)" in backbone
    call = _source(_method(pipeline, "__call__"))
    assert "check_to_use_naive_forward" in call
    assert "set_video_input_parameters" in call
    assert "gather_latents_for_vae" in call
    assert "gather_broadcast_latents" in call


def test_wan_runtime_multiplies_spatial_patch_tokens_by_temporal_tokens():
    runtime = _classes(RUNTIME)["DiTRuntimeState"]
    source = _source(_method(runtime, "_calc_wan_patches_metadata"))
    assert "_calc_cogvideox_patches_metadata" in source
    assert "temporal_tokens" in source
    assert "value * temporal_tokens" in source
