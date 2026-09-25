"""Dependency-light checks for Qwen-Image PipeFusion wiring."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "xfuser/model_executor/models/runner_models/qwen.py"
TRANSFORMER = ROOT / "xfuser/model_executor/models/transformers/transformer_qwen.py"
PIPELINE = ROOT / "xfuser/model_executor/pipelines/pipeline_qwen_image.py"
CACHE_DIT = ROOT / "xfuser/model_executor/cache/adapters/cache_dit.py"


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


def test_qwen_runners_enable_pipefusion_cfg_and_stage_local_loading():
    classes = _classes(RUNNER)
    for name in ("xFuserQwenImageModel", "xFuserQwenImageEditModel"):
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
        values = {
            keyword.arg: keyword.value.value
            for keyword in capabilities.keywords
            if isinstance(keyword.value, ast.Constant)
        }
        assert values["pipefusion_parallel_degree"] is True
        assert values["use_cfg_parallel"] is True
        assert values["ulysses_degree"] is True

        load_source = _source(_method(runner, "_load_model"))
        assert "plan_pipefusion_components" in load_source
        assert "QwenImageTransformer2DModel" in load_source
        assert "mark_pipeline_stage_blockwise" in load_source
        assert "runtime_config.dtype = torch.bfloat16" in load_source

    edit_load_source = _source(
        _method(classes["xFuserQwenImageEditModel"], "_load_model")
    )
    assert "if self.config.use_cfg_parallel" in edit_load_source
    assert "xFuserQwenImageEditPipeline(pipe, self.engine_config)" in edit_load_source


def test_qwen_transformer_has_stage_gates_and_pipefusion_kv_cache():
    classes = _classes(TRANSFORMER)
    wrapper = classes["xFuserQwenImagePipeFusionTransformerWrapper"]
    source = _source(_method(wrapper, "forward"))
    assert "is_pipeline_first_stage()" in source
    assert "is_pipeline_last_stage()" in source
    assert "_xdit_pipefusion_full_img_shapes" in source
    assert "pp_patches_token_start_idx_local" in source
    assert "torch.cat(patch_image_freqs, dim=0)" in source
    assert "transformer_blocks_name" in _source(wrapper)
    assert "transformer_blocks" in _source(wrapper)

    processor = classes["xFuserQwenPipeFusionAttnProcessor"]
    processor_source = _source(_method(processor, "__call__"))
    assert "update_and_get_kv_cache" in processor_source
    assert "reference_patch_start" in processor_source
    assert "reference_patch_end" in processor_source


def test_qwen_pipeline_wraps_text_edit_cfg_and_shared_schedule():
    classes = _classes(PIPELINE)
    base = classes["xFuserQwenImagePipelineBase"]
    call_source = _source(_method(base, "__call__"))
    assert "_sync_pipeline" in call_source
    assert "_async_pipeline" in call_source
    assert "get_classifier_free_guidance_world_size" in call_source
    conversion_source = _source(
        _method(base, "_convert_transformer_backbone")
    )
    assert "xFuserQwenImageTransformerWrapper" in conversion_source

    backbone_source = _source(_method(base, "_backbone_forward"))
    assert "torch.cat([hidden_states, hidden_states], dim=0)" in backbone_source
    assert "get_cfg_group().all_gather" in backbone_source
    assert "noise_pred[:, :target_tokens]" in backbone_source

    async_source = _source(_method(base, "_async_pipeline"))
    assert "PipeFusionTransport" in async_source
    assert "PipeFusionAsyncDriver" in async_source
    assert "PipeFusionAsyncCallbacks" in async_source
    assert "CombinedTensorPayloadCodec" in async_source
    assert "full_img_shapes" in async_source
    assert "payload_codec.pack" in async_source
    assert "payload_codec.unpack" in async_source

    edit = classes["xFuserQwenImageEditPipeline"]
    edit_source = _source(edit)
    assert "calculate_dimensions" in edit_source
    assert "self.prepare_latents" in edit_source


def test_qwen_pipefusion_dbcache_uses_batched_cfg_contexts():
    module = ast.parse(CACHE_DIT.read_text())
    resolver = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_enable_separate_cfg"
    )
    source = _source(resolver)
    assert "get_pipeline_parallel_world_size() == 1" in source
