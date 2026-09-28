"""Dependency-light checks for SD3 PipeFusion payload ordering."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = (
    ROOT
    / "xfuser/model_executor/pipelines/pipeline_stable_diffusion_3.py"
)
RUNNER = ROOT / "xfuser/model_executor/models/runner_models/stable_diffusion.py"


def _source(node):
    return ast.unparse(node)


def test_sd3_async_pipeline_combines_image_and_text_payloads():
    module = ast.parse(PIPELINE.read_text())
    pipeline = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "xFuserStableDiffusion3Pipeline"
    )
    async_pipeline = next(
        node
        for node in pipeline.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_async_pipeline"
    )
    source = _source(async_pipeline)

    assert "PipeFusionImagePatchSchedule" in source
    assert "condition_reuse=True" in source
    assert 'name="encoder_hidden_states"' not in source


def test_sd3_pipefusion_preserves_fp16_runtime_contract():
    module = ast.parse(RUNNER.read_text())
    runner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "xFuserStableDiffusionModel"
    )
    load_model = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_model"
    )
    source = _source(load_model)

    assert "torch.float16 if self.config.pipefusion_parallel_degree > 1" in source


def test_sd3_rejects_unimplemented_stage_local_replicated_load():
    module = ast.parse(RUNNER.read_text())
    runner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "xFuserStableDiffusionModel"
    )
    validator = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_config"
    )
    source = _source(validator)

    assert "config.pipefusion_parallel_degree > 1" in source
    assert "config.memory_efficient_replicated_load" in source
    assert "does not support" in source
