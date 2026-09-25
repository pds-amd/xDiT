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

    assert "CombinedTensorPayloadCodec" in source
    assert "PipeFusionTransport" in source
    assert "PipeFusionAsyncDriver" in source
    assert "PipeFusionAsyncCallbacks" in source
    assert "payload_codec.unpack" in source
    assert "payload_codec.pack" in source
    assert "step_condition[0] = next_encoder_hidden_states" in source
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
