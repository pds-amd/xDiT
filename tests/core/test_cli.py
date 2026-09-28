import pytest

from xfuser.cli import get_nproc_from_args
from xfuser.config.args import FlexibleArgumentParser, xFuserArgs
from xfuser.model_executor.models.runner_models.base_model import xFuserModel


def test_text_encoder_tp_reuses_model_ranks():
    args = [
        "--ulysses_degree",
        "8",
        "--text_encoder_tp_degree",
        "8",
    ]

    assert get_nproc_from_args(args) == 8


def test_runner_exposes_pipefusion_runtime_options():
    parser = FlexibleArgumentParser()
    args = xFuserArgs.add_runner_args(parser).parse_args([
        "--model",
        "black-forest-labs/FLUX.1-dev",
        "--pipefusion_parallel_degree",
        "2",
        "--num_pipeline_patch",
        "4",
        "--pipefusion_sync_steps",
        "3",
        "--disable_pipefusion_image_query_only",
        "--attn_layer_num_for_pp",
        "20",
        "37",
    ])

    assert args.num_pipeline_patch == 4
    assert args.warmup_steps == 3
    assert args.disable_pipefusion_image_query_only is True
    assert args.attn_layer_num_for_pp == [20, 37]


def test_runner_deprecates_warmup_steps_alias():
    parser = FlexibleArgumentParser()
    with pytest.warns(FutureWarning, match="--pipefusion_sync_steps"):
        args = xFuserArgs.add_runner_args(parser).parse_args([
            "--model",
            "black-forest-labs/FLUX.1-dev",
            "--warmup_steps",
            "3",
        ])

    assert args.warmup_steps == 3


def test_runner_accepts_targeted_fsdp_components():
    parser = FlexibleArgumentParser()
    args = xFuserArgs.add_runner_args(parser).parse_args([
        "--model",
        "black-forest-labs/FLUX.2-dev",
        "--fully_shard_degree",
        "2",
        "--fully_shard_components",
        "text_encoder",
    ])

    assert args.fully_shard_components == ["text_encoder"]


def test_targeted_fsdp_requires_a_sharding_degree():
    with pytest.raises(ValueError, match="fully_shard_degree"):
        xFuserArgs(fully_shard_components=["text_encoder"])


def test_pipefusion_requires_a_sync_step_to_initialize_kv_cache():
    config = xFuserArgs(
        pipefusion_parallel_degree=2,
        warmup_steps=0,
    )

    with pytest.raises(ValueError, match="at least one synchronous"):
        xFuserModel._validate_config(object(), config)
