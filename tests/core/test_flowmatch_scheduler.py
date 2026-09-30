from types import SimpleNamespace

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

import xfuser.model_executor.schedulers.base_scheduler as base_scheduler
import xfuser.model_executor.schedulers.scheduling_flow_match_euler_discrete as flowmatch_scheduler
from xfuser.model_executor.schedulers.scheduling_flow_match_euler_discrete import (
    xFuserFlowMatchEulerDiscreteSchedulerWrapper,
)


def test_zero_churn_pipefusion_step_matches_diffusers(monkeypatch):
    monkeypatch.setattr(
        base_scheduler, "get_pipeline_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        base_scheduler, "get_sequence_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        flowmatch_scheduler,
        "get_runtime_state",
        lambda: SimpleNamespace(
            patch_mode=True,
            pipeline_patch_idx=0,
            num_pipeline_patch=2,
        ),
    )

    reference = FlowMatchEulerDiscreteScheduler()
    reference.set_timesteps(3, device="cpu")
    wrapped_module = FlowMatchEulerDiscreteScheduler()
    wrapped_module.set_timesteps(3, device="cpu")
    wrapped = xFuserFlowMatchEulerDiscreteSchedulerWrapper(wrapped_module)
    model_output = torch.tensor([1.5])
    sample = torch.tensor([2.0])
    timestep = reference.timesteps[0]

    expected = reference.step(
        model_output,
        timestep,
        sample,
        s_churn=0.0,
        return_dict=False,
    )[0]
    actual = wrapped.step(
        model_output,
        wrapped.timesteps[0],
        sample,
        s_churn=0.0,
        return_dict=False,
    )[0]

    torch.testing.assert_close(actual, expected)
