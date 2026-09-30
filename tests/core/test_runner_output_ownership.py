from unittest import mock

import xfuser.runner as runner_module
from xfuser.model_executor.models.runner_models.base_model import DiffusionOutput


def _runner():
    instance = object.__new__(runner_module.xFuserModelRunner)
    instance.model = mock.Mock()
    return instance


def _distributed(monkeypatch, *, rank: int, reduced_owner: int):
    monkeypatch.setattr(runner_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(runner_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runner_module.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(runner_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(runner_module.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(
        runner_module.dist,
        "all_reduce",
        lambda candidate, op: candidate.fill_(reduced_owner),
    )


def test_save_writes_only_the_rank_that_owns_output(monkeypatch):
    runner = _runner()
    _distributed(monkeypatch, rank=0, reduced_owner=0)
    output = DiffusionOutput(images=[object()], pipe_args=[{}])

    runner.save(output=output, timings=[1.0])

    runner.model.save_output.assert_called_once_with(output)
    runner.model.save_timings.assert_called_once_with([1.0])


def test_save_skips_non_owner_rank(monkeypatch):
    runner = _runner()
    _distributed(monkeypatch, rank=1, reduced_owner=0)
    output = DiffusionOutput(images=[object()], pipe_args=[{}])

    runner.save(output=output, timings=[1.0])

    runner.model.save_output.assert_not_called()
    runner.model.save_timings.assert_not_called()


def test_save_without_output_falls_back_to_last_rank(monkeypatch):
    runner = _runner()
    _distributed(monkeypatch, rank=1, reduced_owner=-1)

    runner.save(output=None, timings=[1.0])

    runner.model.save_output.assert_not_called()
    runner.model.save_timings.assert_called_once_with([1.0])
