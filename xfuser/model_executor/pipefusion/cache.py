from typing import Callable, Generic, Iterable, Tuple, TypeVar


StagePayload = TypeVar("StagePayload")
_MISSING = object()


def normalize_pipefusion_scm_mask(
    mask: Iterable[int],
    num_timesteps: int,
) -> Tuple[int, ...]:
    """Return a safe mask with isolated, non-edge cache steps."""
    values = tuple(mask)
    if len(values) != num_timesteps:
        raise ValueError(
            "PipeFusion SCM mask length must equal the number of timesteps: "
            f"{len(values)} != {num_timesteps}."
        )

    normalized = []
    for index, value in enumerate(values):
        if value not in (0, 1):
            raise ValueError("PipeFusion SCM mask entries must be 0 or 1.")
        cache_step = (
            value == 0
            and index > 0
            and index < num_timesteps - 1
            and normalized[-1] == 1
        )
        normalized.append(0 if cache_step else 1)
    return tuple(normalized)


def pipefusion_async_computation_mask(
    pipeline,
    num_timesteps: int,
    warmup_steps: int,
) -> Tuple[int, ...]:
    if warmup_steps < 0 or warmup_steps > num_timesteps:
        raise ValueError(
            "PipeFusion warmup steps must be between zero and the timestep count."
        )
    mask = getattr(pipeline, "_xdit_pipefusion_scm_mask", None)
    if mask is None:
        return (1,) * (num_timesteps - warmup_steps)

    async_mask = list(
        normalize_pipefusion_scm_mask(mask, num_timesteps)[warmup_steps:]
    )
    if async_mask:
        async_mask[0] = 1
    return tuple(async_mask)


def install_pipefusion_scm_mask(pipeline, mask: Iterable[int]) -> None:
    pipeline._xdit_pipefusion_scm_mask = tuple(mask)


def supports_pipefusion_stage_cache(pipeline) -> bool:
    """Return whether a pipeline exposes the shared stage-cache contract."""
    return all(
        callable(getattr(pipeline, method, None))
        for method in (
            "_pipefusion_async_computation_mask",
            "_pipefusion_stage_output_cache",
        )
    )


class PipeFusionStageOutputCache(Generic[StagePayload]):
    """Per-patch cache for a pipeline stage's complete forward payload."""

    def __init__(self, computation_mask: Iterable[int], num_patches: int):
        self.computation_mask = tuple(computation_mask)
        if any(value not in (0, 1) for value in self.computation_mask):
            raise ValueError("PipeFusion computation mask entries must be 0 or 1.")
        if num_patches < 1:
            raise ValueError("PipeFusion requires at least one pipeline patch.")
        self._payloads = [_MISSING] * num_patches

    def should_compute(self, step_index: int) -> bool:
        return self.computation_mask[step_index] == 1

    def resolve(
        self,
        step_index: int,
        patch_index: int,
        compute: Callable[[], StagePayload],
    ) -> StagePayload:
        if self.should_compute(step_index):
            payload = compute()
            self._payloads[patch_index] = payload
            return payload

        payload = self._payloads[patch_index]
        if payload is _MISSING:
            raise RuntimeError(
                "PipeFusion SCM cache step has no computed predecessor."
            )
        return payload
