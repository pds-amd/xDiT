from typing import Callable, Generic, Iterable, Tuple, TypeVar

from xfuser.model_executor.cache.presets import (
    PipeFusionCachePlan,
    PipeFusionCacheUnit,
    PipeFusionStaticMask,
)


StagePayload = TypeVar("StagePayload")
_MISSING = object()


def build_pipefusion_scm_mask(num_timesteps: int) -> Tuple[int, ...]:
    """Return the alternating-middle static policy for a PipeFusion invocation."""
    return _build_alternating_mask(
        num_timesteps,
        first_cache_fraction=1 / 3,
        last_cache_fraction=5 / 6,
    )


def _build_alternating_mask(
    num_timesteps: int,
    *,
    first_cache_fraction: float,
    last_cache_fraction: float,
) -> Tuple[int, ...]:
    mask = [1] * num_timesteps
    first_cache_step = round((num_timesteps - 1) * first_cache_fraction)
    last_cache_step = round((num_timesteps - 1) * last_cache_fraction)
    for index in range(first_cache_step, last_cache_step + 1, 2):
        if 0 < index < num_timesteps - 1:
            mask[index] = 0
    return tuple(mask)


def build_pipefusion_static_mask(
    policy: PipeFusionStaticMask,
    num_timesteps: int,
) -> Tuple[int, ...]:
    """Build a model-owned static stage-payload reuse mask."""
    if policy is PipeFusionStaticMask.ALTERNATING_MIDDLE:
        return build_pipefusion_scm_mask(num_timesteps)
    if policy is PipeFusionStaticMask.WIDE_ALTERNATING_MIDDLE:
        return _build_alternating_mask(
            num_timesteps,
            first_cache_fraction=1 / 4,
            last_cache_fraction=11 / 12,
        )
    raise ValueError(f"Unsupported PipeFusion static mask policy: {policy!r}")


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
    total_timesteps: int,
    warmup_steps: int,
    *,
    final_stage: bool,
) -> Tuple[int, ...]:
    """Return a per-stage cache schedule for the asynchronous suffix of a PP run.

    ``total_timesteps`` includes the synchronous PipeFusion warmup prefix;
    the returned mask therefore has ``total_timesteps - warmup_steps`` entries.
    """
    if warmup_steps < 0 or warmup_steps > total_timesteps:
        raise ValueError(
            "PipeFusion warmup steps must be between zero and the timestep count."
        )
    plan = getattr(pipeline, "_xdit_pipefusion_cache_plan", None)
    if (
        plan is None
        or plan.unit is PipeFusionCacheUnit.BLOCK_LOCAL
        or (
            final_stage
            and plan.unit is PipeFusionCacheUnit.INTERMEDIATE_STAGE_OUTPUT
        )
    ):
        return (1,) * (total_timesteps - warmup_steps)
    mask = build_pipefusion_static_mask(plan.static_mask, total_timesteps)

    async_mask = list(
        normalize_pipefusion_scm_mask(mask, total_timesteps)[warmup_steps:]
    )
    if async_mask:
        async_mask[0] = 1
    return tuple(async_mask)


def install_pipefusion_cache_plan(
    pipeline,
    plan: PipeFusionCachePlan,
) -> None:
    """Attach a model-selected PipeFusion cache plan to its pipeline."""
    pipeline._xdit_pipefusion_cache_plan = plan


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
