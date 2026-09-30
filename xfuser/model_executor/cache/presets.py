"""
Step-caching presets and configuration dataclasses.

DBCachePreset: typed config for cache-dit DBCache.

CacheDitAdapterConfig: block layout for one transformer in a cache-dit BlockAdapter.
    transformer_attr: pipe attribute name for this transformer (e.g. "transformer", "transformer_2").
    blocks: (block_attr, ForwardPattern_name) pairs to try on the transformer.
    enable_separate_cfg: True for two-pass CFG models.

DBCacheSettings: bundles adapter + preset for one model.
    adapter: CacheDitAdapterConfig (single transformer) or List[CacheDitAdapterConfig]
             (multi-transformer, e.g. Wan2.2). Must map 1:1 with preset list.
    preset: DBCachePreset or List[DBCachePreset] mapping 1:1 with adapter list.

ModelCacheConfig: dict[str, optional method_config] keyed by cache method name,
    e.g. {"dbcache": DBCacheSettings(...)}. .get(cache_method) returns None for
    in-tree methods that need no model-specific configuration.
"""
import dataclasses
import json
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

PIPEFUSION_CACHE_PLAN_KEY = "pipefusion_cache_plan"


class PipeFusionCacheUnit(str, Enum):
    """The reusable unit selected by a PipeFusion cache plan."""

    BLOCK_LOCAL = "block_local"
    INTERMEDIATE_STAGE_OUTPUT = "intermediate_stage_output"
    FULL_STAGE_OUTPUT = "full_stage_output"


class PipeFusionCacheDecision(str, Enum):
    """How a PipeFusion cache plan decides whether to reuse state."""

    ADAPTIVE = "adaptive"
    STATIC = "static"


class PipeFusionStaticMask(str, Enum):
    """Model-owned static masks supported by the shared PipeFusion runtime."""

    ALTERNATING_MIDDLE = "alternating_middle"
    WIDE_ALTERNATING_MIDDLE = "wide_alternating_middle"


@dataclasses.dataclass(frozen=True)
class PipeFusionTopology:
    """A PP partition for which a cache plan has been validated."""

    pp_degree: int
    num_pipeline_patches: int
    attn_layer_num_for_pp: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.pp_degree < 2:
            raise ValueError("PipeFusion cache topologies require PP degree >= 2.")
        if self.num_pipeline_patches < 1:
            raise ValueError("num_pipeline_patches must be positive.")
        if (
            self.attn_layer_num_for_pp is not None
            and len(self.attn_layer_num_for_pp) != self.pp_degree
        ):
            raise ValueError(
                "attn_layer_num_for_pp must contain one split per PP stage."
            )

    def matches(
        self,
        *,
        pp_degree: int,
        num_pipeline_patches: int,
        attn_layer_num_for_pp: Optional[Tuple[int, ...]],
    ) -> bool:
        return (
            self.pp_degree == pp_degree
            and self.num_pipeline_patches == num_pipeline_patches
            and (
                self.attn_layer_num_for_pp is None
                or self.attn_layer_num_for_pp == attn_layer_num_for_pp
            )
        )


@dataclasses.dataclass(frozen=True)
class PipeFusionCachePlan:
    """Model-owned PipeFusion reuse policy.

    Block-local caching remains adaptive and is implemented by Cache-DiT.
    Stage-payload reuse is static because its scheduler semantics must agree
    across pipeline stages. Intermediate-stage plans keep the final prediction
    stage fresh; full-stage-output reuse requires an explicit quality-validated
    opt-in.
    """

    unit: PipeFusionCacheUnit = PipeFusionCacheUnit.BLOCK_LOCAL
    decision: PipeFusionCacheDecision = PipeFusionCacheDecision.ADAPTIVE
    static_mask: Optional[PipeFusionStaticMask] = None
    allow_final_stage_reuse: bool = False
    pp_degrees: Optional[Tuple[int, ...]] = None
    pipeline_patch_counts: Optional[Tuple[int, ...]] = None
    min_inference_steps: Optional[int] = None
    max_inference_steps: Optional[int] = None
    topologies: Optional[Tuple[PipeFusionTopology, ...]] = None
    # An experimental, explicitly requested plan remains model-gated but is
    # excluded from automatic selection until its quality is promoted.
    auto_select: bool = True
    # For intermediate-stage plans, cache the stage's Mn residual and execute
    # this many Bn refinement blocks on every step instead of reusing a whole
    # stage payload. None retains whole-payload intermediate reuse.
    tail_compute_blocks: Optional[int] = None

    def __post_init__(self) -> None:
        stage_payload = self.unit is not PipeFusionCacheUnit.BLOCK_LOCAL
        if stage_payload != (self.decision is PipeFusionCacheDecision.STATIC):
            raise ValueError(
                "PipeFusion stage-payload caching requires a static decision; "
                "block-local caching requires an adaptive decision."
            )
        if stage_payload != (self.static_mask is not None):
            raise ValueError(
                "PipeFusion stage-payload caching requires a static mask; "
                "block-local caching does not accept one."
            )
        if (
            self.unit is PipeFusionCacheUnit.FULL_STAGE_OUTPUT
            and not self.allow_final_stage_reuse
        ):
            raise ValueError(
                "Full-stage-output reuse requires an explicit final-stage "
                "quality opt-in."
            )
        if (
            self.unit is not PipeFusionCacheUnit.FULL_STAGE_OUTPUT
            and self.allow_final_stage_reuse
        ):
            raise ValueError(
                "Only full-stage-output reuse may reuse the final prediction "
                "stage."
            )
        if (
            self.tail_compute_blocks is not None
            and self.unit is not PipeFusionCacheUnit.INTERMEDIATE_STAGE_OUTPUT
        ):
            raise ValueError(
                "tail_compute_blocks is only supported by "
                "intermediate-stage plans."
            )
        if (
            self.tail_compute_blocks is not None
            and self.tail_compute_blocks < 1
        ):
            raise ValueError("tail_compute_blocks must be positive.")
        if self.min_inference_steps is not None and self.min_inference_steps < 1:
            raise ValueError("min_inference_steps must be positive.")
        if (
            self.max_inference_steps is not None
            and self.max_inference_steps < 1
        ):
            raise ValueError("max_inference_steps must be positive.")
        if (
            self.min_inference_steps is not None
            and self.max_inference_steps is not None
            and self.min_inference_steps > self.max_inference_steps
        ):
            raise ValueError(
                "min_inference_steps cannot exceed max_inference_steps."
            )

    def matches(
        self,
        *,
        pp_degree: int,
        num_pipeline_patches: int,
        num_inference_steps: int,
        attn_layer_num_for_pp: Optional[Tuple[int, ...]],
    ) -> bool:
        """Return whether this plan was validated for a runtime topology."""
        return (
            (self.pp_degrees is None or pp_degree in self.pp_degrees)
            and (
                self.pipeline_patch_counts is None
                or num_pipeline_patches in self.pipeline_patch_counts
            )
            and (
                self.min_inference_steps is None
                or num_inference_steps >= self.min_inference_steps
            )
            and (
                self.max_inference_steps is None
                or num_inference_steps <= self.max_inference_steps
            )
            and (
                self.topologies is None
                or any(
                    topology.matches(
                        pp_degree=pp_degree,
                        num_pipeline_patches=num_pipeline_patches,
                        attn_layer_num_for_pp=attn_layer_num_for_pp,
                    )
                    for topology in self.topologies
                )
            )
        )

    @property
    def uses_block_local_cache(self) -> bool:
        """Whether the plan must retain Cache-DiT block boundaries."""
        return (
            self.unit is PipeFusionCacheUnit.BLOCK_LOCAL
            or self.tail_compute_blocks is not None
        )

    @classmethod
    def block_local(cls) -> "PipeFusionCachePlan":
        return cls()

    @classmethod
    def intermediate_stage_output(
        cls, **applicability
    ) -> "PipeFusionCachePlan":
        return cls(
            unit=PipeFusionCacheUnit.INTERMEDIATE_STAGE_OUTPUT,
            decision=PipeFusionCacheDecision.STATIC,
            static_mask=PipeFusionStaticMask.ALTERNATING_MIDDLE,
            **applicability,
        )

    @classmethod
    def full_stage_output(
        cls,
        *,
        static_mask: PipeFusionStaticMask = PipeFusionStaticMask.ALTERNATING_MIDDLE,
        **applicability,
    ) -> "PipeFusionCachePlan":
        return cls(
            unit=PipeFusionCacheUnit.FULL_STAGE_OUTPUT,
            decision=PipeFusionCacheDecision.STATIC,
            static_mask=static_mask,
            allow_final_stage_reuse=True,
            **applicability,
        )


@dataclasses.dataclass
class DBCachePreset:
    # the first N blocks to use to calculate L1 difference, increase to improve accuracy at the cost of performance
    Fn_compute_blocks: int = 8
    # set to 0 explicitly since we exclusively use the TaylorSeer calibrator which overrides this
    Bn_compute_blocks: int = 0
    # increase to allow more caching
    residual_diff_threshold: float = 0.08
    # steps before cache kicks in
    max_warmup_steps: int = 8
    max_cached_steps: int = -1
    # SCM policy: None | "slow" | "medium" | "fast" | "ultra" | "pipefusion".
    # "pipefusion" alternates cache/compute through the middle denoising window.
    scm_policy: Optional[str] = "fast"
    # "dynamic" also requires the residual threshold to pass; "static" makes
    # SCM the sole decision source, which keeps PipeFusion stages in lockstep.
    steps_computation_policy: str = "dynamic"
    # enable_taylorseer: None/True attaches the TaylorSeer calibrator (default DBCache
    # behavior). Set False to run the plain Fn-block residual cache with no calibrator
    # (e.g. "true" FBCache when Fn_compute_blocks=1 and scm_policy=None).
    enable_taylorseer: Optional[bool] = None
    # enable_separate_cfg: True for models with two separate CFG forward passes (Wan, Qwen-Image-Edit).
    # False for fused-CFG or no-CFG models (FLUX, HunyuanVideo, Qwen-Image).
    # None = infer from CacheDitAdapterConfig.enable_separate_cfg.
    enable_separate_cfg: Optional[bool] = None
    # enable_encoder_calibrator: set False for MMDiT models (SD3.5) with Bn=0
    # Bn-residual buffer is never populated, causing the calibrator to assert.
    enable_encoder_calibrator: Optional[bool] = None
    # PipeFusion-only experimental optimization: patch 0 makes the adaptive
    # decision for a block group and later patches reuse it. Each patch retains
    # its own Cache-DiT buffers and executes the selected path.
    share_pipefusion_decisions: Optional[bool] = None


@dataclasses.dataclass(frozen=True)
class CacheDitAdapterConfig:
    """Block layout for one transformer in a cache-dit BlockAdapter.

    transformer_attr: pipe attribute name used to retrieve this transformer
        (e.g. "transformer", "transformer_2"). Default "transformer" covers all
        single-transformer models.

    blocks: sequence of (block_attr_name, ForwardPattern_name) pairs tried in order;
        first match on the transformer wins. Allows one config to cover model variants
        where optional block groups exist (e.g. Flux1 vs Flux2 single_transformer_blocks).

    enable_separate_cfg: True for models where CFG runs as two separate forward passes
        (Wan, Qwen-Image-Edit). False for fused or no-CFG models (FLUX, SD3, ZImage).

    """
    blocks: Tuple[Tuple[str, str], ...]
    enable_separate_cfg: bool = False
    transformer_attr: str = "transformer"


AdapterValue = Union["CacheDitAdapterConfig", List["CacheDitAdapterConfig"]]
PresetValue = Union[DBCachePreset, List[DBCachePreset]]


@dataclasses.dataclass
class DBCacheSettings:
    """Bundles CacheDitAdapterConfig(s) and DBCachePreset(s) for a model.

    Single transformer: adapter=CacheDitAdapterConfig(...), preset=DBCachePreset(...)
    Multi-transformer:  adapter=[cfg_t1, cfg_t2], preset=[preset_t1, preset_t2]
        Lists must be the same length; index i covers transformer i.
    """
    adapter: AdapterValue
    preset: Optional[PresetValue] = None
    # Plans are model-owned and gate every user request against the active
    # topology. Automatic selection considers only plans marked auto_select;
    # experimental stage plans require an explicit cache_config request.
    pipefusion_cache_plans: Tuple[PipeFusionCachePlan, ...] = ()

    def resolve_pipefusion_cache_plan(
        self,
        *,
        pp_degree: int,
        num_pipeline_patches: int,
        num_inference_steps: int,
        attn_layer_num_for_pp: Optional[Tuple[int, ...]],
        requested_unit: Optional[PipeFusionCacheUnit] = None,
    ) -> PipeFusionCachePlan:
        matching_plans = [
            plan
            for plan in self.pipefusion_cache_plans
            if plan.matches(
                pp_degree=pp_degree,
                num_pipeline_patches=num_pipeline_patches,
                num_inference_steps=num_inference_steps,
                attn_layer_num_for_pp=attn_layer_num_for_pp,
            )
        ]
        if requested_unit is not None:
            requested_plans = [
                plan for plan in matching_plans if plan.unit is requested_unit
            ]
            if not requested_plans:
                supported = sorted(
                    {plan.unit.value for plan in matching_plans}
                )
                raise ValueError(
                    f"PipeFusion cache plan {requested_unit.value!r} is not "
                    f"supported for PP={pp_degree}, patches="
                    f"{num_pipeline_patches}, steps={num_inference_steps}; "
                    f"compatible plans are {supported}."
                )
            if len(requested_plans) > 1:
                raise ValueError(
                    "Multiple requested PipeFusion cache plans match the "
                    "active topology."
                )
            return requested_plans[0]

        matching_stage_plans = [
            plan
            for plan in matching_plans
            if (
                plan.unit is not PipeFusionCacheUnit.BLOCK_LOCAL
                and plan.auto_select
            )
        ]
        if len(matching_stage_plans) > 1:
            raise ValueError(
                "Multiple PipeFusion stage-payload cache plans match the "
                "active topology."
            )
        return (
            matching_stage_plans[0]
            if matching_stage_plans
            else PipeFusionCachePlan.block_local()
        )


def parse_pipefusion_cache_plan_request(
    cache_config_json: Optional[str],
) -> Optional[PipeFusionCacheUnit]:
    """Read the model-gated stage-cache request from ``cache_config``."""
    if not cache_config_json:
        return None
    try:
        overrides = json.loads(cache_config_json)
        if not isinstance(overrides, dict):
            raise TypeError("cache_config must be a JSON object")
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(
            f"--cache_config is not valid JSON: {error}"
        ) from error
    requested = overrides.get(PIPEFUSION_CACHE_PLAN_KEY)
    if requested is None:
        return None
    try:
        return PipeFusionCacheUnit(requested)
    except ValueError as error:
        choices = [unit.value for unit in PipeFusionCacheUnit]
        raise ValueError(
            f"{PIPEFUSION_CACHE_PLAN_KEY} must be one of {choices}; "
            f"got {requested!r}."
        ) from error


# Dict keys declare supported methods; None marks an in-tree method with no
# model-specific configuration.
ModelCacheConfig = Dict[str, Optional[object]]
