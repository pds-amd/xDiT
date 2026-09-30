## PipeFusion: Displaced Patch Pipeline Parallelism for Diffusion Models
[Chinese Blog 1](https://zhuanlan.zhihu.com/p/699612077); [Chinese Blog 2](https://zhuanlan.zhihu.com/p/706475158)

PipeFusion is the innovative method first proposed by us. 
It is a sequence-level pipeline parallel method, similar to [TeraPipe](https://proceedings.mlr.press/v139/li21y.html), demonstrates significant advantages in weakly interconnected network hardware such as PCIe/Ethernet. 

PipeFusion innovatively harnesses input temporal redundancy—the similarity between inputs and activations across diffusion steps, a diffusion-specific characteristics also employed in DistriFusion. PipeFusion not only reduces communication volume but also streamlines pipeline parallelism with TeraPipe, avoiding the load balancing issues inherent in LLM models with Causal Attention.
It significantly surpasses other methods in communication efficiency, particularly in multi-node setups connected via Ethernet and multi-GPU configurations linked with PCIe.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/overview.png" alt="PipeFusion Image">
</div>

The above picture compares DistriFusion and PipeFusion.
(a) DistriFusion replicates DiT parameters on two devices. 
It splits an image into 2 patches and employs asynchronous allgather for activations of every layer.
(b) PipeFusion shards DiT parameters on two devices.
It splits an image into 4 patches and employs asynchronous P2P for activations across two devices.

We briefly explain the workflow of PipeFusion. It partitions an input image into $M$ non-overlapping patches.
The DiT network is partitioned into $N$ stages ($N$ < $L$), which are sequentially assigned to $N$ computational devices. 
Note that $M$ and $N$ can be unequal, which is different from the image-splitting approaches used in sequence parallelism and DistriFusion.
Each device processes the computation task for one patch of its assigned stage in a pipelined manner. 

For joint text/image transformers, later patches may omit text-query, text-output,
and text-FFN work when each block reuses the text state captured from patch 0.
FLUX.1 and SD3 support this image-query-only path while retaining fresh text K/V
conditioning and the normal stale image-KV cache. FLUX.2 is intentionally excluded:
its single-stream block fuses Q, K, V, and MLP input projection into one linear.
Selecting only text K/V would require bypassing that module with manually sliced
weights, which would break quantized, wrapped, and tensor-parallel linear contracts.

### Runtime architecture

The root `xfuser.model_executor.pipefusion` package contains only model-agnostic
mechanics: patch layout, P2P transport, stage-output caching, and the async
driver. Model-family dataflow lives in `pipefusion.schedules`.

`schedules.joint_image_text` is the shared schedule for joint text/image
transformers such as FLUX.1, FLUX.2, and SD3. It explicitly selects whether
text state is reused from patch 0 or forwarded for every patch. Video and
whole-step scheduler families use separate schedule adapters rather than
expanding the joint text/image contract.

### Stage partitioning

Use `--attn_layer_num_for_pp` to provide a measured contiguous block count for
each stage. Parameter count is not a reliable automatic cost estimate: block
families can process different token counts and use different compiled kernels.
If no split is provided, xDiT falls back to a naive equal-block partition and
logs a warning. Production configurations should benchmark and specify the
split explicitly.

### Cache plans

PipeFusion caching is a model-owned cache plan, not a global switch. A plan
selects a cache unit and decision rule:

- **Block-local** uses Cache-DiT's adaptive residual decision with independent
  context for every PipeFusion patch. This is the safe default and the
  validated policy for FLUX.2 and SD3.5.
- **Intermediate-stage output** uses a model-owned static mask to reuse an
  earlier stage's per-patch payload, but always computes the final prediction
  stage. It is available for models that validate this partial reuse policy.
- **Full-stage output** reuses every stage payload for selected steps. This
  makes the scheduler consume a stale prediction and therefore requires an
  explicit quality-validated opt-in; FLUX.1 is the current validated user.

Plans may be restricted to a validated PP degree, patch count, and inference
step range. Outside that domain, xDiT automatically falls back to adaptive
block-local caching rather than applying a result measured under a different
topology.

The shared alternating-middle mask reuses isolated middle denoising steps (at
25 steps: 8, 10, 12, 14, 16, 18, and 20). The first asynchronous step, early
structure-forming steps, and final correction steps always compute.

Users may tune ordinary block-local DBCache thresholds, but cannot promote a
model to a stage-payload policy through `--cache_config`.

Pipeline side channels remain model-specific. In particular, FLUX.2 carries
its jointly updated text state for every patch; reusing patch 0's text state for
later patches is invalid for its fused single-stream blocks.

Stage payloads are converted to the runtime transport dtype before P2P. This is
required for scheduler feedback: several schedulers produce FP32 latents while
PipeFusion receive buffers are BF16 or FP16, and NCCL does not perform implicit
dtype conversion. Schedulers that update one complete latent per timestep use
the driver's deferred whole-step commit, so they advance once after all patch
predictions are assembled rather than once per patch.

Parallel VAE is a separate distributed composition. PipeFusion broadcasts
scheduler latents in the runtime transport dtype before tiled decode so every
rank enters DistVAE with the same finite tensor; NCCL does not convert a
source FP32 scheduler tensor into BF16/FP16 receive buffers implicitly.

`--use_fp8_comms` quantizes Ulysses collectives. Pure PipeFusion uses point-to-
point stage transport and has no FP8 wire protocol, so the flag is rejected.

### Sharding replicated components

PipeFusion already partitions the transformer, but components such as a large
text encoder remain replicated and can still exceed per-rank memory. They can
be sharded without moving the stage-local transformer to CPU:

```bash
--pipefusion_parallel_degree 2 \
--fully_shard_degree 2 \
--fully_shard_components text_encoder \
--memory_efficient_sharding \
--memory_efficient_replicated_load
```

Each selected name must be present in the model's FSDP strategy. A PipeFusion
stage-local transformer cannot also be selected for FSDP.

The PipeFusion pipeline workflow when $M$ = $N$ =4 is shown in the following picture.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/workflow.png" alt="Pipeline Image">
</div>


We have evaluated the accuracy of PipeFusion, DistriFusion and the baseline as shown bolow. To conduct the FID experiment, follow the detailed instructions provided in the [documentation](../../docs/fid/FID.md).

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/image_quality.png" alt="image_quality">
</div>


For more details, please refer to the following paper.

```
@inproceedings{
    fang2025pipefusion,
    title={PipeFusion: Patch-level Pipeline Parallelism for Diffusion Transformers Inference},
    author={Jiarui Fang and Jinzhe Pan and Aoyu Li and Xibo Sun and WANG Jiannan},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=5xwyxupsLL}
}
```
