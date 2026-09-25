---
name: testing
description: Apply xDiT's test and validation policy for any code change, test change, review, or decision about CPU, accelerator, distributed, or end-to-end coverage.
---

# xDiT testing

- Add a regression test for externally observable behavior or a demonstrated failure. Test at the lowest stable API boundary that proves the behavior.
- Put deterministic, offline, CPU-only tests in `tests/unit/`, mirroring the `xfuser/` package. Mock only process, network, filesystem, accelerator, or third-party boundaries.
- Put real cross-component and local Gloo/subprocess tests in `tests/integration/cpu/`, tagging spawned Gloo tests `gloo`. Put real device tests in `tests/integration/accelerator/` and tag exact requirements such as `nvidia`, `rocm`, or `multi_gpu`; remember that PyTorch calls both CUDA and ROCm devices `cuda`.
- Put complete model or pipeline runs in `tests/e2e/`; record the model, hardware, command, and result in the pull request.
- Do not add tests that inspect source text, repeat implementation logic, assert constants or one-line delegation, duplicate an existing contract, download data in unit tests, or only print/benchmark without a correctness assertion.
- Keep collection safe without optional packages or accelerators. Use `python -m pytest` and run the focused test plus `tests/unit -m "not accelerator"` and `tests/integration/cpu -m "not accelerator"` when relevant.
- Before accelerator validation, inspect `torch.cuda.is_available()`, `torch.version.hip`, device count and names, and required optional backends. Do not infer NVIDIA from PyTorch's `cuda` device name.
- On ROCm, run affected focused tests and `python -m pytest tests -m "accelerator and not nvidia"`. On NVIDIA, use `python -m pytest tests -m "accelerator and not rocm"`. Add `and multi_gpu` only when enough visible devices exist.
- Do not emulate unavailable accelerator behavior or count skipped tests as validation. State the hardware, backend, command, pass/skip counts, and anything not run.
- For model, pipeline, kernel, output-quality, or performance changes, identify the affected runnable example or E2E path and run it on suitable hardware; ask for access when the current system cannot provide meaningful validation.
