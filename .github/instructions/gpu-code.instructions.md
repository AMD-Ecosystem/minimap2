---
applyTo: "minimap2/**/gpu/**"
description: "GPU code conventions for HIP/ROCm kernel and device code under minimap2/chain/src/gpu/ and minimap2/align/src/gpu/. Use when editing .cu or .cuh files."
---

# GPU Code

- All `.cu` / `.cuh` files are **HIP/ROCm**, compiled with `hipcc`. Never use CUDA-only APIs (`cudaMalloc`, `cudaMemcpy`, etc.) — use HIP equivalents (`hipMalloc`, `hipMemcpy`).
- GPU chaining kernels live in `minimap2/chain/src/gpu/pl*.cu`. GPU alignment lives in `minimap2/align/src/gpu/`.
- GPU context lifecycle is managed by context manager objects. Understand the init/teardown paths before modifying them.
- GPU config files (`configs/*.json`) control batching, memory limits, and kernel parameters. Don't hard-code GPU tuning constants.
- Build with `cmake --preset amd-default` for GPU. CPU-only builds (`test-cpu`) skip GPU code entirely.
- GPU tests are in `tests/gpu/` — run with `ctest -R gpu` after an AMD build.
- When adding GPU tests, follow the parameterized/behavioral dual-pattern in [`tests/README.md`](../../tests/README.md).
