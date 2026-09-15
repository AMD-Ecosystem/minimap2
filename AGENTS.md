# Agent Instructions

Operating rules for AI coding assistants (Cursor, GitHub Copilot, Claude Code, etc.) working in this repository. For repository layout, build commands, scripts, formatting, and coding style, see [`CONVENTIONS.md`](./CONVENTIONS.md).

## Behavioral rules (agent-specific, do not skip)

1. **Don't commit unless explicitly asked.**
2. **Don't install tools, switch compilers, or change build configurations without confirmation.** If `clang-format`, `cmake`, `hipcc`, `ctest`, `lcov`, etc. is missing, stop and ask the user.
3. **Use existing scripts for common tasks** — check [`scripts/README.md`](./scripts/README.md) before writing one-off shell or Python (benchmarking, profiling, test-data generation, coverage, etc.). Extend an existing script rather than working around it. New common tasks should become new scripts, not chat-only commands.
4. **Don't refactor opportunistically.** Stay focused on the requested change. Don't reformat, rename, or restructure unrelated code in the same diff.
5. **Don't reformat untracked or vendored code.** Only run `clang-format` on tracked first-party files (covered by the batch command in [`CONVENTIONS.md`](./CONVENTIONS.md#code-formatting)).

## Key pitfalls

- **`.cu` files are HIP/ROCm, not CUDA.** GPU code under `minimap2/chain/src/gpu/` and `minimap2/align/src/gpu/` uses HIP compiled via `hipcc`. Don't add CUDA-only APIs or headers.
- **`external/` contains git submodules** (simde, googletest, cJSON, spdlog, ittapi). Never modify files in `external/`.
- **Test data is generated, not hand-written.** Use `scripts/generate_test_data.py` to produce JSON test data from actual minimap2 runs. See [`tests/README.md`](./tests/README.md).
- **Build output goes under `out/<preset>/build/`**, not `build/`. The `build/` directory in the repo root may be stale.

## Architecture awareness

The mapping pipeline has three stages — **seed → chain → align** — each in its own module under `minimap2/`. GPU acceleration is available for chain and align independently. The library dependency chain is: `core → seed → chain → align → libminimap2`.

For architecture details, see [`docs/ARCHITECTURE_GUIDE.md`](./docs/ARCHITECTURE_GUIDE.md).

## Mandatory before finalizing changes

- **Build via CMake presets**, not raw `cmake -S . -B build`. See [`CONVENTIONS.md`](./CONVENTIONS.md#build--test-commands) for the preset list (`amd-default`, `release`, `profile`).
- **Run `clang-format -i`** on every tracked first-party C/C++/CUDA file (`.c`, `.cc`, `.cpp`, `.h`, `.hpp`, `.cu`, `.cuh`) you modified under `minimap2/`. See [`CONVENTIONS.md`](./CONVENTIONS.md#code-formatting) for the batch command.
- **Match existing code style; minimize comments.** Code should be self-documenting; don't add narrative comments that just describe what the code does. Stay close to the surrounding-file style — don't introduce a different style in an existing module. See [`CONVENTIONS.md`](./CONVENTIONS.md#coding-style) for the full style guide.

## Documentation index

| Area | Document |
|------|----------|
| Build & install | [`BUILD_GUIDE.md`](./BUILD_GUIDE.md) |
| Pipeline architecture | [`docs/ARCHITECTURE_GUIDE.md`](./docs/ARCHITECTURE_GUIDE.md) |
| Profiling | [`docs/PROFILING_GUIDE.md`](./docs/PROFILING_GUIDE.md) |
| Tests | [`tests/README.md`](./tests/README.md) |
| Scripts | [`scripts/README.md`](./scripts/README.md) |
| Staged CLI tools | [`cli/README.md`](./cli/README.md) |
| Versioning | [`docs/VERSIONING.md`](./docs/VERSIONING.md) |
## Overview

This is the **AMD fork of minimap2** — a sequence alignment tool with GPU acceleration for chaining and alignment stages via ROCm/HIP. The primary development target is AMD MI300X/MI300A (gfx942). The goal is maximum performance improvement over the upstream v2.24 CPU baseline.

## Build Commands

CMake Presets are the primary build interface. All build outputs go to `out/<preset>/`.

```bash
# Standard AMD GPU build (auto-detects GPU arch)
cmake --preset amd-default && cmake --build --preset amd-default -j$(nproc)

# gfx942 target (MI300X/MI300A) — most common development target
cmake --preset gfx942 && cmake --build --preset gfx942 -j$(nproc)

# CPU-only build (no GPU)
cmake --preset cpu-only && cmake --build --preset cpu-only -j$(nproc)

# Profile build (enables ITT/ROCTX instrumentation + trace logging)
cmake --preset profile && cmake --build --preset profile -j$(nproc)

# Debug presets
cmake --preset debug-info     # Enables DEBUG_PRINT
cmake --preset debug-analyze  # Enables DEBUG_CHECK + DEBUG_PRINT + GPU kernel analysis
cmake --preset debug-verbose  # Full debug output

# Sanitizer builds
cmake --preset asan && cmake --preset tsan
```

## Testing

```bash
# CPU tests only
cmake --preset test-cpu && cmake --build --preset test-cpu -j$(nproc)
ctest --test-dir out/test-cpu/build --output-on-failure

# GPU tests (AMD)
cmake --preset amd-default && cmake --build --preset amd-default -j$(nproc)
ctest --test-dir out/amd-default/build -j$(nproc) --output-on-failure

# Run specific test by gtest filter
./out/amd-default/build/tests/core/test_core --gtest_filter="KallocTest.*"
./out/amd-default/build/tests/core/test_core --gtest_filter="*MT_human_vs_orang*"
./out/amd-default/build/tests/gpu/test_gpu --gtest_filter="*GpuChain*"

# List all available tests
./out/amd-default/build/tests/core/test_core --gtest_list_tests
./out/amd-default/build/tests/gpu/test_gpu --gtest_list_tests

# Quick smoke test (CPU vs GPU output)
./out/amd-default/build/bin/minimap2 -t 1 --gpu-chain --gpu-cfg configs/gpu_config.json \
    test/MT-human.fa test/MT-orang.fa > out_chain.paf

# Code coverage report
./scripts/coverage.sh               # Clang (default, includes host-side GPU code)
./scripts/coverage.sh --compiler gcc   # GCC (CPU-only)
```

## Regenerating Test Data

```bash
# Regenerate all test data JSON fixtures
./scripts/generate_all_test_data.sh

# Regenerate for a single pair
python3 scripts/generate_test_data.py \
    --ref test_suite/original/MT-human.fa \
    --query test_suite/original/MT-orang.fa \
    --output test_suite/original/expected/MT_human_vs_orang.json
```

## Performance Benchmarking

```bash
# Profile stage breakdown (VTune + rocprofv3 + trace logs)
python3 scripts/analyze_stages.py -- -ax map-ont --gpu-chain \
    --gpu-cfg configs/gpu_config.json --gpu-align ref.fa reads.fa

# Generate profiling comparison report (CPU vs GPU)
python3 scripts/generate_profiling_report.py \
    /tmp/cpu_analysis.json /tmp/gpu_analysis.json --output-dir /tmp/report
```

## Architecture

### Three-Stage Mapping Pipeline

Every read goes through three stages in sequence:

1. **Seeding** (`mm_map_seed`) — extracts k-mer minimizers and looks them up in the reference index to produce `mm128_t` anchors.
2. **Chaining** (`mm_map_chain`) — runs DP or RMQ to link anchors into colinear chains. **This is GPU-accelerated with `--gpu-chain`.**
3. **Alignment** (`mm_map_align`) — performs base-level DP alignment and generates CIGAR strings. **This is GPU-accelerated with `--gpu-align`.**

### I/O-Parallel Pipeline (`kt_pipeline`)

The pipeline runs three stages concurrently across batches using worker threads that cycle through all stages (0=Read → 1=Map → 2=Output → repeat). A batch of reads currently being mapped by Stage 1 overlaps with Stage 0 reading the next batch and Stage 2 writing the previous batch. Stage 1 uses `kt_for()` with work-stealing to distribute reads across all threads.

### GPU Batch Pipeline (`mm_trbuf_t`)

When GPU chaining is enabled, each worker thread maintains a three-batch rotation: `acc_batch` (accumulating reads) → `pending_batch` (ready for GPU) → `launched_batch` (on GPU). This allows CPU seeding of new reads to overlap with GPU chaining of the previous batch. After GPU completion, batches rotate in reverse.

### Source Layout

```
minimap2/
├── src/          # Main pipeline: map.c (worker_pipeline, worker_for, init_step, free_step)
│                 #                main.c (CLI flags), splitidx.c, format.c
├── include/      # minimap.h — public API, mm_tbuf_t, mm_mapopt_t
├── core/
│   ├── src/      # kthread.c (kt_pipeline, kt_for), bseq.c, kalloc.c, mm_log.cpp, mm_profiler.cpp
│   └── include/  # mm_log.h, mm_profiler.h, chain_read.h
├── seed/
│   ├── src/      # seed_map.c (mm_map_seed), seed.c, sketch.c
│   └── include/  # mm2_seed.h
├── chain/
│   ├── src/      # chain_map.c (mm_map_chain), lchain.c
│   │   └── gpu/  # plchain.cu (GPU chaining kernel), plmem.cu (GPU memory management)
│   └── include/  # mm2_chain.h
├── align/
│   ├── src/      # align.c (mm_map_align), ksw2_*.c (DP kernels), esterr.c, hit.c, pe.c
│   │   └── gpu/  # ksw2_gpu_wrapper.cpp (GPU alignment dispatch)
│   └── include/  # mm2_align.h
└── ksw2_gpu/     # GPU KSW2 alignment standalone
cli/              # Staged CLI tools: mm2_seed, mm2_chain, mm2_align (output JSON)
tests/
├── core/         # CPU unit tests (test_core binary, Google Test)
└── gpu/          # GPU unit tests (test_gpu binary)
test_suite/       # Test data JSON + FASTA files (auto-discovered by parameterized tests)
configs/          # GPU config files (gpu_config.json, mi210_*.json, a6000_*.json)
scripts/          # Benchmarking, profiling, test data generation, and validation scripts
```

### Key Data Structures

| Structure | Location | Purpose |
|-----------|----------|---------|
| `step_t` | `minimap2/src/map_priv.h` | One pipeline batch: seq[], reg[], buf[], trbuf[] |
| `pipeline_t` | `minimap2/src/map_priv.h` | Pipeline context: fp, opt, mi, n_threads |
| `mm_tbuf_t` | `minimap2/include/minimap.h` | Per-thread memory arena (kalloc) + timing stats |
| `mm_trbuf_t` | `minimap2/src/map_priv.h` | GPU 3-batch pipeline: acc/pending/launched |
| `chain_read_t` | `minimap2/core/include/chain_read.h` | Per-read mapping data passed between stages |

### Logging Rules

Two logging systems coexist — do not mix them:

- **`mm_print()` / `mm_log()`** — Upstream-compatible, plain `%v` format. Output must remain byte-for-byte identical to upstream minimap2 v2.24. **Never add timestamps, levels, or decorations to these.**
- **`mm_log_info/debug/warn/error()`** — AMD-specific code uses these spdlog-backed macros. They include timestamp + source location automatically. Use fmt-style `{}` in C++ files, printf-style `%d/%s` in C files.
- spdlog appends newlines automatically — **do not add `\n`** to log format strings.
- In C++ files, `#include "mm_log.h"` must appear **before** any `extern "C"` block (spdlog templates cannot be inside `extern "C"`).

### Profiling Instrumentation

Build with `cmake --preset profile` to enable. Use stage-colored macros:

```c
// C files
MM_PROFILE_SEED("my_function");   // push green range
MM_PROFILE_RANGE_POP();           // pop

// C++ files (RAII — auto-pops on scope exit)
MM_PROFILE_SCOPE_CHAIN("my_function");
```

Color codes: Seed=Green, Chain=Blue, Align=Red, I/O=Yellow, GPU=Cyan.

## GPU Configuration

GPU chaining behavior is tuned via JSON config files in `configs/`. Key parameters:

- `min_n` — minimum anchor count to use GPU (smaller batches fall back to CPU)
- `max_total_n` — maximum anchors per microbatch
- `max_read` — maximum reads per microbatch

GPU alignment does **not** use a config file. GPU chaining is incompatible with the RMQ algorithm (`-r` flag).

## Staged CLI Tools (Debugging)

Three single-stage executables expose intermediate JSON for debugging:

```bash
# Run staged pipeline to inspect intermediates
./mm2_seed ref.fa query.fa -o seeds.json
./mm2_chain ref.fa -s seeds.json -o chains.json
./mm2_align ref.fa query.fa -c chains.json -o alignments.json

# With GPU chaining
./mm2_chain --use-gpu --gpu-config configs/gpu_config.json ref.fa -s seeds.json -o chains.json
```

## Comparing with Upstream Baseline

The upstream v2.24 baseline lives on branch `upstream/v2.24`. Benchmark generation and comparison scripts (`generate_all_mode_benchmark_data.py`, `compare_benchmark_data.py`) live in this repo under `scripts/`. See [`scripts/README.md`](./scripts/README.md) for usage.

## GPU Performance State (as of May 2026, v2.24.0a4)

Version 2.24.0a4 (commit `71720ef`, PR #92) restored GPU align correctness: per-task `w`/`zdrop`, `uint64_t` p_ps prefix sums, and VRAM budget tuning. Result: **significantly reduced divergence from CPU baseline** across all thread counts.

Full-dataset benchmark results vs upstream v2.24 (MI300X, 3 runs per mode):

| Preset | Threads | Baseline (s) | cpu | gpu_chain | gpu_align | gpu_both |
|--------|---------|-------------|-----|-----------|-----------|----------|
| map-ont | 16 | 530.29 | 334.82 (**1.58×**) | 426.66 (**1.24×**) | 348.38 (**1.52×**) | 478.67 (**1.11×**) |
| map-ont | 32 | 391.68 | 266.53 (**1.47×**) | 394.50 (0.99×) | 284.02 (**1.38×**) | 580.59 (0.67×) |
| map-hifi | 16 | 292.53 | 235.73 (**1.24×**) | 321.15 (0.91×) | 407.80 (0.72×) | 447.37 (0.65×) |
| map-hifi | 32 | 158.92 | 137.92 (**1.15×**) | 282.37 (0.56×) | 433.15 (0.37×) | 497.32 (0.32×) |
| map-pb | 16 | 72.06 | 73.73 (0.98×) | 139.10 (0.52×) | 94.17 (0.77×) | 105.84 (0.68×) |
| map-pb | 32 | 42.56 | 43.91 (0.97×) | 56.88 (0.75×) | 107.02 (0.40×) | 114.75 (0.37×) |

**Key takeaways:**
- **CPU-only mode beats baseline for all presets** at t16/t32, driven by 1.24–1.66× CPU chain improvement.
- **map-ont: all modes beat baseline at t16** — the only preset where every GPU mode outperforms upstream.
- **GPU align stage is slower than CPU in all configurations** (0.13–0.66×). Wall-time wins for map-ont come from the CPU chain improvement, not GPU align acceleration.
- **GPU align serialization worsens at t32** — batching z-drop second passes (AIOSS-4684) remains critical.

**GPU chaining** (`chain/src/gpu/plchain.cu`) is working but limited by:
- 0.3–7.2% DP score divergence from CPU (different chain counts/compositions)
- GPU chain stage is slower than CPU chain at multi-thread (0.14–0.81×)

**Note on `rocprofv3` usage**: use `--runtime-trace` (not `--hip-trace`) and `--` separator before the app:
```bash
rocprofv3 --runtime-trace -- ./out/amd-default/build/bin/minimap2 -a -x map-hifi ...
```

## AMD GPU Optimization Patterns (MI300X / gfx942)

**Hardware context**: 304 CUs total (8 XCDs × 40 CUs). Wavefront = 64 lanes. Optimal block size = 256 (4 waves, fills all 4 SIMDs). LDS = 64 KB/CU. Target ≤ 8-12 KB LDS/block for 5-8 concurrent blocks per CU.

**Pattern 1 — Kernel batching (highest priority)**:
Batch multiple work items (alignments, environments) into a single kernel launch. Each block handles one work item. Eliminates per-item `hipMalloc/hipDeviceSynchronize/hipFree` overhead.
```c
// Anti-pattern: one launch per alignment
for (int i = 0; i < n_aln; i++) {
    hipMalloc(&buf, size);
    hipLaunchKernel(ksw_extd2_gpu, dim3(1), dim3(64), ...);
    hipDeviceSynchronize();
    hipFree(buf);
}

// Target: one launch for all alignments
hipLaunchKernel(ksw_extd2_gpu_batched, dim3(n_aln), dim3(256), ...);
// gridDim.x = alignment index, blockDim.x = threads cooperating on that alignment
hipDeviceSynchronize();  // once only
```

**Pattern 2 — LDS batch size tuning**:
If per-element shared memory > 10 KB, process in sub-batches:
`max_batch_size = (8 * 1024) / bytes_per_element` (rounded down to multiple of 64).
Compile with `-Rpass-analyze=kernel-resource-usage` to see LDS usage, VGPR count, and occupancy.

**Pattern 3 — Async dispatch**:
Pre-allocate all device memory at startup. Use HIP streams to overlap batch N kernel with batch N+1 CPU preparation.

**Pattern 4 — Occupancy vs ILP**:
- Memory-bound kernels (ksw2 DP): prefer high occupancy → reduce VGPR count, use `__launch_bounds__`
- Compute-bound: prefer ILP → multiple accumulators, `#pragma unroll`, FMA (`fma(a,b,c)`)
- Check register spill: `--save-temps` → `.hip-amdgcn-amd-amdhsa-gfx942.s` → `.vgpr_spill_count`

**Pattern 5 — Wave intrinsics for reductions** (avoid LDS + syncthreads):
```c
// Butterfly XOR reduction across 64 lanes
for (int offset = 32; offset >= 1; offset >>= 1)
    val += __shfl_xor(val, offset);
```

**Pattern 6 — Memory coalescing**:
Wavefront threads must access contiguous addresses. Use SoA layout (not AoS) for sequence data. Use `float4`/`int4` vectorized loads.

**Profiling workflow**:
1. `rocprofv3 --runtime-trace -- ./mm2` → verify dispatch counts
2. `rocprofv3 --stats ./mm2` → per-kernel time after changes
3. `-Rpass-analyze=kernel-resource-usage` → VGPR/LDS/occupancy at compile time
4. `rocm-compute-profiler` → roofline chart (memory-bound vs compute-bound)

## GPU Kernel Correctness: Known Bugs and Fixes

### sc_N wildcard mask bug (FIXED — April 2025, AIOSS-4672)

**File**: `align/src/gpu/ksw2_extd2_gpu.cpp`, non-generic-sc kernel, two instances (~lines 186-187 and 710-711)

**Symptom**: On reads with a large insertion gap (e.g., 169-insertion), the GPU alignment split the read into 2 supplementary records (`zd:i:1` + `zd:i:2`) instead of 1 contiguous record. CPU produced correct `AS:i:19931 NM:i:372`; GPU produced wrong `AS:i=-496`.

**Root cause**: The GPU DP kernel computed the "wildcard match" substitution score `sc_N` only when **both** the target and query base were the wildcard base `m1` (AND logic). The CPU SSE kernel (`ksw2_extd2_sse.c`) correctly uses OR: `sc_N` applies when **either** base is the wildcard. This caused a 3-point score error at every position where exactly one base was the wildcard, causing premature z-drop on the first pass and a split supplementary read.

**Fix**:
```cpp
// WRONG (both bases must be wildcard — AND):
uint8_t mask = target[tid] == m1 ? 1 : 0;
mask *= (query[qlen - 1 - r + tid] == m1 ? 1 : 0);

// CORRECT (either base is wildcard — OR, matches CPU):
uint8_t mask = (target[tid] == m1) | (query[qlen - 1 - r + tid] == m1);
```

**Two-pass z-drop context**: `mm_align1` in `align.c` does two passes. Pass 1 uses `KSW_EZ_APPROX_MAX` (approximate max tracking, no exact z-drop). If the GPU kernel's score was wrong in pass 1, `mm_test_zdrop()` received a truncated CIGAR → triggered a second pass that only re-aligned the truncated region → split into supplementary records.

**How to avoid in future GPU kernel ports**: Always diff the substitution score computation (especially the wildcard/N-base path) against `ksw2_extd2_sse.c`'s `__dp_code_block1` macro. The GPU query array is pre-reversed (`qseq_rev[i] = query[qlen-1-i]`) so `GPU_kernel_query[qlen-1-r+t]` == `original_query[r-t]` — this is correct, not a bug.

### 64-thread chunk carry bug (FIXED — April 2025, AIOSS-4672)

**File**: `align/src/gpu/ksw2_extd2_gpu.cpp`

**Symptom**: Alignment scores for reads spanning multiple 64-thread chunks were wrong.

**Root cause**: After writing updated values to `x[tid]`, `x2[tid]`, `v[tid]` for the current diagonal, the code passed those new values (not the prior-diagonal boundary values) as carry into the next chunk. This read current-diagonal boundary values via `__shfl(x_new, last_lane)` when it should read prior-diagonal values.

**Fix**: Save prior-diagonal boundary values **before** any writes using `__shfl(x[tid], last_lane)`, then use those saved values as carry at the end of the diagonal loop.

---

## Known Issues

- **GPU chaining output parity (mostly resolved)**: As of AIOSS-3664 (PR #122), a finite `--max-chain-skip` routes through backward-scan pull kernels that are **bit-exact** with the CPU (byte-identical PAF on full map-ont/map-hifi/map-pb), and the default push path is now **deterministic** run-to-run. The earlier reports of 0.3–7.2% chain-score divergence and hg00438 non-deterministic chain counts (7731/7733/7861) described the pre-fix push DP; they no longer reproduce. The only residual divergence is the push-infinity boundary tie below.
- **GPU push (infinity) chaining is not bit-exact with the CPU**: With the default `--gpu-chain` (which uses `max_chain_skip = infinity` → the forward-push kernel `score_generation_long_push`), a tiny fraction of reads (~0.012% on full map-ont: 6 of 51,253; bit-exact on map-hifi/map-pb) produce a slightly different chain composition (`cm:i` off by 1–2; same locus, identical `dv:f`). Root cause: on high-anchor repeat reads where the `max_chain_iter` window cap (default 5000) binds, the push DP breaks an equal-score predecessor tie at the window boundary differently from the CPU. It is **deterministic** (stable run-to-run) and benign (alignment quality unaffected). Ruled out as causes: FMA/FP-contraction, the forward/backward window-count cap, and cross-reference predecessors. The push path is kept as the default for performance; **any finite `--max-chain-skip` routes through the backward-scan pull kernels, which are bit-exact with the CPU** (verified IDENTICAL on full map-ont/hifi/pb). Use a finite `--max-chain-skip` when byte-identical output is required. Routing infinity through the pull kernel would be bit-exact but ~3.6× slower (full map-ont t32: 297s → ~1080s), so it is intentionally not the default. See the NOTE on `score_generation_long_push` in `minimap2/chain/src/gpu/plscore.cu`.
- **GPU align stage serialization at high thread counts**: The GPU align stage is 0.13–0.66× of CPU align across presets. Serialization worsens at t32 (align stage inflates up to 7.9× slower for map-hifi). Batching z-drop second passes (AIOSS-4684) is the primary fix target.
- **GPU chain/align memory overhead**: GPU chain modes add ~30–38 GB RSS over baseline. `gpu_both` peaks at 55–73 GB at t16/t32. VRAM budget (`MM_GPU_VRAM_UTIL_FRAC`) was tuned to 0.50 in v2.24.0a4 to prevent `grow()` reallocation spikes exceeding 192 GB on MI300X.

## PR Requirements

Every PR commit message must include: ticket number, what changed, why it was needed, and how it was tested. See `.github/PULL_REQUEST_TEMPLATE.md`.
