# Project Conventions

Reference for human contributors to this repository. AI coding assistants follow additional behavioral rules in [`AGENTS.md`](./AGENTS.md).

## Repository Layout

| Path | Description |
|---|---|
| `minimap2/seed/`, `chain/`, `align/`, `core/` | Pipeline stages (CPU) |
| `minimap2/chain/src/gpu/plchain.cu` | GPU chaining (HIP/ROCm) |
| `minimap2/align/src/gpu/` | GPU alignment (HIP/ROCm) |
| `cli/` | CLI entry point (`mm2_chain.cpp`) |
| `tests/core/`, `tests/gpu/` | Google Test unit tests |
| `python/` | Python bindings |
| `external/` | Submodules — do not modify |

See [`BUILD_GUIDE.md`](./BUILD_GUIDE.md) for build details and [`tests/README.md`](./tests/README.md) for tests.

## Build & Test Commands

Build via CMake presets defined in `CMakePresets.json`.

| Task | Command |
|---|---|
| CPU-only build + tests | `cmake --preset test-cpu && cmake --build --preset test-cpu -j$(nproc)` |
| AMD GPU build + tests | `cmake --preset amd-default && cmake --build --preset amd-default -j$(nproc)` |
| Benchmarking | `cmake --preset release && cmake --build --preset release -j$(nproc)` |
| Profiling (ITT/ROCTX) | `cmake --preset profile && cmake --build --preset profile -j$(nproc)` |
| Run tests | `ctest --test-dir out/<preset>/build --output-on-failure` |
| Filter tests | `ctest -R <pattern>` or `<binary> --gtest_filter="<pattern>"` |

**Preset selection:**
- **Benchmarking / performance measurement** → `release`. Don't benchmark on `amd-default` (it's `RelWithDebInfo` with extras enabled — slower than `release`) or any debug/sanitizer/coverage preset.
- **Profiling / tracing** → `profile`. Enables `ENABLE_PROFILING` and `ENABLE_TRACE` for ITT (Intel) and ROCTX (AMD) instrumentation, plus debug symbols.
- **Tests / dev iteration** → `amd-default`.

See [`BUILD_GUIDE.md`](./BUILD_GUIDE.md) for the full preset list.

## Scripts

The `scripts/` directory contains tooling for benchmarking, profiling, test-data generation, coverage, and data utilities. See [`scripts/README.md`](./scripts/README.md) for the full index, arguments, and examples.

| Category | Scripts |
|---|---|
| Benchmarking | `generate_benchmark_data.py`, `generate_all_mode_benchmark_data.py`, `compare_benchmark_data.py` |
| Profiling | `analyze_stages.py`, `generate_profiling_report.py` |
| Test data | `generate_all_test_data.sh`, `generate_test_data.py`, `compare_json_intermediates.py` |
| Validation | `test_output_formats.py`, `coverage.sh` |
| Data utilities | `download_preset_data.py`, `subset_queries.py` |

When adding new common tasks, prefer adding a new script (with an entry in `scripts/README.md`) over chat-only commands.

## Code Formatting

Run `clang-format -i` on first-party C/C++/CUDA files (`.c`, `.cc`, `.cpp`, `.h`, `.hpp`, `.cu`, `.cuh`) using the project's `.clang-format` config. Don't override styles inline.

```bash
clang-format -i path/to/file.c
```

To format all modified first-party files:

```bash
git diff --name-only --diff-filter=ACM \
  | grep -E '^minimap2/.*\.(c|cc|cpp|h|hpp|cu|cuh)$' \
  | xargs -r clang-format -i
```

The `.clang-format` config uses tabs, Linux braces, and minimizes whitespace changes. Don't modify it without coordination.

## Coding Style

- **Minimize comments; be concise.** Code should be self-explanatory and self-documenting.
- **Comments should be useful** — for example, comments that remind the reader about some global context that is non-obvious and can't be inferred locally.
- **Don't make trivial (1–2 LOC) helper functions** that are only used once, unless it significantly improves readability.
- **Prefer clear abstractions. State management should be explicit.** For example, if managing state in a Python class, there should be a clear class definition that has all of the members. Don't dynamically `setattr` a field on an object and then dynamically `getattr` the field on the object.
- **Match existing code style and architectural patterns.** Stay close to the surrounding-file style; don't introduce a different style for new code in an existing module.
- **`stdout` is reserved for SAM/PAF output** in `minimap2/` source files. All diagnostic messages, logging, and status output must go to `stderr`. Writing anything else to `stdout` corrupts downstream pipelines (e.g. `minimap2 ... | samtools`). Refer to the logger destination table:

  | Function | Destination |
  |----------|------------|
  | `mm_print()` | **stdout** |
  | `mm_log_info()` | **stdout** |
  | `mm_log()` | stderr |
  | `mm_log_debug()` | stderr |
  | `mm_log_trace()` | stderr |
  | `mm_log_warn()` | stderr |
  | `mm_log_error()` | stderr |
  | `mm_gpu_log_info()` | stderr |

  Use `mm_log_debug`/`mm_log_warn`/`mm_log_error` for diagnostics — not `mm_log_info`. See [`docs/ARCHITECTURE_GUIDE.md`](./docs/ARCHITECTURE_GUIDE.md#logging-architecture) for full details.

If uncertain, choose the simpler, more concise implementation.
