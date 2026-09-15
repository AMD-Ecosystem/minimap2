---
applyTo: "tests/**"
description: "Testing conventions for minimap2 unit tests (Google Test). Use when writing or modifying test files under tests/core/ or tests/gpu/."
---

# Tests

- Tests use **Google Test** from `external/googletest/` (submodule — don't modify).
- CPU tests are in `tests/core/` (binary: `test_core`). GPU tests are in `tests/gpu/` (binary: `test_gpu`).
- Follow the **dual-pattern**: `TEST_P` for parameterized exact-match verification against baselines, `TEST_F` for behavioral/invariant checks on a single dataset.
- **Test data is generated**, not hand-written. Use `scripts/generate_test_data.py` to produce JSON from actual minimap2 runs. Never manually create expected JSON data.
- Test datasets live under `test_suite/` with subdirectories (`original/`, `small/`, `medium/`, `skip-large/`, `hg00438/`). `skip-large/` is skipped by default.
- Shared fixtures are in `test_fixtures.h`, helpers in `test_helpers.h`, data loading in `test_data_loader.cpp/.h`.
- Build and run: `cmake --preset amd-default && cmake --build --preset amd-default -j$(nproc) && ctest --test-dir out/amd-default/build --output-on-failure`
- Filter tests: `ctest -R <pattern>` or `<binary> --gtest_filter="<pattern>"`
- For full details see [`tests/README.md`](../tests/README.md).
