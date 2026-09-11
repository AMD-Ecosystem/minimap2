# Minimap2 Scripts

Utility scripts for benchmarking, profiling, testing, validating, and
preparing data for the minimap2 aligner.

## Script Index

### Benchmarking

| Script | Purpose |
|--------|---------|
| [generate_benchmark_data.py](#generate_benchmark_datapy) | Build, generate intermediates, and collect averaged timing stats |
| [generate_all_mode_benchmark_data.py](#generate_all_mode_benchmark_datapy) | Run benchmarks across multiple execution modes (CPU, GPU chain, GPU align, etc.) |
| [compare_benchmark_data.py](#compare_benchmark_datapy) | Compare 1 baseline against N test directories (intermediate correctness + timing) |

### Profiling

| Script | Purpose |
|--------|---------|
| [analyze_stages.py](#analyze_stagespy) | Profile the minimap2 pipeline across 6 stages using VTune, rocprofv3, and trace logs |
| [generate_profiling_report.py](#generate_profiling_reportpy) | Compare CPU vs GPU profiling results with optional graphs and Markdown report |

### Testing & Validation

| Script | Purpose |
|--------|---------|
| [generate_all_test_data.sh](#generate_all_test_datash) | Batch-build minimap2 and generate JSON test data for all test suites |
| [generate_test_data.py](#generate_test_datapy) | Generate a single JSON test data file from minimap2 `--dump-intermediates JSON` output |
| [compare_binary_intermediates.py](#compare_binary_intermediatespy) | Compare per-query FlatBuffers `.bin` intermediate files between two runs |
| [compare_json_intermediates.py](#compare_json_intermediatespy) | Compare per-query JSON intermediate files (seeds, chains, alignments) between two runs |
| [flatbuf_to_json.py](#flatbuf_to_jsonpy) | Convert a FlatBuffers intermediate dump (`.bin`) to JSON (single file or batch) |
| [test_output_formats.py](#test_output_formatspy) | Validate SAM/PAF output across CPU and GPU modes using samtools and paftools.js |
| [compare_sam.py](#compare_sampy) | Compare SAM alignment correctness (primary, supplementary set, AS) between a baseline and test run(s) |
| [run_accuracy_eval.py](#run_accuracy_evalpy) | Ground-truth accuracy floor: `mapeval` (simulated ONT/HiFi/PacBio/short reads) + `junceval` (splice vs GENCODE) across one or more minimap2 builds |
| [run_giab_concordance.py](#run_giab_concordancepy) | GIAB HG002 small-variant concordance (`hap.py` + Clair3) across one or more minimap2 builds/modes |
| [coverage.sh](#coveragesh) | Generate code coverage reports (HTML, XML, JSON, Markdown) |

### Data Utilities

| Script | Purpose |
|--------|---------|
| [download_preset_data.py](#download_preset_datapy) | Download GRCh38 reference genome and representative real-read datasets for every minimap2 preset (long-read, short-read, splice RNA, assembly) |
| [download_verification_data.py](#download_verification_datapy) | Acquire ground-truth verification inputs: GENCODE annotation, GIAB benchmark VCF/BED, and opt-in HG002 HiFi region reads |
| [subset_queries.py](#subset_queriespy) | Count or extract a subset of reads from a FASTA/FASTQ file |

### Shared Modules

| Module | Purpose | Used by |
|--------|---------|---------|
| [json_dump_utils.py](#json_dump_utilspy) | Shared utilities for parsing `--dump-intermediates JSON` output | generate_test_data.py, compare_json_intermediates.py, compare_benchmark_data.py |
| [stage_defs.py](#stage_defspy) | Pipeline stage constants and helpers | analyze_stages.py, generate_profiling_report.py |

---

# Benchmarking

The typical workflow to benchmark the AMD fork against the upstream v2.24
baseline:

### 1. Generate baseline data (upstream v2.24)

Check out the `upstream_v2.24_with_benchmarking` branch (which has the
JSON dump and timer instrumentation backported to official v2.24) and run
`generate_benchmark_data.py` to produce intermediates and averaged timing:

```bash
cd tmp/minimap2_v224
python3 scripts/generate_benchmark_data.py \
    -r /data/ref.fa -q /data/reads.fq \
    -o /tmp/benchmark_v224 -n 5
```

### 2. Generate AMD fork data (all modes)

Switch back to the AMD fork and run `generate_all_mode_benchmark_data.py`
to benchmark across CPU and GPU execution modes:

```bash
cd /workspaces/minimap2
python3 scripts/generate_all_mode_benchmark_data.py \
    -r /data/ref.fa -q /data/reads.fq \
    -o /tmp/benchmark_amd -t 1 32 -n 5
```

### 3. Compare results

Compare the baseline against one or more AMD mode directories:

```bash
python3 scripts/compare_benchmark_data.py \
    /tmp/benchmark_v224 \
    /tmp/benchmark_amd/cpu_t32 \
    /tmp/benchmark_amd/gpu_both_t32 \
    --labels AMD-cpu AMD-gpu
```

This reports both **intermediate correctness** (per-query seed/chain/alignment
comparison) and **timing comparison** (wall time, per-stage breakdown, speedup).

---

## generate_benchmark_data.py

Build minimap2, generate intermediates (FLATBUF or JSON), and collect averaged
timing stats across multiple runs.

### Prerequisites

- CMake (for building minimap2)
- Python 3.6+

### Usage

```bash
python3 scripts/generate_benchmark_data.py -r <ref.fa> -q <query.fa> [OPTIONS] [-- MINIMAP2_ARGS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-r, --ref` | — | Reference FASTA (required) |
| `-q, --query` | — | Query FASTA/FASTQ (required) |
| `-n, --runs` | `5` | Number of timing runs (0 to skip timing) |
| `-t, --threads` | `3` | Number of threads |
| `-o, --outdir` | `benchmark_output` | Output directory |
| `-p, --preset` | — | Minimap2 preset (e.g. `map-ont`) |
| `-B, --skip-build` | off | Skip the build step |
| `--skip-intermediates` | off | Skip intermediate dump generation |
| `--build-preset` | `amd-default` | CMake build preset |
| `--binary` | auto-detect | Explicit path to minimap2 binary |
| `--auto-download` | off | Fetch any missing `--ref`/`--query`/`--query2` input via `download_preset_data.py` instead of failing |
| `--dump-format` | `FLATBUF` | Dump format: `FLATBUF` or `JSON` |
| `--compress` | off | Compress intermediates into a `.zip` archive (zstd per file) |
| `-- [args]` | — | Extra arguments passed to minimap2 |

### Output

```
<outdir>/
  intermediates/
    *.bin                          FlatBuffers binary files (default)
    intermediates.zip              (with --compress) zstd-compressed .bin.zst inside zip
    stats/timer_summary.json       Per-stage timing from intermediates run
  timing_runs/
    run_1/stats/timer_summary.json
    run_2/stats/timer_summary.json
    ...
  timer_summary_avg.json           Averaged timing across all runs
```

When `--dump-format JSON` is used, intermediates contain per-query JSON files
in `seeds/`, `chains/`, `alignments/` subdirectories.
When `--compress` is used, all intermediate files are zstd-compressed and packed into
`intermediates.zip` (originals deleted). Requires the `zstandard` Python package.

### Examples

```bash
# Basic benchmark (5 runs, 3 threads, FLATBUF intermediates)
python3 scripts/generate_benchmark_data.py -r ref.fa -q reads.fq

# 10 runs, skip build, use specific binary
python3 scripts/generate_benchmark_data.py -r ref.fa -q reads.fq \
    -n 10 -B --binary ./out/release/build/bin/minimap2

# Intermediates only, no timing runs
python3 scripts/generate_benchmark_data.py -r ref.fa -q reads.fq -n 0

# Timing only, skip intermediates
python3 scripts/generate_benchmark_data.py -r ref.fa -q reads.fq --skip-intermediates

# JSON intermediates instead of FLATBUF
python3 scripts/generate_benchmark_data.py -r ref.fa -q reads.fq --dump-format JSON

# Compress intermediates into zip archive
python3 scripts/generate_benchmark_data.py -r ref.fa -q reads.fq --compress

# With minimap2 preset and extra flags
python3 scripts/generate_benchmark_data.py -r ref.fa -q reads.fq \
    -p map-ont -- --cs --MD
```

---

## generate_all_mode_benchmark_data.py

Run `generate_benchmark_data.py` across multiple minimap2 execution modes
and collect timing + intermediate data for each.

### Base Modes

Four base modes are defined.  Each is expanded across every thread count
given via `-t`, producing concrete modes like `cpu_t1`, `gpu_chain_t32`, etc.

| Base Mode | Description | Extra Flags |
|-----------|-------------|-------------|
| `cpu` | CPU only | — |
| `gpu_chain` | GPU chaining | `--gpu-chain --gpu-cfg <cfg>` |
| `gpu_align` | GPU alignment | `--gpu-align` |
| `gpu_both` | GPU chain + align | `--gpu-chain --gpu-align --gpu-cfg <cfg>` |

### Usage

```bash
python3 scripts/generate_all_mode_benchmark_data.py -r <ref.fa> -q <query.fa> [OPTIONS] [-- MINIMAP2_ARGS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-r, --ref` | — | Reference FASTA (required) |
| `-q, --query` | — | Query FASTA/FASTQ (required) |
| `-n, --runs` | `5` | Number of timing runs per mode (0 to skip timing) |
| `-t, --threads` | `32` | Thread count(s) — accepts a list (e.g. `-t 1 4 32`) |
| `-o, --outdir` | `all_mode_data` | Root output directory |
| `-p, --preset` | — | Minimap2 preset (e.g. `map-ont`) |
| `--build-preset` | `release` | CMake build preset |
| `--binary` | auto-detect | Explicit path to minimap2 binary |
| `--auto-download` | off | Fetch any missing `--ref`/`--query`/`--query2` input via `download_preset_data.py` instead of failing |
| `-B, --skip-build` | off | Skip the build step |
| `--skip-intermediates` | off | Skip intermediate dump generation in each mode |
| `--no-compress` | off | Keep raw intermediate files (default: compress into `.zip`) |
| `--gpu-cfg` | `configs/gpu_config.json` | GPU config file path |
| `--skip-gpu` | off | Skip all GPU modes |
| `--modes` | all | Base modes to run: `cpu`, `gpu_chain`, `gpu_align`, `gpu_both` |
| `--no-color` | off | Disable colored output |
| `--export-json` | off | Export combined results to `all_mode_results.json` |
| `-- [args]` | — | Extra arguments passed to minimap2 |

### Output

```
<outdir>/
  cpu_t1/                    One directory per mode × thread count
    intermediates/
    timing_runs/
    timer_summary_avg.json
  cpu_t32/
  gpu_chain_t32/
  gpu_both_t32/
  all_mode_results.json      Only if --export-json
```

### Examples

```bash
# All 4 modes at t=1 and t=32, 3 timing runs each
python3 scripts/generate_all_mode_benchmark_data.py -r ref.fa -q reads.fq -t 1 32 -n 3

# CPU-only at multiple thread counts
python3 scripts/generate_all_mode_benchmark_data.py -r ref.fa -q reads.fq --modes cpu -t 1 4 8 32

# GPU chain + align at 32 threads, intermediates only (no timing)
python3 scripts/generate_all_mode_benchmark_data.py -r ref.fa -q reads.fq --modes gpu_both -t 32 -n 0

# CPU-only (skip GPU)
python3 scripts/generate_all_mode_benchmark_data.py -r ref.fa -q reads.fq --skip-gpu

# Pass extra minimap2 flags
python3 scripts/generate_all_mode_benchmark_data.py -r ref.fa -q reads.fq -- --cs --MD
```

---

## compare_benchmark_data.py

Compare 1 baseline against N test directories. Checks both intermediate
correctness and timing (wall time, per-stage breakdown, speedup).

Automatically detects intermediate format and storage — FlatBuffers binary
files (`*.bin`) or per-query JSON files (`seeds/`, `chains/`, `alignments/`
directories), stored as loose files or `.zip` archives (zstd-compressed).
Prefers FLATBUF when both exist; prefers zip > directory.

### Prerequisites

- Baseline and test directories containing intermediates (FLATBUF or JSON) and/or timing data
- Python packages: `zstandard`, `tqdm`, `flatbuffers`
- Optional: `stats/timer_summary.json` or `timer_summary_avg.json` for timing

### Usage

```bash
python3 scripts/compare_benchmark_data.py <baseline> <test1> [test2 ...] [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `baseline` | — | Baseline data directory |
| `test_dirs` | — | One or more test data directories to compare |
| `--labels` | auto | Custom labels for each test directory |
| `-s, --stage` | all | Stage(s) to compare: `seeds`, `chains`, `alignments` (repeatable) |
| `-r, --read` | — | Filter to reads matching substring |
| `-v, --verbose` | off | Show per-file match details |
| `--summary-only` | off | Only show summary tables |
| `--no-color` | off | Disable colored output |
| `-o, --output` | — | Export results to JSON file |
| `--timing-only` | off | Skip intermediate comparison |
| `--intermediates-only` | off | Skip timing comparison |

### Examples

```bash
# Compare one test against baseline
python3 scripts/compare_benchmark_data.py /tmp/json_v224 /tmp/json_amd

# Multiple tests with custom labels
python3 scripts/compare_benchmark_data.py baseline/ run1/ run2/ --labels v224 AMD-cpu AMD-gpu

# Chains only, verbose
python3 scripts/compare_benchmark_data.py baseline/ test/ -s chains -v

# Timing comparison only
python3 scripts/compare_benchmark_data.py baseline/ t1/ t2/ --timing-only

# Export results
python3 scripts/compare_benchmark_data.py baseline/ test/ -o results.json
```

---

# Profiling

## analyze_stages.py

Profile the minimap2 pipeline by classifying CPU and GPU functions into
6 stages (seeding, chaining, alignment, I/O, gpu_other, other) using up to
3 profiling backends.

### Prerequisites

- Intel VTune (`vtune`) on PATH — skipped gracefully if absent
- `rocprofv3` on PATH — skipped gracefully if absent
- minimap2 built with profile preset for trace logs
- `ptrace` capability required in containers (for VTune/rocprof)

### Usage

```bash
python3 scripts/analyze_stages.py [OPTIONS] -- <minimap2 args>
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-m, --minimap2` | auto-detect | Path to minimap2 binary |
| `-o, --output` | — | Output directory for results |
| `-f, --format` | `text` | Output format: `text` or `json` |
| `--no-vtune` | off | Skip VTune profiling |
| `--no-rocprof` | off | Skip rocprofv3 GPU profiling |
| `--no-trace` | off | Skip trace log timing |
| `--reuse-data` | off | Reuse existing profiles from output dir (no re-run) |
| `--min-duration` | `0` | Filter functions below this duration in microseconds |
| `mm_args` | — | Minimap2 arguments (placed after `--` separator) |

### Output

| File | Description |
|------|-------------|
| `{output}/analysis.json` | Structured analysis (always saved) |
| `{output}/vtune_*/` | VTune raw data directory |
| `{output}/rocprof_*/` | rocprofv3 raw data directory |
| `{output}/trace.log` | Trace timing log |

### Examples

```bash
# CPU-only analysis
python3 scripts/analyze_stages.py -- -ax map-ont ref.fa reads.fa

# With GPU chaining
python3 scripts/analyze_stages.py -- -ax map-ont --gpu-chain \
    --gpu-cfg gpu/gpu_config.json --gpu-align ref.fa reads.fa

# Skip VTune (rocprof + trace only)
python3 scripts/analyze_stages.py --no-vtune -- -ax map-ont --gpu-chain \
    --gpu-cfg gpu/gpu_config.json --gpu-align ref.fa reads.fa

# Reuse saved profiles, just regenerate analysis
python3 scripts/analyze_stages.py --reuse-data -o /tmp/mm2_analysis
```

---

## generate_profiling_report.py

Compare CPU-only vs GPU profiling results from `analyze_stages.py`. Reads two
`analysis.json` files and prints a 6-section text comparison report to stdout.
Optionally generates matplotlib graphs and a Markdown report.

### Prerequisites

- Two `analysis.json` files generated by `analyze_stages.py`
- `matplotlib` (optional, required only when `--output-dir` is used)

### Usage

```bash
# Text-only comparison (stdout):
python3 scripts/generate_profiling_report.py <cpu_analysis.json> <gpu_analysis.json>

# Full report with graphs and Markdown:
python3 scripts/generate_profiling_report.py <cpu_analysis.json> <gpu_analysis.json> \
    --output-dir <report_dir>
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `cpu_analysis` | — | Path to CPU-only `analysis.json` |
| `gpu_analysis` | — | Path to GPU `analysis.json` |
| `--output-dir, -o` | — | Directory for graphs and Markdown report (optional) |

### Text Report Sections (stdout)

1. **Overall Summary** — Total trace time, GPU kernel time, memory usage, speedup
2. **CPU-Only Mode Functions** — All traced functions by stage
3. **GPU Mode Functions** — All traced functions by stage
4. **GPU Kernel Bottleneck Analysis** — Kernels by stage, top bottlenecks by avg time, hardware counters, transfers, HIP API overhead
5. **Function Comparison** — Replaced functions, GPU overhead, new GPU-only functions
6. **Summary & Bottleneck Identification** — Regressions, potential savings

### Generated Graphs (with `--output-dir`)

| File | Description |
|------|-------------|
| `stage_timing.png` | Per-stage CPU vs GPU bar chart |
| `stage_pie.png` | Time distribution pie charts |
| `speedup.png` | Per-stage speedup bar chart (green = faster, red = slower) |
| `kernel_bottleneck.png` | Top GPU kernels by total time |
| `hip_api_overhead.png` | Top 10 HIP API calls by total time |
| `vtune_hotspots.png` | VTune CPU hotspot functions by stage |
| `memory_timeline.png` | GPU VRAM + CPU RSS over time (copied from GPU run) |
| `PROFILING_REPORT.md` | 12-section Markdown report with embedded graph references |

### Examples

```bash
# Text comparison only
python3 scripts/generate_profiling_report.py \
    /data/profiling/hifi_cpu/analysis.json \
    /data/profiling/hifi_gpu/analysis.json

# Full report with graphs
python3 scripts/generate_profiling_report.py \
    /data/profiling/hifi_cpu/analysis.json \
    /data/profiling/hifi_gpu/analysis.json \
    --output-dir /data/profiling/report
```

---

# Testing & Validation

## generate_all_test_data.sh

Batch-build minimap2 and generate JSON test data for all test suites.
Discovers test suites automatically and pairs matching reference/query files.

### Prerequisites

- CMake (for `cmake --preset` / `--build`)
- Python 3 (runs `generate_test_data.py` per pair)
- Test suites under `test_suite/` with `t*.fa` and `q*.fa` files

### Usage

```bash
./scripts/generate_all_test_data.sh [CMAKE_PRESET]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `CMAKE_PRESET` (arg or env) | `coverage` | CMake preset to build with |
| `TEST_SUITE_BASE` (env) | `test_suite` | Base directory for test suites |

### Output

```
test_suite/{suite}/expected/{ref}_{query}.json
```

For example:
```
test_suite/
  medium/
    t0.fa, q0.fa
    expected/
      t0_q0.json
  small/
    t0.fa, q0.fa
    expected/
      t0_q0.json
```

### Examples

```bash
# Default (coverage preset)
./scripts/generate_all_test_data.sh

# Specific preset
./scripts/generate_all_test_data.sh test-cpu

# Via environment variable
CMAKE_PRESET=debug ./scripts/generate_all_test_data.sh

# Custom test suite location
TEST_SUITE_BASE=/custom/path ./scripts/generate_all_test_data.sh
```

### Integration with Unit Tests

Generated JSON files are auto-discovered by the test framework via
`test_suite/*/expected/*.json`. Each file becomes a parameterized test case:

```cpp
INSTANTIATE_TEST_SUITE_P(
    AllTestData, MapTest,
    ::testing::ValuesIn(getAllTestData()),
    testDataNameGenerator
);
```

To use a custom path for both generation and testing:

```bash
TEST_SUITE_BASE=/custom/path ./scripts/generate_all_test_data.sh
TEST_SUITE_DIR_ENV=/custom/path ./out/test-cpu/build/tests/core/test_core
```

---

## generate_test_data.py

Generate a single JSON test data file by running minimap2 with `--extra-out-dir`
and `--dump-intermediates JSON`, combining per-query JSON output. Most users
should prefer `generate_all_test_data.sh` for batch generation.

### Prerequisites

- minimap2 binary (auto-detected from build tree, or pass `--minimap2`)
- Python 3.6+

### Usage

```bash
python3 scripts/generate_test_data.py \
    --ref <reference.fa> --query <query.fa> --output <output.json> \
    [--minimap2 PATH] [-x PRESET] [--name NAME] [--description DESC] [--dry-run]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--ref` | yes | Reference FASTA file |
| `--query` | yes | Query FASTA file |
| `--output` | yes | Output JSON file path |
| `--minimap2` | no | Path to minimap2 binary (auto-detected if omitted) |
| `-x, --preset` | no | minimap2 preset (e.g. `map-ont`, `map-pb`, `map-hifi`, `sr`) |
| `--name` | no | Test name (auto-generated from file names if omitted) |
| `--description` | no | Human-readable test description |
| `--dry-run` | no | Print combined JSON to stdout without writing file |

### Output JSON Schema

```json
{
  "name": "medium_t0_q0",
  "description": "...",
  "reference": "ref.fa",
  "query": "query.fa",
  "preset": "map-ont",
  "queries": [
    {
      "query_name": "read1",
      "seeds": [ ... ],
      "chains": [ ... ],
      "alignments": [
        {
          "query_name": "...", "query_start": 0, "query_end": 16025,
          "target_name": "...", "target_start": 576, "target_end": 16569,
          "strand": "+", "mapq": 60, "cigar": "...", "score": 3187,
          "y_position": 16065, "NM": 42, "blen": 16200
        }
      ]
    }
  ]
}
```

### Examples

```bash
# Generate test data for medium dataset
python3 scripts/generate_test_data.py \
    --ref test_suite/medium/t0.fa \
    --query test_suite/medium/q0.fa \
    --output test_suite/medium/expected/t0_q0.json \
    --minimap2 ./out/coverage/build/minimap2 \
    --name "medium_t0_q0"

# Auto-detect minimap2 binary
python3 scripts/generate_test_data.py \
    --ref test_suite/original/MT-human.fa \
    --query test_suite/original/MT-orang.fa \
    --output test_suite/original/expected/mt_human_orang.json

# Dry run to verify output
python3 scripts/generate_test_data.py \
    --ref test.fa --query query.fa --output /tmp/test.json --dry-run
```

---

## compare_json_intermediates.py

Compare per-query JSON intermediate files (seeds, chains, alignments) between
two minimap2 runs to identify mismatches between CPU and GPU modes or different
configurations.

### Prerequisites

- Two directories of JSON intermediates produced by `minimap2 --extra-out-dir DIR --dump-intermediates JSON`

### Usage

```bash
python3 scripts/compare_json_intermediates.py <dir1> <dir2> [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `dir1` | — | First directory with JSON intermediates |
| `dir2` | — | Second directory with JSON intermediates |
| `-s, --stage` | all | Stage(s) to compare: `seeds`, `chains`, `alignments` (repeatable) |
| `-r, --read` | — | Filter to reads whose filename contains this substring |
| `-v, --verbose` | off | Show all comparisons including matches |
| `-o, --output` | — | Export mismatches to JSON file |
| `--summary-only` | off | Only show summary, skip per-file output |

### Expected Directory Layout

```
dir1/
  seeds/         query1.json, query2.json, ...
  chains/        query1.json, query2.json, ...
  alignments/    query1.json, query2.json, ...
```

### Comparison Details

| Stage | Fields Compared |
|-------|----------------|
| Seeds | `(x, y)` coordinate pairs, count, query name |
| Chains | score, length, first/last qpos/tpos, strand, rid |
| Alignments | score, dp_score, qs, qe, rs, re, mapq, rev, rname, cigar |

### Examples

```bash
# Compare two full intermediate directories
python3 scripts/compare_json_intermediates.py /path/to/cpu_run /path/to/gpu_run

# Compare only chains with verbose output
python3 scripts/compare_json_intermediates.py dir1 dir2 --stage chains -v

# Compare specific read
python3 scripts/compare_json_intermediates.py dir1 dir2 --read "m64136_200706_123635/1008/ccs"

# Export mismatches for further analysis
python3 scripts/compare_json_intermediates.py dir1 dir2 -o mismatches.json

# Summary only
python3 scripts/compare_json_intermediates.py dir1 dir2 --summary-only
```

---

## compare_binary_intermediates.py

Compare per-query FlatBuffers `.bin` intermediate files (seeds, chains,
alignments) between two minimap2 runs. Used standalone or as a library by
`compare_benchmark_data.py`.

### Prerequisites

- Two directories of FlatBuffers `.bin` files produced by `minimap2 --dump-intermediates FLATBUF`
- FlatBuffers Python bindings (under `scripts/generated/`)

### Usage

```bash
python3 scripts/compare_binary_intermediates.py <dir1> <dir2> [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `dir1` | — | First directory with `.bin` files |
| `dir2` | — | Second directory with `.bin` files |
| `-s, --stage` | all | Stage(s) to compare: `seeds`, `chains`, `alignments` (repeatable) |
| `-r, --read` | — | Filter to reads whose filename contains this substring |
| `-v, --verbose` | off | Show all comparisons including matches |
| `-o, --output` | — | Export mismatches to JSON file |
| `--summary-only` | off | Only show summary, skip per-query output |

### Comparison Details

| Stage | Fields Compared |
|-------|----------------|
| Seeds | `(x, y)` coordinate pairs, count |
| Chains | score, length, first/last qpos/tpos, rev, rid, rname |
| Alignments | seg_id, rid, rname, score, dp_score, qs, qe, rs, re, mapq, rev, mlen, blen, n_sub, sam_pri, cigar, n_cigar_ops, dp_max, n_ambi |

### Examples

```bash
# Compare two FlatBuffers dump directories
python3 scripts/compare_binary_intermediates.py /tmp/cpu_dump/ /tmp/gpu_dump/

# Compare only chains with verbose output
python3 scripts/compare_binary_intermediates.py dir1/ dir2/ --stage chains -v

# Export mismatches
python3 scripts/compare_binary_intermediates.py dir1/ dir2/ -o mismatches.json
```

---

## flatbuf_to_json.py

Convert a FlatBuffers intermediate dump (`.bin`, produced by
`minimap2 --dump-intermediates FLATBUF`) into human-readable JSON. Accepts a
single `.bin` file or a directory of them (batch mode). Useful for inspecting
intermediates by eye or feeding the JSON comparators.

### Usage

```bash
python3 scripts/flatbuf_to_json.py <input.bin | dir/> [-o OUTPUT]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | — | A single `.bin` file or a directory of `.bin` files |
| `-o, --output` | stdout | Output JSON file (single input) or output directory (batch) |

### Examples

```bash
# Print one file to stdout
python3 scripts/flatbuf_to_json.py query.bin

# Convert one file to JSON
python3 scripts/flatbuf_to_json.py query.bin -o query.json

# Batch-convert a directory of .bin files
python3 scripts/flatbuf_to_json.py dir_of_bins/ -o out/
```

---

## test_output_formats.py

Validate minimap2 SAM and PAF output across CPU and GPU modes using
`samtools` and `paftools.js`. Runs a matrix of (mode × format × capture)
combinations as independent test cases.

### Prerequisites

- `samtools` on PATH
- `k8` on PATH (JavaScript runtime for paftools.js)
- `misc/paftools.js` in the repo tree
- Both `samtools` and `k8` are included in the devcontainer

### Usage

```bash
python3 scripts/test_output_formats.py [minimap2] [reference.fa] [query.fa] [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `minimap2` | `out/amd-default/build/bin/minimap2` | Path to minimap2 binary |
| `reference` | `test/MT-human.fa` | Reference FASTA file |
| `query` | `test/MT-orang.fa` | Query FASTA file |
| `--cpu-only` | off | Skip all GPU modes |
| `--json [FILE]` | off | Write JSON report (to FILE, or stdout if no path given) |

### Test Matrix

| Dimension | Values |
|-----------|--------|
| **Modes** | cpu-only, gpu-chain, gpu-align, gpu-chain+align |
| **Formats** | SAM (`-a`), PAF (`-c`), PAF+cs (`-c --cs`) |
| **Capture** | file output (`-o`), stdout pipe |
| **Total** | up to 24 test combinations |

### Validation Checks

| Format | Checks |
|--------|--------|
| SAM | `samtools quickcheck` + `samtools flagstat` + SAM→BAM→SAM round-trip |
| PAF | 12-column format check + `k8 paftools.js stat` |
| PAF+cs | Same as PAF, with `--cs` tag for richer statistics |

### Examples

```bash
# Defaults (amd-default build, test/MT-human.fa, test/MT-orang.fa)
python3 scripts/test_output_formats.py

# CPU-only mode
python3 scripts/test_output_formats.py --cpu-only

# Custom binary and data
python3 scripts/test_output_formats.py /path/to/minimap2 ref.fa query.fa

# JSON report to file
python3 scripts/test_output_formats.py --json results.json

# JSON report to stdout
python3 scripts/test_output_formats.py --json
```

### Example Output

```
Active modes: cpu-only, gpu-chain, gpu-align, gpu-chain+align

Running tests...

  cpu-only             SAM    file   ... PASS (4.8s)
  cpu-only             SAM    stdout ... PASS (4.6s)
  cpu-only             PAF    file   ... PASS (4.6s)
  ...

Total: 24  Passed: 24  Failed: 0  Time: 414.4s
```

---

## compare_sam.py

Compare SAM alignment **correctness** between a baseline run and one or more
test runs produced from the same reads. Built for oracle-vs-fork verification
(e.g. upstream minimap2 vs the AMD fork's CPU/GPU modes), but works on any pair
of SAM files.

Unlike [test_output_formats.py](#test_output_formatspy) (which validates that
each output is *well-formed*), this script checks that two outputs *agree* on
where and how each read aligned.

### Prerequisites

- Python 3.6+ (standard library only)

### Usage

```bash
python3 scripts/compare_sam.py <baseline.sam> <test.sam> [test2.sam ...] [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `baseline` | — | Baseline/oracle SAM file (required) |
| `test` | — | One or more test SAM files (required) |
| `--labels` | file stem | Label per test file; count must match test files |
| `--as-tol` | `0` | AS score tolerance; diffs within this window are not flagged |
| `--max-samples` | `5` | Max AS-diff example rows to print per test |

### What it checks (per read)

| Check | Detail |
|-------|--------|
| Primary hit | Selected by SAM FLAG (not file order); `rname` + `pos` + `strand` + CIGAR must all match for an exact hit |
| Unmapped | FLAG `0x4` handled explicitly (both-unmapped vs unmapped-mismatch) |
| Supplementary/secondary | Full set of non-primary alignments compared (common GPU divergence) |
| AS score | Difference reported with optional tolerance window and signed delta |

Primary diff categories are mutually exclusive, so each read is counted once:
`exact` \| `pos-diff` \| `cigar-only-diff` \| `unmapped-mismatch` \| `missing`.

### Exit code

`0` if all test files were found and compared; `1` if any test file was
missing or the baseline did not exist.

### Memory note

Each SAM is loaded fully into memory (minimap2 output is not sorted by read
name, so streaming comparison is not possible). Peak RAM is roughly
proportional to total SAM size; very large all-vs-all (ava) outputs (>10 GB)
need a machine with comparable free RAM.

### Examples

```bash
# Single baseline vs test
python3 scripts/compare_sam.py oracle.sam fork.sam

# One baseline vs several modes, with labels
python3 scripts/compare_sam.py oracle.sam cpu.sam gpu_chain.sam gpu_both.sam \
    --labels cpu gpu_chain gpu_both

# Tolerate small AS differences and show more examples
python3 scripts/compare_sam.py oracle.sam gpu.sam --as-tol 5 --max-samples 10
```

### Example Output

```
cpu:
  Total baseline queries:    51253
  Exact primary match:       49407 (96.40%)
  Both unmapped:             1846
  Position/strand diff:      0
  CIGAR-only diff:           0
  Supp/secondary set diff:   0

gpu_align:
  Exact primary match:       49323 (96.23%)
  Position/strand diff:      51
  CIGAR-only diff:           33
  Supp/secondary set diff:   154
      <read>: baseline=612032, test=288758 (delta=-323274)
```

---

## coverage.sh

Build minimap2 with coverage instrumentation (GCC or Clang) and generate
code coverage reports in multiple formats.

### Prerequisites

| Toolchain | Requirements |
|-----------|-------------|
| Clang (default) | ROCm `clang`, `llvm-cov`/`llvm-profdata` (>= 19, to match the ROCm Clang profile format), `lcov_cobertura` |
| GCC | `gcc`, `gcov`, `gcovr` >= 8.0 |

### Usage

```bash
./scripts/coverage.sh [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--clean` | off | Remove previous build before generating coverage |
| `--skip-build` | off | Skip build and test, only regenerate reports |
| `--format` | `all` | Output format: `all`, `html`, `xml`, `json`, `markdown` |
| `--compiler` | `clang` | Compiler toolchain: `gcc` or `clang` |
| `--verbose` | off | Show verbose output |

### Output

Reports are written to `out/coverage[-clang]/build/coverage/`:

| File | Description |
|------|-------------|
| `html/index.html` | Interactive HTML report with source annotations |
| `coverage.xml` | Cobertura XML for CI/CD (Jenkins, GitLab, GitHub Actions) |
| `coverage.json` | JSON data (gcovr format for GCC, llvm-cov native for Clang) |
| `coverage.md` | Markdown summary table |
| `coverage.lcov` | LCOV tracefile (Clang only) |

### Examples

```bash
# Clang (default): full build + all reports, includes host-side GPU code
./scripts/coverage.sh

# GCC toolchain (CPU-only)
./scripts/coverage.sh --compiler gcc

# Clean rebuild + reports
./scripts/coverage.sh --clean

# Regenerate reports only (skip build)
./scripts/coverage.sh --skip-build

# HTML report only with GCC
./scripts/coverage.sh --format html --compiler gcc
```

> **Note:** Clang (default) includes host-side GPU/HIP code; device kernel code is
> not instrumented (ROCm provides no device-side profile runtime). GCC is CPU-only
> — gcov cannot instrument HIP/ROCm output.

---

## run_accuracy_eval.py

Measures an **absolute accuracy floor** (truth-based, not just cross-version
agreement) for **one or more** minimap2 builds across execution modes. Builds are
passed generically as `--binary NAME=PATH[=MODES]`, so the script makes no
assumption about which build is the "reference" — any labelled build can be the
baseline (it survives a future upstream merge). Two evaluations, both scored with
`misc/paftools.js`:

- **mapeval** — simulate reads from a chosen reference region, map them back to the
  **full** reference, and score the fraction of wrong mappings (truth coordinates
  ride in the read names). Long reads (`map-ont`, `map-hifi`, `map-pb`) use
  `pbsim3` + `paftools.js pbsim2fq`; short reads (`sr`) use `mason2` +
  `paftools.js mason2fq`.
- **junceval** — map real RNA reads with `-ax splice -uf` and score predicted
  splice junctions against a GENCODE GTF (`paftools.js junceval`).

The summary prints both the aggregate err% and a **per-mapQ accuracy floor** table
(cumulative err% at mapQ ≥ 60/30/10/1/0) — the aggregate is dominated by
deliberately-ambiguous low-mapQ reads, so the high-confidence bands are where a
real regression shows. It also prints a **GPU↔CPU primary divergence** table
(each GPU mode vs the same build's cpu mode: pos/strand, CIGAR-only, and mapQ
differences at the primary locus) so GPU-vs-CPU divergence is reported alongside
the accuracy numbers. Supplementary/secondary-set and AS differences are
additional — use [compare_sam.py](#compare_sampy) for the full SAM-level diff.

Simulated reads and per-mode SAMs are **cached** (reuse unless `--force`). Sim
filenames encode their parameters (depth / read count), and SAMs auto-invalidate
when the binary or reads are newer, so swapping in a rebuilt binary transparently
re-maps.

### Prerequisites

- `k8` + `misc/paftools.js` (scoring), `samtools` (region extraction/indexing).
- `pbsim3` with `ERRHMM-*.model` files (long-read mapeval).
- `mason2` (`mason_simulator`) for short-read mapeval — the `sr` point is skipped
  if it is not on `PATH` (dwgsim read names are **not** parsed by `mapeval`).
- A local reference FASTA + `.fai` on fast storage.
- junceval only: a GENCODE annotation GTF + real RNA reads.

See [Getting the data](#getting-the-data-for-run_accuracy_evalpy) below.

### Usage

```bash
python3 scripts/run_accuracy_eval.py --ref REF.fa --binary NAME=PATH[=MODES] [options]
```

### Key arguments

| Argument | Description |
|----------|-------------|
| `--ref` | Reference FASTA (**required**; mapping target + simulation source) |
| `--binary` | `NAME=PATH[=MODES]`, **repeatable**. NAME is a free-form label; optional `=MODES` (comma-separated subset of `cpu,gpu_chain,gpu_align,gpu_both`) overrides `--modes` for that build. A cpu-only reference build is `NAME=PATH=cpu`. |
| `--modes` | Default modes per build unless overridden (default: `cpu gpu_chain gpu_align`) |
| `--eval` | `mapeval`, `junceval`, or both (default: both) |
| `--sim-region` | Contigs to simulate from for mapeval (default: `chr21,chr22`) |
| `--sim-models` | Subset of `map-ont,map-hifi,map-pb,sr` (default: all four) |
| `--sim-depth` | pbsim3 simulated depth (default: `10`) |
| `--sr-num-reads` / `--sr-read-length` | mason2 fragment pairs / read length (default: `2000000` / `150`) |
| `--pbsim-data` | Dir with `ERRHMM-*.model` files (default: `~/.local/src/pbsim3/data`) |
| `--gpu-cfg` | GPU config JSON for GPU chain modes (default: none — built-in defaults) |
| `--junc-subset` | Map only the first N RNA reads for junceval (default: `100000`) |
| `--splice-reads` / `--gtf` | RNA reads / GENCODE GTF (required when `--eval junceval`) |
| `--workdir` | Output dir for sims/SAMs (default: `accuracy_eval`) |
| `--auto-download` | Fetch any missing `--ref`/`--gtf`/`--splice-reads` input via the `download_*_data.py` helpers instead of failing |
| `--force` | Regenerate sims/SAMs even if present |

### Examples

```bash
# One build, all sim points, default modes (per-mapQ floor printed)
python3 scripts/run_accuracy_eval.py --ref ref.fa \
  --binary dev=out/release-gfx942/build/bin/minimap2 --eval mapeval

# Two builds: a cpu-only reference vs a GPU dev build
python3 scripts/run_accuracy_eval.py --ref ref.fa \
  --binary v2.31=/path/to/upstream/minimap2=cpu \
  --binary dev=out/release-gfx942/build/bin/minimap2 \
  --eval mapeval --modes cpu gpu_chain gpu_align \
  --sim-models map-ont,map-hifi,map-pb,sr -t 32

# junceval only, first 100k cDNA reads
python3 scripts/run_accuracy_eval.py --ref ref.fa \
  --binary dev=out/release-gfx942/build/bin/minimap2 \
  --eval junceval --junc-subset 100000 \
  --splice-reads reads.fastq --gtf gencode.v46.annotation.gtf
```

A correct build yields **cpu err% identical across versions**, with GPU modes
within statistical noise (tie/chain ordering).

### Getting the data for run_accuracy_eval.py

All inputs are local files / user-space tools (no root required).

**Reference FASTA** (GRCh38) — download once and index:

```bash
python3 scripts/download_preset_data.py        # fetches GRCh38 ref.fa
samtools faidx ref.fa                           # build ref.fa.fai
```

**k8 + paftools.js** — `paftools.js` ships in `misc/`; install the `k8`
JavaScript shell on `PATH` from its release page
([attractivechaos/k8](https://github.com/attractivechaos/k8/releases)) — download
the prebuilt binary for your platform and symlink it as `~/.local/bin/k8`.

**pbsim3 + ERRHMM models** (long-read mapeval) — build pbsim3 and point
`--pbsim-data` at its `data/` dir (contains `ERRHMM-ONT`, `ERRHMM-ONT-HQ`,
`ERRHMM-RSII`, ...):

```bash
git clone https://github.com/yukiteruono/pbsim3 ~/.local/src/pbsim3
cd ~/.local/src/pbsim3 && ./configure && make
ln -sf $PWD/src/pbsim ~/.local/bin/pbsim3
# models are in ~/.local/src/pbsim3/data  (the --pbsim-data default)
```

**mason2** (short-read mapeval) — no upstream prebuilt binary tarball exists;
extract the Debian `seqan-apps` package into `~/.local` (no root):

```bash
url=http://ftp.debian.org/debian/pool/main/s/seqan2/seqan-apps_2.4.0+dfsg-15_amd64.deb
curl -fL -o /tmp/seqan-apps.deb "$url"
dpkg-deb -x /tmp/seqan-apps.deb /tmp/seqan-extract
mkdir -p ~/.local/src/seqan-apps ~/.local/bin
cp -a /tmp/seqan-extract/usr/lib/seqan/bin/mason_* ~/.local/src/seqan-apps/
for b in ~/.local/src/seqan-apps/mason_*; do ln -sf "$b" ~/.local/bin/; done
mason_simulator --version    # verify
```

**junceval inputs** (GENCODE GTF + RNA reads) — fetch via
`download_verification_data.py` (GENCODE annotation + SG-NEx RNA reads), then
pass them with `--gtf` / `--splice-reads`.

> Or pass `--auto-download` to fetch any missing `--ref`/`--gtf`/`--splice-reads`
> input on demand (to the exact path given) via the `download_*_data.py` helpers.

---

## run_giab_concordance.py

GIAB **HG002 small-variant concordance** — the real-data genomic accuracy oracle.
For each build/mode it aligns HG002 HiFi reads (`-ax map-hifi`), calls variants
with **Clair3** (`hifi` model), and scores against the **GIAB HG002 v4.2.1**
benchmark with **`hap.py`**, reporting SNP/INDEL recall, precision and F1.
Comparing a dev build to the upstream oracle shows whether a change regresses
variant-calling-relevant alignment; comparing a build's GPU modes to its own CPU
shows whether GPU divergence reaches downstream variant calls.

Builds use the same `--binary NAME=PATH[=MODES]` convention as
[run_accuracy_eval.py](#run_accuracy_evalpy).

**Tooling** (no sudo if `docker` is usable without sudo): Clair3 (`hkubal/clair3`)
and `hap.py` (`jmcdani20/hap.py`, or any `hap.py` on PATH) run as docker images;
local `samtools` for sort/index.

**Data prep** (one-time):

```bash
# GIAB truth (VCF + high-conf BED)
python3 scripts/download_verification_data.py --components giab --dest <preset_data>
# HG002 HiFi reads for one chromosome (streams ~1.5 GB of chr20 from the remote
# PrecisionFDA BAM via byte-range — needs a libcurl samtools or staphb/samtools docker)
python3 scripts/download_verification_data.py --components giab-reads --giab-region chr20 --dest <preset_data>
```

Or skip the manual prep and pass `--auto-download`, which fetches any missing
`--ref`/`--reads`/`--truth-*` input (to the exact path given) via the
`download_*_data.py` helpers before running.

**Run**:

```bash
python3 scripts/run_giab_concordance.py --ref ref.fa \
  --binary upstream=/path/to/upstream/minimap2=cpu \
  --binary dev=out/release-gfx942/build/bin/minimap2 \
  --reads     <preset_data>/giab/hg002.hifi.chr20.fastq.gz \
  --truth-vcf <preset_data>/giab/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz \
  --truth-bed <preset_data>/giab/HG002_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed \
  --region chr20 --modes cpu gpu_chain gpu_align -t 48
```

Outputs a per-build-mode table; full artifacts (BAM, Clair3 VCF, hap.py reports)
land under `--workdir`. The GIAB truth must share the reference assembly
(`GRCh38_no_alt_analysis_set`, as fetched by `download_preset_data.py --presets ref`).

---

# Data Utilities

## download_preset_data.py

Download the GRCh38 reference genome and one representative real-read dataset for
every minimap2 preset/flag combination, so each `-x` mode in USAGE.md can be
exercised end to end. Sources include NCBI (reference), the HPRC S3 bucket
(long-read + assembly), GIAB (PacBio CLR), the SG-NEx AWS Open Data bucket (ONT
cDNA / direct-RNA for splice), and ENCODE (short-read RNA-seq for `splice:sr`).
Verifies SHA-256 checksums where published (SG-NEx/ENCODE have none and are
skipped), and optionally decompresses `.gz` files or converts BAM to FASTQ
(requires `samtools`).

> Ground-truth / accuracy inputs (GENCODE annotation, GIAB benchmark VCF,
> simulated reads) are **not** here — those are fetched by
> `download_verification_data.py`.

### Supported Presets & Datasets

| Preset | Dataset | Sample | Size |
|--------|---------|--------|------|
| `ref` | GRCh38 no-alt analysis set (reference genome) | — | ~833 MB compressed, ~3 GB decompressed |
| `map-hifi` | PacBio HiFi reads | HG02486 | ~898 MB |
| `map-pb` | PacBio CLR subreads (RS II) | HG002 | ~370 MB |
| `map-ont` | Oxford Nanopore reads | NA18565 | ~973 MB |
| `sr` | Illumina Hi-C paired-end reads (R1 + R2) | HG03098 | ~806 MB + ~920 MB |
| `splice` | SG-NEx ONT cDNA (plain `splice`) + direct-RNA (`splice -uf -k14`) | A549 / Hct116 | ~102 MB + ~97 MB |
| `splice:hq` | PacBio IsoSeq/Kinnex FLNC | HG00097 | ~15 GB |
| `splice:sr` | ENCODE polyA short-read RNA-seq (R1 + R2) | ENCFF336WMA/ENCFF732TMT | ~272 MB + ~276 MB |
| `asm5` | Assembly FASTA (paternal) | HG00438 | ~834 MB |

### Prerequisites

- Python 3.10+ (uses `X | None` union syntax)
- `requests` and `tqdm` Python packages
- `samtools` on PATH (required only for `map-ont` and `splice:hq` BAM→FASTQ conversion)

### Usage

```bash
python3 scripts/download_preset_data.py [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dest` | `$MM2_PRESET_DATA` or `<repo>/test/preset_data` | Destination directory for downloaded files |
| `--presets` | `all` | Comma-separated preset names to download, or `all` |
| `--dry-run` | off | Show what would be downloaded without actually downloading |
| `--verify` | off | Verify SHA-256 checksums of existing files |
| `-q, --quiet` | off | Suppress example commands in output |

### Output Directory Layout

```
<dest>/
  ref.fa                                          # Decompressed reference genome
  GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz
  map-hifi/
    HG02486.m64076_200211_192227.dc.q20.fastq     # Decompressed
  map-pb/
    m141224_220915_42177.subreads.fasta            # Decompressed
  map-ont/
    10_10_23_R1041_HPRC_NA18565_1_*.bam
    10_10_23_R1041_HPRC_NA18565_1_*.fastq          # BAM→FASTQ via samtools
  sr/
    SE5138_NWM047-3_S1_L002_R1_001.trimmed.fastq   # Decompressed
    SE5138_NWM047-3_S1_L002_R2_001.trimmed.fastq   # Decompressed
  splice/
    HG00097.lymph.*.flnc.bam
    HG00097.lymph.*.flnc.fastq                     # BAM→FASTQ via samtools
  splice-rna/
    SGNex_A549_directcDNA_replicate3_run1.fastq    # plain `splice` (Decompressed)
    SGNex_Hct116_directRNA_replicate1_run1.fastq   # `splice -uf -k14` (Decompressed)
  splice-sr/
    ENCFF336WMA.fastq                              # `splice:sr` R1 (Decompressed)
    ENCFF732TMT.fastq                              # `splice:sr` R2 (Decompressed)
  asm/
    HG00438.paternal.f1_assembly_v2_genbank.fa      # Decompressed
```

### Features

- **Resumable**: Skips files that are already downloaded or extracted
- **Integrity checks**: SHA-256 verified on download; `--verify` re-checks existing files
- **Post-processing**: Automatically decompresses `.gz` and converts BAM→FASTQ
- **Example commands**: Prints ready-to-run `minimap2 -ax <preset>` commands after download

### Examples

```bash
# Download all preset datasets to the default location
python3 scripts/download_preset_data.py

# Download only HiFi and ONT data to a custom directory
python3 scripts/download_preset_data.py --presets map-hifi,map-ont --dest /data/input

# Dry run to see what would be downloaded
python3 scripts/download_preset_data.py --dry-run

# Verify checksums of previously downloaded files
python3 scripts/download_preset_data.py --verify

# Download reference + short reads only, suppress example commands
python3 scripts/download_preset_data.py --presets ref,sr --dest /data/input -q
```

---

## download_verification_data.py

Acquire the ground-truth / feature-flag inputs that `download_preset_data.py`
deliberately leaves out. These back the correctness-verification flows
(junceval, GIAB variant concordance):

| Component | Contents | In `all` |
|-----------|----------|----------|
| `anno` | GENCODE GRCh38 annotation (GTF → BED12 via `paftools.js gff2bed`). Unblocks `--junc-bed`/`-j`/`--jump-pass1`, derived `.spsc`, and junceval. | yes |
| `giab` | GIAB HG002 v4.2.1 benchmark VCF + high-confidence BED (+ `.tbi`). | yes |
| `giab-reads` | HG002 HiFi reads for one chromosome (default `chr20`), streamed from the remote PrecisionFDA BAM via byte-range (~1.5 GB). Read input for [run_giab_concordance.py](#run_giab_concordancepy). | no (opt-in) |

> Simulated reads for `paftools.js mapeval` are **not** fetched here —
> [run_accuracy_eval.py](#run_accuracy_evalpy) generates them on demand
> (pbsim3/mason2).

Everything is idempotent: existing/extracted outputs are skipped, and steps
whose required tool is missing are skipped with an actionable message. A single
explicit URL can also be fetched with `--url`/`--into`.

### Prerequisites

- Python 3.10+
- `requests` and `tqdm` Python packages
- `paftools.js` (in `misc/`) for the `anno` GTF → BED12 conversion
- `samtools` (local with libcurl, or via docker) for the `giab-reads` component

### Usage

```bash
python3 scripts/download_verification_data.py [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dest` | `$MM2_PRESET_DATA` env var, or `<repo-root>/test/preset_data` | Destination preset_data root |
| `--components` | `all` | Comma-separated: `anno,giab,giab-reads` or `all` (`all` excludes the heavy `giab-reads`) |
| `--giab-region` | `chr20` | Chromosome to stream for `--components giab-reads` |
| `--url` | — | Fetch a single explicit URL (for SG-NEx/ENCODE accessions) |
| `--into` | URL basename | With `--url`: destination path relative to `--dest` |
| `--decompress` | off | With `--url`: gunzip a `.gz` after download |
| `--dry-run` | off | Show actions without downloading/generating |
| `--list` | off | List registered items and exit |
| `-q, --quiet` | off | Suppress the manual-items hint |

### Examples

```bash
# Ground-truth datasets in 'all' (anno + giab)
python3 scripts/download_verification_data.py --components anno,giab

# Opt-in: stream HG002 HiFi chr20 reads for GIAB concordance
python3 scripts/download_verification_data.py --components giab-reads --giab-region chr20

# Fetch a single explicit URL
python3 scripts/download_verification_data.py --url <FASTQ_URL> --into splice-sr/encode_R1.fastq.gz

# Preview without downloading
python3 scripts/download_verification_data.py --dry-run
```

---

## subset_queries.py

Count or extract a subset of reads/sequences from a FASTA or FASTQ file.
Auto-detects file format by inspecting the first byte.

### Usage

```bash
python3 scripts/subset_queries.py <input> [-o OUTPUT] [--start N] [--end N] [--no-count]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | — | Input FASTA or FASTQ file |
| `-o, --output` | — | Output file for the subset (omit to just count) |
| `--start` | `0` | Start index, 0-based inclusive |
| `--end` | total count | End index, exclusive |
| `--no-count` | off | Skip counting total records (faster for large files, requires `--end`) |

### Examples

```bash
# Count queries in a file
python3 scripts/subset_queries.py input.fa

# Extract first 100 reads
python3 scripts/subset_queries.py input.fastq -o subset.fastq --end 100

# Extract reads 50..149
python3 scripts/subset_queries.py input.fa -o subset.fa --start 50 --end 150

# Fast extraction (skip counting the whole file)
python3 scripts/subset_queries.py large.fastq -o subset.fastq --start 0 --end 100 --no-count
```

---

# Shared Modules

## json_dump_utils.py

Shared utilities for parsing minimap2 `--dump-intermediates JSON` output. Provides:

- `load_json(path)` — Load a JSON file, returning `None` on error
- `discover_query_names(dump_dir)` — Discover all query names across stage subdirectories
- `load_stage_json(dump_dir, stage, query_name)` — Load a per-query JSON file for a stage
- `STAGE_DIRS` — Tuple of stage subdirectory names: `("seeds", "chains", "alignments")`
- `SEED_FIELDS`, `CHAIN_FIELDS`, `ALIGNMENT_FIELDS` — Field name tuples for each stage

Used by `generate_test_data.py` and `compare_json_intermediates.py` to avoid
duplicating JSON parsing logic and field name definitions.

## stage_defs.py

Shared pipeline stage constants and helpers for profiling scripts. Provides:

- `PIPELINE_STAGES` — Canonical stage list: `["seeding", "io", "chaining", "alignment", "gpu_other", "other"]`
- `PROFILE_MACRO_TO_STAGE` — Maps C macros (`MM_PROFILE_SEED`, etc.) to stage names
- `reclassify_gpu_chain(name, stage)` — Reclassify `gpu_chain_*` functions from `gpu_other` to `chaining`
- `shorten_kernel_name(name)` — Remove template parameters from GPU kernel names

Used by `analyze_stages.py` and `generate_profiling_report.py` to ensure consistent
stage definitions and classification logic.
