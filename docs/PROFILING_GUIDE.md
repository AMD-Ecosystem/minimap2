# Profiling Guide for minimap2

This guide explains how to use the unified profiling infrastructure in minimap2 for performance analysis with Intel VTune, NVIDIA Nsight, and AMD rocprof.

## Overview

minimap2 includes a unified profiling API that supports multiple profiler backends:

| Backend | Profiler | Platform |
|---------|----------|----------|
| **ITT** | Intel VTune Profiler | Intel CPUs |
| **NVTX** | NVIDIA Nsight Systems/Compute | NVIDIA GPUs |
| **ROCTX** | AMD rocprof/rocprofv3 | AMD GPUs |

All backends use the same API, so instrumented code works across all platforms.

## Building with Profiling Support

### Using CMake Presets (Recommended)

```bash
# Build with profiling and trace logging enabled
cmake --preset profile
cmake --build --preset profile
```

### Manual Configuration

```bash
# Enable profiling instrumentation
cmake -DENABLE_PROFILING=ON ..

# Enable trace logging (console output with timing)
cmake -DENABLE_TRACE=ON ..

# Enable both
cmake -DENABLE_PROFILING=ON -DENABLE_TRACE=ON ..
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_PROFILING` | OFF | Enable profiler instrumentation (ITT/NVTX/ROCTX) |
| `ENABLE_TRACE` | OFF | Enable trace logging with timing to console |

When profiling is enabled, the build system automatically:
- Detects available profiler libraries (ITT, ROCTX)
- Links against the appropriate libraries
- Defines `MM_PROFILE`, `MM_USE_ITT`, `MM_USE_ROCTX` as needed

## Pipeline Stage Colors

The profiling instrumentation uses colors to distinguish pipeline stages:

| Stage | Color | Macro |
|-------|-------|-------|
| **Seeding** | Green | `MM_PROFILE_SEED("name")` |
| **Chaining** | Blue | `MM_PROFILE_CHAIN("name")` |
| **Alignment** | Red | `MM_PROFILE_ALIGN("name")` |
| **I/O** | Yellow | `MM_PROFILE_IO("name")` |
| **GPU** | Cyan | `MM_PROFILE_GPU("name")` |

## Running with Profilers

### AMD rocprof (rocprofv3)

```bash
# Basic profiling (note the required double dash before the application)
rocprofv3 --hip-trace --marker-trace -- ./minimap2 -a ref.fa reads.fa > out.sam

# View results (add '--' before the application if running a command)
rocprofv3 --output-format json --output-directory ./results [other options] -- ./minimap2 -a ref.fa reads.fa
```

### Intel VTune

```bash
# Hotspots analysis
vtune -collect hotspots ./minimap2 -a ref.fa reads.fa > out.sam

# Threading analysis
vtune -collect threading ./minimap2 -a ref.fa reads.fa > out.sam
```

### NVIDIA Nsight Systems

```bash
# Profile with NVTX markers
nsys profile --trace=cuda,nvtx ./minimap2 -a ref.fa reads.fa > out.sam

# Generate report
nsys stats report.nsys-rep
```

## Trace Logging

When built with `ENABLE_TRACE=ON`, minimap2 outputs hierarchical timing information to stderr:

```
[2024-01-15 10:30:45.123] [trace] [map.c:373] >> mm_map_frag
[2024-01-15 10:30:45.124] [trace] [seed_map.c:51]   >> mm_map_seed
[2024-01-15 10:30:45.125] [trace] [seed.c:251]     >> collect_minimizers
[2024-01-15 10:30:45.126] [trace] [seed.c:263]     << collect_minimizers [1.234ms]
[2024-01-15 10:30:45.127] [trace] [seed_map.c:153]   << mm_map_seed [3.456ms]
[2024-01-15 10:30:45.128] [trace] [chain_map.c:39]   >> mm_map_chain
[2024-01-15 10:30:45.129] [trace] [chain_map.c:169]   << mm_map_chain [0.789ms]
[2024-01-15 10:30:45.130] [trace] [align.c:1094]   >> mm_map_align
[2024-01-15 10:30:45.135] [trace] [align.c:1120]   << mm_map_align [5.678ms]
[2024-01-15 10:30:45.136] [trace] [map.c:458] << mm_map_frag [12.345ms]
```

### Trace Output Format

- `>>` indicates function entry
- `<<` indicates function exit with elapsed time
- Indentation shows call hierarchy
- Times are in seconds (s), milliseconds (ms), or microseconds (us)

## API Reference

### C Macros (for .c files)

```c
#include "mm_profiler.h"

void my_function() {
    MM_PROFILE_SEED("my_function");  // Push colored range
    
    // ... work ...
    
    MM_PROFILE_RANGE_POP();          // Pop range
}
```

### C++ RAII (for .cpp files)

```cpp
#include "mm_profiler.h"

void my_function() {
    MM_PROFILE_SCOPE_SEED("my_function");  // Auto-pops on scope exit
    
    // ... work ...
}  // Automatically pops here
```

### Available Macros

| Macro | Description |
|-------|-------------|
| `MM_PROFILE_RANGE_PUSH(name)` | Push a profiler range |
| `MM_PROFILE_RANGE_PUSH_COLOR(name, color)` | Push with specific color |
| `MM_PROFILE_RANGE_POP()` | Pop the current range |
| `MM_PROFILE_MARK(name)` | Insert an instant marker |
| `MM_PROFILE_FUNCTION()` | Push range with function name |
| `MM_PROFILE_FUNCTION_END()` | Pop function range |
| `MM_PROFILE_THREAD_NAME(name)` | Set thread name (NVTX/ITT) |

### Domain-Specific Macros

| Macro | Color | Use Case |
|-------|-------|----------|
| `MM_PROFILE_SEED(name)` | Green | Seeding stage functions |
| `MM_PROFILE_CHAIN(name)` | Blue | Chaining stage functions |
| `MM_PROFILE_ALIGN(name)` | Red | Alignment stage functions |
| `MM_PROFILE_IO(name)` | Yellow | I/O operations |
| `MM_PROFILE_GPU(name)` | Cyan | GPU kernel launches |

### C++ Scoped Variants

| Macro | Description |
|-------|-------------|
| `MM_PROFILE_SCOPE(name)` | RAII scoped range |
| `MM_PROFILE_SCOPE_SEED(name)` | Scoped seeding range |
| `MM_PROFILE_SCOPE_CHAIN(name)` | Scoped chaining range |
| `MM_PROFILE_SCOPE_ALIGN(name)` | Scoped alignment range |

## Instrumented Functions

The following functions are instrumented in the codebase:

### Seeding Stage (Green)
- `mm_map_seed` - Main seeding entry point
- `mm_sketch` - Minimizer sketching
- `collect_minimizers` - Collect query minimizers
- `seed_mz_flt` - Minimizer filtering
- `collect_hits_heap` / `collect_hits_radix` - Seed hit collection
- `dust_minier` - Low-complexity filtering
- `idx_gen` - Index generation
- `sdust_core` - SDUST algorithm

### Chaining Stage (Blue)
- `mm_map_chain` - Main chaining entry point
- `lchain_dp` - Long-chain dynamic programming
- `chain_backtrack` - Chain backtracking

### Alignment Stage (Red)
- `mm_map_align` - Main alignment entry point
- `align_skeleton` - Skeleton alignment
- `align1` - Single alignment
- `chain_post` - Post-chaining processing
- `align_regs` - Region alignment
- `gen_regs` - Generate regions from chains
- `set_parent` - Set parent relationships
- `hit_sort` - Sort hits
- `filter_regs` - Filter regions
- `set_mapq` - Compute mapping quality
- `mm_pair` - Paired-end processing
- `est_err` - Error estimation
- `ksw_extz2_sse` / `ksw_extd2_sse` - KSW2 alignment kernels

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      User-facing Macros                          │
│  MM_PROFILE_SEED/CHAIN/ALIGN/IO/GPU, MM_PROFILE_FUNCTION, etc.  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           mm_profiler.h (Inline Wrappers + RAII)                │
│  - Zero overhead when disabled (inline no-ops)                   │
│  - Thread-local trace stack for timing                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
         ┌──────────┐   ┌──────────┐   ┌──────────┐
         │   ITT    │   │   NVTX   │   │  ROCTX   │
         │ (VTune)  │   │ (NVIDIA) │   │  (AMD)   │
         └──────────┘   └──────────┘   └──────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  mm_log.cpp     │
                    │ (Trace Logging) │
                    │   via spdlog    │
                    └─────────────────┘
```

## Performance Overhead

- **Profiling disabled**: Zero overhead (macros expand to nothing)
- **Profiling enabled, no profiler attached**: Minimal overhead (~1-10 ns per call)
- **Profiling enabled, profiler attached**: Overhead depends on profiler (typically ~100 ns - 1 µs per call)

For production builds, disable profiling:
```bash
cmake --preset amd-default  # Profiling OFF by default
```

## Files

| File | Description |
|------|-------------|
| `minimap2/core/include/mm_profiler.h` | Public API, macros, inline wrappers |
| `minimap2/core/src/mm_profiler.cpp` | Backend implementations (ITT/NVTX/ROCTX) |
| `minimap2/core/include/mm_log.h` | Logging API: mm_print, mm_log, mm_log_* macros, trace logging |
| `minimap2/core/src/mm_log.cpp` | spdlog backend: 5 loggers (print/log/info/debug/error), trace implementation |

## Troubleshooting

### No profiler output

1. Ensure built with `ENABLE_PROFILING=ON`
2. Check that profiler libraries were detected during CMake:
   ```
   -- ITT: using submodule
   -- ROCTX: enabled (linked)
   -- Profiling: ITT=TRUE ROCTX=TRUE
   ```

### Trace logging not appearing

1. Ensure built with `ENABLE_TRACE=ON`
2. Trace goes to stderr by default, or to `--debug-log FILE` if set
3. Redirect: `./minimap2 ... 2>&1 | grep TRACE`

### Missing timing data

If trace shows `<< (file:line)` without timing, ensure `MM_PROFILE_RANGE_POP()` is properly paired with the corresponding `MM_PROFILE_*_PUSH()`.
