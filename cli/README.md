# Minimap2 Staged Mapping CLI Tools

This directory contains three standalone CLI tools that expose minimap2's internal mapping pipeline stages as separate executables. These tools are designed for debugging, testing, and understanding the minimap2 mapping algorithm.

## Overview

Minimap2's mapping process consists of three main stages:

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   SEEDING   │ ───► │  CHAINING   │ ───► │  ALIGNMENT  │
│  mm2_seed   │      │  mm2_chain  │      │  mm2_align  │
└─────────────┘      └─────────────┘      └─────────────┘
     │                     │                     │
     ▼                     ▼                     ▼
 seeds.json          chains.json          alignments.json
```

Each tool performs one stage and outputs JSON, allowing inspection of intermediate results.

## Building

The CLI tools are controlled by the `BUILD_CLI_TOOLS` CMake option (default: **ON**).

### Quick Build

```bash
mkdir build && cd build
cmake ..
make mm2_seed mm2_chain mm2_align
```

### Build Options

```bash
# Enable CLI tools (default)
cmake .. -DBUILD_CLI_TOOLS=ON

# Disable CLI tools
cmake .. -DBUILD_CLI_TOOLS=OFF

# Build all CLI tools at once
cmake --build . --target mm2_seed mm2_chain mm2_align

# Or use make
make mm2_seed mm2_chain mm2_align -j$(nproc)
```

### CMake Option

| Option | Description | Default |
|--------|-------------|---------|
| `BUILD_CLI_TOOLS` | Build CLI tools (mm2_seed, mm2_chain, mm2_align) | ON |

Executables are placed in `build/cli/`.

## Tools

### mm2_seed - Seeding Stage

Finds seed matches (minimizer hits) between query and reference sequences.

```bash
mm2_seed [options] <reference.fa> <query.fa> -o <output.json>
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `-x STR` | Preset configuration (see below) | - |
| `-k INT` | K-mer size | 15 |
| `-w INT` | Minimizer window size | 10 |
| `-d INT` | SDUST threshold for masking low-complexity | 0 (disabled) |
| `-m INT` | Max occurrence threshold | auto-computed |
| `-M INT` | Hard max occurrence limit | 4095 |
| `-D INT` | Occurrence distance for selective filtering | 500 |
| `-o FILE` | Output JSON file | (required) |
| `-h` | Show help | |

**Presets:**
- `map-ont`: Oxford Nanopore reads (k=15, w=10)
- `map-pb`: PacBio CLR reads (k=19, w=10)
- `map-hifi`: PacBio HiFi/CCS reads (k=19, w=19)
- `asm5/asm10/asm20`: Assembly alignment (k=19, w=19/10)
- `sr`: Short reads (k=21, w=11)
- `splice`: Spliced alignment (k=15, w=5)
- `ava-ont/ava-pb`: Read overlap (k=15/19, w=5)

**Example:**
```bash
./mm2_seed ref.fa query.fa -o seeds.json
./mm2_seed -x map-ont ref.fa query.fa -o seeds.json  # use preset
./mm2_seed -k 19 -w 10 -d 12 ref.fa query.fa -o seeds.json  # custom parameters
```

---

### mm2_chain - Chaining Stage

Groups seeds into chains (collinear seed clusters).

```bash
mm2_chain [options] <reference.fa> -s <seeds.json> -o <output.json>
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `-x STR` | Preset configuration (see below) | - |
| `-k INT` | K-mer size | from input JSON or 15 |
| `-w INT` | Minimizer window size | from input JSON or 10 |
| `-g INT` | Maximum gap in reference/query | 5000 |
| `-G INT` | Maximum gap on reference (overrides -g) | 0 (use -g) |
| `-b INT` | Bandwidth for DP chaining | 500 |
| `-B INT` | Long-join bandwidth (for ultra-long reads) | 0 (disabled) |
| `-n INT` | Minimum anchor count per chain | 3 |
| `-c INT` | Minimum chain score | 40 |
| `-r` | Use RMQ algorithm (faster for ultra-long reads) | |
| `-s FILE` | Input seeds JSON (from mm2_seed) | (required) |
| `-o FILE` | Output JSON file | (required) |
| `-h` | Show help | |
**Advanced Chaining Options:**
| Option | Description | Default |
|--------|-------------|---------||
| `--max-chain-skip INT` | Max anchors to skip in chaining | 25 |
| `--max-chain-iter INT` | Max chaining DP iterations | 5000 |
| `--chain-gap-scale FLOAT` | Gap cost scaling factor | 1.0 |
| `--chain-skip-scale FLOAT` | Skip cost scaling factor | 1.0 |

**Presets:**
- `map-ont/map-pb/map-hifi`: Long read mapping (default parameters)
- `asm5/asm10/asm20`: Assembly alignment (bw=1000, max_gap=10000)
- `sr`: Short reads (max_gap=100, bw=100, min_cnt=2)
- `splice`: Spliced alignment (max_gap=2000, bw=200000)
- `ava-ont/ava-pb`: Read overlap (bw=2000)

**Example:**
```bash
./mm2_chain ref.fa -s seeds.json -o chains.json
./mm2_chain -x map-ont ref.fa -s seeds.json -o chains.json  # use preset
./mm2_chain -g 10000 -r ref.fa -s seeds.json -o chains.json  # RMQ for long reads
./mm2_chain --use-gpu ref.fa -s seeds.json -o chains.json    # GPU acceleration
```

**GPU Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--use-gpu` | Enable GPU acceleration for chaining | disabled |
| `--gpu-config FILE` | GPU configuration file | gpu_config.json |

*Note: GPU chaining requires compilation with `-DMM_ENABLE_HIP=ON` and is incompatible with RMQ (`-r`).*

---

### mm2_align - Alignment Stage

Performs base-level alignment on chains to produce final alignments with CIGAR strings.

```bash
mm2_align [options] <reference.fa> <query.fa> -c <chains.json> -o <output.json>
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `-x STR` | Preset configuration (see below) | - |
| `-k INT` | K-mer size | 15 |
| `-w INT` | Minimizer window size | 10 |
| `-c FILE` | Input chains JSON (from mm2_chain) | (required) |
| `-o FILE` | Output JSON file | (required) |
| `-h` | Show help | |

**Presets:**
- `map-ont/map-pb`: Long read mapping (A=2, B=4, O=4,24, E=2,1)
- `map-hifi`: PacBio HiFi reads (A=1, B=4, O=6,26, E=2,1)
- `asm5`: Assembly ~0.1% divergence (A=1, B=19, O=39,81, E=3,1)
- `asm10`: Assembly ~1% divergence (A=1, B=9, O=16,41, E=2,1)
- `asm20`: Assembly ~5% divergence (A=1, B=4, O=6,26, E=2,1)
- `sr`: Short reads (A=2, B=8, O=12,24, E=2,1, zdrop=100)
- `splice`: Spliced alignment (A=1, B=2, O=2,32, E=1,0)
- `splice:hq`: High-quality spliced (A=1, B=4, O=6,24, E=1,0)

**Alignment Scoring:**
| Option | Description | Default |
|--------|-------------|---------|
| `-A INT` | Matching score | 2 |
| `-B INT` | Mismatch penalty | 4 |
| `-O INT[,INT]` | Gap open penalty | 4,24 |
| `-E INT[,INT]` | Gap extension penalty | 2,1 |
| `-z INT[,INT]` | Z-drop score | 400,200 |
| `-s INT` | Minimum DP alignment score | 40 |

**Example:**
```bash
./mm2_align ref.fa query.fa -c chains.json -o alignments.json
./mm2_align -x map-ont ref.fa query.fa -c chains.json -o alignments.json  # use preset
./mm2_align --use-gpu ref.fa query.fa -c chains.json -o alignments.json  # GPU acceleration
./mm2_align -A 1 -B 2 -O 2,10 -E 1,1 ref.fa query.fa -c chains.json -o alignments.json  # Custom scoring
```

**GPU Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--use-gpu` | Enable GPU acceleration for alignment | disabled |

*Note: GPU alignment requires compilation with `-DHAVE_GPU_ALIGNMENT=ON`. Unlike GPU chaining, GPU alignment does not use a configuration file.*

---

## Complete Pipeline Example

```bash
# Run full pipeline
./mm2_seed ref.fa query.fa -o seeds.json
./mm2_chain ref.fa -s seeds.json -o chains.json  
./mm2_align ref.fa query.fa -c chains.json -o alignments.json

# Or as a one-liner
./mm2_seed ref.fa query.fa -o /tmp/s.json && \
./mm2_chain ref.fa -s /tmp/s.json -o /tmp/c.json && \
./mm2_align ref.fa query.fa -c /tmp/c.json -o alignments.json

# With GPU acceleration (chaining only needs config, alignment doesn't)
./mm2_seed ref.fa query.fa -o /tmp/s.json && \
./mm2_chain --use-gpu --gpu-config configs/gpu_config.json ref.fa -s /tmp/s.json -o /tmp/c.json && \
./mm2_align --use-gpu ref.fa query.fa -c /tmp/c.json -o alignments.json
```

## GPU Acceleration

The `mm2_chain` and `mm2_align` tools support GPU acceleration for improved performance on large datasets.

### Quick Start

```bash
# Enable GPU chaining (with config file)
mm2_chain --use-gpu ref.fa -s seeds.json -o chains.json
mm2_chain --use-gpu --gpu-config configs/gpu_config.json ref.fa -s seeds.json -o chains.json

# Enable GPU alignment (no config file needed)
mm2_align --use-gpu ref.fa query.fa -c chains.json -o alignments.json
```

### Requirements

1. **Compilation flags:**
   - GPU chaining: `-DMM_ENABLE_HIP=ON`
   - GPU alignment: `-DHAVE_GPU_ALIGNMENT=ON`

2. **GPU configuration file** (for chaining only)
   - Required only for `mm2_chain --use-gpu`
   - Default: `gpu_config.json`
   - Example configs in `configs/` directory
   - Different configs for different GPU models (A6000, MI210, etc.)

3. **Compatible hardware:**
   - AMD GPUs with ROCm support
   - NVIDIA GPUs with CUDA support

### GPU Configuration

GPU configuration files are **only used for chaining** (`mm2_chain --use-gpu`). They control batch sizes, kernel parameters, and memory usage. Key parameters:

- `min_n`: Minimum anchors to use GPU (smaller batches fall back to CPU)
- `max_total_n`: Maximum anchors per microbatch
- `max_read`: Maximum reads per microbatch
- Kernel-specific tuning parameters

### Important Notes

- **GPU chaining** is incompatible with RMQ algorithm (`-r` flag)
- GPU provides benefit mainly for large anchor counts (>500 anchors)
- Smaller batches automatically fall back to CPU

---

## JSON Format

### Common Structure

All output files share this top-level structure:

```json
{
  "name": "seed_output|chain_output|align_output",
  "description": "...",
  "reference": { "file": "path/to/ref.fa" },
  "query": { "file": "path/to/query.fa" },
  "index_params": { "k": 15, "w": 10 },
  "queries": [ ... ]
}
```

### Seeds Output (`mm2_seed`)

```json
{
  "queries": [
    {
      "name": "query_name",
      "length": 1000,
      "seeds": [
        { "x": "12345", "y": "64424509494" },
        ...
      ],
      "n_seeds": 42
    }
  ]
}
```

### Chains Output (`mm2_chain`)

```json
{
  "queries": [
    {
      "name": "query_name", 
      "length": 1000,
      "seeds": [ ... ],
      "n_seeds": 42,
      "chains": [
        {
          "score": 345,
          "length": 42,
          "first_qpos": 54,
          "first_tpos": 54,
          "last_qpos": 805,
          "last_tpos": 805
        }
      ],
      "n_chains": 1
    }
  ]
}
```

### Alignments Output (`mm2_align`)

```json
{
  "queries": [
    {
      "name": "query_name",
      "length": 1000,
      "seeds": [ ... ],
      "n_seeds": 42,
      "chains": [ ... ],
      "n_chains": 1,
      "alignments": [
        {
          "rid": 0,
          "score": 345,
          "dp_score": 1280,
          "qs": 0,
          "qe": 820,
          "rs": 0,
          "re": 820,
          "mapq": 60,
          "rev": 0,
          "n_cigar_ops": 1,
          "cigar": "820M"
        }
      ]
    }
  ]
}
```

## Data Encoding

### Anchor Format (seeds)

Seeds are stored as `mm128_t` anchors with two 64-bit values:

**`x` field** (reference position):
```
┌────────────┬────────┬─────────────────────────────────┐
│  ref_id    │ strand │         ref_position            │
│ (bits 32-62)│(bit 63)│         (bits 0-31)            │
└────────────┴────────┴─────────────────────────────────┘
```

**`y` field** (query position):
```
┌─────────────────────────────────┬─────────────────────────────────┐
│           q_span                │           q_position            │
│         (bits 32-63)            │           (bits 0-31)           │
└─────────────────────────────────┴─────────────────────────────────┘
```

Both values are stored as strings to preserve full 64-bit precision (JSON numbers lose precision beyond 53 bits).

**Decoding Example (Python):**
```python
x = 4294967328  # example x value
ref_id = x >> 32        # reference sequence id
strand = (x >> 63) & 1  # 0=forward, 1=reverse  
ref_pos = x & 0xFFFFFFFF  # reference position

y = 64424509494  # example y value
q_span = y >> 32          # k-mer span (usually equals k)
q_pos = y & 0xFFFFFFFF    # query position
```

### Chain Format

Chains are represented by a score and the number of anchors they contain:

| Field | Description |
|-------|-------------|
| `score` | Chain score (sum of anchor weights minus gap penalties) |
| `length` | Number of anchors in the chain |
| `first_qpos` | Query position of first anchor |
| `first_tpos` | Target (reference) position of first anchor |
| `last_qpos` | Query position of last anchor |
| `last_tpos` | Target position of last anchor |

### Alignment Format

| Field | Description |
|-------|-------------|
| `rid` | Reference sequence ID (0-indexed) |
| `score` | Alignment score (from chaining) |
| `dp_score` | Dynamic programming alignment score |
| `qs`, `qe` | Query start/end positions |
| `rs`, `re` | Reference start/end positions |
| `mapq` | Mapping quality (0-60) |
| `rev` | Strand (0=forward, 1=reverse) |
| `n_cigar_ops` | Number of CIGAR operations |
| `cigar` | CIGAR string (e.g., "100M2I50M") |

## Use Cases

### 1. Debugging Mapping Issues

Inspect intermediate stages to understand why a read maps (or doesn't map):

```bash
# Check if seeds are found
./mm2_seed ref.fa problem_read.fa -o seeds.json
grep n_seeds seeds.json

# Check if chains form
./mm2_chain ref.fa problem_read.fa -s seeds.json -o chains.json
grep n_chains chains.json
```

### 2. Unit Testing

Generate expected outputs for unit tests:

```bash
./mm2_seed test_ref.fa test_query.fa -o expected_seeds.json
```

### 3. Algorithm Analysis

Study how parameters affect each stage:

```bash
# Compare different k-mer sizes
for k in 13 15 17 19; do
  ./mm2_seed ref.fa query.fa -k $k -o seeds_k${k}.json
  echo "k=$k: $(grep -o '"n_seeds":[^,]*' seeds_k${k}.json | head -1)"
done
```

### 4. GPU Kernel Validation

Compare CPU and GPU implementations at each stage:

```bash
# Run CPU version
./mm2_chain ref.fa query.fa -s seeds.json -o chains_cpu.json

# Compare with GPU results
diff chains_cpu.json chains_gpu.json
```

## Architecture

```
cli/
├── cli_common.h     # Shared utilities (context management, I/O helpers)
├── cli_json.h       # JSON serialization/deserialization (seeds, chains, alignments)
├── mm2_seed.cpp     # Seeding stage CLI
├── mm2_chain.cpp    # Chaining stage CLI
├── mm2_align.cpp    # Alignment stage CLI
├── CMakeLists.txt   # Build configuration
└── README.md        # This file
```

### Key Components

**cli_common.h:**
| Component | Description |
|-----------|-------------|
| `mm2_context` | RAII class managing index, sequences, and buffers |
| `init_chain_read()` | Initialize chain_read_t from sequence |

**cli_json.h:**
| Component | Description |
|-----------|-------------|
| `anchor_to_json()` | Convert mm128_t anchor to JSON |
| `chain_to_json()` | Convert chain data to JSON |
| `alignment_to_json()` | Convert mm_reg1_t to JSON |
| `load_seeds_from_json()` | Parse seeds from JSON input |
| `load_chains_from_json()` | Parse chains from JSON input |
| `create_output_json()` | Create standard output structure |

### Resource Management

The `mm2_context` struct provides RAII-style resource management:

```cpp
mm2_context ctx;
if (!ctx.init(ref_path, query_path, k, w)) return 1;

// Use ctx.mi (index), ctx.seqs (sequences), ctx.tbuf (buffer)
// Resources automatically freed when ctx goes out of scope
```

## Comparison with minimap2

| Feature | minimap2 | CLI Tools |
|---------|----------|-----------|
| Output format | SAM/PAF | JSON |
| Pipeline | All-in-one | Staged |
| Intermediate data | Internal | Inspectable |
| Use case | Production | Debugging/Testing |
| Multi-threading | Yes | Single-threaded |

## Limitations

- **Single-threaded**: Tools process queries sequentially
- **Memory**: Full index loaded for each stage (consider pre-built .mmi files for large references)
- **JSON overhead**: JSON format is verbose; not suitable for production workloads
- **No streaming**: All queries loaded into memory before processing

## Troubleshooting

### No seeds found
- Check k-mer size matches reference complexity
- Verify query sequences are valid FASTA/FASTQ
- Try smaller k value for divergent sequences

### Empty chains
- Ensure seeds JSON contains valid y values with q_span
- Check that reference and query files match between stages

### Alignment errors
- Verify chains JSON has both seeds and chains arrays
- Check that reference file is the same across all stages

## Related Documentation

- [minimap2 man page](../minimap2.1)
- [Algorithm overview](../docs/ARCHITECTURE_GUIDE.md)
