# Minimap2

Minimap2 is a versatile sequence alignment program that aligns DNA or mRNA
sequences against a large reference database. Typical use cases include: (1)
mapping PacBio or Oxford Nanopore genomic reads to the human genome; (2)
finding overlaps between long reads with error rate up to ~15%; (3)
splice-aware alignment of PacBio Iso-Seq or Nanopore cDNA or Direct RNA reads
against a reference genome; (4) aligning Illumina single- or paired-end reads;
(5) assembly-to-assembly alignment; (6) full-genome alignment between two
closely related species with divergence below ~15%.

This release is based on [minimap2 v2.28](https://github.com/lh3/minimap2/releases/tag/v2.28) (8170693) upstream with AMD GPU acceleration extensions. Tested on ROCm 7.0.2 and 7.2 on Ubuntu 24.04.

Minimap2 also supports **AMD GPU acceleration** for the chaining and alignment
stages via ROCm/HIP, delivering significant speedups on supported AMD GPUs.

For ~10kb noisy read sequences, minimap2 is tens of times faster than
mainstream long-read mappers such as BLASR, BWA-MEM, NGMLR and GMAP. It is more
accurate on simulated long reads and produces biologically meaningful alignment
ready for downstream analyses. For >100bp Illumina short reads, minimap2 is
three times as fast as BWA-MEM and Bowtie2, and as accurate on simulated data.
Detailed evaluations are available from the
[minimap2 paper](https://doi.org/10.1093/bioinformatics/bty191).

## Table of Contents

- [Quick Start](#started)
- [Presets Guide](#presets)
- [General Usage](#general)
- [Use Cases](#cases)
  - [Map long noisy genomic reads](#map-long-genomic)
  - [Map long mRNA/cDNA reads](#map-long-splice)
  - [Find overlaps between long reads](#long-overlap)
  - [Map short genomic reads](#short-genomic)
  - [Map short RNA-seq reads](#short-rna-seq)
  - [Full genome/assembly alignment](#full-genome)
- [GPU Acceleration](#gpu)
  - [GPU Configuration](#gpu-config)
  - [Verifying GPU Accuracy](#gpu-verify)
  - [Checking GPU Usage](#gpu-monitor)
  - [GPU Supported Features](#gpu-supported)
  - [GPU Unsupported Features](#gpu-unsupported)
  - [GPU Known Limitations](#gpu-limitations)
- [Command-Line Options](#options)
- [Environment Variables](#env-vars)
- [Output Formats](#output)
- [Common Workflows](#workflows)
- [Advanced Features](#advanced)
  - [Working with >65535 CIGAR operations](#long-cigar)
  - [The cs optional tag](#cs)
- [Debug and Diagnostic Output](#debug-output)
- [CLI Tools (Debugging/Analysis)](#cli-tools)
  - [mm2_seed](#cli-seed)
  - [mm2_chain](#cli-chain)
  - [mm2_align](#cli-align)
  - [Complete CLI Pipeline](#cli-pipeline)
  - [CLI Output Format (JSON)](#cli-json)
- [Algorithm Overview](#algo)
- [Troubleshooting](#troubleshooting)
- [Getting Help](#help)
- [Citing minimap2](#cite)
- [Limitations](#limit)

## <a name="started"></a>Quick Start

```sh
# long sequences against a reference genome
./minimap2 -a test/MT-human.fa test/MT-orang.fa > test.sam
# create an index first and then map
./minimap2 -x map-ont -d MT-human-ont.mmi test/MT-human.fa
./minimap2 -a MT-human-ont.mmi test/MT-orang.fa > test.sam
# use presets (no test data)
./minimap2 -ax map-pb ref.fa pacbio.fq.gz > aln.sam       # PacBio CLR genomic reads
./minimap2 -ax map-ont ref.fa ont.fq.gz > aln.sam         # Oxford Nanopore genomic reads
./minimap2 -ax map-hifi ref.fa pacbio-ccs.fq.gz > aln.sam # PacBio HiFi/CCS genomic reads (v2.19+)
./minimap2 -ax lr:hq ref.fa ont-Q20.fq.gz > aln.sam       # Nanopore Q20 genomic reads (v2.27+)
./minimap2 -ax sr ref.fa read1.fa read2.fa > aln.sam      # short genomic paired-end reads
./minimap2 -ax splice ref.fa rna-reads.fa > aln.sam       # spliced long reads (strand unknown)
./minimap2 -ax splice -uf -k14 ref.fa reads.fa > aln.sam  # noisy Nanopore direct RNA-seq
./minimap2 -ax splice:hq -uf ref.fa query.fa > aln.sam    # PacBio Kinnex/Iso-seq (RNA-seq)
./minimap2 -ax splice --junc-bed=anno.bed12 ref.fa query.fa > aln.sam  # use annotated junctions
./minimap2 -ax splice:sr ref.fa r1.fq r2.fq > aln.sam     # short-read RNA-seq (v2.29+)
./minimap2 -ax splice:sr -j anno.bed12 ref.fa r1.fq r2.fq > aln.sam
./minimap2 -cx asm5 asm1.fa asm2.fa > aln.paf             # intra-species asm-to-asm alignment
./minimap2 -x ava-pb reads.fa reads.fa > overlaps.paf     # PacBio read overlap
./minimap2 -x ava-ont reads.fa reads.fa > overlaps.paf    # Nanopore read overlap

# GPU-accelerated alignment (AMD GPU package only)
minimap2 -ax map-ont --gpu-chain --gpu-align ref.fa ont.fq.gz > aln.sam

# With optional JSON config for expert tuning
minimap2 -ax map-ont --gpu-chain --gpu-cfg /usr/share/minimap2/configs/gpu_config.json --gpu-align \
  ref.fa ont.fq.gz > aln.sam

# View the man page
man minimap2
```

## <a name="presets"></a>Presets Guide

Presets configure all parameters for specific data types. Use **-x** to select.

### Available Presets

| Preset | Data Type | Command Example |
|--------|-----------|------------------|
| `map-ont` | Oxford Nanopore | `minimap2 -ax map-ont ref.fa reads.fq` |
| `map-pb` | PacBio CLR | `minimap2 -ax map-pb ref.fa reads.fq` |
| `map-hifi` | PacBio HiFi/CCS | `minimap2 -ax map-hifi ref.fa reads.fq` |
| `asm5` | Assembly (~0.1% div) | `minimap2 -ax asm5 asm1.fa asm2.fa` |
| `asm10` | Assembly (~1% div) | `minimap2 -ax asm10 asm1.fa asm2.fa` |
| `asm20` | Assembly (~5% div) | `minimap2 -ax asm20 asm1.fa asm2.fa` |
| `sr` | Illumina/Short reads | `minimap2 -ax sr ref.fa r1.fq r2.fq` |
| `splice` | RNA-seq (long reads) | `minimap2 -ax splice ref.fa cdna.fq` |
| `splice:hq` | RNA-seq (HQ long reads) | `minimap2 -ax splice:hq ref.fa cdna.fq` |
| `ava-ont` | ONT read overlap | `minimap2 -x ava-ont reads.fq reads.fq` |
| `ava-pb` | PacBio read overlap | `minimap2 -x ava-pb reads.fq reads.fq` |

### Preset Selection Guide

```
Your Data Type
│
├─ Long Reads?
│  ├─ Oxford Nanopore → -ax map-ont
│  ├─ PacBio CLR → -ax map-pb
│  └─ PacBio HiFi → -ax map-hifi
│
├─ Assembly Alignment?
│  ├─ ~0.1% divergence → -ax asm5
│  ├─ ~1% divergence → -ax asm10
│  └─ ~5% divergence → -ax asm20
│
├─ Short Reads?
│  └─ Illumina/NGS → -ax sr
│
├─ RNA-seq?
│  ├─ Noisy long reads → -ax splice
│  └─ HQ long reads → -ax splice:hq
│
└─ Read Overlap?
   ├─ ONT reads → -x ava-ont
   └─ PacBio reads → -x ava-pb
```

### Default Preset Parameters

Different presets adjust these key parameters:

| Parameter | ONT | PacBio CLR | HiFi | Assembly (asm5/10) | Assembly (asm20) | Short | Splice |
|-----------|-----|------------|------|--------------------|------------------|-------|--------|
| K-mer (k) | 15 | 19 | 19 | 19 | 19 | 21 | 15 |
| Window (w) | 10 | 10 | 19 | 19 | 10 | 11 | 5 |
| Bandwidth | 500 | 500 | 500 | 1000 | 1000 | 100 | 200000 |
| Match score | 2 | 2 | 1 | 1 | 1 | 2 | 1 |
| Error tolerance | ~15% | ~15% | ~1% | ~0.1–1% | ~5% | ~2% | ~15% |

## <a name="general"></a>General Usage

Without any options, minimap2 takes a reference database and a query sequence
file as input and produces approximate mapping, without base-level alignment
(i.e. coordinates are only approximate and no CIGAR in output), in the [PAF format][paf]:
```sh
minimap2 ref.fa query.fq > approx-mapping.paf
```
You can ask minimap2 to generate CIGAR at the `cg` tag of PAF with:
```sh
minimap2 -c ref.fa query.fq > alignment.paf
```
or to output alignments in the [SAM format][sam]:
```sh
minimap2 -a ref.fa query.fq > alignment.sam
```
Minimap2 seamlessly works with gzip'd FASTA and FASTQ formats as input. You
don't need to convert between FASTA and FASTQ or decompress gzip'd files first.

For the human reference genome, minimap2 takes a few minutes to generate a
minimizer index for the reference before mapping. To reduce indexing time, you
can optionally save the index with option **-d** and replace the reference
sequence file with the index file on the minimap2 command line:
```sh
minimap2 -d ref.mmi ref.fa                     # indexing
minimap2 -a ref.mmi reads.fq > alignment.sam   # alignment
```
***Importantly***, it should be noted that once you build the index, indexing
parameters such as **-k**, **-w**, **-H** and **-I** can't be changed during
mapping. If you are running minimap2 for different data types, you will
probably need to keep multiple indexes generated with different parameters.
This makes minimap2 different from BWA which always uses the same index
regardless of query data types.

## <a name="cases"></a>Use Cases

Minimap2 uses the same base algorithm for all applications. However, due to the
different data types it supports (e.g. short vs long reads; DNA vs mRNA reads),
minimap2 needs to be tuned for optimal performance and accuracy. It is usually
recommended to choose a preset with option **-x**, which sets multiple
parameters at the same time. The default setting is the same as `map-ont`.

### <a name="map-long-genomic"></a>Map long noisy genomic reads

```sh
minimap2 -ax map-pb  ref.fa pacbio-reads.fq > aln.sam   # for PacBio CLR reads
minimap2 -ax map-ont ref.fa ont-reads.fq > aln.sam      # for Oxford Nanopore reads
minimap2 -ax map-iclr ref.fa iclr-reads.fq > aln.sam    # for Illumina Complete Long Reads
```
The difference between `map-pb` and `map-ont` is that `map-pb` uses
homopolymer-compressed (HPC) minimizers as seeds, while `map-ont` uses ordinary
minimizers as seeds. Empirical evaluation suggests HPC minimizers improve
performance and sensitivity when aligning PacBio CLR reads, but hurt when aligning
Nanopore reads. `map-iclr` uses an adjusted alignment scoring matrix that
accounts for the low overall error rate in the reads, with transversion errors
being less frequent than transitions.

### <a name="map-long-splice"></a>Map long mRNA/cDNA reads

```sh
minimap2 -ax splice:hq -uf ref.fa iso-seq.fq > aln.sam       # PacBio Iso-seq/traditional cDNA
minimap2 -ax splice ref.fa nanopore-cdna.fa > aln.sam        # Nanopore 2D cDNA-seq
minimap2 -ax splice -uf -k14 ref.fa direct-rna.fq > aln.sam  # Nanopore Direct RNA-seq
minimap2 -ax splice --splice-flank=no SIRV.fa SIRV-seq.fa    # mapping against SIRV control
```
There are different long-read RNA-seq technologies, including traditional
full-length cDNA, EST, PacBio Iso-seq, Nanopore 2D cDNA-seq and Direct RNA-seq.
They produce data of varying quality and properties. By default, `-x splice`
assumes the read orientation relative to the transcript strand is unknown. It
tries two rounds of alignment to infer the orientation and write the strand to
the `ts` SAM/PAF tag if possible. For Iso-seq, Direct RNA-seq and traditional
full-length cDNAs, apply `-u f` to force minimap2 to
consider the forward transcript strand only. This speeds up alignment with
slight improvement to accuracy. For noisy Nanopore Direct RNA-seq reads, it is
recommended to use a smaller k-mer size for increased sensitivity to the first
or the last exons.

Minimap2 rates an alignment by the score of the max-scoring sub-segment,
*excluding* introns, and marks the best alignment as primary in SAM. When a
spliced gene also has unspliced pseudogenes, minimap2 slightly prefers
the spliced alignment. By default, minimap2 outputs up to five secondary
alignments (i.e. likely pseudogenes in the context of RNA-seq mapping). This
can be tuned with option **-N**.

For long RNA-seq reads, minimap2 may produce chimeric alignments potentially
caused by gene fusions/structural variations or by an intron longer than the
max intron length **-G** (200k by default). For now, it is not recommended to
apply an excessively large **-G** as this slows down minimap2 and sometimes
leads to false alignments.

It is worth noting that by default `-x splice` prefers GT[A/G]..[C/T]AG
over GT[C/T]..[A/G]AG, and then over other splicing signals. Considering
one additional base improves the junction accuracy for noisy reads, but
reduces the accuracy when aligning against the widely used SIRV control data.
This is because SIRV does not honor the evolutionarily conservative splicing
signal. If you are studying SIRV, you may apply `--splice-flank=no` to let
minimap2 only model GT..AG, ignoring the additional base.

Minimap2 can optionally take annotated genes as input and
prioritize on annotated splice junctions. To use this feature, prepare a BED12
file from your gene annotation using `paftools.js` (a companion JavaScript
utility included with minimap2):
```sh
paftools.js gff2bed anno.gff > anno.bed
minimap2 -ax splice --junc-bed anno.bed ref.fa query.fa > aln.sam
```
Here, `anno.gff` is the gene annotation in the GTF or GFF3 format (`gff2bed`
automatically tests the format). The output of `gff2bed` is in the 12-column
BED format, or the BED12 format. With the `--junc-bed` option, minimap2 adds a
bonus score (tuned by `--junc-bonus`) if an aligned junction matches a junction
in the annotation. Option `--junc-bed` also takes 5-column BED, including the
strand field. In this case, each line indicates an oriented junction.

**Note:** `--junc-bed` is intended for long noisy RNA-seq reads only.
Applying the option to short RNA-seq reads would increase run time with little
improvement to junction accuracy.

### <a name="long-overlap"></a>Find overlaps between long reads

```sh
minimap2 -x ava-pb  reads.fq reads.fq > ovlp.paf    # PacBio CLR read overlap
minimap2 -x ava-ont reads.fq reads.fq > ovlp.paf    # Oxford Nanopore read overlap
```
Similarly, `ava-pb` uses HPC minimizers while `ava-ont` uses ordinary
minimizers. It is usually not recommended to perform base-level alignment in
the overlapping mode because it is slow and may produce false positive
overlaps. However, if performance is not a concern, you may try to add `-a` or
`-c` anyway.

### <a name="short-genomic"></a>Map short genomic reads

```sh
minimap2 -ax sr ref.fa reads-se.fq > aln.sam           # single-end alignment
minimap2 -ax sr ref.fa read1.fq read2.fq > aln.sam     # paired-end alignment
minimap2 -ax sr ref.fa reads-interleaved.fq > aln.sam  # paired-end alignment
```
When two read files are specified, minimap2 reads from each file in turn and
merges them into an interleaved stream internally. Two reads are considered to
be paired if they are adjacent in the input stream and have the same name (with
the `/[0-9]` suffix trimmed if present). Single- and paired-end reads can be
mixed.

#### <a name="short-rna-seq"></a>Map short RNA-seq reads

```sh
minimap2 -ax splice:sr ref.fa reads-se.fq.gz > aln.sam           # single-end
minimap2 -ax splice:sr ref.fa r1.fq.gz r2.fq.gz > aln.sam        # paired-end
minimap2 -ax splice:sr -j anno.bed ref.fa r1.fq r2.fq > aln.sam  # use annotation
# 2-pass alignment
minimap2 -x splice:sr -j anno.bed --write-junc ref.fa r1.fq r2.fq > junc.bed
minimap2 -ax splice:sr -j anno.bed --pass1=junc.bed ref.fa r1.fq r2.fq > aln.sam
```
The new preset `splice:sr` was added in v2.29. It functions similarly to `sr`
except that it performs spliced alignment.

### <a name="full-genome"></a>Full genome/assembly alignment

```sh
minimap2 -ax asm5 ref.fa asm.fa > aln.sam       # assembly to assembly/ref alignment
```
For cross-species full-genome alignment, the scoring system needs to be tuned
according to the sequence divergence.

## <a name="gpu"></a>GPU Acceleration

The AMD GPU variant of minimap2 delivers hardware-accelerated chaining and
alignment on AMD GPUs via ROCm/HIP.

```bash
# GPU chaining only (auto-derives config from hardware)
minimap2 -ax map-ont --gpu-chain ref.fa reads.fq > aln.sam

# GPU chaining + alignment
minimap2 -ax map-ont --gpu-chain --gpu-align ref.fa reads.fq > aln.sam

# With optional JSON config for expert tuning
minimap2 -ax map-ont --gpu-chain --gpu-cfg /usr/share/minimap2/configs/gpu_config.json --gpu-align \
  ref.fa reads.fq > aln.sam
```

### <a name="gpu-config"></a>GPU Configuration

**GPU chaining auto-derives configuration from hardware properties (CU count, VRAM, host RAM) at startup. An optional JSON config file can be provided via `--gpu-cfg` for expert tuning — any fields specified in the JSON override the auto-derived defaults. GPU alignment does not require additional configuration.**

Sample GPU configuration:

```json
{
    "num_streams": 1,
    "min_n": 512,
    "long_seg_buffer_size": 100000000,
    "max_total_n": 500000000,
    "max_read": 500000,
    "range_kernel": {
        "blockdim": 512,
        "cut_check_anchors": 10,
        "anchor_per_block": 32768
    },
    "score_kernel": {
        "micro_batch": 4,
        "mid_blockdim": 512,
        "short_griddim": 2688,
        "long_griddim": 144,
        "mid_griddim": 2688,
        "long_seg_cutoff": 20,
        "mid_seg_cutoff": 3
    }
}
```

**Key parameters:**
- `min_n`: Minimum anchors for GPU (default 512) — queries with fewer anchors use CPU
- `max_total_n`: Maximum anchors per batch (default 500M) — adjust based on GPU memory
- `max_read`: Maximum reads per batch (default 500k)
- `micro_batch`: Number of micro batches to aggregate (default 4)
- `long_griddim`: Grid dimension for long segments (suggested value: 2× GPU compute units)

Tune these values based on your GPU's memory and compute units.

### <a name="gpu-verify"></a>Verifying GPU Accuracy

GPU chaining uses implicitly `max-chain-skip=infinity`, which provides higher
precision relative to the default CPU configuration. 

To verify the GPU results matches the CPU baseline:

```bash
# GPU chaining
minimap2 -t 1 --gpu-chain ref.fa reads.fq > gpu_out.paf

# CPU baseline (match the implicit max-chain-skip setting)
minimap2 -t 1 --max-chain-skip=infinity ref.fa reads.fq > cpu_out.paf

# Compare — output should be empty (identical results)
diff cpu_out.paf gpu_out.paf
```

**Note:** When enabling `--gpu-chain`, `--gpu-align`, or both, output may differ slightly from CPU-only results. See [GPU Known Limitations](#gpu-limitations) for details.

### <a name="gpu-monitor"></a>Checking GPU Usage

```bash
# Check GPU status
rocm-smi

# Monitor during execution
watch -n 1 rocm-smi
```

### <a name="gpu-supported"></a>GPU Supported Features

- Chaining and alignment stages can run on the GPU.
- Mixed CPU/GPU configuration is supported — indexing and seeding run on the CPU while chaining and alignment run on the GPU.
- The CLI tools `mm2_seed`, `mm2_chain`, and `mm2_align` allow running each pipeline stage separately for debugging and analysis.

### <a name="gpu-unsupported"></a>GPU Unsupported Features

- Indexing and seeding can only run on the CPU.

### <a name="gpu-limitations"></a>GPU Known Limitations

- GPU acceleration performance is slightly behind CPU-only execution in this release. Optimization efforts are ongoing.
- Seeding acceleration is pending.
- RMQ mode in chaining runs on the CPU.
- Spliced mRNA alignment runs on the CPU.
- GPU chaining supports early stopping via `--max-chain-skip`. The default `--gpu-chain` uses `--max-chain-skip=infinity` (a dedicated forward-push kernel that evaluates every predecessor candidate), which is the fastest GPU chaining mode. Any **finite** `--max-chain-skip` (e.g. `--max-chain-skip=25`) routes through a backward-scan kernel that replicates the CPU algorithm exactly.
- **GPU chaining output parity.** A finite `--max-chain-skip` is **bit-exact** with the CPU (byte-identical PAF verified on full map-ont/map-hifi/map-pb). The default `--max-chain-skip=infinity` (push kernel) is bit-exact on the vast majority of reads but **not guaranteed bit-exact**: on a tiny fraction of high-anchor repeat reads (~0.012% on map-ont; none observed on map-hifi/map-pb) it may produce a slightly different chain composition (`cm:i` off by 1–2; same mapping locus, identical `dv:f`) when the `--max-chain-iter` window cap binds and an equal-score predecessor tie is broken differently at the window boundary. This is deterministic and does not affect alignment quality. **Use a finite `--max-chain-skip` when byte-identical output to the CPU is required.**
- **GPU alignment output parity.** With `--gpu-align`, a small discrepancy (<1%) may be observed in CIGAR strings versus CPU (e.g. on the PacBio dataset), because the tie-breaking logic in the CPU and GPU alignment kernels is not identical. This does not affect mapping correctness — multiple CIGAR strings can validly represent the same alignment.
- **Limited Testing:** This version has not undergone rigorous validation across diverse datasets and use cases.
- **Potential Defects:** Users should expect bugs and edge cases that have not been identified or resolved.
- **Production Use:** Strongly discouraged. This release is for evaluation purposes only and should not be deployed in production environments or used for critical research workflows.

## <a name="options"></a>Command-Line Options

### Essential Options

| Option | Description | Example |
|--------|-------------|---------|
| `-a` | Output SAM format | `-a` (use with `-x`) |
| `-c` | Generate CIGAR (for PAF) | `-c` |
| `-x STR` | Preset | `-x map-ont` |
| `-t INT` | Threads | `-t 16` |

### Input/Output Options

| Option | Description | Default |
|--------|-------------|----------|
| `-o FILE` | Output file | stdout |
| `--sam-hit-only` | No unmapped reads in SAM | Include unmapped |
| `-L` | Use hard clipping in SAM | Soft clipping |
| `-Y` | Use soft clipping for supplementary | Hard clipping |

### Indexing Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d FILE` | Save index to file | Don't save |
| `-k INT` | K-mer size | 15 (preset-dependent) |
| `-w INT` | Minimizer window | 10 (preset-dependent) |
| `-I NUM[G\|K\|M]` | Split index every NUM bp | 4G |

### Mapping Options

| Option | Description | Default |
|--------|-------------|---------|
| `-f NUM` | Filter out top NUM fraction | 0.0002 |
| `-g NUM` | Max gap on reference | 5000 |
| `-G NUM` | Max intron length (splice) | 200k |
| `-r NUM` | Bandwidth | 500 |
| `-N NUM` | Max secondary alignments | 5 |
| `--secondary=no` | Disable secondary alignments | Enabled |

### Alignment Options

| Option | Description | Default |
|--------|-------------|---------|
| `-A INT` | Matching score | 2 |
| `-B INT` | Mismatch penalty | 4 |
| `-O INT[,INT]` | Gap open penalty | 4,24 |
| `-E INT[,INT]` | Gap extension penalty | 2,1 |
| `-z INT[,INT]` | Z-drop | 400,200 |
| `-s INT` | Min peak DP score | 40 |

### GPU Options

| Option | Description | Default |
|--------|-------------|---------|
| `--gpu-chain` | GPU chaining | CPU |
| `--gpu-align` | GPU alignment | CPU |
| `--gpu-cfg FILE` | GPU config file (optional; overrides auto-derived defaults) | Auto-derived |
| `--gpu-flush-threshold INT` | Accumulated tasks before GPU flush; 0 = auto-detect | Auto-derived |
| `--gpu-batch-max-align INT` | Max alignments per GPU batch; 0 = auto-detect | Auto-derived |
| `--gpu-batch-max-mem NUM` | GPU memory limit per-thread for alignment batching; 0 = auto-detect | Auto-derived |

### Splicing Options (for splice/splice:hq)

| Option | Description |
|--------|-------------|
| `--junc-bed FILE` | Junction BED file |
| `--junc-bonus INT` | Bonus for known junctions |
| `-G NUM` | Max intron length |
| `-C NUM` | Cost for non-canonical splice |

### Other Options

| Option | Description | When to Use |
|--------|-------------|-------------|
| `--MD` | Generate MD tag | For variant calling |
| `--eqx` | Use =/X instead of M in CIGAR | Differentiate match/mismatch |
| `-p NUM` | Minimal peak to rescue | Increase for low coverage |
| `--cs` | Output cs (short) tag | For pangenome alignments |
| `--debug-log FILE` | Redirect debug output to FILE | For troubleshooting |
| `--extra-out-dir DIR` | Output directory for timer stats and intermediate data | Required for `--dump-intermediates`; timer stats always written |
| `--dump-intermediates[=STR]` | Dump per-query seeds, chains, and alignments; STR is `FLATBUF` (default) or `JSON` | Requires `--extra-out-dir`; if STR is omitted, defaults to FLATBUF |
| `--dump-chained-seeds` | Include per-chain seed indices in FLATBUF dump | Opt-in; adds ~1.6% to dump size. Only affects FLATBUF format |
| `--cap-kalloc NUM` | Free thread-local memory pool if total size exceeds NUM after alignment. Set 0 to disable [1g] | Reduce memory for many small reads |
| `--cap-kalloc-largest NUM` | Free thread-local memory pool if largest allocation block exceeds NUM. Set 0 to disable [256m] | Tune pool reset frequency for large reads |
| `--lj-min-ratio NUM` | Long join min ratio | Ultra-long read optimization |

## <a name="env-vars"></a>Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MM2_EXTRA_OUT_DIR` | Output directory for timer stats and intermediate data (equivalent to `--extra-out-dir`) | Unset |

This variable provides an alternative to the `--extra-out-dir` command-line option for setting the output directory.

### Intermediate Data Dump Details

The `--extra-out-dir` flag sets the output directory for timer stats. Add
`--dump-intermediates` (defaults to FLATBUF) or `--dump-intermediates=JSON` to also
dump per-query seed, chain, and alignment data.

```bash
# Timer stats only
minimap2 -ax map-ont --extra-out-dir /tmp/dump ref.fa reads.fq > aln.sam

# Timer stats + per-query intermediates as JSON (one file per query per stage)
minimap2 -ax map-ont --extra-out-dir /tmp/dump --dump-intermediates=JSON ref.fa reads.fq > aln.sam

# Timer stats + per-query intermediates as FlatBuffers (one binary file per query)
minimap2 -ax map-ont --extra-out-dir /tmp/dump --dump-intermediates ref.fa reads.fq > aln.sam
```

By default, FLATBUF dumps omit per-chain seed data to minimize size. To include
chain→seed traceability (each chain's anchors stored as uint32 indices into the
seeds array), add `--dump-chained-seeds`:

```bash
# FLATBUF with chain→seed lineage
minimap2 -ax map-ont --extra-out-dir /tmp/dump --dump-intermediates --dump-chained-seeds ref.fa reads.fq > aln.sam
```

**FLATBUF output directory layout:**

```
<dump-dir>/
  stats/
    timer_summary.json        # Per-stage timing (seed/chain/align min/max/avg)
  <query_name>.bin            # Only with --dump-intermediates FLATBUF
```

**JSON output directory layout:**

```
<dump-dir>/
  stats/
    timer_summary.json        # Per-stage timing (seed/chain/align min/max/avg)
  seeds/                       # Only with --dump-intermediates JSON
    <query_name>.json          # Per-query seed matches
  chains/                      # Only with --dump-intermediates JSON
    <query_name>.json          # Per-query chains
  alignments/                  # Only with --dump-intermediates JSON
    <query_name>.json          # Per-query alignments
```

The per-query files are used by the comparison scripts
(see [scripts/README.md](scripts/README.md)).

## <a name="output"></a>Output Formats

### SAM Format

Default format when using the `-a` flag:

```bash
minimap2 -ax map-ont ref.fa reads.fq > output.sam
```

**SAM fields:**
- QNAME: Query name
- FLAG: Alignment flags
- RNAME: Reference name
- POS: Alignment position
- MAPQ: Mapping quality (0–60)
- CIGAR: Alignment operations
- Additional tags: NM, MD, AS, etc.

**Common SAM processing:**
```bash
# Convert to BAM
samtools view -b output.sam > output.bam

# Sort BAM
samtools sort output.bam -o sorted.bam

# Index BAM
samtools index sorted.bam

# View specific region
samtools view sorted.bam chr1:1000-2000
```

### PAF Format

Default format without the `-a` flag (faster, no base-level alignment):

```bash
minimap2 -x map-ont ref.fa reads.fq > output.paf
```

**PAF columns:**
1. Query name
2. Query length
3. Query start
4. Query end
5. Strand (+/-)
6. Target name
7. Target length
8. Target start
9. Target end
10. Number of matches
11. Alignment block length
12. Mapping quality

**Example PAF line:**
```
read1  1000  50  980  +  chr1  10000000  1000  1930  890  930  60
```

**When to use PAF:**
- Assembly-to-assembly alignment (no CIGAR needed)
- Read overlap detection
- Faster processing (no base-level alignment)
- Lower memory usage

## <a name="workflows"></a>Common Workflows

### Batch Processing

```bash
# Build index once
minimap2 -d ref.mmi ref.fa

# Process samples in parallel
for sample in sample*.fq; do
  minimap2 -ax map-ont -t 4 ref.mmi "$sample" | \
    samtools sort -@ 2 -o "${sample%.fq}.bam" &
done
wait

# Index all BAMs
for bam in *.bam; do
  samtools index "$bam"
done
```

### Quality Filtering

```bash
# Align and filter MAPQ >= 20
minimap2 -ax map-ont ref.fa reads.fq | \
  samtools view -b -q 20 | \
  samtools sort -o filtered.bam

# Primary alignments only, high quality
minimap2 -ax map-ont --secondary=no ref.fa reads.fq | \
  samtools view -b -q 30 -F 2308 | \
  samtools sort -o primary_hq.bam
```

### Complete Pipeline with Statistics

```bash
# Align
minimap2 -ax map-ont -t 16 ref.fa reads.fq | \
  samtools sort -@ 4 -o aligned.bam

# Index
samtools index aligned.bam

# Statistics
samtools flagstat aligned.bam > stats.txt
samtools stats aligned.bam > detailed_stats.txt
samtools coverage aligned.bam > coverage.txt

# Extract mapped reads
samtools view -b -F 4 aligned.bam > mapped.bam
```

## <a name="advanced"></a>Advanced Features

### <a name="long-cigar"></a>Working with >65535 CIGAR operations

Due to a design flaw, BAM does not work with CIGAR strings with >65535
operations (SAM and CRAM work). However, for ultra-long nanopore reads minimap2
may align ~1% of read bases with long CIGARs beyond the capability of BAM. If
you convert such SAM/CRAM to BAM, Picard and recent samtools will throw an
error and abort. Older samtools and other tools may create corrupted BAM.

To avoid this issue, you can add option `-L` at the minimap2 command line.
This option moves a long CIGAR to the `CG` tag and leaves a fully clipped CIGAR
at the SAM CIGAR column. Current tools that don't read CIGAR (e.g. merging and
sorting) still work with such BAM records; tools that read CIGAR will
effectively ignore these records. It has been decided that future tools
will seamlessly recognize long-cigar records generated by option `-L`.

**TL;DR**: if you work with ultra-long reads and use tools that only process
BAM files, please add option `-L`.

### <a name="cs"></a>The cs optional tag

The `cs` SAM/PAF tag encodes bases at mismatches and INDELs. It matches regular
expression `/(:[0-9]+|\*[a-z][a-z]|[=\+\-][A-Za-z]+)+/`. Like CIGAR, `cs`
consists of series of operations.  Each leading character specifies the
operation; the following sequence is the one involved in the operation.

The `cs` tag is enabled by command line option `--cs`. The following alignment,
for example:
```txt
CGATCGATAAATAGAGTAG---GAATAGCA
||||||   ||||||||||   |||| |||
CGATCG---AATAGAGTAGGTCGAATtGCA
```
is represented as `:6-ata:10+gtc:4*at:3`, where `:[0-9]+` represents an
identical block, `-ata` represents a deletion, `+gtc` an insertion and `*at`
indicates reference base `a` is substituted with a query base `t`. It is
similar to the `MD` SAM tag but is standalone and easier to parse.

If `--cs=long` is used, the `cs` string also contains identical sequences in
the alignment. The above example will become
`=CGATCG-ata=AATAGAGTAG+gtc=GAAT*at=GCA`. The long form of `cs` encodes both
reference and query sequences in one string. The `cs` tag also encodes intron
positions and splicing signals (see `man minimap2` for details).

## <a name="debug-output"></a>Debug and Diagnostic Output

Minimap2 provides several mechanisms for controlling output verbosity and diagnostics.

### Verbosity Levels (`-v`)

Controlled by the `-v` flag (default: 3):

```bash
minimap2 -v 0 ...  # Suppress all non-essential messages (quiet mode)
minimap2 -v 1 ...  # Minimal messages
minimap2 -v 2 ...  # Normal messages
minimap2 -v 3 ...  # Verbose messages (default)
```

### Diagnostic Flags

User-facing flags that print internal data to stdout:

| Flag | Output |
|------|--------|
| `--print-chains` | Chaining information |
| `--print-seeds` | Seeding information |
| `--print-qname` | Query name information |
| `--print-aln-seq` | Full CIGAR strings from GPU alignment |

### Debug Log (`--debug-log`)

Internal debug messages can be redirected to a file:

```bash
minimap2 --debug-log debug.txt -ax map-ont ref.fa reads.fq > aln.sam
```

Without this flag, debug messages go to stderr.

### GPU Diagnostic Messages

GPU-specific diagnostics (memory allocations, kernel launches, processing
statistics) are written to the debug log file, If no debug log is specified,
these messages are emitted to stderr.

### Output Stream Separation

For clean output separation:

```bash
# Redirect stdout to SAM file, stderr to error log
minimap2 -v 0 -a ref.fa query.fq > output.sam 2> errors.log

# Also capture debug messages to a separate file
minimap2 -v 0 -a --debug-log debug.txt ref.fa query.fq > output.sam 2> errors.log

# Print chains with clean separation
minimap2 -v 0 -c --print-chains ref.fa query.fq > chains.paf 2> errors.log
```

## <a name="cli-tools"></a>CLI Tools (Debugging/Analysis)

For debugging and algorithm analysis, minimap2 provides three CLI tools that expose the internal pipeline stages:

```
SEEDING    →    CHAINING    →    ALIGNMENT
mm2_seed       mm2_chain        mm2_align
```

These tools are designed for:
- Debugging mapping issues at each stage
- Understanding algorithm behavior
- Parameter optimization research
- Analyzing intermediate results in JSON format

### <a name="cli-seed"></a>mm2_seed

Find minimizer seed matches:

```bash
# Basic usage
mm2_seed -x map-ont ref.fa query.fa -o seeds.json

# Custom parameters
mm2_seed -k 19 -w 10 ref.fa query.fa -o seeds.json

# With advanced filtering
mm2_seed -x map-ont -d 50 -m 100 ref.fa query.fa -o seeds.json
```

**Key options:**
- `-x` preset configuration
- `-k` k-mer size
- `-w` minimizer window size
- `-d` SDUST threshold for low-complexity masking (default: 0, disabled)
- `-m` max occurrence threshold (default: auto from index)
- `-M` hard max occurrence limit (default: 4095)
- `-D` occurrence distance for filtering (default: 500)
- `-o` output JSON file

### <a name="cli-chain"></a>mm2_chain

Connect seeds into chains:

```bash
# Basic usage
mm2_chain -x map-ont ref.fa -s seeds.json -o chains.json

# With GPU (config required)
mm2_chain -x map-ont --use-gpu --gpu-cfg gpu_config.json ref.fa -s seeds.json -o chains.json

# Ultra-long reads with RMQ
mm2_chain -x map-ont -r ref.fa -s seeds.json -o chains.json

# Custom chaining parameters
mm2_chain -x map-ont -g 10000 -b 1000 -n 2 -c 20 ref.fa -s seeds.json -o chains.json
```

**Key options:**
- `-s` input seeds JSON file
- `-o` output chains JSON file
- `-r` use RMQ algorithm (faster for ultra-long reads)
- `-g` maximum gap in reference/query (default: 5000)
- `-b` bandwidth for DP chaining (default: 500)
- `-B` long-join bandwidth for ultra-long reads (default: 0=disabled)
- `-n` minimum anchor count per chain (default: 3)
- `-c` minimum chain score (default: 40)
- `--use-gpu` enable GPU acceleration
- `--gpu-cfg` GPU configuration file (optional; overrides auto-derived defaults)

**Advanced options:**
- `--max-chain-skip` max anchors to skip (default: 25)
- `--max-chain-iter` max DP iterations (default: 5000)
- `--chain-gap-scale` gap cost scaling (default: 1.0)
- `--chain-skip-scale` skip cost scaling (default: 1.0)

### <a name="cli-align"></a>mm2_align

Perform base-level alignment:

```bash
# Basic usage
mm2_align -x map-ont ref.fa query.fa -c chains.json -o alignments.json

# With GPU
mm2_align -x map-ont --use-gpu ref.fa query.fa -c chains.json -o alignments.json

# Custom scoring
mm2_align -A 1 -B 2 -O 2,10 -E 1,1 -z 200,100 ref.fa query.fa -c chains.json -o out.json
```

**Key options:**
- `-c` input chains JSON file
- `-o` output JSON file
- `-A` matching score (default: 2)
- `-B` mismatch penalty (default: 4)
- `-O` gap open penalty, comma-separated for short/long gaps (default: 4,24)
- `-E` gap extension penalty, comma-separated (default: 2,1)
- `-z` Z-drop score, comma-separated for normal/inversion (default: 400,200)
- `-s` minimum DP alignment score (default: 40)
- `--use-gpu` enable GPU acceleration

### <a name="cli-pipeline"></a>Complete CLI Pipeline

```bash
# Full staged pipeline
mm2_seed -x map-ont ref.fa reads.fq -o seeds.json
mm2_chain -x map-ont ref.fa -s seeds.json -o chains.json
mm2_align -x map-ont ref.fa reads.fq -c chains.json -o alignments.json

# With GPU acceleration
mm2_seed -x map-ont ref.fa reads.fq -o seeds.json
mm2_chain -x map-ont --use-gpu --gpu-cfg gpu_config.json ref.fa -s seeds.json -o chains.json
mm2_align -x map-ont --use-gpu ref.fa reads.fq -c chains.json -o alignments.json
```

### <a name="cli-json"></a>CLI Output Format (JSON)

All CLI tools output JSON for easy parsing:

**Seeds output:**
```json
{
  "queries": [{
    "name": "read1",
    "length": 1000,
    "n_seeds": 42,
    "seeds": [...]
  }]
}
```

**Chains output:**
```json
{
  "queries": [{
    "name": "read1",
    "n_chains": 1,
    "chains": [{
      "score": 345,
      "length": 42
    }]
  }]
}
```

**Alignments output:**
```json
{
  "queries": [{
    "name": "read1",
    "alignments": [{
      "mapq": 60,
      "cigar": "800M",
      "qs": 0, "qe": 800,
      "rs": 1000, "re": 1800
    }]
  }]
}
```

### Debugging with CLI Tools

```bash
# 1. Check if seeds are found
mm2_seed -x map-ont ref.fa problem_read.fa -o seeds.json
echo "Seeds found: $(jq '.queries[0].n_seeds' seeds.json)"

# 2. Check if chains form
mm2_chain -x map-ont ref.fa -s seeds.json -o chains.json
echo "Chains formed: $(jq '.queries[0].n_chains' chains.json)"

# 3. Check alignment quality
mm2_align -x map-ont ref.fa problem_read.fa -c chains.json -o align.json
echo "MAPQ: $(jq '.queries[0].alignments[0].mapq' align.json)"
```

### Parameter Analysis Example

```bash
# Compare k-mer sizes
for k in 13 15 17 19; do
  mm2_seed -k $k ref.fa query.fa -o seeds_k${k}.json
  seeds=$(jq '.queries[0].n_seeds' seeds_k${k}.json)
  echo "k=$k: $seeds seeds"
done
```

## <a name="algo"></a>Algorithm Overview

In the following, minimap2 command line options have a dash ahead and are
highlighted in bold. The description may help to tune minimap2 parameters.

1. Read **-I** [=*4G*] reference bases, extract (**-k**,**-w**)-minimizers and
   index them in a hash table.

2. Read **-K** [=*200M*] query bases. For each query sequence, do step 3
   through 7:

3. For each (**-k**,**-w**)-minimizer on the query, check against the reference
   index. If a reference minimizer is not among the top **-f** [=*2e-4*] most
   frequent, collect the occurrences in the reference, which are called
   *seeds*.

4. Sort seeds by position in the reference. Chain them with dynamic
   programming. Each chain represents a potential mapping. For read
   overlapping, report all chains and then go to step 8. For reference mapping,
   do step 5 through 7:

5. Let *P* be the set of primary mappings, which is an empty set initially. For
   each chain from the best to the worst according to their chaining scores: if
   on the query, the chain overlaps with a chain in *P* by **--mask-level**
   [=*0.5*] or higher fraction of the shorter chain, mark the chain as
   *secondary* to the chain in *P*; otherwise, add the chain to *P*.

6. Retain all primary mappings. Also retain up to **-N** [=*5*] top secondary
   mappings if their chaining scores are higher than **-p** [=*0.8*] of their
   corresponding primary mappings.

7. If alignment is requested, filter out an internal seed if it potentially
   leads to both a long insertion and a long deletion. Extend from the
   left-most seed. Perform global alignments between internal seeds.  Split the
   chain if the accumulative score along the global alignment drops by **-z**
   [=*400*], disregarding long gaps. Extend from the right-most seed.  Output
   chains and their alignments.

8. If there are more query sequences in the input, go to step 2 until no more
   queries are left.

9. If there are more reference sequences, reopen the query file from the start
   and go to step 1; otherwise stop.

## <a name="troubleshooting"></a>Troubleshooting

### No alignments found

**Symptoms:** Empty SAM/PAF output, no alignment records.

**Solutions:**
1. Verify correct preset: Use `-x map-ont` for ONT, `-x map-hifi` for HiFi, etc.
2. Check input file format: Ensure FASTA/FASTQ is valid.
3. Verify reference and query aren't swapped.
4. Try more sensitive settings: Smaller k-mer (`-k 13`).
5. Check sequence divergence matches preset expectations.

```bash
# More sensitive mapping
minimap2 -ax map-ont -k 13 ref.fa reads.fq > aligned.sam

# Check alignment statistics
samtools flagstat aligned.sam
```

### Low mapping quality

**Symptoms:** MAPQ < 20 in SAM output, many secondary alignments.

**Solutions:**
1. Verify correct preset for your data type.
2. Check for repetitive sequences or contamination.
3. Filter for primary alignments only: `--secondary=no`.
4. Increase minimum alignment score: `-s 60`.

```bash
# Primary alignments only
minimap2 -ax map-ont --secondary=no ref.fa reads.fq > primary.sam

# Filter by MAPQ in post-processing
samtools view -b -q 30 aligned.sam > high_quality.bam
```

### No chains formed (CLI tools)

**Symptoms:** `"n_chains": 0` in mm2_chain JSON output.

**Solutions:**
1. Increase max gap: `-g 10000` for fragmented alignments.
2. Lower chain score threshold: `-c 20`.
3. Reduce minimum anchor count: `-n 2`.
4. Check if seeds exist in input JSON.

```bash
# More permissive chaining
mm2_chain -g 10000 -c 20 -n 2 ref.fa -s seeds.json -o chains.json
```

### Poor alignment quality (CLI tools)

**Symptoms:** Low mapping quality (mapq < 20) or incorrect alignments in mm2_align output.

**Solutions:**
1. Verify you're using the correct preset for your data type.
2. Adjust alignment scoring parameters.
3. Check for contamination or adapter sequences.
4. Try different k-mer size in seeding.

```bash
# More sensitive alignment scoring
mm2_align -A 2 -B 3 -O 4,8 -E 2,1 ref.fa query.fa -c chains.json -o align.json
```

### Slow performance

**Solutions:**
1. Build index once and reuse: `-d ref.mmi`
2. Increase threads: `-t 16`
3. Disable secondary alignments: `--secondary=no`
4. Use PAF instead of SAM (skip `-a` flag)

```bash
# Optimize for speed
minimap2 -d ref.mmi ref.fa
minimap2 -x map-ont --secondary=no -t 16 ref.mmi reads.fq > aligned.paf
```

### Out of memory

**Solutions:**
1. Pre-build index and split into smaller chunks: `-I 2G`
2. Process queries in smaller batches.
3. Reduce k-mer size to reduce index size.

```bash
# Build smaller index chunks
minimap2 -I 2G -d ref.mmi ref.fa

# Process in batches, then merge
split -l 1000 reads.fq batch_
for batch in batch_*; do
  minimap2 -ax map-ont ref.mmi "$batch" | samtools sort -o "${batch}.bam"
done
samtools merge aligned.bam batch_*.bam
```

## <a name="help"></a>Getting Help

The `minimap2` man page (`man minimap2`) provides a detailed description of all
command-line options and optional tags.

If you encounter bugs or have further questions or requests, you can raise an
issue at the [issue page][issue].

## <a name="cite"></a>Citing minimap2

If you use minimap2 in your work, please cite:

> Li, H. (2018). Minimap2: pairwise alignment for nucleotide sequences.
> *Bioinformatics*, **34**:3094-3100. [doi:10.1093/bioinformatics/bty191](https://doi.org/10.1093/bioinformatics/bty191)

and/or:

> Li, H. (2021). New strategies to improve minimap2 alignment accuracy.
> *Bioinformatics*, **37**:4572-4574. [doi:10.1093/bioinformatics/btab705](https://doi.org/10.1093/bioinformatics/btab705)

If you use the GPU acceleration features, please also cite:

> Dong, J., Liu, X., Sadasivan, H., Sitaraman, S., & Narayanasamy, S. (2024).
> mm2-gb: GPU Accelerated Minimap2 for Long Read DNA Mapping.
> *bioRxiv*. [doi:10.1101/2024.03.23.586366](https://doi.org/10.1101/2024.03.23.586366)

## <a name="limit"></a>Limitations

* Minimap2 may produce suboptimal alignments through long low-complexity
  regions where seed positions may be suboptimal. This should not be a big
  concern because even the optimal alignment may be wrong in such regions.

* Minimap2 does not work with a single query or database sequence ~2
  billion bases or longer (2,147,483,647 to be exact). The total length of all
  sequences can well exceed this threshold.

* Minimap2 often misses small exons.

[paf]: https://github.com/lh3/miniasm/blob/master/PAF.md
[sam]: https://samtools.github.io/hts-specs/SAMv1.pdf
[issue]: https://github.com/ROCm-LS/minimap2/issues
