# Minimap2
**ALERT:** `minimap2.com` is a [phishing site](https://github.com/lh3/minimap2/issues/1316). Please don't use anything from that website.

Minimap2 is a versatile sequence alignment program that aligns DNA or mRNA
sequences against a large reference database. Typical use cases include: (1)
mapping PacBio or Oxford Nanopore genomic reads to the human genome; (2)
finding overlaps between long reads with error rate up to ~15%; (3)
splice-aware alignment of PacBio Iso-Seq or Nanopore cDNA or Direct RNA reads
against a reference genome; (4) aligning Illumina single- or paired-end reads;
(5) assembly-to-assembly alignment; (6) full-genome alignment between two
closely related species with divergence below ~15%.

Minimap2 also supports **AMD GPU acceleration** for the chaining and alignment
stages via ROCm/HIP, delivering significant speedups on supported AMD GPUs.

For ~10kb noisy read sequences, minimap2 is tens of times faster than
mainstream long-read mappers such as BLASR, BWA-MEM, NGMLR and GMAP. It is more
accurate on simulated long reads and produces biologically meaningful alignment
ready for downstream analyses. For >100bp Illumina short reads, minimap2 is
three times as fast as BWA-MEM and Bowtie2, and as accurate on simulated data.
Detailed evaluations are available from the
[minimap2 paper](https://doi.org/10.1093/bioinformatics/bty191).

## Quick Start

```sh
# Map Oxford Nanopore reads to a reference genome
minimap2 -ax map-ont ref.fa ont.fq.gz > aln.sam

# GPU-accelerated alignment (AMD GPU package only)
minimap2 -ax map-ont --gpu-chain --gpu-cfg /usr/share/minimap2/configs/gpu_config.json --gpu-align \
  ref.fa ont.fq.gz > aln.sam
```

See [USAGE.md](USAGE.md) for the complete usage guide — presets, command-line
options, output formats, GPU configuration, troubleshooting, and more.

## Documentation

### User Guides

| Document | Description |
|----------|-------------|
| [USAGE.md](USAGE.md) | Complete usage guide — presets, options, GPU acceleration, output formats, workflows, troubleshooting |
| [cookbook.md](cookbook.md) | Step-by-step recipes for common mapping tasks |
| [FAQ.md](FAQ.md) | Frequently asked questions |
| [NEWS.md](NEWS.md) | Release notes and changelog |

### Build & Installation

| Document | Description |
|----------|-------------|
| [BUILD_GUIDE.md](BUILD_GUIDE.md) | Building minimap2 from source (CPU, AMD GPU), packaging, installing |

### Developer Guides

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE_GUIDE.md](docs/ARCHITECTURE_GUIDE.md) | Internal pipeline architecture — batching, worker threads, memory management |
| [docs/PROFILING_GUIDE.md](docs/PROFILING_GUIDE.md) | Performance profiling with VTune, Nsight, and rocprof |
| [docs/VERSIONING.md](docs/VERSIONING.md) | Versioning system — git tags, CMake integration, release workflow |
| [cli/README.md](cli/README.md) | Staged CLI tools (mm2_seed, mm2_chain, mm2_align) for debugging and analysis |
| [scripts/README.md](scripts/README.md) | Test data generation and comparison scripts |
| [tests/README.md](tests/README.md) | Unit test framework and test suites |

### Other

| Document | Description |
|----------|-------------|
| [misc/README.md](misc/README.md) | Companion utilities — paftools.js, mmphase.js |
| [code_of_conduct.md](code_of_conduct.md) | Contributor Code of Conduct |

## Citing minimap2

If you use minimap2 in your work, please cite:

> Li, H. (2018). Minimap2: pairwise alignment for nucleotide sequences.
> *Bioinformatics*, **34**:3094-3100. [doi:10.1093/bioinformatics/bty191](https://doi.org/10.1093/bioinformatics/bty191)

> Li, H. (2021). New strategies to improve minimap2 alignment accuracy.
> *Bioinformatics*, **37**:4572-4574. [doi:10.1093/bioinformatics/btab705](https://doi.org/10.1093/bioinformatics/btab705)

If you use the GPU acceleration features, please also cite:

> Dong, J., Liu, X., Sadasivan, H., Sitaraman, S., & Narayanasamy, S. (2024).
> mm2-gb: GPU Accelerated Minimap2 for Long Read DNA Mapping.
> *bioRxiv*. [doi:10.1101/2024.03.23.586366](https://doi.org/10.1101/2024.03.23.586366)

## License

See [LICENSE.txt](LICENSE.txt).
