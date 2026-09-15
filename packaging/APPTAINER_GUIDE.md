# Minimap2 Docker Image Tar — Apptainer Guide

Guide to running the minimap2 Docker image tar (`minimap2.tar`) with Apptainer on HPC clusters and workstations.

For minimap2 usage details — presets, command-line options, output formats, GPU flags, CLI tools, and workflows — see the bundled documentation:

```bash
apptainer exec docker-archive:minimap2.tar cat /usr/share/doc/minimap2/USAGE.md
```

## Table of Contents

- [Quick Start](#quick-start)
- [Converting to SIF (Optional)](#converting-to-sif-optional)
- [Prerequisites](#prerequisites)
- [Bind-Mounting Data](#bind-mounting-data)
- [GPU Mode](#gpu-mode)
- [Troubleshooting](#troubleshooting)
- [Quick Reference Card](#quick-reference-card)

---

## Quick Start

Apptainer can run a Docker image tar directly using the `docker-archive:` prefix:

```bash
# Verify the container
apptainer exec docker-archive:minimap2.tar minimap2 --version

# CPU alignment
apptainer exec --bind /path/to/data:/data docker-archive:minimap2.tar \
  minimap2 -ax map-ont /data/ref.fa /data/reads.fq > aln.sam

# GPU-accelerated alignment
apptainer exec --rocm --bind /path/to/data:/data docker-archive:minimap2.tar \
  minimap2 -ax map-ont \
    --gpu-chain --gpu-cfg /usr/share/minimap2/configs/gpu_config.json --gpu-align \
    /data/ref.fa /data/reads.fq > aln.sam
```

---

## Converting to SIF (Optional)

For repeated use, convert the Docker image tar to a native SIF file to avoid the per-invocation conversion overhead:

```bash
apptainer build minimap2.sif docker-archive:minimap2.tar
```

After conversion, use `minimap2.sif` in place of `docker-archive:minimap2.tar` in all commands below.

---

## Prerequisites

- [Apptainer](https://apptainer.org/) (1.0+) installed on the host
- For GPU acceleration: AMD GPU with ROCm driver and `/dev/kfd`, `/dev/dri` accessible to your user

### What's in the Image

| Component | Path |
|-----------|------|
| `minimap2` | `/usr/bin/minimap2` |
| `mm2_seed`, `mm2_chain`, `mm2_align` | `/usr/bin/` |
| GPU configs | `/usr/share/minimap2/configs/` |
| Usage documentation | `/usr/share/doc/minimap2/USAGE.md` |

---

## Bind-Mounting Data

Apptainer does **not** automatically mount arbitrary host paths. Use `--bind` to make data accessible inside the container.

```bash
# Single directory
apptainer exec --bind /scratch/project:/data docker-archive:minimap2.tar \
  minimap2 -ax map-ont /data/ref.fa /data/reads.fq > aln.sam

# Multiple directories
apptainer exec --bind /scratch/ref:/ref,/scratch/reads:/reads docker-archive:minimap2.tar \
  minimap2 -ax map-ont /ref/hg38.fa /reads/sample.fq > aln.sam

# Current working directory
apptainer exec --bind "$PWD":/work docker-archive:minimap2.tar \
  minimap2 -ax map-ont /work/ref.fa /work/reads.fq > aln.sam
```

> **Tip:** Some HPC systems auto-bind `$HOME`, `/tmp`, and `$PWD`. Check your cluster documentation.

---

## GPU Mode

Add `--rocm` to expose AMD GPU devices inside the container:

```bash
apptainer exec --rocm --bind /path/to/data:/data docker-archive:minimap2.tar \
  minimap2 -ax map-ont \
    --gpu-chain --gpu-cfg /usr/share/minimap2/configs/gpu_config.json --gpu-align \
    /data/ref.fa /data/reads.fq > aln.sam
```

### Verify GPU Access

```bash
apptainer exec --rocm docker-archive:minimap2.tar rocminfo | grep "Name:.*gfx"
```

### Available GPU Configs

```bash
apptainer exec docker-archive:minimap2.tar ls /usr/share/minimap2/configs/
```

| Config File | Use Case |
|-------------|----------|
| `gpu_config.json` | General-purpose default |

For GPU flags, supported features, and limitations, see the GPU Acceleration section in USAGE.md.

---

## Troubleshooting

### Files not found inside the container

```
[ERROR] failed to open file '/data/ref.fa': No such file or directory
```

Use `--bind` to mount host directories:

```bash
apptainer exec --bind /scratch/mydata:/data docker-archive:minimap2.tar \
  minimap2 -ax map-ont /data/ref.fa /data/reads.fq
```

### GPU not detected

```
Unable to open /dev/kfd read-write
```

1. Add `--rocm` to the Apptainer command
2. Verify GPU access on the host: `ls /dev/kfd /dev/dri`
3. Check group membership: `id` (need `video` and `render`)
4. On Slurm, request GPU resources: `#SBATCH --gres=gpu:1`

### `setgroups: Permission denied`

AppArmor restricts unprivileged user namespaces (default on Ubuntu 24.04). Ask your system administrator to add an AppArmor exception for Apptainer.

### `fuse2fs not found` warning

Non-fatal warning — the container functions correctly without it.

---

## Quick Reference Card

### Apptainer Flags

| Flag | Purpose |
|------|---------|
| `--rocm` | Expose AMD GPU devices inside the container |
| `--bind /host:/container` | Bind-mount a host directory |
| `exec <image> <cmd>` | Run a command inside the container |
| `shell <image>` | Open an interactive shell |

### Common Commands

```bash
# Verify version
apptainer exec docker-archive:minimap2.tar minimap2 --version

# View full usage documentation
apptainer exec docker-archive:minimap2.tar cat /usr/share/doc/minimap2/USAGE.md

# List GPU configs
apptainer exec docker-archive:minimap2.tar ls /usr/share/minimap2/configs/

# CPU alignment
apptainer exec --bind /data:/data docker-archive:minimap2.tar \
  minimap2 -ax map-ont -t 16 /data/ref.fa /data/reads.fq > aln.sam

# GPU alignment
apptainer exec --rocm --bind /data:/data docker-archive:minimap2.tar \
  minimap2 -ax map-ont \
    --gpu-chain --gpu-cfg /usr/share/minimap2/configs/gpu_config.json --gpu-align \
    /data/ref.fa /data/reads.fq > aln.sam

# Interactive shell
apptainer shell --rocm --bind /data:/data docker-archive:minimap2.tar
```
