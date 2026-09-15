# Docker Guide

This guide covers building and using minimap2 Docker images for testing
DEB, RPM, and SLES RPM packages.

## Overview

Packaging files are located in the `packaging/` directory:

| File | Build OS | Package | Test OS |
|------|----------|---------|---------|
| `Dockerfile.deb` | Ubuntu (jammy/noble) | `.deb` | Ubuntu |
| `Dockerfile.rpm` | RHEL UBI (8/9/10) | `.rpm` | RHEL UBI |
| `Dockerfile.sles` | SLES-compatible SUSE BCI 15.x | `.rpm` (SLES) | SLES-compatible SUSE BCI 15.x |
| `minimap2.def` | — | Apptainer SIF | — |

Each Dockerfile uses a **multi-stage build**. `Dockerfile.deb` has four stages:

1. **rocm-base** — Shared ROCm repo setup + dpkg-exclude patches (cached across downstream stages)
2. **builder** — Full SDK; compiles minimap2; produces the `.deb` and a CI artifact bundle (`.deb`, `version.txt`, `minimap2-tests.tar.gz`)
3. **artifacts** — `scratch` stage used by CI (`buildctl --output type=local` / `docker build --output type=local`) to extract artifacts directly to the host; never produced by a plain `docker build` unless selected with `--target artifacts`
4. **runtime** — Slim image; installs only the `.deb` and runs an assertive smoke test (verifies binaries, man pages, docs, shared library, GPU config)

`Dockerfile.rpm` and `Dockerfile.sles` use a simpler **builder** → **test** two-stage layout. The DEB **builder** stage name matches, so `--target builder` works consistently across all three Dockerfiles.

The runtime/test image contains:

- `minimap2` — main aligner
- `mm2_seed`, `mm2_chain`, `mm2_align` — staged mapping CLI tools
- `libminimap2.so` — shared library
- Man pages, documentation, GPU config
- ROCm HIP runtime libraries for GPU acceleration

## Prerequisites

- Docker 19.03+ (with BuildKit recommended)
- For runtime GPU access: AMD GPU visible to Docker

## Building the DEB Image (Ubuntu)

```bash
# Ubuntu 24.04 (Noble) — default
docker build -f packaging/Dockerfile.deb \
  --build-arg BASE_IMAGE=ubuntu:noble \
  --build-arg ROCM_VERSION=7.2.1 \
  -t minimap2-deb-test .

# Ubuntu 22.04 (Jammy)
docker build -f packaging/Dockerfile.deb \
  --build-arg BASE_IMAGE=ubuntu:jammy \
  --build-arg ROCM_VERSION=7.2.1 \
  -t minimap2-deb-test-jammy .
```

The Ubuntu codename is derived automatically inside the image via `lsb_release -cs`,
so you only need to pass the base image tag.

### DEB build arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `BASE_IMAGE` | `ubuntu:noble` | Ubuntu base image (e.g. `ubuntu:noble`, `ubuntu:jammy`) |
| `ROCM_VERSION` | `7.2.1` | ROCm version (matching packages must exist at `repo.radeon.com`) |
| `BUILD_TESTING` | `OFF` | `ON` builds the CTest tree shipped inside `minimap2-tests.tar.gz` (used by CI) |
| `GPUARCH` | `gfx942;gfx950` | Semicolon-separated list of AMD GPU archs (MI300X/MI325X + MI350X/MI355X) |

## Building the RPM Image (RHEL/UBI)

```bash
# RHEL UBI 9 — default
docker build -f packaging/Dockerfile.rpm \
  --build-arg RHEL_VERSION=9 \
  --build-arg ROCM_VERSION=7.2.1 \
  -t minimap2-rpm-test .

# RHEL UBI 8
docker build -f packaging/Dockerfile.rpm \
  --build-arg RHEL_VERSION=8 \
  --build-arg ROCM_VERSION=7.2.1 \
  -t minimap2-rpm-test-el8 .

# RHEL UBI 10
docker build -f packaging/Dockerfile.rpm \
  --build-arg RHEL_VERSION=10 \
  --build-arg ROCM_VERSION=7.2.1 \
  -t minimap2-rpm-test-el10 .
```

### RPM build arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `RHEL_VERSION` | `9` | RHEL major version (`8`, `9`, or `10`) |
| `ROCM_VERSION` | `7.2.1` | ROCm version |
| `GPUARCH` | `gfx942;gfx950` | Semicolon-separated list of AMD GPU archs (MI300X/MI325X + MI350X/MI355X) |

## Building the SLES RPM Image (SLES 15 / SUSE BCI)

```bash
# SLES-compatible 15.7 — default
docker build -f packaging/Dockerfile.sles \
  --build-arg SLES_VERSION=15.7 \
  --build-arg ROCM_VERSION=7.2.1 \
  -t minimap2-sles-test .
```

### SLES build arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `SLES_VERSION` | `15.7` | SLES-compatible SUSE 15.x base version |
| `ROCM_VERSION` | `7.2.1` | ROCm version (must have SLE packages at `repo.radeon.com`) |
| `GPUARCH` | `gfx942;gfx950` | Semicolon-separated list of AMD GPU archs (MI300X/MI325X + MI350X/MI355X) |

## Running the Image

### GPU mode

Requires `--device` flags for GPU access. `--group-add=44` (`video`) and
`--group-add=109` (`render`) grant the container process permission to open
`/dev/dri` and `/dev/kfd` respectively (see the
[Docker run flags reference](#docker-run-flags-reference) for details).

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=44 \
  --group-add=109 \
  -v /path/to/data:/data \
  minimap2-deb-test \
  minimap2 -ax map-ont \
    --gpu-chain --gpu-cfg /usr/share/minimap2/configs/gpu_config.json --gpu-align \
    /data/ref.fa /data/reads.fq > /data/aln.sam
```

### CPU-only mode (no GPU flags)

```bash
docker run --rm \
  -v /path/to/data:/data \
  minimap2-deb-test \
  minimap2 -ax map-ont /data/ref.fa /data/reads.fq > /data/aln.sam
```

### Interactive shell

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=44 \
  --group-add=109 \
  -v /path/to/data:/data \
  minimap2-deb-test \
  bash
```

### Using staged CLI tools

```bash
# Seeding
docker run --rm -v /path/to/data:/data minimap2-deb-test \
  mm2_seed -x map-ont /data/ref.fa /data/reads.fq -o /data/seeds.json

# Chaining
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add=44 --group-add=109 \
  -v /path/to/data:/data minimap2-deb-test \
  mm2_chain -x map-ont /data/ref.fa -s /data/seeds.json -o /data/chains.json

# Alignment
docker run --rm -v /path/to/data:/data minimap2-deb-test \
  mm2_align -x map-ont /data/ref.fa /data/reads.fq -c /data/chains.json -o /data/aln.sam
```

## Verifying the Image

> The DEB `runtime` stage already runs an assertive smoke test during build:
> it verifies `minimap2 --version`, that `mm2_seed` / `mm2_chain` / `mm2_align`
> respond to `-h`, and that the `.deb` payload contains the expected man pages,
> docs, shared library, and GPU config. A successful `docker build` therefore
> implies these checks passed — the commands below are for manual
> re-verification and for GPU checks that can only run on a real host.

```bash
# Check version
docker run --rm minimap2-deb-test minimap2 --version

# Verify GPU access
docker run --rm --device=/dev/kfd --device=/dev/dri \
  --group-add=44 --group-add=109 \
  minimap2-deb-test rocminfo | head -10

# List installed binaries
docker run --rm minimap2-deb-test which minimap2 mm2_seed mm2_chain mm2_align
```

## Extracting Packages from Docker

### DEB — BuildKit `artifacts` stage (recommended)

`Dockerfile.deb` exposes a dedicated `scratch`-based `artifacts` stage that
ships the `.deb`, `version.txt`, and `minimap2-tests.tar.gz`. This is how CI
extracts artifacts, and it works locally too — no intermediate container
needed:

```bash
DOCKER_BUILDKIT=1 docker build -f packaging/Dockerfile.deb \
  --target artifacts \
  --output type=local,dest=./out \
  --build-arg BASE_IMAGE=ubuntu:noble \
  --build-arg ROCM_VERSION=7.2.1 \
  .

ls ./out
# amd-minimap2_<ver>_amd64.deb  version.txt  minimap2-tests.tar.gz
```

`BUILD_TESTING` is optional (default `OFF`). Pass `--build-arg BUILD_TESTING=ON`
to populate the CTest tree inside `minimap2-tests.tar.gz` (this is what CI
does); with `OFF` the tarball still ships with the source tree for
reproducibility but omits the CTest metadata.

### DEB / RPM / SLES — classic `docker cp` extraction

The builder stages expose the generated packages at `/build/packages/`. To
copy them to the host without rebuilding:

```bash
# Build only the builder stage, tagged for extraction. All three Dockerfiles
# use the stage name `builder`.
docker build -f packaging/Dockerfile.deb  --target builder -t minimap2-deb-builder  .
docker build -f packaging/Dockerfile.rpm  --target builder -t minimap2-rpm-builder  .
docker build -f packaging/Dockerfile.sles --target builder -t minimap2-sles-builder .

# Create a temporary container and copy packages out
docker create --name tmp-pkg minimap2-deb-builder
docker cp tmp-pkg:/build/packages/ ./packages/
docker rm tmp-pkg
```

Replace `minimap2-deb-builder` with `minimap2-rpm-builder` or
`minimap2-sles-builder` as needed.

> **Note:** The runtime/test stage deletes the `.rpm`/`.deb` after install,
> so extracting from the **builder** stage (or, for DEB, the **artifacts**
> stage above) is the reliable method.

## Docker run flags reference

| Flag | Purpose |
|------|---------|
| `--device=/dev/kfd` | AMD GPU kernel driver access |
| `--device=/dev/dri` | GPU render node access |
| `--group-add=44` | Add `video` group (GID 44) |
| `--group-add=109` | Add `render` group (GID 109) |
| `-v /host/path:/container/path` | Bind-mount data directory |
| `--ipc=host` | Shared memory (recommended for large datasets) |

---

## Apptainer / Singularity

The minimap2 Docker image can be converted to an Apptainer (formerly Singularity) SIF container for use on HPC clusters and environments where Docker is unavailable.

### Prerequisites

- A built Docker image (`minimap2-deb-test:latest`) — see [Building the DEB Image](#building-the-deb-image-ubuntu)
- [Apptainer](https://apptainer.org/) installed on the host, **or** use the [`kaczmarj/apptainer`](https://github.com/kaczmarj/apptainer-in-docker) Docker image to build without a native install

### Building the SIF

#### With native Apptainer

```bash
apptainer build minimap2.sif packaging/minimap2.def
```

#### Without native Apptainer (via Docker)

```bash
docker run --rm --privileged \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD":/work \
  kaczmarj/apptainer:latest \
  build /work/minimap2.sif /work/packaging/minimap2.def
```

### Running the SIF

#### GPU mode

```bash
apptainer run --rocm \
  --bind /path/to/data:/data \
  minimap2.sif \
  minimap2 -ax map-ont \
    --gpu-chain --gpu-cfg /usr/share/minimap2/configs/gpu_config.json --gpu-align \
    /data/ref.fa /data/reads.fq > /data/aln.sam
```

#### CPU-only mode

```bash
apptainer run \
  --bind /path/to/data:/data \
  minimap2.sif \
  minimap2 -ax map-ont /data/ref.fa /data/reads.fq > /data/aln.sam
```

#### Interactive shell

```bash
apptainer shell --rocm --bind /path/to/data:/data minimap2.sif
```

#### Using staged CLI tools

```bash
apptainer exec --bind /path/to/data:/data minimap2.sif \
  mm2_seed -x map-ont /data/ref.fa /data/reads.fq -o /data/seeds.json

apptainer exec --rocm --bind /path/to/data:/data minimap2.sif \
  mm2_chain -x map-ont /data/ref.fa -s /data/seeds.json -o /data/chains.json

apptainer exec --bind /path/to/data:/data minimap2.sif \
  mm2_align -x map-ont /data/ref.fa /data/reads.fq -c /data/chains.json -o /data/aln.sam
```

### Verifying the SIF

```bash
apptainer exec minimap2.sif minimap2 --version
apptainer exec minimap2.sif which minimap2 mm2_seed mm2_chain mm2_align
```

### Apptainer flags reference

| Flag | Purpose |
|------|---------|
| `--rocm` | Expose AMD GPU devices inside the container |
| `--bind /host/path:/container/path` | Bind-mount a directory (Apptainer does not auto-bind arbitrary paths) |
