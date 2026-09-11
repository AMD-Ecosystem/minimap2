# Installing minimap2 from a Debian Package

This guide covers installing the minimap2 `.deb` package on Ubuntu 24.04 systems.

## Table of Contents

- [Prerequisites](#prerequisites)
- [CPU-only Package](#cpu-only-package)
- [AMD GPU Package](#amd-gpu-package)
  - [Option A: ROCm Docker Image (Recommended)](#option-a-rocm-docker-image-recommended)
  - [Option B: Bare Ubuntu 24.04](#option-b-bare-ubuntu-2404)
- [Verifying the Installation](#verifying-the-installation)
- [Uninstalling](#uninstalling)
- [Next Steps](#next-steps)

## Prerequisites

- Ubuntu 22.04 (Jammy) or 24.04 (Noble Numbat), amd64
- For the **AMD GPU** build:
  - ROCm 7.0.2 or 7.2 apt repository configured (see below)
  - Supported AMD GPU: **MI300X / MI325X** (gfx942) or **MI350X / MI355X** (gfx950)

---

## CPU-only Package

The CPU-only `.deb` has no ROCm dependency:

```bash
sudo apt install ./minimap2-*-Linux.deb
```

---

## AMD GPU Package

The AMD GPU `.deb` depends on `hip-runtime-amd`, which is provided by the
ROCm apt repository. Choose the installation path that matches your environment.

### Option A: ROCm Docker Image (Recommended)

If you are using a ROCm Docker image (e.g. `rocm/dev-ubuntu-24.04:7.2-complete`),
the ROCm packages are already installed. Install the `.deb` directly:

```bash
docker run -it --rm --entrypoint bash \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=44 --group-add=109 \
  --ipc=host --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=128G \
  -v ./:/workspace \
  rocm/dev-ubuntu-24.04:7.2-complete
```

| Flag | Purpose |
|------|---------|
| `--device=/dev/kfd` | AMD GPU kernel fusion driver — required for GPU compute access |
| `--device=/dev/dri` | GPU render nodes — required for GPU device access |
| `--group-add=44` | Add the `video` group (GID 44) — grants permission to access `/dev/dri` |
| `--group-add=109` | Add the `render` group (GID 109) — grants permission to access `/dev/kfd` |

Inside the container:

```bash
apt update
apt install -y /workspace/amd-minimap2-*-Linux.deb
```

### Option B: Bare Ubuntu 24.04

On a system without ROCm, you must first add the ROCm apt repository so that
`hip-runtime-amd` and its dependencies can be resolved.

#### 1. Install prerequisites

```bash
sudo apt update
sudo apt install -y wget lsb-release
```

#### 2. Add the ROCm repository

Download and install the `amdgpu-install` helper package, which configures
the ROCm apt sources. Set `ROCM_VERSION` to the ROCm 7.x release you want
to use, then run the script to compute the correct URL automatically:

```bash
# Set the desired ROCm version (any 7.x release)
ROCM_VERSION=7.2
```

```bash
UBUNTU_CODENAME=$(lsb_release -cs) && \
    echo "Detected ROCm version: ${ROCM_VERSION}, Ubuntu codename: ${UBUNTU_CODENAME}" && \
    MAJOR=$(echo ${ROCM_VERSION} | cut -d. -f1) && \
    MINOR=$(echo ${ROCM_VERSION} | cut -d. -f2) && \
    PATCH=$(echo ${ROCM_VERSION} | cut -d. -f3) && \
    PATCH=${PATCH:-0} && \
    VERNUM=$((MAJOR * 10000 + MINOR * 100 + PATCH)) && \
    if [ "${PATCH}" = "0" ]; then SHORT_VERSION="${MAJOR}.${MINOR}"; else SHORT_VERSION="${MAJOR}.${MINOR}.${PATCH}"; fi && \
    AMDGPU_URL="https://repo.radeon.com/amdgpu-install/${SHORT_VERSION}/ubuntu/${UBUNTU_CODENAME}/amdgpu-install_${SHORT_VERSION}.${VERNUM}-1_all.deb" && \
    echo "Downloading: ${AMDGPU_URL}" && \
    wget "${AMDGPU_URL}" -O amdgpu-install.deb && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y ./amdgpu-install.deb && \
    rm amdgpu-install.deb && \
    apt-get update
```

> **Tip:** If ROCm is already installed on the system, you can read the
> version automatically instead:  
> `ROCM_VERSION=$(cat /opt/rocm/.info/version)`

#### 3. Install minimap2

```bash
DEBIAN_FRONTEND=noninteractive sudo apt install -y ./amd-minimap2-*-Linux.deb
```

`apt` will automatically pull in the required ROCm packages
(`hip-runtime-amd`, `hsa-rocr`, `comgr`, `rocm-core`, etc.).

---

## Verifying the Installation

```bash
# Check the installed package
dpkg -s amd-minimap2

# Verify the binary runs
minimap2 --version

# List installed files
dpkg -L amd-minimap2

# View man pages (requires man-db)
man minimap2
man mm2_seed
man mm2_chain
man mm2_align
```

**Note:** The Debian package name is `amd-minimap2`, not `minimap2`. However, the binary, shared library, and headers still use the `minimap2` name (e.g. `/usr/bin/minimap2`, `/usr/lib/libminimap2.so`). Querying the wrong package name will show an error:

```
$ dpkg -s minimap2
dpkg-query: package 'minimap2' is not installed and no information is available
```

**Expected output:**

```
$ dpkg -s amd-minimap2
Package: amd-minimap2
Status: install ok installed
Priority: optional
Section: science
Installed-Size: 3274
Maintainer: AMD Corporation
Architecture: amd64
Version: 2.24.0
Depends: zlib1g (>= 1:1.2.11), hip-runtime-amd, libc6 (>= 2.38), libgcc-s1 (>= 3.0), libstdc++6 (>= 12), zlib1g (>= 1:1.1.4)
Description: A versatile pairwise aligner for genomic and spliced nucleotide sequences
 minimap2 is a fast sequence mapping and alignment program that can find overlaps
 between long noisy reads, or map long reads or their assemblies to a reference
 genome optionally with GPU acceleration.
Homepage: https://github.com/ROCm-LS/minimap2
```

```
$ minimap2 --version
2.24.0
```

```
$ dpkg -L amd-minimap2
/usr
/usr/bin
/usr/bin/minimap2
/usr/bin/mm2_align
/usr/bin/mm2_chain
/usr/bin/mm2_seed
/usr/include
/usr/include/minimap2
/usr/include/minimap2/minimap.h
/usr/lib
/usr/lib/cmake
/usr/lib/cmake/minimap2
/usr/lib/cmake/minimap2/minimap2Config.cmake
/usr/lib/cmake/minimap2/minimap2ConfigVersion.cmake
/usr/lib/cmake/minimap2/minimap2Targets-release.cmake
/usr/lib/cmake/minimap2/minimap2Targets.cmake
/usr/lib/libminimap2.so
/usr/share
/usr/share/doc
/usr/share/doc/minimap2
/usr/share/doc/minimap2/LICENSE.txt
/usr/share/doc/minimap2/NEWS.md
/usr/share/doc/minimap2/USAGE.md
/usr/share/man
/usr/share/man/man1
/usr/share/man/man1/minimap2.1.gz
/usr/share/man/man1/mm2_align.1.gz
/usr/share/man/man1/mm2_chain.1.gz
/usr/share/man/man1/mm2_seed.1.gz
/usr/share/minimap2
/usr/share/minimap2/configs
/usr/share/minimap2/configs/gpu_config.json
```

> **Note:** Minimal Docker images (e.g. `ubuntu:24.04`) exclude man pages
> via dpkg filters and replace `/usr/bin/man` with a stub. To enable man pages:
> ```bash
> # Remove the dpkg exclusion rule for man pages
> sed -i '/path-exclude=\/usr\/share\/man/d' /etc/dpkg/dpkg.cfg.d/excludes
> # Remove the man-page stub diversion so man-db installs the real binary
> rm -f /usr/bin/man
> dpkg-divert --quiet --remove --rename /usr/bin/man
> # Install man-db and reinstall minimap2 so the pages are unpacked
> apt install -y man-db
> apt install --reinstall -y ./amd-minimap2-*-Linux.deb
> man minimap2
> ```

## Uninstalling

```bash
sudo dpkg -r amd-minimap2
```

Or, to also remove dependencies that are no longer needed:

```bash
sudo apt remove amd-minimap2
sudo apt autoremove
```

---

## Next Steps

Once minimap2 is installed, see the following guides:

- [USAGE.md](/usr/share/doc/minimap2/USAGE.md) — Full usage documentation covering presets, GPU acceleration, command-line options, output formats, workflows, and troubleshooting.
- `man minimap2` — Man page with a complete option reference.
- `man mm2_seed`, `man mm2_chain`, `man mm2_align` — Man pages for the CLI pipeline tools.
