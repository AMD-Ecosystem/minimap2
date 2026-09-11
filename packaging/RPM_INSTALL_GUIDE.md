# Installing minimap2 from an RPM Package

This guide covers installing the minimap2 `.rpm` package on RPM-based Linux
distributions (RHEL, Rocky Linux, CentOS, Fedora, SLES 15).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Standard RPM (RHEL / Rocky / Fedora)](#standard-rpm-rhel--rocky--fedora)
  - [AMD GPU Package](#amd-gpu-package)
    - [Option A: ROCm Docker Image (Recommended)](#option-a-rocm-docker-image-recommended)
    - [Option B: Bare RHEL / Rocky 9](#option-b-bare-rhel--rocky-9)
- [SLES 15 RPM](#sles-15-rpm)
  - [AMD GPU Package (SLES 15)](#amd-gpu-package-sles-15)
- [Verifying the Installation](#verifying-the-installation)
- [Uninstalling](#uninstalling)
- [Next Steps](#next-steps)

## Prerequisites

- **Standard RPM:** RHEL 9, Rocky Linux 9, or Fedora 38+ (x86_64)
- **SLES RPM:** SLES 15 SP5+ (x86_64)
- For the **AMD GPU** build:
  - ROCm 7.0.2 or 7.2 repository configured
  - Supported AMD GPU: **MI300X / MI325X** (gfx942) or **MI350X / MI355X** (gfx950)

---

## Standard RPM (RHEL / Rocky / Fedora)

### AMD GPU Package

The AMD GPU `.rpm` depends on `hip-runtime-amd`, which is provided by the
ROCm repository.

#### Option A: ROCm Docker Image (Recommended)

```bash
docker run -it --rm --entrypoint bash \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=44 --group-add=109 \
  --ipc=host --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=128G \
  -v ./:/workspace \
  rocm/dev-centos-9:7.2-complete
```

| Flag | Purpose |
|------|---------|
| `--device=/dev/kfd` | AMD GPU kernel fusion driver — required for GPU compute access |
| `--device=/dev/dri` | GPU render nodes — required for GPU device access |
| `--group-add=44` | Add the `video` group (GID 44) — grants permission to access `/dev/dri` |
| `--group-add=109` | Add the `render` group (GID 109) — grants permission to access `/dev/kfd` |

Inside the container:

```bash
dnf install -y /workspace/amd-minimap2-*.rpm
```

#### Option B: Bare RHEL / Rocky 9

On a system without ROCm, add the ROCm repository first.

##### 1. Add the ROCm repository

```bash
ROCM_VERSION=7.2

cat > /etc/yum.repos.d/amdgpu.repo << EOF
[amdgpu]
name=amdgpu
baseurl=https://repo.radeon.com/amdgpu/${ROCM_VERSION}/rhel/9.7/main/x86_64/
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

cat > /etc/yum.repos.d/rocm.repo << EOF
[rocm]
name=rocm
baseurl=https://repo.radeon.com/rocm/rhel9/${ROCM_VERSION}/main
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

dnf clean all
```

##### 2. Install minimap2

```bash
sudo dnf install -y ./amd-minimap2-*.rpm
```

`dnf` will automatically pull in the required ROCm packages.

---

## SLES 15 RPM

SLES 15 uses `zypper` as its package manager and has different package
naming conventions (e.g. `libz1` instead of `zlib`).

### AMD GPU Package (SLES 15)

#### 1. Add the ROCm repository

```bash
ROCM_VERSION=7.0.2

zypper addrepo --no-gpgcheck \
  "https://repo.radeon.com/amdgpu/${ROCM_VERSION}/sle/15.6/main/x86_64/" \
  amdgpu

zypper addrepo --no-gpgcheck \
  "https://repo.radeon.com/rocm/zyp/${ROCM_VERSION}/main" \
  rocm

zypper refresh
```

#### 2. Install minimap2

```bash
sudo zypper install ./amd-minimap2-*.rpm
```

---

## Verifying the Installation

```bash
# Check the installed package
rpm -qi amd-minimap2      # AMD GPU build

# Verify the binary runs
minimap2 --version

# List installed files
rpm -ql amd-minimap2

# View man pages
man minimap2
man mm2_seed
man mm2_chain
man mm2_align
```

**Note:** The RPM package name is `amd-minimap2`, not `minimap2`. However, the binary, shared library, and headers still use the `minimap2` name (e.g. `/usr/bin/minimap2`, `/usr/lib64/libminimap2.so`).

**Expected output:**

```
$ rpm -qi amd-minimap2
Name        : amd-minimap2
Version     : 2.24.0
Release     : 0.1.alpha1
Architecture: x86_64
License     : MIT
Group       : Applications/Engineering
URL         : https://github.com/ROCm-LS/minimap2
Summary     : A versatile pairwise aligner for genomic and spliced nucleotide sequences
Description :
minimap2 is a fast sequence mapping and alignment program that can find
overlaps between long noisy reads, or map long reads or their assemblies
to a reference genome optionally with GPU acceleration.
```

```
$ minimap2 --version
2.24.0
```

```
$ rpm -ql amd-minimap2
/usr/bin/minimap2
/usr/bin/mm2_align
/usr/bin/mm2_chain
/usr/bin/mm2_seed
/usr/include/minimap2/minimap.h
/usr/lib/cmake/minimap2/minimap2Config.cmake
/usr/lib/cmake/minimap2/minimap2ConfigVersion.cmake
/usr/lib/cmake/minimap2/minimap2Targets-release.cmake
/usr/lib/cmake/minimap2/minimap2Targets.cmake
/usr/lib64/libminimap2.so
/usr/share/doc/minimap2/LICENSE.txt
/usr/share/doc/minimap2/NEWS.md
/usr/share/doc/minimap2/USAGE.md
/usr/share/man/man1/minimap2.1.gz
/usr/share/man/man1/mm2_align.1.gz
/usr/share/man/man1/mm2_chain.1.gz
/usr/share/man/man1/mm2_seed.1.gz
/usr/share/minimap2/configs/gpu_config.json
```

---

## Uninstalling

**Fedora / RHEL / Rocky:**

```bash
sudo dnf remove amd-minimap2
```

**SLES 15:**

```bash
sudo zypper remove amd-minimap2
```

**Direct RPM removal (any distro):**

```bash
sudo rpm -e amd-minimap2
```

---

## Next Steps

Once minimap2 is installed, see the following guides:

- [USAGE.md](/usr/share/doc/minimap2/USAGE.md) — Full usage documentation covering presets, GPU acceleration, command-line options, output formats, workflows, and troubleshooting.
- `man minimap2` — Man page with a complete option reference.
