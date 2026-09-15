# Build Guide

This document describes how to build, install, and package minimap2 using CMake.

## Prerequisites

- CMake 3.18 or higher
- For AMD GPU: ROCm with `hipcc`
- Standard build tools (gcc/clang, make)
- zlib development libraries

## Quick Start

### Using CMake Presets (Recommended)

CMake Presets provide convenient pre-configured build options:

```bash
# List available presets
cmake --list-presets

# Configure with a preset (e.g., gfx942 for MI300X/MI300A)
cmake --preset gfx942

# Build
cmake --build --preset gfx942
```

### Manual Configuration

#### Basic build (AMD GPU, auto-detect architecture):
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

#### CPU-only build:
```bash
mkdir build && cd build
cmake -DGPU=NONE ..
cmake --build .
```

## Build Options

### GPU Configuration

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `GPU` | AMD, NONE | AMD | GPU backend (AMD for GPU, NONE for CPU-only) |
| `GPUARCH` | gfx942, gfx90a, etc. | auto-detected | GPU architecture (AMD only) |

Example:
```bash
cmake -DGPU=AMD -DGPUARCH=gfx942 ..
```

### Profiling Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `ENABLE_PROFILING` | ON/OFF | OFF | Enable profiler instrumentation (ITT/NVTX/ROCTX) |
| `ENABLE_TRACE` | ON/OFF | OFF | Enable trace logging with timing to console |

See [docs/PROFILING_GUIDE.md](docs/PROFILING_GUIDE.md) for detailed profiling documentation.

**Using Presets:**
```bash
# Build with profiling and trace logging
cmake --preset profile
cmake --build --preset profile
```

**Manual Configuration:**
```bash
# Enable profiling for VTune/rocprof
cmake -DENABLE_PROFILING=ON ..

# Enable trace logging (console output)
cmake -DENABLE_TRACE=ON ..

# Enable both
cmake -DENABLE_PROFILING=ON -DENABLE_TRACE=ON ..
```

### Debug Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `DEBUG_MODE` | NONE, info, analyze, verbose | NONE | Debug level |
| `ENABLE_ASAN` | ON/OFF | OFF | Enable AddressSanitizer |
| `ENABLE_TSAN` | ON/OFF | OFF | Enable ThreadSanitizer |

#### Debug Mode Details

The `DEBUG_MODE` option controls the level of debug output and checks:

| Mode | Defines | Description |
|------|---------|-------------|
| `NONE` | (none) | Production build, no debug output |
| `info` | `DEBUG_PRINT` | Enables debug print statements |
| `analyze` | `DEBUG_CHECK`, `DEBUG_PRINT` | Enables debug checks and print statements, plus GPU kernel analysis |
| `verbose` | `DEBUG_CHECK`, `DEBUG_PRINT`, `DEBUG_VERBOSE` | Full debug output with verbose logging and GPU test flags |

**Using Presets (Recommended):**
```bash
# Debug with print statements only
cmake --preset debug-info
cmake --build --preset debug-info

# Debug with analysis (checks + prints + GPU kernel analysis)
cmake --preset debug-analyze
cmake --build --preset debug-analyze

# Full verbose debug output
cmake --preset debug-verbose
cmake --build --preset debug-verbose
```

**Manual Configuration:**
```bash
cmake -DDEBUG_MODE=info ..      # Enable debug prints
cmake -DDEBUG_MODE=analyze ..   # Enable debug checks + prints + GPU analysis
cmake -DDEBUG_MODE=verbose ..   # Full verbose debug output
```

**Combining with Sanitizers:**
```bash
cmake -DDEBUG_MODE=verbose -DENABLE_ASAN=ON ..
```

#### Mapping Stage Debug Output

For detailed debug output of specific mapping stages (seed, chain, align), use these options:

| Option | Default | Description |
|--------|---------|-------------|
| `MM_DEBUG_SEED_VERBOSE` | OFF | Verbose output for `mm_map_seed` (seeding stage) |
| `MM_DEBUG_CHAIN_VERBOSE` | OFF | Verbose output for `mm_map_chain` (chaining stage) |
| `MM_DEBUG_ALIGN_VERBOSE` | OFF | Verbose output for `mm_map_align` (alignment stage) |

These options enable the stage-specific debug call sites that print detailed information about anchors, chains, and alignments during mapping. The debug sink remains the standard debug logger, which writes to stderr by default or to `--debug-log FILE` when that option is used.

**Example:**
```bash
# Enable verbose output for seeding stage
cmake -DMM_DEBUG_SEED_VERBOSE=ON ..

# Enable verbose output for chaining stage
cmake -DMM_DEBUG_CHAIN_VERBOSE=ON ..

# Enable verbose output for alignment stage
cmake -DMM_DEBUG_ALIGN_VERBOSE=ON ..

# Enable all mapping stage debug output
cmake -DMM_DEBUG_SEED_VERBOSE=ON -DMM_DEBUG_CHAIN_VERBOSE=ON -DMM_DEBUG_ALIGN_VERBOSE=ON ..
```

### Other Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `BUILD_EXTRA` | ON/OFF | OFF | Build sdust and minimap2-lite |
| `MAX_MICRO_BATCH` | integer | (empty) | Maximum micro batch size |
| `SUFFIX` | string | (empty) | Suffix for executable name |

### CLI Tools

Standalone CLI tools for debugging individual mapping stages are available in the `cli/` directory.

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `BUILD_CLI_TOOLS` | ON/OFF | ON | Build mm2_seed, mm2_chain, mm2_align |

These tools expose minimap2's internal pipeline stages (seeding → chaining → alignment) as separate executables, useful for debugging and testing.

```bash
# Build CLI tools
cmake --build . --target mm2_seed mm2_chain mm2_align
```

See [cli/README.md](cli/README.md) for full documentation on usage and JSON output formats.

## CMake Presets

CMakePresets.json provides pre-configured build configurations for common scenarios.

### Available Configure Presets

**Build Presets:**
- `cpu-only` - CPU-only build without GPU support
- `amd-default` - AMD GPU with auto-detected architecture
- `gfx942` - **gfx942 architecture** (MI300X, MI300A)

**Debug Presets:**
- `debug-info` - Build with debug print statements
- `debug-analyze` - Build with debug checks and kernel analysis
- `debug-verbose` - Build with full debug info and verbose output

**Sanitizer Presets:**
- `asan` - Build with AddressSanitizer
- `tsan` - Build with ThreadSanitizer

**Testing Presets:**
- `coverage` - Build with code coverage (GCC/gcov) and unit tests
- `coverage-clang` - Build with code coverage (Clang/llvm-cov) and unit tests
- `test-cpu` - CPU build with unit tests enabled

**Architecture Presets:**
- `sse2-only` - x86 with SSE2 only (no SSE4.1)

**Profiling Presets:**
- `profile` - Build with profiling (ITT/ROCTX) and trace logging enabled

**Other Presets:**
- `with-extras` - Build with sdust and minimap2-lite
- `release` - Optimized release build (AMD auto-detect)
- `release-gfx942` - Optimized release build for gfx942

**Packaging Presets:**
- `deb-package` - Build a Debian package (CPU-only)
- `deb-package-amd` - Build a Debian package with AMD GPU support
- `rpm-package` - Build an RPM package (CPU-only)
- `rpm-package-amd` - Build an RPM package with AMD GPU support
- `sles-rpm-package` - Build a SLES 15 RPM package (CPU-only)
- `sles-rpm-package-amd` - Build a SLES 15 RPM package with AMD GPU support

### Using Presets

```bash
# List all available presets
cmake --list-presets

# Configure with a preset
cmake --preset gfx942

# Build with a preset
cmake --build --preset gfx942

# One-liner: configure and build
cmake --preset gfx942 && cmake --build --preset gfx942
```

## Common Build Scenarios

### Using Presets (Recommended):

```bash
# CPU-only build (no GPU)
cmake --preset cpu-only
cmake --build --preset cpu-only

# AMD GPU with auto-detection
cmake --preset amd-default
cmake --build --preset amd-default

# gfx942 architecture (MI300X, MI300A)
cmake --preset gfx942
cmake --build --preset gfx942

# Debug with analysis
cmake --preset debug-analyze
cmake --build --preset debug-analyze

# Release build
cmake --preset release
cmake --build --preset release
```

### Manual Configuration:

```bash
# Development build with debug symbols
cmake -DDEBUG_MODE=verbose ..
cmake --build .

# Release build with all extras
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXTRA=ON ..
cmake --build .

# gfx942 architecture (MI300X, MI300A)
cmake -DGPU=AMD -DGPUARCH=gfx942 ..
cmake --build .

# CPU-only build
cmake -DGPU=NONE ..
cmake --build .

# x86 build with SSE2 only
cmake -DSSE2_ONLY=ON ..
cmake --build .

# Profile/analyze build
cmake -DDEBUG_MODE=analyze ..
cmake --build .
```

## Building a Debian Package (.deb)

minimap2 supports building Debian packages using CPack. The `.deb` can be
installed on Debian/Ubuntu systems with `dpkg -i`.

### Using Presets (Recommended)

```bash
# CPU-only .deb
cmake --preset deb-package
cmake --build --preset deb-package
cd out/deb-package/build && cpack -G DEB

# AMD GPU .deb (requires ROCm)
cmake --preset deb-package-amd
cmake --build --preset deb-package-amd
cd out/deb-package-amd/build && cpack -G DEB
```

The `.deb` file is written to `out/<preset>/build/packages/`.

### Manual Configuration

```bash
mkdir build-deb && cd build-deb
cmake -DCMAKE_BUILD_TYPE=Release -DGPU=NONE -DCMAKE_INSTALL_PREFIX=/usr ..
cmake --build . -j $(nproc)
cpack -G DEB          # creates the .deb in packages/
```

### Installing / Removing the Package

```bash
# Install
sudo dpkg -i packages/amd-minimap2-*.deb
sudo apt-get install -f   # resolve any missing dependencies

# Remove
sudo dpkg -r amd-minimap2
```

### Customizing the Package

The following CMake variables can be overridden at configure time:

| Variable | Default | Description |
|----------|---------|-------------|
| `BUILD_DEB_PACKAGE` | `OFF` | Enable/disable CPack DEB generation |
| `CPACK_PACKAGE_VERSION` | `${PROJECT_VERSION}` | Package version string |
| `CPACK_DEBIAN_PACKAGE_MAINTAINER` | `AMD Corporation` | Maintainer field |
| `CPACK_DEBIAN_PACKAGE_DEPENDS` | `zlib1g (>= 1:1.2.11)` | Runtime dependencies |

When building with `GPU=AMD`, the dependency `hip-runtime-amd` is
appended automatically.

## Building an RPM Package (.rpm)

minimap2 supports building RPM packages using CPack. The `.rpm` can be
installed on RHEL, CentOS, Fedora, Rocky Linux, and other RPM-based systems
with `rpm -i` or `dnf install`.

### Using Presets (Recommended)

```bash
# CPU-only .rpm
cmake --preset rpm-package
cmake --build --preset rpm-package
cd out/rpm-package/build && cpack -G RPM

# AMD GPU .rpm (requires ROCm)
cmake --preset rpm-package-amd
cmake --build --preset rpm-package-amd
cd out/rpm-package-amd/build && cpack -G RPM
```

The `.rpm` file is written to `out/<preset>/build/packages/`.

### Manual Configuration

```bash
mkdir build-rpm && cd build-rpm
cmake -DCMAKE_BUILD_TYPE=Release -DGPU=NONE -DBUILD_RPM_PACKAGE=ON -DCMAKE_INSTALL_PREFIX=/usr ..
cmake --build . -j $(nproc)
cpack -G RPM          # creates the .rpm in packages/
```

### Installing / Removing the Package

```bash
# Install (Fedora/RHEL 8+/Rocky)
sudo dnf install packages/amd-minimap2-*.rpm

# Install (RHEL 7/CentOS 7)
sudo yum install packages/amd-minimap2-*.rpm

# Install (direct)
sudo rpm -ivh packages/amd-minimap2-*.rpm

# Remove
sudo rpm -e amd-minimap2
```

### Customizing the Package

| Variable | Default | Description |
|----------|---------|-------------|
| `BUILD_RPM_PACKAGE` | `OFF` | Enable/disable CPack RPM generation |
| `CPACK_PACKAGE_VERSION` | `${PROJECT_VERSION}` | Package version string |
| `CPACK_RPM_PACKAGE_RELEASE` | `1` (or `0.1.alphaN`) | RPM release field |
| `CPACK_RPM_PACKAGE_REQUIRES` | `zlib >= 1.2.11` | Runtime dependencies |

When building with `GPU=AMD`, the dependency `hip-runtime-amd` is
appended automatically.

## Building a SLES 15 RPM Package (.rpm)

SLES 15 (SUSE Linux Enterprise Server) uses different package naming
conventions. A dedicated preset handles these differences.

### Using Presets (Recommended)

```bash
# CPU-only SLES 15 .rpm
cmake --preset sles-rpm-package
cmake --build --preset sles-rpm-package
cd out/sles-rpm-package/build && cpack -G RPM

# AMD GPU SLES 15 .rpm (requires ROCm)
cmake --preset sles-rpm-package-amd
cmake --build --preset sles-rpm-package-amd
cd out/sles-rpm-package-amd/build && cpack -G RPM
```

The `.rpm` file is written to `out/<preset>/build/packages/`.

### Manual Configuration

```bash
mkdir build-sles-rpm && cd build-sles-rpm
cmake -DCMAKE_BUILD_TYPE=Release -DGPU=NONE -DBUILD_SLES_RPM_PACKAGE=ON -DCMAKE_INSTALL_PREFIX=/usr ..
cmake --build . -j $(nproc)
cpack -G RPM          # creates the .rpm in packages/
```

### Installing / Removing the Package

```bash
# Install
sudo zypper install packages/minimap2-*.rpm

# Or direct
sudo rpm -ivh packages/minimap2-*.rpm

# Remove
sudo zypper remove minimap2
```

### Customizing the Package

| Variable | Default | Description |
|----------|---------|-------------|
| `BUILD_SLES_RPM_PACKAGE` | `OFF` | Enable/disable CPack SLES RPM generation |
| `CPACK_PACKAGE_VERSION` | `${PROJECT_VERSION}` | Package version string |
| `CPACK_RPM_PACKAGE_RELEASE` | `1` (or `0.1.alphaN`) | RPM release field |
| `CPACK_RPM_PACKAGE_REQUIRES` | `libz1 >= 1.2.11` | Runtime dependencies (SLES naming) |

When building with `GPU=AMD`, the dependency `hip-runtime-amd` is
appended automatically.

---

## Installing

### Using Presets (Recommended)

When using presets, build and install directories are preset-specific:

```bash
# Build and install with preset
cmake --preset gfx942
cmake --build --preset gfx942
cmake --install out/gfx942/build

# Installation will be in: out/gfx942/install/
```

### Manual Build

```bash
cmake --build . --target install
```

Or with custom prefix:
```bash
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
cmake --build .
cmake --install .
```

### Install Structure

```
out/${presetName}/install/
├── bin/                    # Executables
│   ├── minimap2
│   ├── minimap2-lite      # (if BUILD_EXTRA=ON)
│   └── sdust              # (if BUILD_EXTRA=ON)
├── lib/
│   ├── libminimap2.a      # Static library
│   └── cmake/minimap2/    # CMake package config
│       ├── minimap2Config.cmake
│       ├── minimap2ConfigVersion.cmake
│       └── minimap2Targets.cmake
├── include/
│   └── minimap2/
│       └── minimap.h      # Public header
└── share/
    ├── doc/minimap2/      # Documentation
    └── man/man1/          # Man pages
```

## Using minimap2 as a Library

minimap2 can be used as a library in your own CMake projects.

### With find_package (Recommended)

After installing minimap2, you can use `find_package` to find and link against it:

```cmake
find_package(minimap2 REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE minimap2::minimap2)
```

Make sure the install prefix is in your `CMAKE_PREFIX_PATH`:

```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/minimap2/install ..
```

### Exported Targets

The following targets are available:

| Target | Description |
|--------|-------------|
| `minimap2::minimap2` | Static library |
| `minimap2::minimap2_tool` | Main executable |
| `minimap2::minimap2-lite` | Lite executable (if BUILD_EXTRA=ON) |
| `minimap2::sdust` | sdust executable (if BUILD_EXTRA=ON) |

## Out-of-Source Builds (Recommended)

CMake supports out-of-source builds, keeping your source tree clean:

```bash
# Create multiple build configurations
mkdir build-debug && cd build-debug
cmake -DDEBUG_MODE=verbose ..
cmake --build .

cd ..
mkdir build-release && cd build-release
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

## Parallel Builds

Speed up compilation with parallel builds:
```bash
cmake --build . -j $(nproc)
```

## Cleaning

```bash
# Clean build artifacts
cmake --build . --target clean

# Or simply delete the build directory
cd .. && rm -rf build
```

## Comparing with Makefile

| Makefile Command | CMake Preset | CMake Manual |
|------------------|--------------|--------------|
| `make` | `cmake --preset amd-default && cmake --build --preset amd-default` | `cmake .. && cmake --build .` |
| `make DEBUG=info` | `cmake --preset debug-info && cmake --build --preset debug-info` | `cmake -DDEBUG_MODE=info ..` |
| `make DEBUG=analyze` | `cmake --preset debug-analyze && cmake --build --preset debug-analyze` | `cmake -DDEBUG_MODE=analyze ..` |
| `make DEBUG=verbose` | `cmake --preset debug-verbose && cmake --build --preset debug-verbose` | `cmake -DDEBUG_MODE=verbose ..` |
| `make sse2only=1` | `cmake --preset sse2-only && cmake --build --preset sse2-only` | `cmake -DSSE2_ONLY=ON ..` |
| `make asan=1` | `cmake --preset asan && cmake --build --preset asan` | `cmake -DENABLE_ASAN=ON ..` |
| `make tsan=1` | `cmake --preset tsan && cmake --build --preset tsan` | `cmake -DENABLE_TSAN=ON ..` |
| `make extra` | `cmake --preset with-extras && cmake --build --preset with-extras` | `cmake -DBUILD_EXTRA=ON ..` |
| (profiling) | `cmake --preset profile && cmake --build --preset profile` | `cmake -DENABLE_PROFILING=ON -DENABLE_TRACE=ON ..` |
| (deb package) | `cmake --preset deb-package && cmake --build --preset deb-package && cd out/deb-package/build && cpack -G DEB` | `cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . && cpack -G DEB` |
| (rpm package) | `cmake --preset rpm-package && cmake --build --preset rpm-package && cd out/rpm-package/build && cpack -G RPM` | `cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_RPM_PACKAGE=ON .. && cmake --build . && cpack -G RPM` |
| (sles rpm) | `cmake --preset sles-rpm-package && cmake --build --preset sles-rpm-package && cd out/sles-rpm-package/build && cpack -G RPM` | `cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SLES_RPM_PACKAGE=ON .. && cmake --build . && cpack -G RPM` |
| `make clean` | `rm -rf out/${presetName}` | `cmake --build . --target clean` |

## Troubleshooting

### GPU compiler not found
If CMake can't find `hipcc`, ensure it's in your PATH:
```bash
export PATH=/opt/rocm/bin:$PATH
cmake ..
```

### Architecture detection fails
Manually specify the architecture:
```bash
cmake -DGPUARCH=gfx942 ..  # for MI300X/MI300A
```

### ROCm libraries not found
Set ROCM_PATH:
```bash
export ROCM_PATH=/opt/rocm
cmake ..
```

## Notes

- The traditional Makefile is still maintained and fully supported
- CMake provides better cross-platform support and IDE integration
- Both build systems should produce identical binaries
- For CI/CD, CMake offers better integration with modern tools
