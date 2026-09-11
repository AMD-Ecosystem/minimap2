# Versioning

## Overview

minimap2 uses a unified versioning system where **git tags are the single source
of truth**. Running `cmake --preset <preset>` derives the version from the
nearest `v*` tag and propagates it to every build artifact:

| Consumer | Format | Example |
|---|---|---|
| C/C++ (`-V` flag) | SemVer + build metadata | `2.25.0-alpha.1+g34cc1a3.dirty` |
| CMake `project(VERSION)` | MAJOR.MINOR.PATCH | `2.25.0` |
| Python `__version__` | PEP 440 | `2.25.0a1` |
| Python `__git_version__` | SemVer + build metadata | `2.25.0-alpha.1+g34cc1a3.dirty` |
| Debian `.deb` | Debian policy | `2.25.0~alpha1` |
| VERSION file | PEP 440 | `2.25.0a1` |

## How It Works

```
git tag v2.25.0-alpha.1
        │
        ▼
  cmake/version.cmake
  ┌─────────────────────────────────┐
  │ git describe --tags --match v*  │──(fallback)──▶ reads VERSION file
  │   ▼                             │
  │ Parse: MAJOR.MINOR.PATCH[-PRE]  │
  │ Extract: git hash, dirty flag   │
  │   ▼                             │
  │ Generate format-specific vars:  │
  │  • MM_VERSION_FULL   (SemVer)   │
  │  • MM_CMAKE_VERSION  (M.m.p)    │
  │  • MM_PYTHON_VERSION (PEP 440)  │
  │  • MM_DEB_VERSION    (Debian)   │
  │   ▼                             │
  │ Write VERSION file (for Python) │
  │ Set MM_VERSION_DEFINITIONS      │
  └─────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────┐
  │ mm2_version OBJECT library  │
  │ (mm_version.c compiled with │
  │  -D flags from above)       │
  └─────────────────────────────┘
        │
        ├──▶ libminimap2.so  ──▶  minimap2 -V
        ├──▶ mm2_seed        ──▶  mm2_seed -V
        ├──▶ mm2_chain       ──▶  mm2_chain -V
        ├──▶ mm2_align       ──▶  mm2_align -V
        └──▶ mappy.so        ──▶  mappy.__version__
```

### Version Source Priority

1. **Git tags** (primary) — `git describe --tags --match "v*" --long`
2. **VERSION file** (fallback) — used when `.git/` is absent (tarballs, sdists,
   shallow clones without tags)

The VERSION file is auto-generated during cmake configure and committed to the
repository so that `pip install .` (which reads VERSION before cmake runs) always
has a valid version.

## Creating a Release

### Stable Release

```bash
git tag v2.25.0
cmake --preset <preset>
# VERSION file updated to "2.25.0" (PEP 440)
```

### Pre-release

```bash
git tag v2.25.0-alpha.1    # first alpha
git tag v2.25.0-alpha.2    # second alpha
git tag v2.25.0-beta.1     # beta
git tag v2.25.0-rc.1       # release candidate
git tag v2.25.0             # final release
```

### Tag Format

```
v<MAJOR>.<MINOR>[.<PATCH>][-<PRERELEASE>]
```

- `PATCH` defaults to `0` if omitted (e.g. `v2.25` → `2.25.0`)
- `PRERELEASE` supports: `alpha.N`, `beta.N`, `rc.N`

### After Tagging

```bash
cmake --preset <preset>           # regenerates VERSION file from tag
git add VERSION && git commit     # commit the updated VERSION
```

## Version Formats

### SemVer (C/C++ binaries)

Used by `minimap2 -V`, `mm2_seed -V`, `mm2_chain -V`, `mm2_align -V`.

```
MAJOR.MINOR.PATCH[-PRERELEASE][+gHASH[.dirty]]
```

Examples:
- `2.25.0` — clean release
- `2.25.0+g34cc1a3` — release, 0 commits ahead
- `2.25.0-alpha.1+g34cc1a3.dirty` — alpha with uncommitted changes

Build metadata (after `+`) does **not** affect version precedence per SemVer spec.

### PEP 440 (Python)

Used by `mappy.__version__` and the wheel filename.

| Pre-release tag | PEP 440 |
|---|---|
| `alpha.1` | `a1` |
| `beta.2` | `b2` |
| `rc.1` | `rc1` |
| *(none)* | *(none)* |

Examples: `2.25.0`, `2.25.0a1`, `2.25.0b2`, `2.25.0rc1`

The full SemVer string (with git hash) is available as `mappy.__git_version__`.

### Debian

Used in `.deb` package metadata. The `~` character sorts before the release
version in Debian's version comparison:

```
2.25.0~alpha1 < 2.25.0~beta1 < 2.25.0~rc1 < 2.25.0
```

## C API

Version information is available at runtime via functions declared in `minimap.h`:

```c
#include "minimap.h"

// Simple version string
const char *mm_version(void);           // e.g. "2.25.0-alpha.1+g34cc1a3"

// Detailed version struct
const mm_version_t *mm_version_info(void);
```

The `mm_version_t` struct contains:

| Field | Type | Example |
|---|---|---|
| `major` | `int` | `2` |
| `minor` | `int` | `25` |
| `patch` | `int` | `0` |
| `full` | `const char *` | `"2.25.0-alpha.1+g34cc1a3"` |
| `prerelease` | `const char *` | `"alpha.1"` (or `""`) |
| `git_hash` | `const char *` | `"34cc1a3..."` (full hash) |
| `git_short` | `const char *` | `"34cc1a3"` |
| `git_dirty` | `int` | `1` if uncommitted changes |

## Python API

```python
import mappy

mappy.__version__        # "2.25.0"           (PEP 440, clean)
mappy.__git_version__    # "2.25.0+g34cc1a3"  (SemVer with git metadata)
mappy.version_info()     # dict with all fields
```

## Architecture

### mm2_version OBJECT library

`mm_version.c` is compiled as a CMake OBJECT library (`mm2_version`) with
version values passed as `-D` compile definitions. This avoids generating header
files and ensures version info is baked into the binary at compile time.

The OBJECT library pattern is necessary because CMake does not transitively
propagate object files through chains of static libraries. Each final link
target (shared lib, CLI executables, Python extension) explicitly includes
`$<TARGET_OBJECTS:mm2_version>`.

### File Inventory

| File | Role |
|---|---|
| `cmake/version.cmake` | Parses git tag, generates all version format strings |
| `minimap2/core/src/mm_version.c` | Implements `mm_version()` and `mm_version_info()` |
| `minimap2/include/minimap.h` | Declares `mm_version_t`, `mm_version()`, `mm_version_info()` |
| `minimap2/core/CMakeLists.txt` | Defines `mm2_version` OBJECT library |
| `VERSION` | Auto-generated fallback for non-git builds and Python packaging |
| `pyproject.toml` | Reads VERSION file via scikit-build-core regex provider |
| `python/mappy.pyx` | Exposes `version()`, `git_version()`, `version_info()` |
| `python/cmappy.pxd` | Cython declarations for C version functions |
