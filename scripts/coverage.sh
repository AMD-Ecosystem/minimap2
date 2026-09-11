#!/bin/bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Script to generate code coverage reports for minimap2
#
# Builds minimap2 with coverage instrumentation (GCC --coverage or Clang
# -fprofile-instr-generate), runs CTest, then produces coverage reports
# in HTML, XML (Cobertura), JSON, Markdown, and LCOV formats.
#
# Usage: ./scripts/coverage.sh [OPTIONS]
#
# Options:
#   --clean       Remove previous build before generating coverage
#   --skip-build  Skip build and test, only regenerate reports
#   --skip-pytest Do not run the Python (mappy) pytest suite for coverage
#   --format FMT  Output format: all, html, xml, json, markdown (default: all)
#   --compiler CC Use gcc or clang for coverage (default: clang)
#   --verbose     Show verbose output
#   -h, --help    Show this help message
#
# Prerequisites:
#   Clang toolchain (default): ROCm clang, llvm-cov/llvm-profdata (>= 19, to match
#                              the ROCm Clang profile format), lcov_cobertura
#   GCC toolchain:             gcc, gcov, gcovr >= 8.0 (pipx install gcovr)
#   Python coverage (optional): pytest-cov + coverage in the test interpreter
#                              (pip install pytest-cov) for pytest_coverage.json
#
# Output directory: out/coverage[-clang]/build/coverage/
#   html/index.html       - Interactive HTML with annotated source
#   coverage.xml          - Cobertura XML (CI/CD integration)
#   coverage.json         - Structured JSON (gcovr or llvm-cov native, C/C++)
#   coverage.md           - Markdown summary table
#   coverage.lcov         - LCOV tracefile (Clang only)
#   pytest_coverage.json  - Python (coverage.py) JSON for the mappy bindings
#                           (requires pytest-cov; mappy.pyx covered via Cython
#                            line tracing, MM_PYTHON_COVERAGE=ON)
#
# Key behaviors:
#   - Selects CMake preset coverage-gcc or coverage-clang automatically
#   - Clang (default): includes host-side GPU (.cu/.cpp) code; device kernels excluded
#   - GCC: CPU-only, excludes GPU files (gcov cannot instrument HIP/ROCm output)
#   - Clang uses llvm-cov/llvm-profdata only; GCC uses gcovr (>= 8.0 for markdown)
#   - Clang: merges .profraw files via llvm-profdata before reporting
#   - Falls back gracefully if optional report formats fail

set -e

# Default options
CLEAN=false
SKIP_BUILD=false
RUN_PYTEST=true
FORMAT="all"
COMPILER="clang"
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            cat << 'EOF'
Usage: coverage.sh [OPTIONS]

Generate code coverage reports for minimap2

OPTIONS
  --clean         Remove previous build before generating coverage
  --skip-build    Skip build and test, only regenerate reports
  --skip-pytest   Do not run the Python (mappy) pytest suite for coverage
  --format FMT    Output format: all, html, xml, json, markdown (default: all)
  --compiler CC   Compiler toolchain: gcc or clang (default: clang)
  --verbose       Show verbose output
  -h, --help      Show this help message

COMPILER TOOLCHAINS
  clang  llvm-cov/llvm-profdata toolchain (default; includes host-side GPU code)
  gcc    gcov/gcovr toolchain (CPU-only)

OUTPUT FILES (in out/coverage[-clang]/build/coverage/)
  html/index.html       Interactive HTML report with source annotations
  coverage.xml          Cobertura XML for CI/CD (Jenkins, GitLab, GitHub Actions)
  coverage.json         JSON data (gcovr format for GCC, llvm-cov native for Clang)
  coverage.md           Markdown summary table
  coverage.lcov         LCOV tracefile (Clang only, intermediate for XML)
  pytest_coverage.json  Python coverage.py JSON for the mappy bindings (needs pytest-cov)

REQUIREMENTS
  Clang (default): ROCm clang, llvm-cov/llvm-profdata (>= 19), lcov_cobertura (pipx install lcov_cobertura)
  GCC:             gcc, gcov, gcovr >= 8.0 (pipx install gcovr)

EXAMPLES
  coverage.sh                              # Clang: full build + all reports (CPU + GPU host code)
  coverage.sh --compiler gcc               # GCC: CPU-only full build + all reports
  coverage.sh --clean                      # Clean rebuild + reports
  coverage.sh --skip-build                 # Regenerate reports only
  coverage.sh --format html                # HTML report only
  coverage.sh --format xml --compiler gcc  # Cobertura XML with GCC

NOTE: Clang (default) includes host-side GPU code; device kernels are not
      instrumented. GCC is CPU-only (gcov cannot instrument HIP/ROCm output).
EOF
            exit 0
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-pytest)
            RUN_PYTEST=false
            shift
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --compiler)
            COMPILER="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate format option
case "$FORMAT" in
    all|html|xml|json|markdown) ;;
    *)
        echo "ERROR: Invalid format '$FORMAT'. Must be: all, html, xml, json, or markdown"
        exit 1
        ;;
esac

# Validate compiler option
case "$COMPILER" in
    gcc|clang) ;;
    *)
        echo "ERROR: Invalid compiler '$COMPILER'. Must be: gcc or clang"
        exit 1
        ;;
esac

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# The Python (mappy) coverage path relies on the active environment: python3
# (and cython, pytest) must already be on PATH, e.g. an activated virtualenv or
# the devcontainer's /opt/venv. No venv is auto-selected here.

# Set preset based on compiler
if [[ "$COMPILER" == "clang" ]]; then
    PRESET="coverage-clang"
else
    PRESET="coverage"
fi
BUILD_DIR="out/${PRESET}/build"

echo "================================================================================"
echo "                     minimap2 Code Coverage Report Generator"
echo "================================================================================"
echo ""
echo "Compiler: ${COMPILER^^}"
echo ""

# Check prerequisites
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "ERROR: $1 is not installed"
        echo "$2"
        exit 1
    fi
}

check_command cmake "Install with: sudo apt-get install cmake"
check_command ctest "Install with: sudo apt-get install cmake"

# Check Clang-specific tools
if [[ "$COMPILER" == "clang" ]]; then
    check_command clang "Install with: sudo apt-get install clang"

    # The coverage-clang preset builds with the ROCm Clang toolchain (so the
    # host-side pass of GPU/HIP sources is instrumented in the same profile
    # format as the rest of the code). ROCm Clang emits a newer raw-profile
    # version than the system llvm-18 tools can read, so prefer the newest
    # available versioned llvm-profdata/llvm-cov.
    pick_llvm_tool() {
        local base="$1" c
        for c in "${base}-22" "${base}-21" "${base}-20" "${base}-19" "$base"; do
            if command -v "$c" &> /dev/null; then echo "$c"; return 0; fi
        done
        return 1
    }
    LLVM_PROFDATA="$(pick_llvm_tool llvm-profdata)"
    LLVM_COV="$(pick_llvm_tool llvm-cov)"
    if [[ -z "$LLVM_PROFDATA" || -z "$LLVM_COV" ]]; then
        echo "ERROR: llvm-profdata/llvm-cov not found"
        echo "Install a recent LLVM (e.g. sudo apt-get install llvm-20) to match the ROCm Clang profile format"
        exit 1
    fi
    echo "Using LLVM coverage tools: ${LLVM_PROFDATA}, ${LLVM_COV}"

    # Check for lcov_cobertura (for XML output)
    if [[ -x "$HOME/.local/bin/lcov_cobertura" ]]; then
        LCOV_COBERTURA="$HOME/.local/bin/lcov_cobertura"
    elif command -v lcov_cobertura &> /dev/null; then
        LCOV_COBERTURA="lcov_cobertura"
    else
        LCOV_COBERTURA=""
        if [[ "$FORMAT" == "xml" ]]; then
            echo "ERROR: lcov_cobertura is not installed (required for XML output with Clang)"
            echo "Install with: pipx install lcov_cobertura"
            exit 1
        fi
    fi
fi

# Check GCC-specific tools (gcovr) — only needed for the GCC coverage path.
# Clang mode uses llvm-cov/llvm-profdata and generates Markdown directly, so
# gcovr is not required.
if [[ "$COMPILER" == "gcc" ]]; then
    # Check if gcovr is installed - prefer pipx version (8.4+) for markdown support
    if [[ -x "$HOME/.local/bin/gcovr" ]]; then
        GCOVR="$HOME/.local/bin/gcovr"
    elif command -v gcovr &> /dev/null; then
        GCOVR="gcovr"
    else
        echo "ERROR: gcovr is not installed"
        echo "Install with: pipx install gcovr (recommended) or sudo apt-get install gcovr"
        exit 1
    fi

    # Check gcovr version for markdown support
    GCOVR_VERSION=$($GCOVR --version | head -1 | sed -n 's/.*gcovr \([0-9]*\.[0-9]*\).*/\1/p')
    GCOVR_MAJOR=$(echo "$GCOVR_VERSION" | cut -d. -f1)
    if [[ "$GCOVR_MAJOR" -lt 8 ]] && [[ "$FORMAT" == "all" || "$FORMAT" == "markdown" ]]; then
        echo "WARNING: gcovr $GCOVR_VERSION detected. Markdown output requires gcovr 8.0+"
        echo "         Install latest with: pipx install gcovr"
        if [[ "$FORMAT" == "markdown" ]]; then
            exit 1
        fi
        FORMAT="html"
        echo "         Falling back to HTML only"
        echo ""
    fi

    echo "Using: $($GCOVR --version | head -1)"
    echo ""
fi

# Clean previous build if requested
if [[ "$CLEAN" == "true" ]]; then
    echo "==> Cleaning previous build..."
    rm -rf "out/${PRESET}"
    echo ""
fi

# Build and test unless skipped
if [[ "$SKIP_BUILD" == "false" ]]; then
    echo "==> Building minimap2 with coverage instrumentation..."
    START_TIME=$(date +%s)
    # When the Python (mappy) suite is part of coverage, build the bindings with
    # the same instrumentation so the C/Cython glue is counted. MM_PYTHON_COVERAGE
    # additionally enables Cython line tracing so coverage.py / pytest-cov can
    # report line-level coverage for mappy.pyx.
    PY_CMAKE_ARGS=()
    if [[ "$RUN_PYTEST" == "true" ]]; then
        PY_CMAKE_ARGS+=(-DBUILD_PYTHON=ON -DMM_PYTHON_COVERAGE=ON)
    fi
    cmake --preset ${PRESET} "${PY_CMAKE_ARGS[@]}"
    cmake --build --preset ${PRESET} -j$(nproc)
    BUILD_TIME=$(($(date +%s) - START_TIME))
    echo "    Build completed in ${BUILD_TIME}s"

    echo ""
    echo "==> Running tests..."
    START_TIME=$(date +%s)
    
    # For Clang, set LLVM_PROFILE_FILE to collect profile data
    if [[ "$COMPILER" == "clang" ]]; then
        mkdir -p "${BUILD_DIR}/coverage"
        # Run ctest with LLVM_PROFILE_FILE set - the %p gets replaced with the process ID
        LLVM_PROFILE_FILE="${PROJECT_ROOT}/${BUILD_DIR}/coverage/default_%p.profraw" \
            ctest --test-dir ${BUILD_DIR} --output-on-failure
    else
        ctest --test-dir ${BUILD_DIR} --output-on-failure
    fi
    
    TEST_TIME=$(($(date +%s) - START_TIME))
    echo "    Tests completed in ${TEST_TIME}s"

    # Run the Python (mappy) test suite against the freshly built, instrumented
    # module so the C/Cython binding code (cmappy, mm_map_aux_ctx,
    # mm_map_batch_ctx, ...) is exercised under coverage. For Clang the profile
    # lands in the same coverage dir as the ctest run and is picked up by the
    # merge below; for GCC the .gcda files are written into the build tree.
    if [[ "$RUN_PYTEST" == "true" ]]; then
        # Use the Python already on PATH (an activated venv or /opt/venv); the
        # build's cython/pytest come from the same environment.
        PYTHON_BIN="$(command -v python3 || true)"
        MAPPY_SO_PATH="$(find "${BUILD_DIR}/python" -name 'mappy*.so' 2>/dev/null | head -1)"
        if [[ -z "$PYTHON_BIN" ]]; then
            echo "    WARNING: no Python interpreter found; skipping pytest coverage"
        elif ! "$PYTHON_BIN" -m pytest --version &> /dev/null; then
            echo "    WARNING: pytest not available in ${PYTHON_BIN}; skipping pytest coverage"
            echo "             install with: ${PYTHON_BIN} -m pip install pytest"
        elif [[ -z "$MAPPY_SO_PATH" ]]; then
            echo "    WARNING: built mappy module not found under ${BUILD_DIR}/python; skipping pytest coverage"
        else
            echo ""
            echo "==> Running Python (mappy) tests for coverage..."
            START_PYTEST=$(date +%s)

            # Optionally produce a Python-side coverage JSON (coverage.py) for the
            # mappy bindings. The module is built with Cython line tracing
            # (MM_PYTHON_COVERAGE=ON), and the Cython coverage plugin maps the
            # compiled module back to mappy.pyx source lines. The plugin locates
            # the generated C file next to the .pyx and resolves the embedded
            # (relative) "mappy.pyx" filename against the current directory, so we
            # stage mappy.c beside python/mappy.pyx and run pytest from python/.
            PYTEST_COV_ARGS=()
            PYTEST_COV_JSON="${PROJECT_ROOT}/${BUILD_DIR}/coverage/pytest_coverage.json"
            STAGED_MAPPY_C=""
            GENERATED_MAPPY_C="$(find "${BUILD_DIR}/python" -name 'mappy.c' 2>/dev/null | head -1)"
            if "$PYTHON_BIN" -c 'import pytest_cov, coverage, Cython.Coverage' &> /dev/null \
               && [[ -n "$GENERATED_MAPPY_C" ]]; then
                COVERAGERC="${PROJECT_ROOT}/${BUILD_DIR}/coverage/.coveragerc"
                cat > "$COVERAGERC" << 'EOF'
[run]
plugins = Cython.Coverage

[json]
pretty_print = True
EOF
                # Stage the Cython-generated C next to the .pyx so the plugin can
                # build the line map. Don't clobber a pre-existing file.
                if [[ ! -e "${PROJECT_ROOT}/python/mappy.c" ]]; then
                    cp "$GENERATED_MAPPY_C" "${PROJECT_ROOT}/python/mappy.c"
                    STAGED_MAPPY_C="${PROJECT_ROOT}/python/mappy.c"
                fi
                PYTEST_COV_ARGS=(
                    --cov=mappy
                    --cov-config="${COVERAGERC}"
                    "--cov-report=json:${PYTEST_COV_JSON}"
                    --cov-report=term-missing
                )
            elif ! "$PYTHON_BIN" -c 'import pytest_cov, coverage, Cython.Coverage' &> /dev/null; then
                echo "    NOTE: pytest-cov/coverage/Cython.Coverage not available; skipping Python coverage JSON"
                echo "          install with: ${PYTHON_BIN} -m pip install pytest-cov coverage"
            fi

            set +e
            # Run from python/ so the Cython plugin resolves the relative
            # "mappy.pyx" co_filename and finds the staged mappy.c beside it. The
            # test suite uses __file__-relative paths, so cwd does not matter to it.
            if [[ "$COMPILER" == "clang" ]]; then
                ( cd "${PROJECT_ROOT}/python" && \
                  MAPPY_SO="${PROJECT_ROOT}/${MAPPY_SO_PATH}" \
                  LLVM_PROFILE_FILE="${PROJECT_ROOT}/${BUILD_DIR}/coverage/pytest_%p.profraw" \
                      "$PYTHON_BIN" -m pytest tests -v "${PYTEST_COV_ARGS[@]}" )
            else
                ( cd "${PROJECT_ROOT}/python" && \
                  MAPPY_SO="${PROJECT_ROOT}/${MAPPY_SO_PATH}" \
                      "$PYTHON_BIN" -m pytest tests -v "${PYTEST_COV_ARGS[@]}" )
            fi
            PYTEST_RC=$?
            set -e
            # Remove the staged C file (leave a pre-existing one untouched).
            [[ -n "$STAGED_MAPPY_C" ]] && rm -f "$STAGED_MAPPY_C"
            if [[ $PYTEST_RC -ne 0 ]]; then
                echo "    WARNING: pytest exited with code ${PYTEST_RC} (continuing with coverage report)"
            fi
            if [[ -f "$PYTEST_COV_JSON" ]]; then
                echo "    Python coverage JSON: ${BUILD_DIR}/coverage/pytest_coverage.json"
            fi
            echo "    Python tests completed in $(($(date +%s) - START_PYTEST))s"
        fi
    fi

    # For Clang, merge profile data
    if [[ "$COMPILER" == "clang" ]]; then
        echo ""
        echo "==> Merging profile data..."
        PROFRAW_FILES=$(find "${BUILD_DIR}/coverage" -name "*.profraw" 2>/dev/null)
        if [[ -z "$PROFRAW_FILES" ]]; then
            echo "ERROR: No .profraw files found. Coverage instrumentation may not be working."
            echo "       Check that the build used Clang with coverage flags."
            exit 1
        fi
        echo "    Found profile files: $(echo "$PROFRAW_FILES" | wc -l) files"
        ${LLVM_PROFDATA} merge -sparse ${BUILD_DIR}/coverage/*.profraw -o ${BUILD_DIR}/coverage/coverage.profdata
    fi
else
    echo "==> Skipping build and tests (--skip-build)"
    if [[ ! -d "${BUILD_DIR}" ]]; then
        echo "ERROR: Build directory does not exist. Run without --skip-build first."
        exit 1
    fi
    if [[ "$COMPILER" == "clang" && ! -f "${BUILD_DIR}/coverage/coverage.profdata" ]]; then
        echo "ERROR: Profile data not found. Run without --skip-build first."
        exit 1
    fi
fi

echo ""
echo "==> Generating coverage reports..."
START_TIME=$(date +%s)
cd ${BUILD_DIR}

# Coverage output directory
COVERAGE_DIR="coverage"
mkdir -p "${COVERAGE_DIR}/html"

if [[ "$COMPILER" == "clang" ]]; then
    # =========================================================================
    # LLVM/Clang coverage using llvm-cov (native LLVM instrumentation)
    # =========================================================================
    
    PROFDATA="${COVERAGE_DIR}/coverage.profdata"
    TEST_BINARY="tests/core/test_core"

    IGNORE_REGEX='tests/|external/|googletest|googlemock|_deps|python/mappy\.c'

    # llvm-cov needs every instrumented binary/shared library passed as -object
    # (unlike gcovr, which reads .gcda files from the build tree). Most of the
    # mapping code lives in libminimap2.so, so it must be included or the report
    # collapses to just the few statically-linked translation units.
    BINARIES=(-object "${TEST_BINARY}")
    # GPU test binary (built with GPU=AMD) exercises the host-side pass of the
    # instrumented HIP sources.
    if [[ -f "tests/gpu/test_gpu" ]]; then
        BINARIES+=(-object "tests/gpu/test_gpu")
    fi
    # Shared library holding the bulk of the instrumented pipeline code.
    for so in $(find . -name 'libminimap2.so' 2>/dev/null); do
        BINARIES+=(-object "$so")
    done
    # Python mappy module (instrumented when BUILD_PYTHON=ON); the pytest suite
    # exercises the C/Cython binding code through it.
    for so in $(find . -path '*/python/mappy*.so' 2>/dev/null); do
        BINARIES+=(-object "$so")
    done
    # Main executable (lives under bin/ with CMake presets).
    for exe in "bin/minimap2" "minimap2"; do
        if [[ -f "$exe" ]]; then
            BINARIES+=(-object "$exe")
            break
        fi
    done
    
    # Helper function to generate Markdown from llvm-cov report
    generate_markdown_from_llvm_cov() {
        local report_file="$1"
        local output_file="$2"
        
        # Get totals line
        local totals=$(tail -1 "$report_file")
        local regions=$(echo "$totals" | awk '{print $2}')
        local missed_regions=$(echo "$totals" | awk '{print $3}')
        local region_pct=$(echo "$totals" | awk '{print $4}')
        local functions=$(echo "$totals" | awk '{print $5}')
        local missed_funcs=$(echo "$totals" | awk '{print $6}')
        local func_pct=$(echo "$totals" | awk '{print $7}')
        local lines=$(echo "$totals" | awk '{print $8}')
        local missed_lines=$(echo "$totals" | awk '{print $9}')
        local line_pct=$(echo "$totals" | awk '{print $10}')
        local branches=$(echo "$totals" | awk '{print $11}')
        local missed_branches=$(echo "$totals" | awk '{print $12}')
        local branch_pct=$(echo "$totals" | awk '{print $13}')
        
        cat > "$output_file" << EOF
# minimap2 Code Coverage

Generated with Clang/LLVM coverage (llvm-cov)

## 📂 Overall Coverage

| Metric        | Covered/Total | Coverage |
|---------------|---------------|----------|
| **Lines**     | $((lines - missed_lines))/$lines | $line_pct |
| **Functions** | $((functions - missed_funcs))/$functions | $func_pct |
| **Branches**  | $((branches - missed_branches))/$branches | $branch_pct |
| **Regions**   | $((regions - missed_regions))/$regions | $region_pct |

## 📁 File Coverage

| File | Line Coverage | Function Coverage | Branch Coverage |
|------|---------------|-------------------|-----------------|
EOF
        # Add per-file data (skip header, separator lines, and totals)
        # Filter out lines starting with dash, blank lines, and non-file lines
        tail -n +3 "$report_file" | head -n -1 | while IFS= read -r line; do
            # Skip separator lines and special lines
            [[ "$line" =~ ^[[:space:]]*[-]+ ]] && continue
            [[ "$line" =~ ^[[:space:]]*$ ]] && continue
            [[ "$line" =~ ^Files ]] && continue
            [[ "$line" =~ ^TOTAL ]] && continue
            
            local filepath=$(echo "$line" | awk '{print $1}')
            # Skip if not a valid file path
            [[ -z "$filepath" ]] && continue
            [[ "$filepath" =~ ^[[:space:]]*$ ]] && continue
            
            local file=$(basename "$filepath" 2>/dev/null || echo "$filepath")
            local file_line_pct=$(echo "$line" | awk '{print $10}')
            local file_func_pct=$(echo "$line" | awk '{print $7}')
            local file_branch_pct=$(echo "$line" | awk '{print $13}')
            
            # Only add row if we have valid data
            if [[ -n "$file" && -n "$file_line_pct" ]]; then
                echo "| $file | $file_line_pct | $file_func_pct | $file_branch_pct |" >> "$output_file"
            fi
        done
        
        cat >> "$output_file" << EOF

---
*Coverage data generated by llvm-cov. Includes host-side GPU/HIP code; device kernel code is not instrumented.*
EOF
    }
    
    # Generate reports based on format selection
    case "$FORMAT" in
        all)
            echo "    Generating HTML, XML, JSON, and Markdown reports..."
            
            # HTML via llvm-cov
            ${LLVM_COV} show "${BINARIES[@]}" -instr-profile="${PROFDATA}" \
                -format=html -output-dir="${COVERAGE_DIR}/html" \
                -ignore-filename-regex="${IGNORE_REGEX}" \
                ${VERBOSE:+--show-line-counts-or-regions}
            
            # JSON via llvm-cov (native llvm-cov format)
            ${LLVM_COV} export "${BINARIES[@]}" -instr-profile="${PROFDATA}" \
                -ignore-filename-regex="${IGNORE_REGEX}" \
                > "${COVERAGE_DIR}/coverage.json"
            
            # XML (Cobertura) via llvm-cov lcov export + lcov_cobertura
            if [[ -n "$LCOV_COBERTURA" ]]; then
                ${LLVM_COV} export "${BINARIES[@]}" -instr-profile="${PROFDATA}" \
                    -format=lcov \
                    -ignore-filename-regex="${IGNORE_REGEX}" \
                    > "${COVERAGE_DIR}/coverage.lcov"
                $LCOV_COBERTURA "${COVERAGE_DIR}/coverage.lcov" -o "${COVERAGE_DIR}/coverage.xml"
            fi
            
            # Generate text report for parsing
            ${LLVM_COV} report "${BINARIES[@]}" -instr-profile="${PROFDATA}" \
                -ignore-filename-regex="${IGNORE_REGEX}" \
                > "${COVERAGE_DIR}/coverage_report.txt"
            
            # Generate Markdown from llvm-cov report
            generate_markdown_from_llvm_cov "${COVERAGE_DIR}/coverage_report.txt" "${COVERAGE_DIR}/coverage.md"
            
            # Print summary
            echo ""
            cat "${COVERAGE_DIR}/coverage_report.txt"
            ;;
        html)
            echo "    Generating HTML report..."
            ${LLVM_COV} show "${BINARIES[@]}" -instr-profile="${PROFDATA}" \
                -format=html -output-dir="${COVERAGE_DIR}/html" \
                -ignore-filename-regex="${IGNORE_REGEX}"
            ${LLVM_COV} report "${BINARIES[@]}" -instr-profile="${PROFDATA}" \
                -ignore-filename-regex="${IGNORE_REGEX}"
            ;;
        xml)
            echo "    Generating XML (Cobertura) report..."
            ${LLVM_COV} export "${BINARIES[@]}" -instr-profile="${PROFDATA}" \
                -format=lcov \
                -ignore-filename-regex="${IGNORE_REGEX}" \
                > "${COVERAGE_DIR}/coverage.lcov"
            $LCOV_COBERTURA "${COVERAGE_DIR}/coverage.lcov" -o "${COVERAGE_DIR}/coverage.xml"
            ${LLVM_COV} report "${BINARIES[@]}" -instr-profile="${PROFDATA}" \
                -ignore-filename-regex="${IGNORE_REGEX}"
            ;;
        json)
            echo "    Generating JSON report (llvm-cov format)..."
            ${LLVM_COV} export "${BINARIES[@]}" -instr-profile="${PROFDATA}" \
                -ignore-filename-regex="${IGNORE_REGEX}" \
                > "${COVERAGE_DIR}/coverage.json"
            ${LLVM_COV} report "${BINARIES[@]}" -instr-profile="${PROFDATA}" \
                -ignore-filename-regex="${IGNORE_REGEX}"
            ;;
        markdown)
            echo "    Generating Markdown report..."
            ${LLVM_COV} report "${BINARIES[@]}" -instr-profile="${PROFDATA}" \
                -ignore-filename-regex="${IGNORE_REGEX}" \
                > "${COVERAGE_DIR}/coverage_report.txt"
            generate_markdown_from_llvm_cov "${COVERAGE_DIR}/coverage_report.txt" "${COVERAGE_DIR}/coverage.md"
            cat "${COVERAGE_DIR}/coverage_report.txt"
            ;;
    esac
else
    # =========================================================================
    # GCC coverage using gcovr
    # =========================================================================
    
    # Common gcovr options for filtering
    # Note: GPU .cu files compiled with HIP/ROCm don't generate gcov data
    GCOVR_OPTS=(
        --root "${PROJECT_ROOT}"
        --filter "${PROJECT_ROOT}/.*\\.[ch]$"
        --filter "${PROJECT_ROOT}/gpu/.*\\.h$"
        --exclude "${PROJECT_ROOT}/tests/.*"
        --exclude "${PROJECT_ROOT}/external/.*"
        --exclude ".*/googletest.*"
        --exclude ".*/googlemock.*"
        --exclude ".*/_deps/.*"
        --exclude ".*/python/mappy\\.c"
        --merge-mode-functions=merge-use-line-0
        --gcov-ignore-parse-errors=negative_hits.warn
    )

    # Add verbose flag if requested
    if [[ "$VERBOSE" == "true" ]]; then
        GCOVR_OPTS+=(--verbose)
    fi

    # Generate reports based on format selection
    # Note: gcovr can generate multiple outputs in a single run for efficiency
    case "$FORMAT" in
        all)
            echo "    Generating HTML, XML, JSON, and Markdown reports..."
            $GCOVR "${GCOVR_OPTS[@]}" \
                --html-details ${COVERAGE_DIR}/html/index.html \
                --html-title "minimap2 Code Coverage" \
                --cobertura-pretty --output ${COVERAGE_DIR}/coverage.xml \
                --json-pretty --json ${COVERAGE_DIR}/coverage.json \
                --markdown ${COVERAGE_DIR}/coverage.md \
                --markdown-title "minimap2 Code Coverage" \
                --print-summary
            ;;
        html)
            echo "    Generating HTML report..."
            $GCOVR "${GCOVR_OPTS[@]}" \
                --html-details ${COVERAGE_DIR}/html/index.html \
                --html-title "minimap2 Code Coverage" \
                --print-summary
            ;;
        xml)
            echo "    Generating XML (Cobertura) report..."
            $GCOVR "${GCOVR_OPTS[@]}" \
                --cobertura-pretty --output ${COVERAGE_DIR}/coverage.xml \
                --print-summary
            ;;
        json)
            echo "    Generating JSON report (gcovr format)..."
            $GCOVR "${GCOVR_OPTS[@]}" \
                --json-pretty --json ${COVERAGE_DIR}/coverage.json \
                --print-summary
            ;;
        markdown)
            echo "    Generating Markdown report..."
            $GCOVR "${GCOVR_OPTS[@]}" \
                --markdown ${COVERAGE_DIR}/coverage.md \
                --markdown-title "minimap2 Code Coverage" \
                --print-summary
            ;;
    esac
fi

REPORT_TIME=$(($(date +%s) - START_TIME))
echo "    Reports generated in ${REPORT_TIME}s"

cd "${PROJECT_ROOT}"

echo ""
echo "================================================================================"
echo "                        Coverage Report Generated"
echo "================================================================================"
echo ""

# Show generated reports
COVERAGE_PATH="${BUILD_DIR}/coverage"
if [[ "$FORMAT" == "all" || "$FORMAT" == "html" ]]; then
    echo "HTML report:     ${COVERAGE_PATH}/html/index.html"
fi
if [[ "$FORMAT" == "all" || "$FORMAT" == "xml" ]]; then
    if [[ "$COMPILER" == "clang" && -z "$LCOV_COBERTURA" ]]; then
        echo "XML report:      (not available - install lcov_cobertura: pipx install lcov_cobertura)"
    else
        echo "XML report:      ${COVERAGE_PATH}/coverage.xml"
    fi
fi
if [[ "$FORMAT" == "all" || "$FORMAT" == "json" ]]; then
    if [[ "$COMPILER" == "clang" ]]; then
        echo "JSON report:     ${COVERAGE_PATH}/coverage.json (llvm-cov format)"
    else
        echo "JSON report:     ${COVERAGE_PATH}/coverage.json (gcovr format)"
    fi
fi
if [[ "$FORMAT" == "all" || "$FORMAT" == "markdown" ]]; then
    echo "Markdown report: ${COVERAGE_PATH}/coverage.md"
fi
if [[ -f "${COVERAGE_PATH}/pytest_coverage.json" ]]; then
    echo "Python report:   ${COVERAGE_PATH}/pytest_coverage.json (coverage.py)"
fi

echo ""
if [[ "$COMPILER" == "clang" ]]; then
    echo "Note: Host-side GPU/HIP code is included. Device kernel code is not"
    echo "      instrumented (ROCm provides no device-side profile runtime)."
else
    echo "Note: GPU code (.cu files) is excluded - GCC gcov cannot instrument HIP."
    echo "      Use --compiler clang for host-side GPU coverage."
fi
echo ""
echo "To view HTML report:"
echo "  • Open in browser: file://$(realpath ${COVERAGE_PATH}/html/index.html 2>/dev/null || echo "${COVERAGE_PATH}/html/index.html")"
echo "  • Or run: xdg-open ${COVERAGE_PATH}/html/index.html"
echo ""
echo "================================================================================"
