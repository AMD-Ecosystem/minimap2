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

# Generate test data for all test/query combinations in multiple test suites.
# Automatically builds minimap2 using specified or default preset.
# No special build flags are required -- the script uses --extra-out-dir
# and --dump-intermediates JSON to extract intermediate data as JSON.
#
# Usage:
#   ./scripts/generate_all_test_data.sh [CMAKE_PRESET]
#
# Arguments:
#   CMAKE_PRESET  CMake preset to build with (default: coverage)
#                 Can also be set via CMAKE_PRESET environment variable.
#
# Environment Variables:
#   CMAKE_PRESET      CMake preset override (default: coverage)
#   TEST_SUITE_BASE   Base directory for test suites (default: test_suite)
#
# Prerequisites:
#   - CMake (for cmake --preset / --build)
#   - Python 3 (runs generate_test_data.py per pair)
#   - Test suites under test_suite/ with t*.fa and q*.fa files
#
# Output:
#   test_suite/{suite}/expected/{ref}_{query}.json for each matching pair
#
# Key behaviors:
#   - Pairs only matching numbers: t0.fa <-> q0.fa, t1.fa <-> q1.fa, etc.
#   - Shows [current/total] progress counter per suite
#   - Exits on first error (set -e), prints error context
#   - Skips suites with no .fa files (warning, not failure)
#
# Examples:
#   ./scripts/generate_all_test_data.sh                  # coverage preset
#   ./scripts/generate_all_test_data.sh test-cpu         # test-cpu preset
#   CMAKE_PRESET=debug ./scripts/generate_all_test_data.sh
#   TEST_SUITE_BASE=/custom/path ./scripts/generate_all_test_data.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Use preset from environment or argument, default to "coverage"
CMAKE_PRESET="${1:-${CMAKE_PRESET:-coverage}}"
BUILD_DIR="$WORKSPACE_DIR/out/$CMAKE_PRESET/build"
MINIMAP2="$BUILD_DIR/minimap2"

echo "=== Building minimap2 ==="
echo "Preset: $CMAKE_PRESET"
echo "Build dir: $BUILD_DIR"
echo ""

cd "$WORKSPACE_DIR"

# Configure (no special flags needed — --extra-out-dir is always available)
cmake --preset "$CMAKE_PRESET"

# Build
cmake --build "$BUILD_DIR" -j$(nproc)

echo ""
echo "Build complete: $MINIMAP2"
echo ""

# Verify minimap2 binary exists
if [ ! -x "$MINIMAP2" ]; then
    echo "Error: minimap2 binary not found after build: $MINIMAP2" >&2
    exit 1
fi

generate_suite() {
    local suite_dir="$1"
    local suite_name=$(basename "$suite_dir")
    
    echo "=== Processing $suite_name ==="
    
    # Create expected directory
    mkdir -p "$suite_dir/expected"
    
    # Get all t*.fa and q*.fa files
    local refs=($suite_dir/t*.fa)
    local queries=($suite_dir/q*.fa)
    
    if [ ${#refs[@]} -eq 0 ] || [ ${#queries[@]} -eq 0 ]; then
        echo "Warning: No test/query pairs found in $suite_dir" >&2
        return
    fi
    
    local total_pairs=$((${#refs[@]} * ${#queries[@]}))
    local current=0
    
    # Generate only same-number pairs (t0 vs q0, t1 vs q1, etc.)
    for ref in "${refs[@]}"; do
        for query in "${queries[@]}"; do
            local ref_base=$(basename "$ref" .fa)
            local query_base=$(basename "$query" .fa)
            
            # Only process if ref and query have the same number (e.g., t0 vs q0)
            local ref_num="${ref_base#t}"
            local query_num="${query_base#q}"
            if [ "$ref_num" != "$query_num" ]; then
                continue
            fi
            
            current=$((current + 1))
            
            local output="$suite_dir/expected/${ref_base}_${query_base}.json"
            local test_name="${suite_name}_${ref_base}_${query_base}"
            
            echo "[$current/$total_pairs] Generating $test_name..."
            
            python3 "$SCRIPT_DIR/generate_test_data.py" \
                --ref "$ref" \
                --query "$query" \
                --output "$output" \
                --minimap2 "$MINIMAP2" \
                --name "$test_name" \
                --description "Test data for $suite_name: $ref_base vs $query_base" \
                2>&1 | grep -E "(Found|Generated|Error)" || true
        done
    done
    
    echo "Completed $suite_name: Generated $total_pairs test data files"
    echo ""
}

# Discover test suite directories dynamically
# Look for directories in test_suite/ that contain .fa files
TEST_SUITE_BASE="${TEST_SUITE_BASE:-test_suite}"

if [ ! -d "$TEST_SUITE_BASE" ]; then
    echo "Error: Test suite base directory not found: $TEST_SUITE_BASE" >&2
    exit 1
fi

echo "Discovering test suites in $TEST_SUITE_BASE..."
SUITES=()
for dir in "$TEST_SUITE_BASE"/*; do
    if [ -d "$dir" ] && ls "$dir"/*.fa >/dev/null 2>&1; then
        SUITES+=("$dir")
    fi
done

if [ ${#SUITES[@]} -eq 0 ]; then
    echo "Error: No test suites found in $TEST_SUITE_BASE (directories with .fa files)" >&2
    exit 1
fi

echo "Found ${#SUITES[@]} test suite(s): ${SUITES[*]}"
echo ""

# Process each discovered test suite
for suite in "${SUITES[@]}"; do
    generate_suite "$suite"
done

echo "=== Summary ==="
echo "Generated files:"
find test_suite/*/expected -name "*.json" -type f | wc -l
echo ""
echo "Done!"
