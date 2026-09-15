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

# Python bindings for minimap2 (mappy module)
#
# Build with:
#   cmake -B build -DBUILD_PYTHON=ON
#   cmake --build build --target mappy
#
# Install with:
#   cmake --install build --component python
#
# Or use pip:
#   pip install .

cmake_minimum_required(VERSION 3.18)

# Find Python and required components
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Find Cython
find_program(CYTHON_EXECUTABLE NAMES cython cython3 REQUIRED)
message(STATUS "Found Cython: ${CYTHON_EXECUTABLE}")

# Get Cython version
execute_process(
    COMMAND ${CYTHON_EXECUTABLE} --version
    OUTPUT_VARIABLE CYTHON_VERSION_OUTPUT
    ERROR_VARIABLE CYTHON_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Cython version: ${CYTHON_VERSION_OUTPUT}")

# Generate C source from Cython .pyx file
set(MAPPY_PYX "${CMAKE_CURRENT_SOURCE_DIR}/mappy.pyx")
set(MAPPY_C "${CMAKE_CURRENT_BINARY_DIR}/mappy.c")

# When MM_PYTHON_COVERAGE is enabled (used by scripts/coverage.sh), build the
# Cython module with line tracing so coverage.py / pytest-cov can collect
# line-level coverage for mappy.pyx. This adds per-line trace callbacks, so it
# is opt-in and only used for coverage builds.
set(MAPPY_CYTHON_FLAGS "")
if(MM_PYTHON_COVERAGE)
    message(STATUS "mappy: Cython line tracing enabled (MM_PYTHON_COVERAGE=ON)")
    set(MAPPY_CYTHON_FLAGS -X linetrace=True -X profile=True)
endif()

add_custom_command(
    OUTPUT ${MAPPY_C}
    COMMAND ${CYTHON_EXECUTABLE} -3 ${MAPPY_CYTHON_FLAGS} -o ${MAPPY_C} ${MAPPY_PYX}
    DEPENDS 
        ${MAPPY_PYX}
        ${CMAKE_CURRENT_SOURCE_DIR}/cmappy.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/cmappy.h
    COMMENT "Generating mappy.c from mappy.pyx with Cython"
    VERBATIM
)

# Create Python extension module
Python3_add_library(mappy MODULE
    ${MAPPY_C}
    ${CMAKE_SOURCE_DIR}/minimap2/src/map.c
    ${CMAKE_SOURCE_DIR}/minimap2/src/format.c
    ${CMAKE_SOURCE_DIR}/minimap2/src/splitidx.c
    $<TARGET_OBJECTS:mm2_version>
)

# Enable the Cython trace macros in the generated C so line tracing is actually
# compiled in (the -X linetrace directive only emits guarded callbacks).
if(MM_PYTHON_COVERAGE)
    set_source_files_properties(${MAPPY_C} PROPERTIES
        COMPILE_DEFINITIONS "CYTHON_TRACE=1;CYTHON_TRACE_NOGIL=1")
endif()

# Include directories
target_include_directories(mappy PRIVATE
    ${CMAKE_SOURCE_DIR}/minimap2/include
    ${CMAKE_SOURCE_DIR}/minimap2/src
    ${CMAKE_SOURCE_DIR}/external/cJSON
    ${CMAKE_CURRENT_SOURCE_DIR}
)
# Note: Other includes (core, seed, chain, align) inherited via mm2_align

# Link against mm2_align (transitively pulls in chain -> seed -> core)
target_link_libraries(mappy PRIVATE
    mm2_align
    cjson
)

# GPU-specific linking
if(GPU STREQUAL "AMD")
    # map_ctx.cpp defines mm_mapctx_* and is HIP-compiled into this object lib
    # by minimap2/CMakeLists.txt; pull it in so the shared mapping context the
    # Aligner owns is available. MM_ENABLE_HIP activates the per-read GPU chain
    # branch in map.c (HAVE_GPU_ALIGNMENT is already PUBLIC from mm2_align).
    target_sources(mappy PRIVATE $<TARGET_OBJECTS:mm2_map_gpu_objects>)
    target_compile_definitions(mappy PRIVATE MM_ENABLE_HIP)
    target_include_directories(mappy PRIVATE
        ${CMAKE_SOURCE_DIR}/minimap2/core/include
        ${CMAKE_SOURCE_DIR}/minimap2/seed/include
        ${CMAKE_SOURCE_DIR}/minimap2/chain/include
        ${CMAKE_SOURCE_DIR}/minimap2/chain/include/gpu
        ${CMAKE_SOURCE_DIR}/minimap2/align/include
        ${CMAKE_SOURCE_DIR}/external/spdlog/include
        ${ROCM_PATH}/include
    )
    target_link_libraries(mappy PRIVATE ${GPU_LIBS} stdc++)
    # Set rpath for HIP runtime
    set_target_properties(mappy PROPERTIES
        BUILD_RPATH "${ROCM_PATH}/lib"
        INSTALL_RPATH "${ROCM_PATH}/lib"
    )
else()
    # CPU-only build: map_ctx.cpp is not built as a separate object library, so
    # compile it directly into the module.
    target_sources(mappy PRIVATE ${CMAKE_SOURCE_DIR}/minimap2/src/map_ctx.cpp)
endif()

# Set output name (without 'lib' prefix)
set_target_properties(mappy PROPERTIES
    PREFIX ""
    OUTPUT_NAME "mappy"
)

# Optimization flags
target_compile_options(mappy PRIVATE -O3)

# Get Python extension suffix for this platform
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set_target_properties(mappy PROPERTIES SUFFIX "${PYTHON_EXT_SUFFIX}")

message(STATUS "Python extension suffix: ${PYTHON_EXT_SUFFIX}")

# Install the Python module
# Use CMAKE_INSTALL_PREFIX for wheel building compatibility
install(TARGETS mappy
    LIBRARY DESTINATION .
    COMPONENT python
)

# Install the minimap2.py script
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/minimap2.py
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT python
    OPTIONAL
)

# Copy module to source directory for development (optional target)
add_custom_target(mappy_inplace
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mappy> ${CMAKE_SOURCE_DIR}/
    DEPENDS mappy
    COMMENT "Copying mappy module to source directory for development"
)
