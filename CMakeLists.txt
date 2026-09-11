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

cmake_minimum_required(VERSION 3.18)

# Unified version management — single source of truth is the VERSION file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.cmake)
project(minimap2 VERSION ${MM_CMAKE_VERSION} LANGUAGES C CXX)

# Options
option(BUILD_EXTRA "Build extra programs (sdust, minimap2-lite)" OFF)
option(BUILD_TESTING "Build unit tests" OFF)
option(BUILD_CLI_TOOLS "Build CLI tools for staged mapping (mm2_seed, mm2_chain, mm2_align)" ON)
option(BUILD_GPU_BENCH "Build GPU kernel config benchmark tool" OFF)
option(BUILD_PYTHON "Build Python bindings (mappy module)" OFF)
option(BUILD_PYTHON_ONLY "Build only Python module (skip shared lib and executables)" OFF)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer (host code)" OFF)
option(MM_ENABLE_WARNINGS "Enable an extended compiler warning set (narrowing/overflow checks)" OFF)
option(ENABLE_PROFILING "Enable profiling instrumentation (ITT/NVTX/ROCTX)" OFF)
option(ENABLE_TRACE "Enable trace logging with timing" OFF)
option(MM_ENABLE_CLANG_TIDY "Run clang-tidy on first-party C++/HIP sources during the build (requires clang-tidy on PATH)" OFF)
option(MM_DEBUG_SEED_VERBOSE "Enable verbose debug output for mm_map_seed" OFF)
option(MM_DEBUG_CHAIN_VERBOSE "Enable verbose debug output for mm_map_chain" OFF)
option(MM_DEBUG_ALIGN_VERBOSE "Enable verbose debug output for mm_map_align" OFF)
option(MM_ENABLE_SEED_GPU "Enable GPU support for mm_map_seed" OFF)
set(GPU "AMD" CACHE STRING "GPU backend: AMD or NONE for CPU-only")
set(GPUARCH "" CACHE STRING "GPU architecture (e.g., gfx942, gfx90a)")
set(MAX_MICRO_BATCH "" CACHE STRING "Maximum micro batch size")
set(DEBUG_MODE "NONE" CACHE STRING "Debug mode: NONE, info, analyze, verbose")
set(SUFFIX "" CACHE STRING "Suffix for executable name")

# Architecture options
option(SSE2_ONLY "Build for SSE2 only (no SSE4.1)" OFF)

# Detect ARM / aarch64 so the SSE kernels can be built through the sse2neon
# shim (mirrors the upstream Makefile arm_neon / aarch64 targets).
set(MM_ARM_NEON OFF)
set(MM_AARCH64 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv8|armv7|^arm")
    set(MM_ARM_NEON ON)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv8")
        set(MM_AARCH64 ON)
    endif()
endif()
option(ARM_NEON "Build the SSE kernels for ARM via sse2neon (auto-detected)" ${MM_ARM_NEON})

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# clang-tidy integration (opt-in; default OFF so normal builds are unaffected).
# When ON, CMake runs clang-tidy on every first-party C++ translation unit using
# the repo-root .clang-tidy config. hipcc is clang-based, so the checks also
# cover HIP device code under minimap2/**/gpu/. Applied to CXX only — the C
# translation units are left untouched.
if(MM_ENABLE_CLANG_TIDY)
    find_program(MM_CLANG_TIDY_EXE NAMES clang-tidy
        HINTS /opt/rocm/llvm/bin /opt/rocm/bin)
    if(MM_CLANG_TIDY_EXE)
        # The -extra-arg flags make clang-tidy tolerant of the hipcc/clang
        # compile flags it re-parses each TU with (HIP offload flags etc. it
        # would otherwise report as unknown/unused), and pin the C++ standard.
        set(CMAKE_CXX_CLANG_TIDY
            "${MM_CLANG_TIDY_EXE}"
            "-extra-arg=-Wno-unknown-warning-option"
            "-extra-arg=-Wno-ignored-optimization-argument"
            "-extra-arg=-Wno-unused-command-line-argument"
            "-extra-arg=-std=c++${CMAKE_CXX_STANDARD}")
        message(STATUS "clang-tidy enabled: ${MM_CLANG_TIDY_EXE}")
    else()
        message(WARNING "MM_ENABLE_CLANG_TIDY=ON but clang-tidy was not found on PATH; skipping")
    endif()
endif()

# Base definitions
add_compile_definitions(HAVE_KALLOC)

# ARM NEON build flags (mirrors the upstream Makefile arm_neon / aarch64 targets).
if(ARM_NEON)
    add_compile_definitions(_FILE_OFFSET_BITS=64)
    add_compile_options(-fsigned-char)
    if(NOT MM_AARCH64)
        add_compile_options(-mfpu=neon)
    endif()
    message(STATUS "ARM NEON build enabled (SSE kernels via sse2neon)")
endif()

# Debug verbose flags for split kernel functions
if(MM_DEBUG_SEED_VERBOSE)
    add_compile_definitions(MM_DEBUG_SEED_VERBOSE)
    message(STATUS "MM_DEBUG_SEED_VERBOSE enabled")
endif()
if(MM_DEBUG_CHAIN_VERBOSE)
    add_compile_definitions(MM_DEBUG_CHAIN_VERBOSE)
    message(STATUS "MM_DEBUG_CHAIN_VERBOSE enabled")
endif()
if(MM_DEBUG_ALIGN_VERBOSE)
    add_compile_definitions(MM_DEBUG_ALIGN_VERBOSE)
    message(STATUS "MM_DEBUG_ALIGN_VERBOSE enabled")
endif()

# MAX_MICRO_BATCH definition
if(MAX_MICRO_BATCH)
    add_compile_definitions(MAX_MICRO_BATCH=${MAX_MICRO_BATCH})
    set(CONFIG_DEFS "-DMICRO_BATCH=${MAX_MICRO_BATCH}")
endif()

# =============================================================================
# Build Configuration
# =============================================================================
# CMAKE_BUILD_TYPE controls optimization and debug symbols (standard CMake):
#   - Release: -O3 -DNDEBUG (default, optimized, no debug symbols)
#   - RelWithDebInfo: -O2 -g -DNDEBUG (optimized with debug symbols for profiling/VTune)
#   - Debug: -O0 -g (debug symbols, no optimization)
#
# DEBUG_MODE controls minimap2-specific debug defines (independent of build type):
#   - info: DEBUG_PRINT (enables debug print statements)
#   - analyze: DEBUG_PRINT + DEBUG_CHECK (adds runtime checks)
#   - verbose: DEBUG_PRINT + DEBUG_CHECK + DEBUG_VERBOSE (full debug output)
# =============================================================================

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# DEBUG_MODE: Only adds debug defines, does NOT affect optimization/symbols
if(DEBUG_MODE STREQUAL "info")
    add_compile_definitions(DEBUG_PRINT)
    message(STATUS "Debug mode: info (DEBUG_PRINT enabled)")
elseif(DEBUG_MODE STREQUAL "analyze")
    add_compile_definitions(DEBUG_PRINT DEBUG_CHECK)
    message(STATUS "Debug mode: analyze (DEBUG_PRINT + DEBUG_CHECK enabled)")
elseif(DEBUG_MODE STREQUAL "verbose")
    add_compile_definitions(DEBUG_PRINT DEBUG_CHECK DEBUG_VERBOSE)
    message(STATUS "Debug mode: verbose (DEBUG_PRINT + DEBUG_CHECK + DEBUG_VERBOSE enabled)")
endif()

# Code Coverage
# LLVM/Clang path supports GPU=AMD: HIP sources are instrumented on the HOST
# compilation pass only (via -Xarch_host). The device (amdgcn) pass is left
# uninstrumented because ROCm ships no device-side profile runtime — the bulk
# of the GPU source files is host-side orchestration code that runs on the CPU.
# GCC path stays CPU-only (GPU=NONE): GCC's gcov cannot consume the .gcno files
# that ROCm clang emits for HIP sources.
if(ENABLE_COVERAGE)
    if(COVERAGE_TYPE STREQUAL "LLVM" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
        # Non-HIP (C/C++) sources: instrument normally.
        add_compile_options(
            $<$<NOT:$<COMPILE_LANGUAGE:HIP>>:-g>
            $<$<NOT:$<COMPILE_LANGUAGE:HIP>>:-O0>
            $<$<NOT:$<COMPILE_LANGUAGE:HIP>>:-fprofile-instr-generate>
            $<$<NOT:$<COMPILE_LANGUAGE:HIP>>:-fcoverage-mapping>
        )
        # HIP sources: instrument the host pass only; the amdgcn device pass has
        # no profile runtime (would fail with undefined __llvm_profile_* symbols).
        add_compile_options(
            "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-g -O0 -Xarch_host -fprofile-instr-generate -Xarch_host -fcoverage-mapping>"
        )
        add_link_options(-fprofile-instr-generate)
        message(STATUS "Code coverage enabled (LLVM/Clang, GPU host code instrumented)")
    else()
        # GCC coverage flags (CPU-only build)
        add_compile_options(-g -O0 --coverage -fprofile-arcs -ftest-coverage)
        add_link_options(--coverage)
        message(STATUS "Code coverage enabled (GCC/gcov)")
    endif()
endif()

# Sanitizers
if(ENABLE_ASAN)
    # Host code only. The amdgcn device pass has no usable ASan runtime in this
    # ROCm (device-instrumented kernels fail to load with INVALID_ISA, and the
    # device report hostcall has no handler), so HIP device compilation is
    # excluded — same pattern as the coverage / UBSan blocks. Catches host-side
    # heap/stack OOB and use-after-free in the CPU pipeline and GPU host
    # orchestration code.
    add_compile_options(
        $<$<NOT:$<COMPILE_LANGUAGE:HIP>>:-fsanitize=address>
        $<$<NOT:$<COMPILE_LANGUAGE:HIP>>:-fno-omit-frame-pointer>
        "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-Xarch_host -fsanitize=address>"
    )
    add_link_options(-fsanitize=address)
    message(STATUS "AddressSanitizer enabled (host code)")
endif()

if(ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()

# UndefinedBehaviorSanitizer — host code only. The amdgcn device pass has no
# UBSan runtime, so HIP device compilation is excluded (same pattern as the
# coverage block). This keeps ENABLE_UBSAN usable on GPU builds to catch
# host-side integer-overflow / UB in the CPU pipeline at runtime.
if(ENABLE_UBSAN)
    add_compile_options(
        $<$<NOT:$<COMPILE_LANGUAGE:HIP>>:-fsanitize=undefined>
        $<$<NOT:$<COMPILE_LANGUAGE:HIP>>:-fno-omit-frame-pointer>
        "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-Xarch_host -fsanitize=undefined>"
    )
    add_link_options(-fsanitize=undefined)
    message(STATUS "UndefinedBehaviorSanitizer enabled (host code)")
endif()

# Extended compiler warnings (opt-in; default OFF). -Wall/-Wextra are portable;
# the precise narrowing/overflow warnings — the class behind the GA-4 32-bit
# index bug (a 32-bit value multiplied into a >2 GB device buffer) — are
# Clang-only, so they are gated on an all-Clang toolchain. Warnings only, never
# errors, so the build is unaffected during incremental adoption.
if(MM_ENABLE_WARNINGS)
    add_compile_options(-Wall -Wextra)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-Wshorten-64-to-32 -Wimplicit-int-conversion -Wfloat-conversion)
    endif()
    message(STATUS "Extended compiler warnings enabled")
endif()

# Find required packages
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

# GPU compiler setup
if(GPU STREQUAL "AMD")
    # Set ROCm path - can be overridden via -DROCM_PATH=...
    if(NOT DEFINED ROCM_PATH)
        if(DEFINED ENV{ROCM_PATH})
            set(ROCM_PATH $ENV{ROCM_PATH})
        elseif(EXISTS "/opt/rocm")
            set(ROCM_PATH "/opt/rocm")
        else()
            message(FATAL_ERROR "ROCM_PATH not found. Please set ROCM_PATH or install ROCm.")
        endif()
    endif()
    
    # Detect GPU architecture if not specified
    if(NOT GPUARCH)
        execute_process(
            COMMAND bash -c "rocminfo | grep -m 1 -E 'gfx[^0]{1}' | sed -e 's/ *Name: *//'"
            OUTPUT_VARIABLE GPUARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(NOT GPUARCH)
            set(GPUARCH "gfx942")
            message(WARNING "Could not detect GPU architecture, defaulting to ${GPUARCH}")
        endif()
        # Cache the detected architecture
        set(GPUARCH "${GPUARCH}" CACHE STRING "GPU architecture" FORCE)
    endif()
    message(STATUS "GPU Architecture: ${GPUARCH}")
    
    # Enable HIP language support (CMake 3.21+)
    # Note: CMAKE_HIP_ARCHITECTURES must be set BEFORE enable_language(HIP)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
        set(CMAKE_HIP_ARCHITECTURES ${GPUARCH})
        enable_language(HIP)
        message(STATUS "HIP language enabled (CMake ${CMAKE_VERSION})")
    else()
        message(STATUS "CMake < 3.21: Using hipcc via CXX compiler path")
    endif()
    
    # Fallback compiler and flags (for CMake < 3.21)
    find_program(HIPCC hipcc REQUIRED)
    set(GPU_COMPILER ${HIPCC})
    set(GPU_FLAGS -DUSEHIP --offload-arch=${GPUARCH})
    set(GPU_ANALYZE_FLAGS -Rpass-analysis=kernel-resource-usage)
    set(GPU_TEST_FLAGS -G -ggdb)
    
    # Note: MM_ENABLE_HIP is passed only to GPU sources and files that need it,
    # not globally, to avoid SSE files trying to include hip_runtime.h
    
    # Find HIP runtime library (required)
    find_library(HIP_RT_LIB amdhip64 HINTS ${ROCM_PATH}/lib /opt/rocm/lib REQUIRED)
    set(GPU_LIBS ${HIP_RT_LIB})
    
    # Find ROCm profiling libraries (optional – only needed when profiling is enabled)
    if(ENABLE_PROFILING)
        find_library(ROCTX_LIB roctx64 HINTS ${ROCM_PATH}/lib /opt/rocm/lib)
        find_library(ROCTRACER_LIB roctracer64 HINTS ${ROCM_PATH}/lib /opt/rocm/lib)
        if(ROCTX_LIB AND ROCTRACER_LIB)
            list(APPEND GPU_LIBS ${ROCTX_LIB} ${ROCTRACER_LIB})
            set(HAVE_ROCTX TRUE CACHE BOOL "ROCm profiling libraries found")
            message(STATUS "Found ROCm profiling libraries")
        else()
            set(HAVE_ROCTX FALSE CACHE BOOL "ROCm profiling libraries not found")
            message(STATUS "ROCm profiling libraries not found (optional)")
        endif()
    else()
        set(HAVE_ROCTX FALSE CACHE BOOL "Profiling disabled – ROCm profiling libraries not linked")
    endif()
elseif(GPU STREQUAL "NONE")
    message(STATUS "Building CPU-only version (no GPU support)")
    set(GPU_COMPILER ${CMAKE_C_COMPILER})
    set(GPU_FLAGS "")
    set(GPU_LIBS "")
else()
    message(FATAL_ERROR "Invalid GPU option: ${GPU}. Must be AMD or NONE")
endif()

# Add analyze/verbose flags if needed
if(DEBUG_MODE STREQUAL "analyze" OR DEBUG_MODE STREQUAL "verbose")
    set(GPU_FLAGS "${GPU_FLAGS} ${GPU_ANALYZE_FLAGS}")
endif()
if(DEBUG_MODE STREQUAL "verbose")
    set(GPU_FLAGS "${GPU_FLAGS} ${GPU_TEST_FLAGS}")
endif()

# =============================================================================
# Helper: detect flatc version from its --version output
# Both v2.x (apt/Ubuntu) and v25.x (FetchContent/latest) share identical
# flag syntax (--cpp, --python, -o PATH). Version detection is used for
# logging and future-proofing only.
# =============================================================================
function(detect_flatc_version FLATC_EXECUTABLE_PATH FLATC_VERSION_OUT)
    execute_process(
        COMMAND ${FLATC_EXECUTABLE_PATH} --version
        OUTPUT_VARIABLE FLATC_VERSION_STR
        ERROR_VARIABLE  FLATC_VERSION_ERR
        RESULT_VARIABLE FLATC_VERSION_RESULT
    )
    if(NOT FLATC_VERSION_RESULT EQUAL 0)
        message(WARNING "Could not detect flatc version: ${FLATC_VERSION_ERR}")
        set(${FLATC_VERSION_OUT} "unknown" PARENT_SCOPE)
        return()
    endif()
    # Parse "flatc version X.Y.Z" or bare "X.Y.Z"
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" FLATC_VERSION_MATCH "${FLATC_VERSION_STR}")
    if(FLATC_VERSION_MATCH)
        set(${FLATC_VERSION_OUT} "${FLATC_VERSION_MATCH}" PARENT_SCOPE)
        message(STATUS "FlatBuffers flatc version detected: ${FLATC_VERSION_MATCH}")
    else()
        message(WARNING "Could not parse flatc version from: ${FLATC_VERSION_STR}")
        set(${FLATC_VERSION_OUT} "unknown" PARENT_SCOPE)
    endif()
endfunction()

find_package(FlatBuffers QUIET)
if(NOT FlatBuffers_FOUND)
    message(STATUS "System FlatBuffers not found — fetching via FetchContent")
    include(FetchContent)
    set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(FLATBUFFERS_BUILD_FLATLIB ON CACHE BOOL "" FORCE)
    set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
    set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        flatbuffers
        GIT_REPOSITORY https://github.com/google/flatbuffers.git
        GIT_TAG        v25.12.19
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(flatbuffers)
    if(NOT TARGET flatbuffers::flatbuffers)
        add_library(flatbuffers::flatbuffers ALIAS flatbuffers)
    endif()
    set(FLATC_TARGET flatc)
    set(FLATC_VERSION "25.12.19")
else()
    find_program(FLATC_EXECUTABLE flatc REQUIRED)
    set(FLATC_TARGET)
    detect_flatc_version(${FLATC_EXECUTABLE} FLATC_VERSION)
endif()

# Generate C++ and Python bindings from FlatBuffers schema at build time.
set(FLATBUF_SCHEMA "${CMAKE_SOURCE_DIR}/schemas/query_dump.fbs")
set(FLATBUF_CPP_OUT "${CMAKE_SOURCE_DIR}/minimap2/core/include/generated")
set(FLATBUF_PY_OUT  "${CMAKE_SOURCE_DIR}/scripts/generated")
set(FLATBUF_CPP_HEADER "${FLATBUF_CPP_OUT}/query_dump_generated.h")
set(FLATBUF_PY_INIT1   "${FLATBUF_PY_OUT}/mm2/__init__.py")
set(FLATBUF_PY_INIT2   "${FLATBUF_PY_OUT}/mm2/dump/__init__.py")

file(MAKE_DIRECTORY ${FLATBUF_CPP_OUT})
file(MAKE_DIRECTORY ${FLATBUF_PY_OUT})

if(FLATC_TARGET)
    # FetchContent: generate at build time (flatc is a build target)
    add_custom_command(
        OUTPUT  ${FLATBUF_CPP_HEADER}
        COMMAND flatc --cpp -o ${FLATBUF_CPP_OUT} ${FLATBUF_SCHEMA}
        DEPENDS ${FLATBUF_SCHEMA} flatc
        COMMENT "Generating FlatBuffers C++ header (v${FLATC_VERSION})"
    )
    add_custom_command(
        OUTPUT  ${FLATBUF_PY_INIT2}
        COMMAND flatc --python -o ${FLATBUF_PY_OUT} ${FLATBUF_SCHEMA}
        COMMAND ${CMAKE_COMMAND} -E touch ${FLATBUF_PY_INIT1}
        COMMAND ${CMAKE_COMMAND} -E touch ${FLATBUF_PY_INIT2}
        DEPENDS ${FLATBUF_SCHEMA} flatc
        COMMENT "Generating FlatBuffers Python bindings (v${FLATC_VERSION})"
    )
else()
    # System flatc: run at configure time
    message(STATUS "Generating FlatBuffers with system flatc v${FLATC_VERSION}")
    execute_process(
        COMMAND ${FLATC_EXECUTABLE} --cpp -o ${FLATBUF_CPP_OUT} ${FLATBUF_SCHEMA}
        RESULT_VARIABLE FLATC_CPP_RESULT
        ERROR_VARIABLE  FLATC_CPP_ERROR
    )
    if(NOT FLATC_CPP_RESULT EQUAL 0)
        message(FATAL_ERROR "flatc --cpp failed (v${FLATC_VERSION}): ${FLATC_CPP_ERROR}")
    endif()
    execute_process(
        COMMAND ${FLATC_EXECUTABLE} --python -o ${FLATBUF_PY_OUT} ${FLATBUF_SCHEMA}
        RESULT_VARIABLE FLATC_PY_RESULT
        ERROR_VARIABLE  FLATC_PY_ERROR
    )
    if(NOT FLATC_PY_RESULT EQUAL 0)
        message(FATAL_ERROR "flatc --python failed (v${FLATC_VERSION}): ${FLATC_PY_ERROR}")
    endif()
endif()

add_custom_target(flatbuf_generated
    DEPENDS ${FLATBUF_CPP_HEADER} ${FLATBUF_PY_INIT2}
)

message(STATUS "FlatBuffers: C++ header -> ${FLATBUF_CPP_HEADER}")
message(STATUS "FlatBuffers: Python -> ${FLATBUF_PY_OUT}/mm2/dump/")
message(STATUS "FlatBuffers: Version ${FLATC_VERSION}")

# ============================================================================
# Component Libraries (Static Libraries)
# ============================================================================
# Each component is built as a separate static library in its own subdirectory.
# Dependency chain: core -> seed -> chain -> align -> libminimap2

# Common internal include paths for CLI tools, tests, and executables
# that need access to private headers
set(MM2_INTERNAL_INCLUDES
    ${CMAKE_SOURCE_DIR}/minimap2/include
    ${CMAKE_SOURCE_DIR}/minimap2/src
    ${CMAKE_SOURCE_DIR}/minimap2/core/include
    ${CMAKE_SOURCE_DIR}/minimap2/seed/include
    ${CMAKE_SOURCE_DIR}/minimap2/chain/include
    ${CMAKE_SOURCE_DIR}/minimap2/align/include
    ${CMAKE_SOURCE_DIR}/external/cJSON
)

# cJSON library (third-party dependency)
add_library(cjson STATIC external/cJSON/cJSON.c)
target_include_directories(cjson 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external/cJSON>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/minimap2>
)
set_target_properties(cjson PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Add minimap2 library subdirectory (builds components and libminimap2)
add_subdirectory(minimap2)

# Install targets
include(GNUInstallDirs)

# Skip installing C/C++ artifacts if only building Python module
if(NOT BUILD_PYTHON_ONLY)
    # Documentation (man page) - gzip for Debian policy compliance
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/minimap2.1.gz
        COMMAND gzip -9 -c ${CMAKE_CURRENT_SOURCE_DIR}/man/minimap2.1
                > ${CMAKE_CURRENT_BINARY_DIR}/minimap2.1.gz
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/man/minimap2.1
        COMMENT "Compressing man page"
    )
    add_custom_target(man_page ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/minimap2.1.gz
    )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/minimap2.1.gz
        DESTINATION ${CMAKE_INSTALL_MANDIR}/man1
    )

    # Install documentation
    install(FILES USAGE.md LICENSE.txt NEWS.md
        DESTINATION ${CMAKE_INSTALL_DOCDIR}
        COMPONENT Documentation
    )

    # Install GPU configuration file (AMD GPU builds only)
    if(GPU STREQUAL "AMD")
        install(FILES configs/gpu_config.json
            DESTINATION ${CMAKE_INSTALL_DATADIR}/minimap2/configs
        )
    endif()

    # ============================================================================
    # CMake Package Configuration (for find_package support)
    # ============================================================================
    include(CMakePackageConfigHelpers)

    # Export targets - shared library
    install(TARGETS minimap2
        EXPORT minimap2Targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/minimap2
    )

    # Export targets - main executable
    install(TARGETS minimap2_tool
        EXPORT minimap2Targets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    # Export targets - extra programs (if built)
    if(BUILD_EXTRA)
        install(TARGETS minimap2-lite sdust
            EXPORT minimap2Targets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()

    # Generate and install package config files
    configure_package_config_file(
        "${CMAKE_SOURCE_DIR}/cmake/minimap2Config.cmake.in"
        "${CMAKE_BINARY_DIR}/minimap2Config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/minimap2
    )

    write_basic_package_version_file(
        "${CMAKE_BINARY_DIR}/minimap2ConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(EXPORT minimap2Targets
        FILE minimap2Targets.cmake
        NAMESPACE minimap2::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/minimap2
    )

    install(FILES
        "${CMAKE_BINARY_DIR}/minimap2Config.cmake"
        "${CMAKE_BINARY_DIR}/minimap2ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/minimap2
    )
endif() # NOT BUILD_PYTHON_ONLY

# Print configuration summary
message(STATUS "=== minimap2 Build Configuration ===")
message(STATUS "GPU Backend: ${GPU}")
if(GPU STREQUAL "AMD")
    message(STATUS "GPU Architecture: ${GPUARCH}")
endif()
message(STATUS "Debug Mode: ${DEBUG_MODE}")
message(STATUS "SSE2 Only: ${SSE2_ONLY}")
message(STATUS "AddressSanitizer: ${ENABLE_ASAN}")
message(STATUS "ThreadSanitizer: ${ENABLE_TSAN}")
message(STATUS "Build Extra: ${BUILD_EXTRA}")
message(STATUS "Build Testing: ${BUILD_TESTING}")
message(STATUS "Build CLI Tools: ${BUILD_CLI_TOOLS}")
message(STATUS "Build Python: ${BUILD_PYTHON}")
if(BUILD_DEB_PACKAGE OR BUILD_RPM_PACKAGE OR BUILD_SLES_RPM_PACKAGE)
    if(GPU STREQUAL "AMD")
        set(_pkg_name "amd-minimap2")
    else()
        set(_pkg_name "minimap2")
    endif()
    if(BUILD_DEB_PACKAGE)
        message(STATUS "Package: DEB (${_pkg_name} ${MM_DEB_VERSION})")
    endif()
    if(BUILD_RPM_PACKAGE)
        message(STATUS "Package: RPM (${_pkg_name} ${MM_RPM_VERSION}-${MM_RPM_RELEASE})")
    endif()
    if(BUILD_SLES_RPM_PACKAGE)
        message(STATUS "Package: SLES RPM (${_pkg_name} ${MM_RPM_VERSION}-${MM_RPM_RELEASE})")
    endif()
endif()

# Testing support
if(BUILD_TESTING)
    enable_testing()
    
    # Use Google Test from submodule
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    add_subdirectory(external/googletest ${CMAKE_BINARY_DIR}/external/googletest)
    
    # Add test subdirectory
    add_subdirectory(tests)
endif()

# CLI tools for staged mapping (independent of testing)
if(BUILD_CLI_TOOLS)
    add_subdirectory(cli)
endif()

# GPU kernel benchmark tool
if(BUILD_GPU_BENCH)
    add_subdirectory(tools)
endif()

# Python bindings
if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

# =============================================================================
# CPack — Package Generation (.deb / .rpm)
# =============================================================================
# Supported packaging targets:
#   BUILD_DEB_PACKAGE       — Debian/Ubuntu .deb
#   BUILD_RPM_PACKAGE       — RHEL/Fedora/Rocky .rpm
#   BUILD_SLES_RPM_PACKAGE  — SLES 15 / openSUSE .rpm
#
# Build with:
#   cmake --build <build-dir> --target package
# or:
#   cd <build-dir> && cpack -G DEB   (or RPM)
# =============================================================================
option(BUILD_DEB_PACKAGE      "Enable CPack DEB package generation"              OFF)
option(BUILD_RPM_PACKAGE      "Enable CPack RPM package generation"              OFF)
option(BUILD_SLES_RPM_PACKAGE "Enable CPack RPM package generation for SLES 15"  OFF)

# BUILD_RPM_PACKAGE and BUILD_SLES_RPM_PACKAGE both target the RPM generator
# with conflicting distro-specific settings, so they are mutually exclusive.
# BUILD_DEB_PACKAGE may be combined with either RPM option.
if(BUILD_RPM_PACKAGE AND BUILD_SLES_RPM_PACKAGE)
    message(FATAL_ERROR
        "BUILD_RPM_PACKAGE and BUILD_SLES_RPM_PACKAGE are mutually exclusive "
        "(both target the RPM generator with conflicting settings). "
        "Use separate build directories for RHEL and SLES packages.")
endif()

if(BUILD_DEB_PACKAGE OR BUILD_RPM_PACKAGE OR BUILD_SLES_RPM_PACKAGE)

    # ---- Common package metadata (shared across all generators) ----
    if(GPU STREQUAL "AMD")
        set(CPACK_PACKAGE_NAME "amd-minimap2")
    else()
        set(CPACK_PACKAGE_NAME "minimap2")
    endif()
    set(CPACK_PACKAGE_VERSION_MAJOR ${MM_VERSION_MAJOR})
    set(CPACK_PACKAGE_VERSION_MINOR ${MM_VERSION_MINOR})
    set(CPACK_PACKAGE_VERSION_PATCH ${MM_VERSION_PATCH})
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
        "A versatile pairwise aligner for genomic and spliced nucleotide sequences")
    set(CPACK_PACKAGE_DESCRIPTION
        "minimap2 is a fast sequence mapping and alignment program that can find \
overlaps between long noisy reads, or map long reads or their assemblies to a \
reference genome optionally with GPU acceleration.")
    set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/ROCm-LS/minimap2")
    set(CPACK_PACKAGE_CONTACT "AMD Corporation")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE.txt")
    set(CPACK_RESOURCE_FILE_README  "${CMAKE_SOURCE_DIR}/USAGE.md")
    set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")
    set(CPACK_PACKAGE_DIRECTORY "${CMAKE_BINARY_DIR}/packages")
    set(CPACK_STRIP_FILES TRUE)

    # Build CPACK_GENERATOR as a list so multiple formats can be produced
    # from a single build directory (e.g. DEB + RPM).
    set(CPACK_GENERATOR "")

    # ---- Debian (.deb) ----
    if(BUILD_DEB_PACKAGE)
        list(APPEND CPACK_GENERATOR "DEB")
        set(CPACK_DEBIAN_PACKAGE_VERSION "${MM_DEB_VERSION}")

        set(CPACK_DEBIAN_PACKAGE_MAINTAINER "AMD Corporation")
        set(CPACK_DEBIAN_PACKAGE_SECTION "science")
        set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")

        # Runtime dependencies – shlibdeps will auto-detect linked shared libs,
        # but we also declare the obvious ones explicitly as a safety net.
        # shlibdeps cannot resolve ROCm libs under /opt/rocm, so those must be
        # listed manually.
        set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "zlib1g (>= 1:1.2.11)")

        if(GPU STREQUAL "AMD")
            string(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS ", hip-runtime-amd")
            if(HAVE_ROCTX)
                string(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS ", roctracer")
            endif()
        endif()
    endif()

    # ---- RPM (.rpm) — Standard (RHEL/Fedora/Rocky) or SLES 15 (openSUSE) ----
    if(BUILD_RPM_PACKAGE OR BUILD_SLES_RPM_PACKAGE)

        list(APPEND CPACK_GENERATOR "RPM")
        set(CPACK_RPM_PACKAGE_VERSION "${MM_RPM_VERSION}")

        set(CPACK_RPM_PACKAGE_LICENSE "MIT")
        set(CPACK_RPM_PACKAGE_RELEASE "${MM_RPM_RELEASE}")
        set(CPACK_RPM_PACKAGE_URL "${CPACK_PACKAGE_HOMEPAGE_URL}")
        set(CPACK_RPM_PACKAGE_DESCRIPTION "${CPACK_PACKAGE_DESCRIPTION}")
        set(CPACK_RPM_PACKAGE_AUTOREQPROV ON)

        # RPM packaging workaround:
        # By default, CPack RPM generator may add parent directories (e.g., /usr, /usr/share/man,
        # /usr/lib64/cmake, /usr/lib64/pkgconfig) to the package file list. This causes conflicts
        # because these directories are owned by system packages like 'filesystem',
        # 'pkgconf-pkg-config', and 'cmake-filesystem'.
        #
        # Using CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION ensures:
        #   - These system directories are NOT owned by this package
        #   - Actual installed files are still included automatically
        #   - RPM installs succeed without file conflicts
        #
        # This is required for all RPM-based distros (RHEL, Rocky, SLES, openSUSE).
        set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
            /usr
            /usr/bin
            /usr/share
            /usr/share/man
            /usr/share/man/man1
            /usr/lib
            /usr/lib64
            /usr/lib64/cmake
            /usr/lib64/pkgconfig
        )

        # Distro-specific: RPM group and zlib package name
        if(BUILD_SLES_RPM_PACKAGE)
            set(CPACK_RPM_PACKAGE_GROUP "Productivity/Scientific/Other")
            set(CPACK_RPM_PACKAGE_REQUIRES "libz1 >= 1.2.11")
        else()
            set(CPACK_RPM_PACKAGE_GROUP "Applications/Engineering")
            set(CPACK_RPM_PACKAGE_REQUIRES "zlib >= 1.2.11")
        endif()

        if(GPU STREQUAL "AMD")
            string(APPEND CPACK_RPM_PACKAGE_REQUIRES ", hip-runtime-amd")
            if(HAVE_ROCTX)
                string(APPEND CPACK_RPM_PACKAGE_REQUIRES ", roctracer")
            endif()
        endif()
    endif()

    include(CPack)
endif()
