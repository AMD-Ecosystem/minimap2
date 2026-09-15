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

# minimap2 Core Library
# Contains: memory management, threading, sequence I/O, logging, profiling, options
# This is the base library - all other mm2 components depend on this

# ============================================================================
# Source Files
# ============================================================================
set(MM2_CORE_SOURCES
    src/kalloc.c
    src/index.c
    src/kthread.c
    src/bseq.c
    src/misc.c
    src/options.c
    src/mm_log.cpp
    src/mm_json_dump.cpp
    src/mm_dump.cpp
    src/mm_dump_util.cpp
    src/mm_flatbuf_dump.cpp
    src/core_ctx.cpp
)

# Only include profiler backend when profiling is enabled (avoids dead code)
if(ENABLE_PROFILING)
    list(APPEND MM2_CORE_SOURCES src/mm_profiler.cpp)
endif()

add_library(mm2_core STATIC ${MM2_CORE_SOURCES})
add_dependencies(mm2_core flatbuf_generated)

target_include_directories(mm2_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/minimap2>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/minimap2/include
        ${CMAKE_SOURCE_DIR}/external/cJSON
)

target_link_libraries(mm2_core
    PUBLIC
        ZLIB::ZLIB
        Threads::Threads
        m
        spdlog::spdlog
        cjson
        flatbuffers::flatbuffers
)

target_compile_features(mm2_core PRIVATE cxx_std_11)
# Note: HAVE_KALLOC is defined globally in root CMakeLists.txt
set_target_properties(mm2_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Version info is compiled into mm_version.c via -D macros from cmake/version.cmake.
# Built as an OBJECT library so the .o is directly injected into every final link
# target (shared lib, CLI executables, Python extension).  CMake does not
# transitively propagate OBJECT targets through chains of static libraries,
# so each consumer must add $<TARGET_OBJECTS:mm2_version> explicitly.
add_library(mm2_version OBJECT src/mm_version.c)
target_include_directories(mm2_version PRIVATE
    ${CMAKE_SOURCE_DIR}/minimap2/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(mm2_version PRIVATE ${MM_VERSION_DEFINITIONS})
set_target_properties(mm2_version PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Profiling Support (ITT, ROCTX)
# ============================================================================
if(ENABLE_PROFILING)
    target_compile_definitions(mm2_core PUBLIC MM_PROFILE)
    
    # Intel ITT - prefer submodule, fallback to system
    if(EXISTS ${CMAKE_SOURCE_DIR}/external/ittapi/CMakeLists.txt)
        set(ITT_API_IPT_SUPPORT OFF CACHE BOOL "" FORCE)
        set(ITT_API_FORTRAN_SUPPORT OFF CACHE BOOL "" FORCE)
        add_subdirectory(${CMAKE_SOURCE_DIR}/external/ittapi ${CMAKE_BINARY_DIR}/external/ittapi)
        target_link_libraries(mm2_core PUBLIC ittnotify)
        target_include_directories(mm2_core PUBLIC ${CMAKE_SOURCE_DIR}/external/ittapi/include)
        target_compile_definitions(mm2_core PUBLIC MM_USE_ITT)
        set(HAVE_ITT TRUE CACHE BOOL "" FORCE)
        message(STATUS "ITT: using submodule")
    else()
        find_path(ITT_INCLUDE_DIR ittnotify.h
            HINTS $ENV{VTUNE_PROFILER_DIR}/include $ENV{ONEAPI_ROOT}/vtune/latest/include
                  /opt/intel/oneapi/vtune/latest/include /opt/intel/vtune_profiler/include)
        find_library(ITT_LIBRARY ittnotify
            HINTS $ENV{VTUNE_PROFILER_DIR}/lib64 $ENV{ONEAPI_ROOT}/vtune/latest/lib64
                  /opt/intel/oneapi/vtune/latest/lib64 /opt/intel/vtune_profiler/lib64)
        if(ITT_INCLUDE_DIR AND ITT_LIBRARY)
            target_link_libraries(mm2_core PUBLIC ${ITT_LIBRARY})
            target_include_directories(mm2_core PUBLIC ${ITT_INCLUDE_DIR})
            target_compile_definitions(mm2_core PUBLIC MM_USE_ITT)
            set(HAVE_ITT TRUE CACHE BOOL "" FORCE)
            message(STATUS "ITT: using system (${ITT_LIBRARY})")
        else()
            set(HAVE_ITT FALSE CACHE BOOL "" FORCE)
        endif()
    endif()
    
    # AMD ROCTX - HAVE_ROCTX set by top-level CMakeLists.txt
    # ROCTX is a general user-annotation API that works from any C/C++ code
    if(HAVE_ROCTX)
        target_compile_definitions(mm2_core PUBLIC MM_USE_ROCTX)
        target_link_libraries(mm2_core PUBLIC ${ROCTX_LIB})
        target_include_directories(mm2_core PRIVATE ${ROCM_PATH}/include)
        message(STATUS "ROCTX: enabled (linked)")
    endif()
    
    message(STATUS "Profiling: ITT=${HAVE_ITT} ROCTX=${HAVE_ROCTX}")
endif()

# ============================================================================
# Trace Logging Support
# ============================================================================
if(ENABLE_TRACE)
    target_compile_definitions(mm2_core PUBLIC MM_TRACE)
    if(NOT ENABLE_PROFILING)
        target_compile_definitions(mm2_core PUBLIC MM_PROFILE)
    endif()
    message(STATUS "Trace logging: enabled")
endif()
