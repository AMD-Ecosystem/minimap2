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

# minimap2 Library CMakeLists.txt
# This file builds libminimap2 and the main executable
# Dependency chain: core -> seed -> chain -> align -> libminimap2

# ============================================================================
# Logging Library (C++ wrapper around spdlog for C code)
# ============================================================================
# Use spdlog from submodule - MUST be built before component libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/external/spdlog ${CMAKE_BINARY_DIR}/external/spdlog)

# ============================================================================
# Component Libraries (Static Libraries)
# ============================================================================
# Each component is built as a separate static library in its own subdirectory.

# Add component subdirectories (order matters for dependencies)
# 1. Core - foundational utilities, no dependencies on other mm2 components
add_subdirectory(core)

# 2. Seed - depends on core
add_subdirectory(seed)

# 3. Chain - depends on seed (which depends on core)
add_subdirectory(chain)

# 4. Align - depends on chain (which depends on seed, core)
add_subdirectory(align)

# ============================================================================
# Main minimap2 Library (combines all component libraries)
# ============================================================================
# Build libminimap2.so - combines remaining sources with all component static libraries
# Sources here: map.c (high-level mapping orchestration)
# Also includes: format.c, splitidx.c (minimap2-level utilities)
# Note: esterr.c is now in mm2_align library, options.c is in mm2_core library

set(MM2_MAIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/map.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/format.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/splitidx.c
)

# map_ctx.cpp uses gpu_device_descriptor.h which requires HIP headers.
# Compile it via hipcc on AMD builds using the same OBJECT library pattern
# used by mm2_chain_gpu_objects and mm2_align_gpu_objects.
if(GPU STREQUAL "AMD")
    add_library(mm2_map_gpu_objects OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/src/map_ctx.cpp
    )
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/map_ctx.cpp
        PROPERTIES LANGUAGE HIP
    )
    set_target_properties(mm2_map_gpu_objects PROPERTIES
        HIP_STANDARD 17
        HIP_STANDARD_REQUIRED ON
    )
    target_compile_options(mm2_map_gpu_objects PRIVATE ${GPU_FLAGS})
    target_compile_definitions(mm2_map_gpu_objects PRIVATE MM_ENABLE_HIP)
    target_include_directories(mm2_map_gpu_objects PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core/include
        ${CMAKE_CURRENT_SOURCE_DIR}/seed/include
        ${CMAKE_CURRENT_SOURCE_DIR}/chain/include
        ${CMAKE_CURRENT_SOURCE_DIR}/chain/include/gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/align/include
        ${CMAKE_SOURCE_DIR}/external/spdlog/include
        ${CMAKE_SOURCE_DIR}/external/cJSON
        ${ROCM_PATH}/include
    )
else()
    list(APPEND MM2_MAIN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/map_ctx.cpp)
endif()

add_library(minimap2 SHARED 
    ${MM2_MAIN_SOURCES}
    $<TARGET_OBJECTS:cjson>
    $<TARGET_OBJECTS:mm2_version>
)

if(GPU STREQUAL "AMD")
    target_sources(minimap2 PRIVATE $<TARGET_OBJECTS:mm2_map_gpu_objects>)
endif()

# Set public header for the library
set_target_properties(minimap2 PROPERTIES 
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/minimap.h
)

# Set RPATH so installed binaries can find ROCm libraries under /opt/rocm/lib
# without requiring LD_LIBRARY_PATH or an ldconfig entry.
if(GPU STREQUAL "AMD")
    set_target_properties(minimap2 PROPERTIES
        INSTALL_RPATH "${ROCM_PATH}/lib"
        BUILD_RPATH   "${ROCM_PATH}/lib"
    )
endif()

# Include directories for libminimap2
target_include_directories(minimap2 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/minimap2>
    PRIVATE
        # Implementation includes for compiling map.c, pe.c, esterr.c
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/seed/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/chain/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/chain/include/gpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/align/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external/cJSON>
)

# Link component libraries (chain of dependencies: align->chain->seed->core)
# Only need to link mm2_align since it transitively pulls in all others
# PRIVATE linkage embeds static libs into shared lib, avoiding export issues
target_link_libraries(minimap2 
    PRIVATE 
        mm2_align
    PUBLIC
        ${GPU_LIBS}
)

# Add GPU-specific compile definitions for map.c which uses MM_ENABLE_HIP guards
if(GPU STREQUAL "AMD")
    target_compile_definitions(minimap2 PRIVATE MM_ENABLE_HIP)
endif()

# ============================================================================
# Main Executable
# ============================================================================

add_executable(minimap2_tool ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c)
# Add GPU-specific compile definitions for main.c which uses MM_ENABLE_HIP guards
if(GPU STREQUAL "AMD")
    target_compile_definitions(minimap2_tool PRIVATE MM_ENABLE_HIP)
endif()
if(SUFFIX)
    set_target_properties(minimap2_tool PROPERTIES OUTPUT_NAME minimap2${SUFFIX})
else()
    set_target_properties(minimap2_tool PROPERTIES OUTPUT_NAME minimap2)
endif()
# Ensure executable goes to bin/ to avoid conflict with minimap2/ directory
set_target_properties(minimap2_tool PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
# Set RPATH so the installed executable can find libminimap2.so and ROCm libs
if(GPU STREQUAL "AMD")
    set_target_properties(minimap2_tool PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR};${ROCM_PATH}/lib"
        BUILD_RPATH   "${CMAKE_BINARY_DIR}/minimap2;${ROCM_PATH}/lib"
    )
else()
    set_target_properties(minimap2_tool PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        BUILD_RPATH   "${CMAKE_BINARY_DIR}/minimap2"
    )
endif()
# Need internal headers not exposed by libminimap2
target_include_directories(minimap2_tool PRIVATE ${MM2_INTERNAL_INCLUDES})
target_link_libraries(minimap2_tool PRIVATE minimap2)
if(GPU STREQUAL "AMD")
    target_link_libraries(minimap2_tool PRIVATE stdc++)
endif()

# ============================================================================
# Extra Programs (optional)
# ============================================================================

if(BUILD_EXTRA)
    # minimap2-lite
    add_executable(minimap2-lite ${CMAKE_CURRENT_SOURCE_DIR}/src/example.c)
    target_include_directories(minimap2-lite PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core/include
    )
    target_link_libraries(minimap2-lite PRIVATE minimap2 ZLIB::ZLIB)
    if(GPU STREQUAL "AMD")
        target_link_libraries(minimap2-lite PRIVATE stdc++)
    endif()
    
    # sdust
    add_executable(sdust ${CMAKE_CURRENT_SOURCE_DIR}/seed/src/sdust.c)
    target_sources(sdust PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/core/src/kalloc.c)
    target_compile_definitions(sdust PRIVATE _SDUST_MAIN)
    target_include_directories(sdust PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/core/include
        ${CMAKE_CURRENT_SOURCE_DIR}/seed/include
    )
    target_link_libraries(sdust PRIVATE ZLIB::ZLIB)
endif()
