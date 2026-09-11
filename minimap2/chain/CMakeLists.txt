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

# minimap2 Chaining Library
# Contains: anchor chaining (DP, RMQ), hit processing, GPU chaining
# Dependency chain: core -> seed -> chain
#
# Public API: mm2_chain.h with mm_map_chain() entry point

set(MM2_CHAIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/chain_map.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lchain.c
)

# Build as STATIC library
add_library(mm2_chain STATIC ${MM2_CHAIN_SOURCES})

target_include_directories(mm2_chain
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/gpu>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/minimap2/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external/cJSON>
)

# Link to seed (inherits seed's and core's public includes and dependencies)
target_link_libraries(mm2_chain
    PUBLIC
        mm2_seed
)

set_target_properties(mm2_chain PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# GPU Chaining Support (AMD HIP)
# ============================================================================
if(GPU STREQUAL "AMD")
    target_compile_definitions(mm2_chain PRIVATE MM_ENABLE_HIP)

    # Collect GPU source files
    file(GLOB MM2_CHAIN_GPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/*.cu")
    list(APPEND MM2_CHAIN_GPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/gpu_device_descriptor.cpp")
    list(APPEND MM2_CHAIN_GPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/chain_ctx.cpp")
    
    # Add debug.c if debug mode is analyze or verbose
    if(DEBUG_MODE STREQUAL "analyze" OR DEBUG_MODE STREQUAL "verbose")
        list(APPEND MM2_CHAIN_GPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/debug.c")
        message(STATUS "Chain GPU debug support enabled (including debug.c)")
    endif()
    
    # Build GPU object library using modern CMake HIP support
    add_library(mm2_chain_gpu_objects OBJECT ${MM2_CHAIN_GPU_SOURCES})
    
    # Use native HIP language (CMake 3.21+) or CXX with hipcc flags
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
        set_source_files_properties(${MM2_CHAIN_GPU_SOURCES} PROPERTIES LANGUAGE HIP)
        set_target_properties(mm2_chain_gpu_objects PROPERTIES
            HIP_ARCHITECTURES "${GPUARCH}"
        )
    else()
        set_target_properties(mm2_chain_gpu_objects PROPERTIES
            LINKER_LANGUAGE CXX
        )
        target_compile_options(mm2_chain_gpu_objects PRIVATE ${GPU_FLAGS})
    endif()
    
    # Inherit includes from dependencies (seed -> core, cjson)
    target_link_libraries(mm2_chain_gpu_objects PRIVATE mm2_seed cjson)
    
    target_include_directories(mm2_chain_gpu_objects
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu
            ${CMAKE_SOURCE_DIR}/minimap2/include  # Common headers (not PUBLIC anywhere)
            ${ROCM_PATH}/include                   # HIP runtime headers
    )
    
    target_compile_definitions(mm2_chain_gpu_objects PRIVATE
        MM_ENABLE_HIP
        USEHIP
    )
    # Note: HAVE_KALLOC and DEBUG_* definitions are inherited from root CMakeLists.txt
    
    set_target_properties(mm2_chain_gpu_objects PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    
    # Add GPU objects to main chain library
    target_sources(mm2_chain PRIVATE $<TARGET_OBJECTS:mm2_chain_gpu_objects>)
    
    # Set HIP_ARCHITECTURES on main target (required when linking HIP runtime)
    set_target_properties(mm2_chain PROPERTIES HIP_ARCHITECTURES "${GPUARCH}")
    
    # Link GPU runtime libraries
    target_link_libraries(mm2_chain PUBLIC ${GPU_LIBS})
endif()

# chain_ctx.cpp builds on all targets (CPU stub path when MM_ENABLE_HIP absent)
if(NOT GPU STREQUAL "AMD")
    target_sources(mm2_chain PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/chain_ctx.cpp)
endif()
