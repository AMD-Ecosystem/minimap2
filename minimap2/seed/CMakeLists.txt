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

# minimap2 Seeding Library
# Contains: minimizer sketching, seed collection, sdust filtering, index building
# Dependency chain: core -> seed

set(MM2_SEED_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/seed.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/seed_map.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sketch.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sdust.c
)

# Build as STATIC library
add_library(mm2_seed STATIC ${MM2_SEED_SOURCES})

target_include_directories(mm2_seed
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/minimap2/include>
)

# Link to core (inherits core's public includes and dependencies)
target_link_libraries(mm2_seed
    PUBLIC
        mm2_core
)

set_target_properties(mm2_seed PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# GPU Support (AMD HIP)
# ============================================================================
if(GPU STREQUAL "AMD" AND MM_ENABLE_SEED_GPU)
    set(MM2_SEED_GPU_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/seed_gpu.hip
    )
    
    # Build GPU object library
    add_library(mm2_seed_gpu OBJECT ${MM2_SEED_GPU_SOURCES})
    
    # Inherit includes from dependencies (core)
    target_link_libraries(mm2_seed_gpu PRIVATE mm2_core)
    
    target_include_directories(mm2_seed_gpu
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/minimap2/include  # Common headers (not PUBLIC anywhere)
            ${ROCM_PATH}/include                   # HIP runtime headers
    )
    
    # Use native HIP language (CMake 3.21+) or CXX with hipcc flags
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
        set_source_files_properties(${MM2_SEED_GPU_SOURCES} PROPERTIES LANGUAGE HIP)
        set_target_properties(mm2_seed_gpu PROPERTIES
            HIP_ARCHITECTURES "${GPUARCH}"
            POSITION_INDEPENDENT_CODE ON
        )
    else()
        set_source_files_properties(${MM2_SEED_GPU_SOURCES} PROPERTIES LANGUAGE CXX)
        set_target_properties(mm2_seed_gpu PROPERTIES
            LINKER_LANGUAGE CXX
            POSITION_INDEPENDENT_CODE ON
        )
        target_compile_options(mm2_seed_gpu PRIVATE ${GPU_FLAGS})
    endif()
    
    target_compile_definitions(mm2_seed_gpu PRIVATE MM_ENABLE_SEED_GPU)
    
    # Add GPU objects to main seed library
    target_sources(mm2_seed PRIVATE $<TARGET_OBJECTS:mm2_seed_gpu>)
    
    # Add MM_ENABLE_SEED_GPU to seed.c compilation
    target_compile_definitions(mm2_seed PRIVATE MM_ENABLE_SEED_GPU)
    
    # Set HIP_ARCHITECTURES on main target (required when linking HIP runtime)
    set_target_properties(mm2_seed PROPERTIES HIP_ARCHITECTURES "${GPUARCH}")
    
    # Link HIP runtime
    target_link_libraries(mm2_seed PUBLIC ${GPU_LIBS})
endif()
