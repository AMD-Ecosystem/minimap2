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

# minimap2 Alignment Library
# Contains: KSW2 alignment, CIGAR generation, alignment scoring
# Dependency chain: core -> seed -> chain -> align

# ============================================================================
# KSW2 SSE Objects - compiled separately with specific CPU flags
# These are created as OBJECT libraries for inclusion in the align library
# KSW2 objects link to mm2_core to get kalloc.h and other core headers
# ============================================================================

set(KSW2_INCLUDE_DIRS 
    ${CMAKE_SOURCE_DIR}/minimap2/include
    ${CMAKE_SOURCE_DIR}/minimap2/align/include
)

if(ARM_NEON)
    # ARM NEON build: compile the SSE kernels through the sse2neon shim
    # (mirrors the Makefile arm_neon / aarch64 targets). KSW_SSE2_ONLY + __SSE2__
    # select the SSE2 code path, which sse2neon translates to NEON. No CPU
    # dispatch object is built (that is x86-only).
    set(KSW2_NEON_DEFS KSW_SSE2_ONLY __SSE2__)
    set(KSW2_NEON_INCLUDES ${KSW2_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/sse2neon)

    add_library(ksw2_extz2_sse OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/ksw2_extz2_sse.c)
    target_compile_definitions(ksw2_extz2_sse PRIVATE ${KSW2_NEON_DEFS})
    target_include_directories(ksw2_extz2_sse PRIVATE ${KSW2_NEON_INCLUDES})
    target_link_libraries(ksw2_extz2_sse PRIVATE mm2_core)

    add_library(ksw2_extd2_sse OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/ksw2_extd2_sse.c)
    target_compile_definitions(ksw2_extd2_sse PRIVATE ${KSW2_NEON_DEFS})
    target_include_directories(ksw2_extd2_sse PRIVATE ${KSW2_NEON_INCLUDES})
    target_link_libraries(ksw2_extd2_sse PRIVATE mm2_core)

    add_library(ksw2_exts2_sse OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/ksw2_exts2_sse.c)
    target_compile_definitions(ksw2_exts2_sse PRIVATE ${KSW2_NEON_DEFS})
    target_include_directories(ksw2_exts2_sse PRIVATE ${KSW2_NEON_INCLUDES})
    target_link_libraries(ksw2_exts2_sse PRIVATE mm2_core)

    # ksw2_ll_sse is compiled through the sse2neon shim too (no -msse2 on ARM)
    add_library(ksw2_ll_sse OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/ksw2_ll_sse.c)
    target_include_directories(ksw2_ll_sse PRIVATE ${KSW2_NEON_INCLUDES})
    target_link_libraries(ksw2_ll_sse PRIVATE mm2_core)

    set(KSW2_OBJECTS
        $<TARGET_OBJECTS:ksw2_extz2_sse>
        $<TARGET_OBJECTS:ksw2_extd2_sse>
        $<TARGET_OBJECTS:ksw2_exts2_sse>
        $<TARGET_OBJECTS:ksw2_ll_sse>
    )
elseif(SSE2_ONLY)
    # SSE2 only versions
    add_library(ksw2_extz2_sse OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/ksw2_extz2_sse.c)
    target_compile_options(ksw2_extz2_sse PRIVATE -msse2)
    target_include_directories(ksw2_extz2_sse PRIVATE ${KSW2_INCLUDE_DIRS})
    target_link_libraries(ksw2_extz2_sse PRIVATE mm2_core)
    
    add_library(ksw2_extd2_sse OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/ksw2_extd2_sse.c)
    target_compile_options(ksw2_extd2_sse PRIVATE -msse2)
    target_include_directories(ksw2_extd2_sse PRIVATE ${KSW2_INCLUDE_DIRS})
    target_link_libraries(ksw2_extd2_sse PRIVATE mm2_core)
    
    add_library(ksw2_exts2_sse OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/ksw2_exts2_sse.c)
    target_compile_options(ksw2_exts2_sse PRIVATE -msse2)
    target_include_directories(ksw2_exts2_sse PRIVATE ${KSW2_INCLUDE_DIRS})
    target_link_libraries(ksw2_exts2_sse PRIVATE mm2_core)
    
    # ksw2_ll_sse with SSE2
    add_library(ksw2_ll_sse OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/ksw2_ll_sse.c)
    target_compile_options(ksw2_ll_sse PRIVATE -msse2)
    target_include_directories(ksw2_ll_sse PRIVATE ${KSW2_INCLUDE_DIRS})
    target_link_libraries(ksw2_ll_sse PRIVATE mm2_core)
    
    set(KSW2_OBJECTS 
        $<TARGET_OBJECTS:ksw2_extz2_sse>
        $<TARGET_OBJECTS:ksw2_extd2_sse>
        $<TARGET_OBJECTS:ksw2_exts2_sse>
        $<TARGET_OBJECTS:ksw2_ll_sse>
    )
else()
    # CPU dispatch with SSE2 and SSE4.1 versions
    # SSE4.1 versions
    add_library(ksw2_extz2_sse41 OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/ksw2_extz2_sse.c)
    target_compile_options(ksw2_extz2_sse41 PRIVATE -msse4.1)
    target_compile_definitions(ksw2_extz2_sse41 PRIVATE KSW_CPU_DISPATCH)
    target_include_directories(ksw2_extz2_sse41 PRIVATE ${KSW2_INCLUDE_DIRS})
    target_link_libraries(ksw2_extz2_sse41 PRIVATE mm2_core)
    
    add_library(ksw2_extd2_sse41 OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/ksw2_extd2_sse.c)
    target_compile_options(ksw2_extd2_sse41 PRIVATE -msse4.1)
    target_compile_definitions(ksw2_extd2_sse41 PRIVATE KSW_CPU_DISPATCH)
    target_include_directories(ksw2_extd2_sse41 PRIVATE ${KSW2_INCLUDE_DIRS})
    target_link_libraries(ksw2_extd2_sse41 PRIVATE mm2_core)
    
    add_library(ksw2_exts2_sse41 OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/ksw2_exts2_sse.c)
    target_compile_options(ksw2_exts2_sse41 PRIVATE -msse4.1)
    target_compile_definitions(ksw2_exts2_sse41 PRIVATE KSW_CPU_DISPATCH)
    target_include_directories(ksw2_exts2_sse41 PRIVATE ${KSW2_INCLUDE_DIRS})
    target_link_libraries(ksw2_exts2_sse41 PRIVATE mm2_core)
    
    # SSE2 versions
    add_library(ksw2_extz2_sse2 OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/ksw2_extz2_sse.c)
    target_compile_options(ksw2_extz2_sse2 PRIVATE -msse2 -mno-sse4.1)
    target_compile_definitions(ksw2_extz2_sse2 PRIVATE KSW_CPU_DISPATCH KSW_SSE2_ONLY)
    target_include_directories(ksw2_extz2_sse2 PRIVATE ${KSW2_INCLUDE_DIRS})
    target_link_libraries(ksw2_extz2_sse2 PRIVATE mm2_core)
    
    add_library(ksw2_extd2_sse2 OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/ksw2_extd2_sse.c)
    target_compile_options(ksw2_extd2_sse2 PRIVATE -msse2 -mno-sse4.1)
    target_compile_definitions(ksw2_extd2_sse2 PRIVATE KSW_CPU_DISPATCH KSW_SSE2_ONLY)
    target_include_directories(ksw2_extd2_sse2 PRIVATE ${KSW2_INCLUDE_DIRS})
    target_link_libraries(ksw2_extd2_sse2 PRIVATE mm2_core)
    
    add_library(ksw2_exts2_sse2 OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/ksw2_exts2_sse.c)
    target_compile_options(ksw2_exts2_sse2 PRIVATE -msse2 -mno-sse4.1)
    target_compile_definitions(ksw2_exts2_sse2 PRIVATE KSW_CPU_DISPATCH KSW_SSE2_ONLY)
    target_include_directories(ksw2_exts2_sse2 PRIVATE ${KSW2_INCLUDE_DIRS})
    target_link_libraries(ksw2_exts2_sse2 PRIVATE mm2_core)
    
    # Dispatch object
    add_library(ksw2_dispatch OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/ksw2_dispatch.c)
    target_compile_options(ksw2_dispatch PRIVATE -msse4.1)
    target_compile_definitions(ksw2_dispatch PRIVATE KSW_CPU_DISPATCH)
    target_include_directories(ksw2_dispatch PRIVATE ${KSW2_INCLUDE_DIRS})
    target_link_libraries(ksw2_dispatch PRIVATE mm2_core)
    
    # ksw2_ll_sse with SSE2
    add_library(ksw2_ll_sse OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/ksw2_ll_sse.c)
    target_compile_options(ksw2_ll_sse PRIVATE -msse2)
    target_include_directories(ksw2_ll_sse PRIVATE ${KSW2_INCLUDE_DIRS})
    target_link_libraries(ksw2_ll_sse PRIVATE mm2_core)
    
    set(KSW2_OBJECTS 
        $<TARGET_OBJECTS:ksw2_extz2_sse41>
        $<TARGET_OBJECTS:ksw2_extd2_sse41>
        $<TARGET_OBJECTS:ksw2_exts2_sse41>
        $<TARGET_OBJECTS:ksw2_extz2_sse2>
        $<TARGET_OBJECTS:ksw2_extd2_sse2>
        $<TARGET_OBJECTS:ksw2_exts2_sse2>
        $<TARGET_OBJECTS:ksw2_dispatch>
        $<TARGET_OBJECTS:ksw2_ll_sse>
    )
endif()

# ============================================================================
# Alignment Library - includes align.c, esterr.c and KSW2 objects
# ============================================================================

set(MM2_ALIGN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/align.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/jump.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/align_ctx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/esterr.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hit.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pe.c
)

add_library(mm2_align STATIC 
    ${MM2_ALIGN_SOURCES}
    ${KSW2_OBJECTS}
)

target_include_directories(mm2_align
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/minimap2/include>
)

# Link to chain (inherits chain's, seed's and core's public includes and dependencies)
target_link_libraries(mm2_align
    PUBLIC
        mm2_chain
)

set_target_properties(mm2_align PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# GPU Alignment Support (AMD HIP)
# ============================================================================
if(GPU STREQUAL "AMD")
    # Collect GPU source files (.cpp files compiled as HIP)
    file(GLOB MM2_ALIGN_GPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/*.cpp")
    # align_ctx.cpp includes gpu_device_descriptor.h; compile this TU with HIP too.
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/align_ctx.cpp PROPERTIES LANGUAGE HIP)
    
    # Build GPU object library using modern CMake HIP support
    add_library(mm2_align_gpu_objects OBJECT ${MM2_ALIGN_GPU_SOURCES})
    
    # Use native HIP language (CMake 3.21+) or CXX with hipcc flags
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
        set_source_files_properties(${MM2_ALIGN_GPU_SOURCES} PROPERTIES LANGUAGE HIP)
        set_target_properties(mm2_align_gpu_objects PROPERTIES
            HIP_ARCHITECTURES "${GPUARCH}"
        )
    else()
        set_target_properties(mm2_align_gpu_objects PROPERTIES
            LINKER_LANGUAGE CXX
        )
        target_compile_options(mm2_align_gpu_objects PRIVATE ${GPU_FLAGS})
    endif()
    
    # Inherit includes from dependencies (chain -> seed -> core, cjson)
    target_link_libraries(mm2_align_gpu_objects PRIVATE mm2_chain cjson)
    
    target_include_directories(mm2_align_gpu_objects
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu
            ${CMAKE_SOURCE_DIR}/minimap2/include  # Common headers (not PUBLIC anywhere)
            ${ROCM_PATH}/include                   # HIP runtime headers
    )
    
    target_compile_definitions(mm2_align_gpu_objects PRIVATE
        MM_ENABLE_HIP
        USEHIP
        HAVE_GPU_ALIGNMENT
    )
    # Note: HAVE_KALLOC and DEBUG_* definitions are inherited from root CMakeLists.txt
    
    set_target_properties(mm2_align_gpu_objects PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    
    # Add GPU objects to main align library
    target_sources(mm2_align PRIVATE $<TARGET_OBJECTS:mm2_align_gpu_objects>)
    
    # HAVE_GPU_ALIGNMENT means GPU code is compiled and available
    # Runtime flag MM_F_GPU_ALIGN controls whether to use it
    target_compile_definitions(mm2_align PUBLIC HAVE_GPU_ALIGNMENT)
    
    # Set HIP_ARCHITECTURES on main target (required when linking HIP runtime)
    set_target_properties(mm2_align PROPERTIES HIP_ARCHITECTURES "${GPUARCH}")
    
    # Link GPU runtime libraries (conditional - only when GPU is enabled)
    target_link_libraries(mm2_align PUBLIC ${GPU_LIBS})
endif()
