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

# CLI tools for staged minimap2 mapping
# Each executable performs one stage of the mapping pipeline
# CLI tools now work without minimap2 lib - they use mm2_seed/mm2_chain/mm2_align

# Use common internal include paths defined in top-level CMakeLists.txt

# Seed stage executable - uses mm2_seed (which includes mm2_core)
add_executable(mm2_seed_cli mm2_seed.cpp)
set_target_properties(mm2_seed_cli PROPERTIES OUTPUT_NAME mm2_seed)
target_link_libraries(mm2_seed_cli PRIVATE mm2_seed cjson)
target_sources(mm2_seed_cli PRIVATE $<TARGET_OBJECTS:mm2_version>)
target_include_directories(mm2_seed_cli PRIVATE ${MM2_INTERNAL_INCLUDES})

# Chain stage executable - uses mm_map_chain and the public
# mm_gpu_chain_ctx_* lifecycle from mm2_chain (chain_ctx.cpp).
add_executable(mm2_chain_cli mm2_chain.cpp)
set_target_properties(mm2_chain_cli PROPERTIES OUTPUT_NAME mm2_chain)
target_link_libraries(mm2_chain_cli PRIVATE mm2_chain cjson)
target_sources(mm2_chain_cli PRIVATE $<TARGET_OBJECTS:mm2_version>)
target_include_directories(mm2_chain_cli PRIVATE ${MM2_INTERNAL_INCLUDES})

# Align stage executable - uses mm2_align (which includes mm2_chain -> mm2_seed -> mm2_core)
add_executable(mm2_align_cli mm2_align.cpp)
set_target_properties(mm2_align_cli PROPERTIES OUTPUT_NAME mm2_align)
target_link_libraries(mm2_align_cli PRIVATE mm2_align cjson)
target_sources(mm2_align_cli PRIVATE $<TARGET_OBJECTS:mm2_version>)
target_include_directories(mm2_align_cli PRIVATE ${MM2_INTERNAL_INCLUDES})

# Set RPATH so installed binaries can find ROCm libraries
if(GPU STREQUAL "AMD")
    target_compile_definitions(mm2_chain_cli PRIVATE MM_ENABLE_HIP)
    target_compile_definitions(mm2_align_cli PRIVATE MM_ENABLE_HIP)
    set_target_properties(mm2_chain_cli mm2_align_cli PROPERTIES
        INSTALL_RPATH "${ROCM_PATH}/lib"
        BUILD_RPATH   "${ROCM_PATH}/lib"
    )
endif()

# Install targets (optional)
install(TARGETS mm2_seed_cli mm2_chain_cli mm2_align_cli
    RUNTIME DESTINATION bin
)

# Man pages (gzipped for Debian policy compliance)
foreach(tool mm2_seed mm2_chain mm2_align)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${tool}.1.gz
        COMMAND gzip -9 -c ${CMAKE_SOURCE_DIR}/man/${tool}.1
                > ${CMAKE_CURRENT_BINARY_DIR}/${tool}.1.gz
        DEPENDS ${CMAKE_SOURCE_DIR}/man/${tool}.1
        COMMENT "Compressing ${tool} man page"
    )
endforeach()
add_custom_target(cli_man_pages ALL
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/mm2_seed.1.gz
        ${CMAKE_CURRENT_BINARY_DIR}/mm2_chain.1.gz
        ${CMAKE_CURRENT_BINARY_DIR}/mm2_align.1.gz
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/mm2_seed.1.gz
    ${CMAKE_CURRENT_BINARY_DIR}/mm2_chain.1.gz
    ${CMAKE_CURRENT_BINARY_DIR}/mm2_align.1.gz
    DESTINATION ${CMAKE_INSTALL_MANDIR}/man1
)
