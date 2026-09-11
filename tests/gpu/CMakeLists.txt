# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

# GPU-specific unit tests for chaining and alignment

add_executable(test_gpu
    test_gpu_main.cpp
    test_gpu_chain.cpp
    test_gpu_chain_context.cpp
    test_gpu_align.cpp
    test_gpu_align_context.cpp
    test_gpu_combined.cpp
    test_gpu_mapctx_concurrency.cpp
    # Reuse core test infrastructure (data loader, JSON loader)
    ${CMAKE_SOURCE_DIR}/tests/core/test_data_loader.cpp
    ${CMAKE_SOURCE_DIR}/tests/core/test_json_loader.cpp
)

target_link_libraries(test_gpu
    PRIVATE
    minimap2
    spdlog::spdlog
    GTest::gtest
)

target_include_directories(test_gpu
    PRIVATE
    ${MM2_INTERNAL_INCLUDES}
    ${CMAKE_SOURCE_DIR}/tests/core   # Reuse test fixtures and helpers
)

# Define test data and GPU config paths
target_compile_definitions(test_gpu PRIVATE
    TEST_SUITE_DIR="${CMAKE_SOURCE_DIR}/test_suite"
    TEST_GPU_CONFIG_FILE="${CMAKE_SOURCE_DIR}/configs/gpu_config.json"
)

# GPU-specific compile definitions
target_compile_definitions(test_gpu PRIVATE MM_ENABLE_HIP)

# Register GPU tests as a single CTest entry instead of per-test discovery.
# GPU initialization is expensive (~9.5s, ~35GB pinned memory) and must happen
# once per process. gtest_discover_tests would create 742 separate processes,
# each repeating init. Running the full binary as one test avoids this.
add_test(NAME gpu_tests
    COMMAND test_gpu
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
set_tests_properties(gpu_tests PROPERTIES
    TIMEOUT 600
    RESOURCE_LOCK gpu    # Prevent parallel execution with other GPU tests
)
