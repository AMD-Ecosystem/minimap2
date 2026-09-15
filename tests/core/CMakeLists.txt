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

# Core minimap2 functionality tests

# Test for basic data structures and utilities
# Test data is loaded from JSON and FASTA files in test_suite/*/expected/ and test_suite/original/
add_executable(test_core
    test_kalloc.cpp
    test_map_seed.cpp
    test_map_chain.cpp
    test_map_align.cpp
    test_map.cpp
    test_data_loader.cpp
    test_json_loader.cpp
    test_sr_regression.cpp
    test_sam_hdr.cpp
)

target_link_libraries(test_core
    PRIVATE
    minimap2
    spdlog::spdlog
    GTest::gtest_main
)

target_include_directories(test_core PRIVATE ${MM2_INTERNAL_INCLUDES})

# Set directories for tests - test data is now in test_suite/*/expected/
target_compile_definitions(test_core PRIVATE
    TEST_SUITE_DIR="${CMAKE_SOURCE_DIR}/test_suite"
)

# Discover tests
include(GoogleTest)
gtest_discover_tests(test_core
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DISCOVERY_TIMEOUT 60
    NO_PRETTY_VALUES
)
