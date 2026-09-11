/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * Tests for JSON test data loader functionality.
 * 
 * These tests verify that test data can be correctly loaded from JSON files
 * at runtime, parsed correctly, and used in the test infrastructure.
 */
#include <gtest/gtest.h>
#include "test_data_loader.h"
#include <fstream>

// ============================================================================
// JSON Parsing Tests
// ============================================================================

class JsonLoaderTest : public ::testing::Test
{
      protected:
	void SetUp() override {}
	void TearDown() override {}
};

// Test loading from a valid JSON string with inline sequences
TEST_F(JsonLoaderTest, LoadFromValidJsonString)
{
	const char* json = R"({
        "name": "test-case",
        "description": "A test case",
        "reference": {
            "name": "ref1",
            "file": "../test_ref.fa"
        },
        "query": {
            "name": "query1",
            "file": "../test_query.fa"
        },
        "index_params": {
            "k": 15,
            "w": 10
        },
        "expected_anchors": [
            {"x": 100, "y": 50},
            {"x": 200, "y": 150}
        ],
        "expected_chains": [
            {"score": 500, "length": 10, "first_qpos": 0, "first_tpos": 0, "last_qpos": 100, "last_tpos": 100}
        ],
        "expected_alignments": [
            {"score": 500, "dp_score": 1000, "qs": 0, "qe": 8, "rs": 0, "re": 12, "mapq": 60, "rev": 0, "n_cigar_ops": 5, "blen": 12, "mlen": 8, "dp_max": 1050, "n_ambi": 0}
        ]
    })";

	RuntimeTestData data = loadTestDataFromJsonString(json);

	EXPECT_EQ(data.name, "test-case");
	EXPECT_EQ(data.description, "A test case");
	EXPECT_EQ(data.ref_name, "ref1");
	EXPECT_NE(data.ref_path.find("test_ref.fa"), std::string::npos);
	EXPECT_EQ(data.query_name, "query1");
	EXPECT_NE(data.query_path.find("test_query.fa"), std::string::npos);
	EXPECT_EQ(data.k, 15);
	EXPECT_EQ(data.w, 10);

	ASSERT_EQ(data.expected_anchors.size(), 2);
	EXPECT_EQ(data.expected_anchors[0].x, 100);
	EXPECT_EQ(data.expected_anchors[0].y, 50);
	EXPECT_EQ(data.expected_anchors[1].x, 200);
	EXPECT_EQ(data.expected_anchors[1].y, 150);

	ASSERT_EQ(data.expected_chains.size(), 1);
	EXPECT_EQ(data.expected_chains[0].score, 500);
	EXPECT_EQ(data.expected_chains[0].length, 10);

	ASSERT_EQ(data.expected_alignments.size(), 1);
	EXPECT_EQ(data.expected_alignments[0].score, 500);
	EXPECT_EQ(data.expected_alignments[0].dp_score, 1000);
	EXPECT_EQ(data.expected_alignments[0].mapq, 60);
	EXPECT_EQ(data.expected_alignments[0].n_cigar_ops, 5);
	EXPECT_EQ(data.expected_alignments[0].blen, 12);
	EXPECT_EQ(data.expected_alignments[0].mlen, 8);
	EXPECT_EQ(data.expected_alignments[0].dp_max, 1050);
	EXPECT_EQ(data.expected_alignments[0].n_ambi, 0);
}

// Test loading with missing optional fields
TEST_F(JsonLoaderTest, LoadWithOptionalFieldsMissing)
{
	const char* json = R"({
        "name": "minimal-test",
        "reference": {
            "name": "ref1",
            "file": "../ref.fa"
        },
        "query": {
            "name": "query1",
            "file": "../query.fa"
        }
    })";

	RuntimeTestData data = loadTestDataFromJsonString(json);

	EXPECT_EQ(data.name, "minimal-test");
	EXPECT_EQ(data.description, "");
	EXPECT_EQ(data.k, 0); // Default
	EXPECT_EQ(data.w, 0); // Default
	EXPECT_EQ(data.expected_anchors.size(), 0);
	EXPECT_EQ(data.expected_chains.size(), 0);
	EXPECT_EQ(data.expected_alignments.size(), 0);
}

// Test error handling for invalid JSON
TEST_F(JsonLoaderTest, ErrorOnInvalidJson)
{
	const char* invalid_json = "{ this is not valid json }";

	EXPECT_THROW(loadTestDataFromJsonString(invalid_json), TestDataLoadError);
}

// Test error handling for missing required fields
TEST_F(JsonLoaderTest, ErrorOnMissingRequiredFields)
{
	// Missing "name" field
	const char* json = R"({
        "reference": {
            "name": "ref1",
            "file": "../ref.fa"
        },
        "query": {
            "name": "query1",
            "file": "../query.fa"
        }
    })";

	EXPECT_THROW(loadTestDataFromJsonString(json), TestDataLoadError);
}

// Test conversion to MappingTestData
TEST_F(JsonLoaderTest, ConversionToMappingTestData)
{
	const char* json = R"({
        "name": "conversion-test",
        "description": "Test conversion",
        "reference": {
            "name": "ref1",
            "file": "../ref.fa"
        },
        "query": {
            "name": "query1",
            "file": "../query.fa"
        },
        "expected_chains": [
            {"score": 100, "length": 5, "first_qpos": 0, "first_tpos": 0, "last_qpos": 8, "last_tpos": 12}
        ]
    })";

	RuntimeTestData runtime_data = loadTestDataFromJsonString(json);
	MappingTestData mapping_data = runtime_data.toMappingTestData();

	EXPECT_STREQ(mapping_data.name, "conversion-test");
	EXPECT_STREQ(mapping_data.description, "Test conversion");
	EXPECT_NE(mapping_data.ref_path, nullptr);
	EXPECT_NE(mapping_data.query_path, nullptr);
	EXPECT_EQ(mapping_data.n_expected_chains, 1);
	EXPECT_NE(mapping_data.expected_chains, nullptr);
	EXPECT_EQ(mapping_data.expected_chains[0].score, 100);
}

// ============================================================================
// Test Data Registry Tests
// ============================================================================

TEST_F(JsonLoaderTest, RegistryLoadAndAccess)
{
	TestDataRegistry& registry = TestDataRegistry::getInstance();
	registry.loadAll();

	// Should have loaded at least the mt_human_orang.json
	if (registry.isLoaded() && registry.size() > 0) {
		auto pointers = registry.getMappingTestDataPointers();
		EXPECT_GT(pointers.size(), 0);

		// Verify first test data has expected properties
		const MappingTestData* first = pointers[0];
		EXPECT_NE(first, nullptr);
		EXPECT_NE(first->name, nullptr);
		EXPECT_NE(first->ref_path, nullptr);
		EXPECT_NE(first->query_path, nullptr);
	}
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(JsonLoaderTest, LoadMtHumanOrangFromDirectory)
{
	std::string test_suite_dir = getTestDataDirectory();

	// Try to load using the registry (which scans test_suite/*/expected)
	try {
		TestDataRegistry& registry = TestDataRegistry::getInstance();
		registry.loadAll();

		// We expect at least one test data file (mt_human_orang.json)
		EXPECT_GT(registry.size(), 0) << "Expected to load test data from " << test_suite_dir << "/*/expected/";

		// Get all test data and verify the mt_human_orang test data
		auto all_data = registry.getMappingTestDataPointers();

		bool found_mt_human_orang = false;
		for (const auto* data : all_data) {
			if (std::string(data->name) == "MT_human_vs_orang") {
				found_mt_human_orang = true;

				// Verify key properties
				EXPECT_STREQ(data->ref_name, "MT_human");
				EXPECT_STREQ(data->query_name, "MT_orang");
				// File paths should be set
				EXPECT_NE(data->ref_path, nullptr);
				EXPECT_NE(data->query_path, nullptr);
				EXPECT_GT(data->n_expected_anchors, 300);
				EXPECT_EQ(data->n_expected_chains, 1);
				EXPECT_EQ(data->n_expected_alignments, 1);

				// Verify chain values
				EXPECT_EQ(data->expected_chains[0].score, 3187);
				EXPECT_EQ(data->expected_chains[0].length, 342);

				// Verify alignment values
				EXPECT_EQ(data->expected_alignments[0].score, 3187);
				EXPECT_EQ(data->expected_alignments[0].dp_score, 18198);
				EXPECT_EQ(data->expected_alignments[0].mapq, 60);

				break;
			}
		}

		if (!found_mt_human_orang) {
			GTEST_SKIP() << "MT_human_vs_orang test data not found in test suite";
		}

	} catch (const TestDataLoadError& e) {
		// Directory might not exist in some test environments
		GTEST_SKIP() << "Test data directory not available: " << e.what();
	}
}
