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
 * JSON/FASTA Test Data Loader
 *
 * Provides functionality to load test data from JSON files at runtime.
 * Each JSON file describes a test case and references FASTA files for sequences.
 * This allows test data to be maintained separately from test code and
 * makes it easy to add new test cases without recompiling.
 *
 * By default, test data is loaded from test_suite/<category>/expected/<name>.json, but this can be overridden
 * by setting the TEST_DATA_DIR environment variable.
 *
 * JSON format:
 * {
 *   "name": "test-name",
 *   "description": "Test description",
 *   "reference": {
 *     "name": "ref_name",
 *     "file": "path/to/reference.fa"    // Path to FASTA file (relative to JSON or absolute)
 *   },
 *   "query": {
 *     "name": "query_name", 
 *     "file": "path/to/query.fa"        // Path to FASTA file (relative to JSON or absolute)
 *   },
 *   "index_params": {
 *     "k": 15,
 *     "w": 10
 *   },
 *   "expected_anchors": [
 *     {"x": 123, "y": 456},
 *     ...
 *   ],
 *   "expected_chains": [
 *     {"score": 100, "length": 10, "first_qpos": 0, "first_tpos": 0, "last_qpos": 100, "last_tpos": 100},
 *     ...
 *   ],
 *   "expected_alignments": [
 *     {"score": 100, "dp_score": 200, "qs": 0, "qe": 100, "rs": 0, "re": 100, "mapq": 60, "rev": 0, "n_cigar_ops": 10, "blen": 100, "mlen": 90, "dp_max": 210, "n_ambi": 0},
 *     ...
 *   ]
 * }
 * 
 * Note: For reference and query, use "file" to specify a FASTA file path.
 * Paths can be absolute or relative to the JSON file location.
 */
#ifndef TEST_DATA_LOADER_H
#define TEST_DATA_LOADER_H

#include "test_data_types.h"
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

// Forward declaration
struct cJSON;

/**
 * Runtime test data container.
 * Holds dynamically allocated test data loaded from JSON.
 * Stores file paths instead of sequences for large data support.
 */
class RuntimeTestData
{
      public:
	// Test identification
	std::string name;
	std::string description;

	// Reference sequence (file path)
	std::string ref_name;
	std::string ref_path; // Path to FASTA file

	// Query sequence (file path)
	std::string query_name;
	std::string query_path; // Path to FASTA file

	// Index parameters
	int k = 0; // 0 = use default
	int w = 0; // 0 = use default

	// Preset name (empty = no preset, use defaults)
	std::string preset; // e.g. "map-ont", "map-pb", "sr"

	// Query index within multi-query FASTA files (0-based)
	int query_index = 0;

	// Persistent storage for preset C-string used by MappingTestData
	mutable std::string preset_cstr_;

	// Expected anchors
	std::vector<ExpectedAnchor> expected_anchors;

	// Expected chains
	std::vector<ExpectedChain> expected_chains;

	// Expected alignments
	std::vector<ExpectedAlignment> expected_alignments;

	/**
     * Convert to MappingTestData for use with existing test infrastructure.
     * Note: The returned MappingTestData references internal data, so
     * this RuntimeTestData must outlive the returned object.
     */
	MappingTestData toMappingTestData() const;

	/**
     * Get k-mer size (returns default 15 if not specified).
     */
	int getKmerSize() const { return (k > 0) ? k : 15; }

	/**
     * Get minimizer window (returns default 10 if not specified).
     */
	int getMinimizerWindow() const { return (w > 0) ? w : 10; }
};

/**
 * Exception thrown when JSON loading fails.
 */
class TestDataLoadError : public std::runtime_error
{
      public:
	explicit TestDataLoadError(const std::string& message)
	    : std::runtime_error(message) {}
};

/**
 * Load test data from a JSON file.
 * For multi-query files, returns only the first query.
 * 
 * @param filepath Path to the JSON file
 * @return RuntimeTestData containing the loaded test data
 * @throws TestDataLoadError if loading fails
 */
RuntimeTestData loadTestDataFromJson(const std::string& filepath);

/**
 * Load all query entries from a JSON file.
 * For multi-query files, returns one RuntimeTestData per query.
 * For single-query files, returns a vector with one element.
 * 
 * @param filepath Path to the JSON file
 * @return Vector of RuntimeTestData, one per query
 * @throws TestDataLoadError if loading fails
 */
std::vector<RuntimeTestData> loadAllQueriesFromJson(const std::string& filepath);

/**
 * Load test data from a JSON string.
 * For multi-query strings, returns only the first query.
 * 
 * @param json_str JSON string containing test data
 * @return RuntimeTestData containing the loaded test data
 * @throws TestDataLoadError if parsing fails
 */
RuntimeTestData loadTestDataFromJsonString(const std::string& json_str);

/**
 * Load all test data files from a directory.
 * Loads all .json files in the specified directory.
 * 
 * @param dirpath Path to the directory containing JSON test data files
 * @return Vector of RuntimeTestData
 * @throws TestDataLoadError if any file fails to load
 */
std::vector<RuntimeTestData> loadAllTestDataFromDirectory(const std::string& dirpath);

/**
 * Get the path to the test data directory.
 * Uses TEST_DATA_DIR environment variable if set, otherwise
 * returns a default path relative to the build directory.
 */
std::string getTestDataDirectory();

/**
 * Registry of loaded test data for parameterized tests.
 * Singleton pattern to ensure test data is loaded only once.
 */
class TestDataRegistry
{
      public:
	static TestDataRegistry& getInstance();

	/**
     * Load all test data from the test data directory.
     * Safe to call multiple times - only loads once.
     */
	void loadAll();

	/**
     * Get all loaded test data.
     */
	const std::vector<RuntimeTestData>& getAllTestData() const { return test_data_; }

	/**
     * Get pointers to MappingTestData for parameterized tests.
     * The returned pointers are valid for the lifetime of the registry.
     */
	std::vector<const MappingTestData*> getMappingTestDataPointers();

	/**
     * Get number of loaded test data sets.
     */
	size_t size() const { return test_data_.size(); }

	/**
     * Check if test data has been loaded.
     */
	bool isLoaded() const { return loaded_; }

      private:
	TestDataRegistry() = default;
	TestDataRegistry(const TestDataRegistry&) = delete;
	TestDataRegistry& operator=(const TestDataRegistry&) = delete;

	std::vector<RuntimeTestData> test_data_;
	std::vector<MappingTestData> mapping_test_data_; // Cached conversions
	bool loaded_ = false;
};

#endif // TEST_DATA_LOADER_H
