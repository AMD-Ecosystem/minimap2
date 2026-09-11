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
 * JSON Test Data Loader Implementation
 */
#include "test_data_loader.h"
#include "mm_log.h"

extern "C" {
#include "cJSON.h"
}

#include <fstream>
#include <sstream>
#include <cstdlib>
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <cstring>

// ============================================================================
// Helper functions
// ============================================================================

namespace
{

/**
 * Read entire file contents into a string.
 */
std::string readFileContents(const std::string& filepath)
{
	std::ifstream file(filepath, std::ios::binary);
	if (!file.is_open()) {
		throw TestDataLoadError("Failed to open file: " + filepath);
	}

	std::stringstream buffer;
	buffer << file.rdbuf();
	return buffer.str();
}

/**
 * Get string value from JSON object, throwing if not found.
 */
std::string getRequiredString(cJSON* obj, const char* key, const std::string& context)
{
	cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
	if (!item || !cJSON_IsString(item)) {
		throw TestDataLoadError("Missing or invalid string field '" + std::string(key) +
		    "' in " + context);
	}
	return item->valuestring;
}

/**
 * Get optional string value from JSON object.
 */
std::string getOptionalString(cJSON* obj, const char* key, const std::string& default_value = "")
{
	cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
	if (item && cJSON_IsString(item)) {
		return item->valuestring;
	}
	return default_value;
}

/**
 * Get integer value from JSON object, throwing if not found.
 */
int getRequiredInt(cJSON* obj, const char* key, const std::string& context)
{
	cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
	if (!item || !cJSON_IsNumber(item)) {
		throw TestDataLoadError("Missing or invalid integer field '" + std::string(key) +
		    "' in " + context);
	}
	return item->valueint;
}

/**
 * Get optional integer value from JSON object.
 */
int getOptionalInt(cJSON* obj, const char* key, int default_value = 0)
{
	cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
	if (item && cJSON_IsNumber(item)) {
		return item->valueint;
	}
	return default_value;
}

/**
 * Get uint64_t value from JSON object.
 */
uint64_t getRequiredUint64(cJSON* obj, const char* key, const std::string& context)
{
	cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
	if (!item) {
		throw TestDataLoadError("Missing uint64 field '" + std::string(key) + "' in " + context);
	}
	if (cJSON_IsString(item)) {
		// Parse string as uint64_t
		const char* str = item->valuestring;
		char* endptr = nullptr;
		uint64_t val = strtoull(str, &endptr, 10);
		if (!str || !endptr || *endptr != '\0') {
			throw TestDataLoadError("Invalid uint64 string value for '" + std::string(key) + "' in " + context);
		}
		return val;
	} else if (cJSON_IsNumber(item)) {
		// Fallback: parse as double (legacy)
		return static_cast<uint64_t>(item->valuedouble);
	} else {
		throw TestDataLoadError("Invalid type for uint64 field '" + std::string(key) + "' in " + context);
	}
}

/**
 * Get uint32_t value from JSON object (supports both number and string formats).
 */
uint32_t getRequiredUint32(cJSON* obj, const char* key, const std::string& context)
{
	cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
	if (!item) {
		throw TestDataLoadError("Missing uint32 field '" + std::string(key) + "' in " + context);
	}
	if (cJSON_IsString(item)) {
		// Parse string as uint32_t
		const char* str = item->valuestring;
		char* endptr = nullptr;
		unsigned long val = strtoul(str, &endptr, 10);
		if (!str || !endptr || *endptr != '\0') {
			throw TestDataLoadError("Invalid uint32 string value for '" + std::string(key) + "' in " + context);
		}
		return static_cast<uint32_t>(val);
	} else if (cJSON_IsNumber(item)) {
		// Fallback: parse as double (legacy)
		return static_cast<uint32_t>(item->valuedouble);
	} else {
		throw TestDataLoadError("Invalid type for uint32 field '" + std::string(key) + "' in " + context);
	}
}

/**
 * Parse anchors from JSON array.
 */
std::vector<ExpectedAnchor> parseAnchors(cJSON* arr, const std::string& context)
{
	std::vector<ExpectedAnchor> anchors;
	if (!arr || !cJSON_IsArray(arr)) {
		return anchors; // Return empty if not present
	}

	cJSON* item;
	int idx = 0;
	cJSON_ArrayForEach(item, arr)
	{
		std::string item_context = context + " anchor[" + std::to_string(idx) + "]";
		ExpectedAnchor anchor;
		anchor.x = getRequiredUint64(item, "x", item_context);
		anchor.y = getRequiredUint32(item, "y", item_context);
		anchors.push_back(anchor);
		idx++;
	}

	return anchors;
}

/**
 * Parse chains from JSON array.
 */
std::vector<ExpectedChain> parseChains(cJSON* arr, const std::string& context)
{
	std::vector<ExpectedChain> chains;
	if (!arr || !cJSON_IsArray(arr)) {
		return chains; // Return empty if not present
	}

	cJSON* item;
	int idx = 0;
	cJSON_ArrayForEach(item, arr)
	{
		std::string item_context = context + " chain[" + std::to_string(idx) + "]";
		ExpectedChain chain;
		chain.score = getRequiredInt(item, "score", item_context);
		chain.length = getRequiredInt(item, "length", item_context);
		chain.first_qpos = getRequiredUint32(item, "first_qpos", item_context);
		chain.first_tpos = getRequiredUint32(item, "first_tpos", item_context);
		chain.last_qpos = getRequiredUint32(item, "last_qpos", item_context);
		chain.last_tpos = getRequiredUint32(item, "last_tpos", item_context);
		chains.push_back(chain);
		idx++;
	}

	return chains;
}

/**
 * Parse alignments from JSON array.
 */
std::vector<ExpectedAlignment> parseAlignments(cJSON* arr, const std::string& context)
{
	std::vector<ExpectedAlignment> alignments;
	if (!arr || !cJSON_IsArray(arr)) {
		return alignments; // Return empty if not present
	}

	cJSON* item;
	int idx = 0;
	cJSON_ArrayForEach(item, arr)
	{
		std::string item_context = context + " alignment[" + std::to_string(idx) + "]";
		ExpectedAlignment alignment;
		alignment.score = getRequiredInt(item, "score", item_context);
		alignment.dp_score = getRequiredInt(item, "dp_score", item_context);
		alignment.qs = getRequiredInt(item, "qs", item_context);
		alignment.qe = getRequiredInt(item, "qe", item_context);
		alignment.rs = getRequiredInt(item, "rs", item_context);
		alignment.re = getRequiredInt(item, "re", item_context);
		alignment.mapq = getRequiredInt(item, "mapq", item_context);
		alignment.rev = getRequiredInt(item, "rev", item_context);
		alignment.n_cigar_ops = getOptionalInt(item, "n_cigar_ops", 0);
		alignment.blen = getOptionalInt(item, "blen", 0);
		alignment.mlen = getOptionalInt(item, "mlen", 0);
		alignment.dp_max = getOptionalInt(item, "dp_max", 0);
		alignment.n_ambi = getOptionalInt(item, "n_ambi", 0);
		alignments.push_back(alignment);
		idx++;
	}

	return alignments;
}

/**
 * Check if a file has .json extension.
 */
bool hasJsonExtension(const std::string& filename)
{
	if (filename.length() < 5) return false;
	std::string ext = filename.substr(filename.length() - 5);
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
	return ext == ".json";
}

/**
 * Check if path is a regular file.
 */
bool isRegularFile(const std::string& path)
{
	struct stat st;
	if (stat(path.c_str(), &st) != 0) return false;
	return S_ISREG(st.st_mode);
}

/**
 * Get the directory part of a file path.
 */
std::string getDirectoryPath(const std::string& filepath)
{
	size_t pos = filepath.find_last_of("/\\");
	if (pos == std::string::npos) {
		return ".";
	}
	return filepath.substr(0, pos);
}

/**
 * Resolve a path relative to a base directory.
 * If the path is absolute, return it as-is.
 * Otherwise, prepend the base directory.
 * 
 * Supports special variable ${TEST_SUITE_DIR} which is replaced with the
 * test_suite directory path using the following resolution order:
 * 1. Compile-time constant TEST_SUITE_DIR (if defined)
 * 2. Environment variable TEST_SUITE_DIR_ENV (if set)
 * 3. Fallback: base_dir + "/../../test_suite"
 * 
 * Expected directory structure (for fallback resolution):
 *   workspace/
 *     test_suite/          <- base for relative paths
 *       expected/
 *         *.json           <- JSON files (base_dir points here)
 *     out/test-cpu/build/  <- where test binaries run (typical case)
 *       tests/core/
 * 
 * When loading JSON from out/test-cpu/build/tests/core/expected/test.json:
 *   base_dir = "out/test-cpu/build/tests/core/expected"
 *   ../../test_suite resolves to: workspace/test_suite
 * 
 * To override the fallback path, set environment variable:
 *   export TEST_SUITE_DIR_ENV=/absolute/path/to/test_suite
 */
std::string resolvePath(const std::string& base_dir, const std::string& path)
{
	if (path.empty()) return path;

	std::string resolved = path;

	// Replace ${TEST_SUITE_DIR} placeholder
	const std::string placeholder = "${TEST_SUITE_DIR}";
	size_t pos = resolved.find(placeholder);
	if (pos != std::string::npos) {
#ifdef TEST_SUITE_DIR
		// Compile-time constant takes precedence
		resolved.replace(pos, placeholder.length(), TEST_SUITE_DIR);
#else
		// Try environment variable (allows runtime configuration)
		const char* env_test_suite_dir = std::getenv("TEST_SUITE_DIR_ENV");
		if (env_test_suite_dir) {
			resolved.replace(pos, placeholder.length(), env_test_suite_dir);
		} else {
			// Fallback: assume relative path from JSON location to workspace test_suite
			// This works for the typical build directory structure:
			//   workspace/test_suite/small/expected/test.json (source)
			//   workspace/out/test-cpu/build/ (build location)
			// When test runs from build dir and loads from installed JSON:
			//   base_dir = ".../out/test-cpu/build/tests/core/expected"
			//   ../../test_suite = "...test_suite" (after resolving ../../../)
			resolved.replace(pos, placeholder.length(), base_dir + "/../../test_suite");
		}
#endif
	}

	// Check if path is now absolute
	if (!resolved.empty() && resolved[0] == '/') {
		return resolved;
	}

	return base_dir + "/" + resolved;
}

/**
 * Read a FASTA file and extract the sequence name and sequence.
 * Only reads the first sequence in the file.
 * 
 * @param filepath Path to the FASTA file
 * @param out_name Output: sequence name (from header line)
 * @param out_seq Output: sequence (concatenated, no whitespace)
 */
void readFastaFile(const std::string& filepath, std::string& out_name, std::string& out_seq)
{
	std::ifstream file(filepath);
	if (!file.is_open()) {
		throw TestDataLoadError("Failed to open FASTA file: " + filepath);
	}

	out_name.clear();
	out_seq.clear();

	std::string line;
	bool in_sequence = false;

	while (std::getline(file, line)) {
		// Remove trailing whitespace
		while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
			line.pop_back();
		}

		if (line.empty()) continue;

		if (line[0] == '>') {
			if (in_sequence) {
				// We've already read one sequence, stop here
				break;
			}
			// Parse header: >name [description]
			size_t space_pos = line.find_first_of(" \t");
			if (space_pos != std::string::npos) {
				out_name = line.substr(1, space_pos - 1);
			} else {
				out_name = line.substr(1);
			}
			in_sequence = true;
		} else if (in_sequence) {
			// Sequence line - append to sequence (removing any whitespace)
			for (char c : line) {
				if (!std::isspace(c)) {
					out_seq += c;
				}
			}
		}
	}

	if (out_name.empty()) {
		throw TestDataLoadError("No sequence found in FASTA file: " + filepath);
	}
}

/**
 * Check if path is a directory.
 */
bool isDirectory(const std::string& path)
{
	struct stat st;
	if (stat(path.c_str(), &st) != 0) return false;
	return S_ISDIR(st.st_mode);
}

/**
 * Scan test_suite subdirectories for expected/*.json files.
 */
std::vector<std::string> findAllTestDataJsonFiles(const std::string& test_suite_dir)
{
	std::vector<std::string> result;

	DIR* dir = opendir(test_suite_dir.c_str());
	if (!dir) {
		return result;
	}

	struct dirent* entry;
	while ((entry = readdir(dir)) != nullptr) {
		std::string subdir_name = entry->d_name;

		// Skip . and ..
		if (subdir_name == "." || subdir_name == "..") {
			continue;
		}

		// Check if this is a directory
		std::string subdir_path = test_suite_dir + "/" + subdir_name;
		if (!isDirectory(subdir_path)) {
			continue;
		}

		// Check for expected subdirectory
		std::string expected_dir = subdir_path + "/expected";
		if (!isDirectory(expected_dir)) {
			continue;
		}

		// Scan expected directory for JSON files
		DIR* expected_dirp = opendir(expected_dir.c_str());
		if (!expected_dirp) {
			continue;
		}

		struct dirent* json_entry;
		while ((json_entry = readdir(expected_dirp)) != nullptr) {
			std::string filename = json_entry->d_name;
			if (hasJsonExtension(filename)) {
				std::string filepath = expected_dir + "/" + filename;
				if (isRegularFile(filepath)) {
					result.push_back(filepath);
				}
			}
		}
		closedir(expected_dirp);
	}
	closedir(dir);

	// Sort for consistent ordering
	std::sort(result.begin(), result.end());

	return result;
}

} // anonymous namespace

// ============================================================================
// RuntimeTestData implementation
// ============================================================================

MappingTestData RuntimeTestData::toMappingTestData() const
{
	MappingTestData data;

	data.name = name.c_str();
	data.description = description.c_str();

	data.ref_name = ref_name.c_str();
	data.ref_path = ref_path.c_str();

	data.query_name = query_name.c_str();
	data.query_path = query_path.c_str();

	data.expected_anchors = expected_anchors.empty() ? nullptr : expected_anchors.data();
	data.n_expected_anchors = expected_anchors.size();

	data.n_expected_chains = static_cast<int>(expected_chains.size());
	data.expected_chains = expected_chains.empty() ? nullptr : expected_chains.data();

	data.n_expected_alignments = static_cast<int>(expected_alignments.size());
	data.expected_alignments = expected_alignments.empty() ? nullptr : expected_alignments.data();

	data.k = k;
	data.w = w;

	// Store a persistent copy so the const char* in MappingTestData
	// remains valid for the lifetime of this RuntimeTestData.
	preset_cstr_ = preset;
	data.preset = preset_cstr_.empty() ? nullptr : preset_cstr_.c_str();

	data.query_index = query_index;

	return data;
}

// ============================================================================
// Loading functions
// ============================================================================

/**
 * Internal function to load test data from JSON with a base path for resolving relative paths.
 * Returns one RuntimeTestData per query in the JSON file.
 * For single-query (flat or queries[0]-only) files, returns a single-element vector.
 * For multi-query files, returns one entry per query with name suffixed "_q{name}".
 */
std::vector<RuntimeTestData> loadTestDataFromJsonInternal(const std::string& json_str, const std::string& base_dir)
{
	cJSON* root = cJSON_Parse(json_str.c_str());
	if (!root) {
		const char* error_ptr = cJSON_GetErrorPtr();
		std::string error_msg = "JSON parse error";
		if (error_ptr) {
			error_msg += " near: " + std::string(error_ptr).substr(0, 50);
		}
		throw TestDataLoadError(error_msg);
	}

	// Use unique_ptr for automatic cleanup
	std::unique_ptr<cJSON, decltype(&cJSON_Delete)> root_guard(root, cJSON_Delete);

	// Parse common fields shared by all queries
	const std::string context = "root";
	std::string base_name = getRequiredString(root, "name", context);
	std::string description = getOptionalString(root, "description", "");

	// Parse reference - store resolved file path
	cJSON* ref = cJSON_GetObjectItemCaseSensitive(root, "reference");
	if (!ref || !cJSON_IsObject(ref)) {
		throw TestDataLoadError("Missing or invalid 'reference' object");
	}

	cJSON* ref_file = cJSON_GetObjectItemCaseSensitive(ref, "file");
	if (!ref_file || !cJSON_IsString(ref_file)) {
		throw TestDataLoadError("Missing 'file' field in 'reference' object");
	}
	std::string ref_path = resolvePath(base_dir, ref_file->valuestring);

	// Get reference name from JSON or extract from FASTA header
	std::string ref_name = getOptionalString(ref, "name", "");
	if (ref_name.empty()) {
		std::string dummy_seq;
		readFastaFile(ref_path, ref_name, dummy_seq);
	}

	// Parse query - store resolved file path
	cJSON* query = cJSON_GetObjectItemCaseSensitive(root, "query");
	if (!query || !cJSON_IsObject(query)) {
		throw TestDataLoadError("Missing or invalid 'query' object");
	}

	cJSON* query_file = cJSON_GetObjectItemCaseSensitive(query, "file");
	if (!query_file || !cJSON_IsString(query_file)) {
		throw TestDataLoadError("Missing 'file' field in 'query' object");
	}
	std::string query_path = resolvePath(base_dir, query_file->valuestring);

	// Get query name from JSON or extract from FASTA header
	std::string query_name_default = getOptionalString(query, "name", "");
	if (query_name_default.empty()) {
		std::string dummy_seq;
		readFastaFile(query_path, query_name_default, dummy_seq);
	}

	// Parse optional index parameters
	int k = 0, w = 0;
	cJSON* params = cJSON_GetObjectItemCaseSensitive(root, "index_params");
	if (params && cJSON_IsObject(params)) {
		k = getOptionalInt(params, "k", 0);
		w = getOptionalInt(params, "w", 0);
	}

	// Parse optional preset name (e.g. "map-ont", "map-pb", "sr")
	std::string preset = getOptionalString(root, "preset", "");

	// Helper to populate a RuntimeTestData with shared fields
	auto populateCommon = [&](RuntimeTestData& data, int qi) {
		data.description = description;
		data.ref_name = ref_name;
		data.ref_path = ref_path;
		data.query_path = query_path;
		data.k = k;
		data.w = w;
		data.preset = preset;
		data.query_index = qi;
	};

	std::vector<RuntimeTestData> results;

	// Parse expected values - support both old and new JSON formats
	// New format: per-query structure with "queries" array
	// Old format: flat structure with "expected_anchors", "expected_chains", "expected_alignments"
	cJSON* queries_array = cJSON_GetObjectItemCaseSensitive(root, "queries");
	if (queries_array && cJSON_IsArray(queries_array) && cJSON_GetArraySize(queries_array) > 0) {
		int n_queries = cJSON_GetArraySize(queries_array);
		// Create one RuntimeTestData per query
		for (int qi = 0; qi < n_queries; qi++) {
			cJSON* query_obj = cJSON_GetArrayItem(queries_array, qi);
			if (!query_obj || !cJSON_IsObject(query_obj)) continue;

			RuntimeTestData data;
			populateCommon(data, qi);

			// Get per-query name if available, otherwise use default
			std::string qname = getOptionalString(query_obj, "name", "");
			if (n_queries == 1) {
				data.name = base_name;
				data.query_name = qname.empty() ? query_name_default : qname;
			} else {
				// Multi-query: append query name/index to dataset name
				if (!qname.empty()) {
					data.name = base_name + "_" + qname;
					data.query_name = qname;
				} else {
					data.name = base_name + "_q" + std::to_string(qi);
					data.query_name = query_name_default;
				}
			}

			std::string qi_ctx = context + " queries[" + std::to_string(qi) + "]";
			cJSON* qa = cJSON_GetObjectItemCaseSensitive(query_obj, "expected_anchors");
			data.expected_anchors = parseAnchors(qa, qi_ctx);

			cJSON* qc = cJSON_GetObjectItemCaseSensitive(query_obj, "expected_chains");
			data.expected_chains = parseChains(qc, qi_ctx);

			cJSON* qal = cJSON_GetObjectItemCaseSensitive(query_obj, "expected_alignments");
			data.expected_alignments = parseAlignments(qal, qi_ctx);

			results.push_back(std::move(data));
		}
	} else {
		// Old format: flat structure (single query)
		RuntimeTestData data;
		populateCommon(data, 0);
		data.name = base_name;
		data.query_name = query_name_default;

		cJSON* anchors = cJSON_GetObjectItemCaseSensitive(root, "expected_anchors");
		data.expected_anchors = parseAnchors(anchors, context);

		cJSON* chains = cJSON_GetObjectItemCaseSensitive(root, "expected_chains");
		data.expected_chains = parseChains(chains, context);

		cJSON* alignments = cJSON_GetObjectItemCaseSensitive(root, "expected_alignments");
		data.expected_alignments = parseAlignments(alignments, context);

		results.push_back(std::move(data));
	}

	return results;
}

RuntimeTestData loadTestDataFromJsonString(const std::string& json_str)
{
	// When loading from string, use current directory as base
	auto results = loadTestDataFromJsonInternal(json_str, ".");
	if (results.empty()) {
		throw TestDataLoadError("No test data entries produced from JSON string");
	}
	return std::move(results[0]);
}

std::vector<RuntimeTestData> loadAllQueriesFromJson(const std::string& filepath)
{
	std::string contents = readFileContents(filepath);
	std::string base_dir = getDirectoryPath(filepath);
	try {
		return loadTestDataFromJsonInternal(contents, base_dir);
	} catch (const TestDataLoadError& e) {
		throw TestDataLoadError("Error loading " + filepath + ": " + e.what());
	}
}

RuntimeTestData loadTestDataFromJson(const std::string& filepath)
{
	auto results = loadAllQueriesFromJson(filepath);
	if (results.empty()) {
		throw TestDataLoadError("No test data entries in " + filepath);
	}
	return std::move(results[0]);
}

std::vector<RuntimeTestData> loadAllTestDataFromDirectory(const std::string& dirpath)
{
	std::vector<RuntimeTestData> result;

	DIR* dir = opendir(dirpath.c_str());
	if (!dir) {
		throw TestDataLoadError("Failed to open directory: " + dirpath);
	}

	std::vector<std::string> json_files;
	struct dirent* entry;
	while ((entry = readdir(dir)) != nullptr) {
		std::string filename = entry->d_name;
		if (hasJsonExtension(filename)) {
			std::string filepath = dirpath + "/" + filename;
			if (isRegularFile(filepath)) {
				json_files.push_back(filepath);
			}
		}
	}
	closedir(dir);

	// Sort for consistent ordering
	std::sort(json_files.begin(), json_files.end());

	for (const auto& filepath : json_files) {
		result.push_back(loadTestDataFromJson(filepath));
	}

	return result;
}

std::string getTestDataDirectory()
{
	const char* env_dir = std::getenv("TEST_DATA_DIR");
	if (env_dir) {
		return env_dir;
	}

#ifdef TEST_SUITE_DIR
	// Use compile-time definition if available
	return TEST_SUITE_DIR;
#else
	// Default path - relative to where tests are typically run
	return "test_suite";
#endif
}

// ============================================================================
// TestDataRegistry implementation
// ============================================================================

TestDataRegistry& TestDataRegistry::getInstance()
{
	static TestDataRegistry instance;
	return instance;
}

void TestDataRegistry::loadAll()
{
	if (loaded_) {
		return;
	}

	std::string test_suite_dir = getTestDataDirectory();

	// Find all JSON files in test_suite/*/expected directories
	std::vector<std::string> json_files = findAllTestDataJsonFiles(test_suite_dir);

	if (json_files.empty()) {
		mm_log_warn("No test data JSON files found in {}/*/expected/",
		    test_suite_dir.c_str());
	}

	// Load each JSON file (may produce multiple entries for multi-query files)
	for (const auto& filepath : json_files) {
		try {
			auto entries = loadAllQueriesFromJson(filepath);
			for (auto& entry : entries) {
				test_data_.push_back(std::move(entry));
			}
		} catch (const TestDataLoadError& e) {
			mm_log_warn("Could not load test data from {}: {}",
			    filepath.c_str(), e.what());
		}
	}

	loaded_ = true;

	// Build mapping test data cache
	mapping_test_data_.clear();
	mapping_test_data_.reserve(test_data_.size());
	for (const auto& td : test_data_) {
		mapping_test_data_.push_back(td.toMappingTestData());
	}
}

std::vector<const MappingTestData*> TestDataRegistry::getMappingTestDataPointers()
{
	if (!loaded_) {
		loadAll();
	}

	std::vector<const MappingTestData*> result;
	result.reserve(mapping_test_data_.size());
	for (const auto& td : mapping_test_data_) {
		result.push_back(&td);
	}
	return result;
}
