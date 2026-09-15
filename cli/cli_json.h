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
 * @file cli_json.h
 * @brief Common JSON utilities for minimap2 CLI tools
 *
 * Provides shared JSON functionality that does NOT depend on minimap2.
 * Can be used by both standalone CLIs (mm2_seed, mm2_chain, mm2_align)
 * and the full pipeline CLIs.
 *
 * For minimap2-dependent utilities (index loading, sequence reading),
 * see cli_common.h instead.
 */
#ifndef CLI_JSON_H
#define CLI_JSON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>

#include "mm_log.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "cJSON.h"

#ifdef __cplusplus
}
#endif

/* ============================================================================
 * JSON Value Helpers
 * ============================================================================ */

/** Get uint64_t from cJSON item (handles both string and number). */
static inline uint64_t json_get_uint64(cJSON* item)
{
	if (!item) return 0;
	if (cJSON_IsString(item)) return strtoull(item->valuestring, NULL, 10);
	if (cJSON_IsNumber(item)) return (uint64_t)item->valuedouble;
	return 0;
}

/** Get int64_t from cJSON item (handles both string and number). */
static inline int64_t json_get_int64(cJSON* item)
{
	if (!item) return 0;
	if (cJSON_IsString(item)) return strtoll(item->valuestring, NULL, 10);
	if (cJSON_IsNumber(item)) return (int64_t)item->valuedouble;
	return 0;
}

/** Get int from cJSON item with default value. */
static inline int json_get_int(cJSON* item, int default_val)
{
	if (!item || !cJSON_IsNumber(item)) return default_val;
	return item->valueint;
}

/** Get string from cJSON item with default value. */
static inline const char* json_get_string(cJSON* item, const char* default_val)
{
	if (!item || !cJSON_IsString(item)) return default_val;
	return item->valuestring;
}

/** Add uint64_t to JSON as string (preserves full 64-bit precision). */
static inline void json_add_uint64(cJSON* obj, const char* name, uint64_t value)
{
	char buf[32];
	snprintf(buf, sizeof(buf), "%" PRIu64, value);
	cJSON_AddStringToObject(obj, name, buf);
}

/** Add int64_t to JSON as string (preserves full 64-bit precision). */
static inline void json_add_int64(cJSON* obj, const char* name, int64_t value)
{
	char buf[32];
	snprintf(buf, sizeof(buf), "%" PRId64, value);
	cJSON_AddStringToObject(obj, name, buf);
}

/* ============================================================================
 * JSON Object Helpers
 * ============================================================================ */

/** Create JSON object for a single anchor (x, y pair). */
static inline cJSON* json_anchor(uint64_t x, uint64_t y)
{
	cJSON* anchor = cJSON_CreateObject();
	json_add_uint64(anchor, "x", x);
	json_add_uint64(anchor, "y", y);
	return anchor;
}

/** Create JSON object for a chain summary. */
static inline cJSON* json_chain(int32_t score, int32_t length,
    uint32_t first_qpos, uint32_t first_tpos,
    uint32_t last_qpos, uint32_t last_tpos)
{
	cJSON* chain = cJSON_CreateObject();
	cJSON_AddNumberToObject(chain, "score", score);
	cJSON_AddNumberToObject(chain, "length", length);
	cJSON_AddNumberToObject(chain, "first_qpos", first_qpos);
	cJSON_AddNumberToObject(chain, "first_tpos", first_tpos);
	cJSON_AddNumberToObject(chain, "last_qpos", last_qpos);
	cJSON_AddNumberToObject(chain, "last_tpos", last_tpos);
	return chain;
}

/* ============================================================================
 * JSON File I/O
 * ============================================================================ */

/**
 * Read JSON from file.
 * @param path File path to read
 * @return Parsed JSON object (caller must cJSON_Delete), or NULL on error
 */
static inline cJSON* json_read_file(const char* path)
{
	FILE* fp = fopen(path, "r");
	if (!fp) {
		mm_log_error("Cannot open file '{}': {}", path, strerror(errno));
		return NULL;
	}

	if (fseek(fp, 0, SEEK_END) != 0) {
		mm_log_error("Cannot seek in file '{}': {}", path, strerror(errno));
		fclose(fp);
		return NULL;
	}

	long size = ftell(fp);
	if (size < 0) {
		mm_log_error("Cannot determine file size '{}': {}", path, strerror(errno));
		fclose(fp);
		return NULL;
	}
	if (size == 0) {
		mm_log_error("File is empty: '{}'", path);
		fclose(fp);
		return NULL;
	}
	rewind(fp);

	char* content = (char*)malloc((size_t)size + 1);
	if (!content) {
		mm_log_error("Memory allocation failed for {} bytes", size);
		fclose(fp);
		return NULL;
	}

	size_t read_size = fread(content, 1, (size_t)size, fp);
	content[read_size] = '\0';
	fclose(fp);

	cJSON* json = cJSON_Parse(content);
	free(content);

	if (!json) {
		const char* error_ptr = cJSON_GetErrorPtr();
		mm_log_error("JSON parse failed in '{}'", path);
		if (error_ptr) mm_log_error("      Near: {:.40}", error_ptr);
	}
	return json;
}

/**
 * Write JSON to file.
 * @param path File path to write
 * @param json JSON object to serialize
 * @return 0 on success, -1 on error
 */
static inline int json_write_file(const char* path, cJSON* json)
{
	char* str = cJSON_Print(json);
	if (!str) {
		mm_log_error("JSON serialization failed");
		return -1;
	}

	FILE* fp = fopen(path, "w");
	if (!fp) {
		mm_log_error("Cannot write to '{}': {}", path, strerror(errno));
		free(str);
		return -1;
	}

	size_t len = strlen(str);
	size_t written = fwrite(str, 1, len, fp);
	int close_err = fclose(fp);
	free(str);

	if (written != len || close_err != 0) {
		mm_log_error("Write failed to '{}'", path);
		return -1;
	}
	return 0;
}

/**
 * Write JSON to file (compact format, no whitespace).
 * @param path File path to write
 * @param json JSON object to serialize
 * @return 0 on success, -1 on error
 */
static inline int json_write_file_compact(const char* path, cJSON* json)
{
	char* str = cJSON_PrintUnformatted(json);
	if (!str) {
		mm_log_error("JSON serialization failed");
		return -1;
	}

	FILE* fp = fopen(path, "w");
	if (!fp) {
		mm_log_error("Cannot write to '{}': {}", path, strerror(errno));
		free(str);
		return -1;
	}

	size_t len = strlen(str);
	size_t written = fwrite(str, 1, len, fp);
	int close_err = fclose(fp);
	free(str);

	if (written != len || close_err != 0) {
		mm_log_error("Write failed to '{}'", path);
		return -1;
	}
	return 0;
}

/* ============================================================================
 * Query/Anchor Lookup Helpers
 * ============================================================================ */

/** Find query by name in JSON queries array. */
static inline cJSON* json_find_query(cJSON* queries_arr, const char* name)
{
	if (!queries_arr || !name) return NULL;

	cJSON* q;
	cJSON_ArrayForEach(q, queries_arr)
	{
		cJSON* name_item = cJSON_GetObjectItem(q, "name");
		if (name_item && cJSON_IsString(name_item) &&
		    strcmp(name_item->valuestring, name) == 0) {
			return q;
		}
	}
	return NULL;
}

/** Get index params (k, w) from JSON. */
static inline void get_params_from_json(cJSON* json, int* k, int* w)
{
	cJSON* params = cJSON_GetObjectItem(json, "index_params");
	if (!params) return;
	*k = json_get_int(cJSON_GetObjectItem(params, "k"), *k);
	*w = json_get_int(cJSON_GetObjectItem(params, "w"), *w);
}

/* ============================================================================
 * JSON Output Helpers
 * ============================================================================ */

/** Create base output JSON with common fields. */
static inline cJSON* create_output_json(const char* name, const char* description,
    const char* ref_path, const char* query_path,
    int k, int w)
{
	cJSON* root = cJSON_CreateObject();
	cJSON_AddStringToObject(root, "name", name);
	cJSON_AddStringToObject(root, "description", description);

	cJSON* ref_obj = cJSON_CreateObject();
	cJSON_AddStringToObject(ref_obj, "file", ref_path);
	cJSON_AddItemToObject(root, "reference", ref_obj);

	cJSON* query_obj = cJSON_CreateObject();
	cJSON_AddStringToObject(query_obj, "file", query_path);
	cJSON_AddItemToObject(root, "query", query_obj);

	cJSON* params = cJSON_CreateObject();
	cJSON_AddNumberToObject(params, "k", k);
	cJSON_AddNumberToObject(params, "w", w);
	cJSON_AddItemToObject(root, "index_params", params);

	return root;
}

/* ============================================================================
 * RAII Wrappers (C++ only)
 * ============================================================================ */

#ifdef __cplusplus

/** RAII wrapper for cJSON objects. */
class JsonGuard
{
	cJSON* json_;

      public:
	explicit JsonGuard(cJSON* json = nullptr) : json_(json) {}
	~JsonGuard()
	{
		if (json_) cJSON_Delete(json_);
	}

	// Move semantics
	JsonGuard(JsonGuard&& other) noexcept : json_(other.json_) { other.json_ = nullptr; }
	JsonGuard& operator=(JsonGuard&& other) noexcept
	{
		if (this != &other) {
			if (json_) cJSON_Delete(json_);
			json_ = other.json_;
			other.json_ = nullptr;
		}
		return *this;
	}

	// No copy
	JsonGuard(const JsonGuard&) = delete;
	JsonGuard& operator=(const JsonGuard&) = delete;

	cJSON* get() const { return json_; }
	cJSON* operator->() const { return json_; }
	operator cJSON*() const { return json_; }
	explicit operator bool() const { return json_ != nullptr; }

	cJSON* release()
	{
		cJSON* p = json_;
		json_ = nullptr;
		return p;
	}
	void reset(cJSON* json = nullptr)
	{
		if (json_) cJSON_Delete(json_);
		json_ = json;
	}
};

#endif /* __cplusplus */

#endif /* CLI_JSON_H */
