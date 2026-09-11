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

#include "mm_dump_util.h"
#include "mm_log.h"
#include "cJSON.h"

#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <sys/stat.h>
#include <cerrno>

int mm_dump_mkdir_p(const std::string& path)
{
	std::string tmp = path;
	if (!tmp.empty() && tmp.back() == '/')
		tmp.pop_back();
	for (size_t i = 1; i < tmp.size(); i++) {
		if (tmp[i] == '/') {
			tmp[i] = '\0';
			if (mkdir(tmp.c_str(), 0755) != 0 && errno != EEXIST)
				return -1;
			tmp[i] = '/';
		}
	}
	if (mkdir(tmp.c_str(), 0755) != 0 && errno != EEXIST)
		return -1;
	return 0;
}

int mm_dump_write_json_file(const std::string& filepath, cJSON* json)
{
	char* json_str = cJSON_Print(json);
	if (!json_str) {
		mm_log_warn("Failed to serialize JSON");
		return -1;
	}
	FILE* fp = fopen(filepath.c_str(), "w");
	if (!fp) {
		mm_log_warn("Failed to open file for writing: {}", filepath.c_str());
		free(json_str);
		return -1;
	}
	fputs(json_str, fp);
	fclose(fp);
	free(json_str);
	return 0;
}

void mm_dump_json_add_uint64(cJSON* obj, const char* name, uint64_t value)
{
	char buf[32];
	snprintf(buf, sizeof(buf), "%" PRIu64, value);
	cJSON_AddStringToObject(obj, name, buf);
}

std::string mm_dump_build_cigar(const uint32_t* cigar, uint32_t n_cigar)
{
	static constexpr char ops[] = "MIDNSHP=X";
	static constexpr int n_ops = sizeof(ops) - 1;
	std::string result;
	if (!cigar || n_cigar == 0) return result;
	result.reserve(n_cigar * 8);
	for (uint32_t i = 0; i < n_cigar; i++) {
		int op = cigar[i] & 0xf;
		int len = cigar[i] >> 4;
		if (op >= n_ops) op = 0;
		char buf[16];
		snprintf(buf, sizeof(buf), "%d%c", len, ops[op]);
		result += buf;
	}
	return result;
}
