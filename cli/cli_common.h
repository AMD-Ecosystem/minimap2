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
 * @file cli_common.h
 * @brief Common utilities for minimap2 stage CLI tools
 *
 * Provides shared functionality for:
 * - JSON input/output using cJSON (via cli_json.h)
 * - Index loading and sequence reading
 * - Resource management (RAII wrappers)
 * - JSON helpers for core/seed types (mm128_t, mm_bseq1_t, mm_idx_t)
 *
 * Note: For chain_read_t helpers, see mm2_chain.cpp
 *       For mm_reg1_t (alignment) helpers, see mm2_align.cpp
 */
#ifndef CLI_COMMON_H
#define CLI_COMMON_H

#include "cli_json.h" /* JSON utilities (no minimap2 dependency) */

#include "mm_log.h"

extern "C" {
#include "minimap.h"
#include "kalloc.h"
#include "bseq.h"
#include "index.h"
#include "option.h" /* For mm_idxopt_init, mm_mapopt_init */
#include "chain_read.h" /* For chain_read_t */
}

/* ============================================================================
 * Constants
 * ============================================================================ */

/** Default batch size for reading query sequences (1MB) */
constexpr size_t CLI_QUERY_BATCH_SIZE = 1024 * 1024;

/** Default fragment gap for chaining */
constexpr int CLI_DEFAULT_FRAG_GAP = 1000;

/** Mid-occ fraction for automatic threshold calculation */
constexpr float CLI_MID_OCC_FRAC = 2e-4f;

/** Min/max bounds for mid_occ */
constexpr int CLI_MIN_MID_OCC = 10;
constexpr int CLI_MAX_MID_OCC = 1000000;

/* ============================================================================
 * RAII Wrapper for kalloc Memory Pool (requires kalloc.h)
 * ============================================================================ */

/** RAII wrapper for kalloc memory pool. */
class KallocGuard
{
	void* km_;

      public:
	KallocGuard() : km_(km_init()) {}
	~KallocGuard()
	{
		if (km_) km_destroy(km_);
	}

	// No copy or move (memory pool should not be transferred)
	KallocGuard(const KallocGuard&) = delete;
	KallocGuard& operator=(const KallocGuard&) = delete;
	KallocGuard(KallocGuard&&) = delete;
	KallocGuard& operator=(KallocGuard&&) = delete;

	void* get() const { return km_; }
	operator void*() const { return km_; }
};

/* ============================================================================
 * Index and Sequence Loading
 * ============================================================================ */

/** Load minimap2 index from reference file. */
static inline mm_idx_t* load_index(const char* ref_path, mm_idxopt_t* iopt)
{
	mm_idx_reader_t* idx_rdr = mm_idx_reader_open(ref_path, iopt, NULL);
	if (!idx_rdr) {
		mm_log_error("Failed to open reference file: {}", ref_path);
		return NULL;
	}
	mm_idx_t* mi = mm_idx_reader_read(idx_rdr, 1);
	mm_idx_reader_close(idx_rdr);
	if (!mi) mm_log_error("Failed to build index from: {}", ref_path);
	return mi;
}

/** Load query sequences from FASTA/FASTQ file. */
static inline mm_bseq1_t* load_queries(const char* query_path, int* n_seqs)
{
	mm_bseq_file_t* fp = mm_bseq_open(query_path);
	if (!fp) {
		mm_log_error("Failed to open query file: {}", query_path);
		return NULL;
	}
	mm_bseq1_t* seqs = mm_bseq_read(fp, CLI_QUERY_BATCH_SIZE, 0, n_seqs);
	mm_bseq_close(fp);
	return seqs;
}

/** Free query sequences. */
static inline void free_queries(mm_bseq1_t* seqs, int n_seqs)
{
	if (!seqs) return;
	for (int i = 0; i < n_seqs; i++) {
		free(seqs[i].seq);
		free(seqs[i].qual);
		free(seqs[i].name);
		free(seqs[i].comment);
	}
	free(seqs);
}

/* ============================================================================
 * chain_read_t Helpers
 * ============================================================================ */

/** Initialize a chain_read_t structure from a mm_bseq1_t sequence. */
static inline void init_chain_read(chain_read_t* read, const mm_bseq1_t* seq, void* km)
{
	memset(read, 0, sizeof(*read));
	if (seq) {
		strncpy(read->seq.name, seq->name ? seq->name : "", sizeof(read->seq.name) - 1);
		read->seq.len = seq->l_seq;
		read->seq.qlen_sum = seq->l_seq;
		read->n_seg = 1;
		read->qlens = (int*)malloc(sizeof(int));
		read->qseqs = (const char**)malloc(sizeof(char*));
		if (read->qlens && read->qseqs) {
			read->qlens[0] = seq->l_seq;
			read->qseqs[0] = seq->seq;
		}
	}
}

/** Free resources in a chain_read_t (but not the struct itself). */
static inline void free_chain_read(chain_read_t* read)
{
	if (read) {
		free(read->qlens);
		free((void*)read->qseqs);
		read->qlens = NULL;
		read->qseqs = NULL;
	}
}

/** Add chains array to query JSON from chain_read_t. */
static inline void add_chains_to_query_json(cJSON* query_item, const chain_read_t* read)
{
	cJSON* chains_arr = cJSON_CreateArray();
	int64_t anchor_offset = 0;

	for (int j = 0; j < read->n_u; j++) {
		int32_t score = (int32_t)(read->u[j] >> 32);
		int32_t len = (int32_t)(read->u[j] & 0xFFFFFFFF);

		if (len > 0 && anchor_offset + len <= read->n) {
			uint32_t first_qpos = (uint32_t)(read->a[anchor_offset].y & 0xFFFFFFFF);
			uint32_t first_tpos = (uint32_t)(read->a[anchor_offset].x);
			uint32_t last_qpos = (uint32_t)(read->a[anchor_offset + len - 1].y & 0xFFFFFFFF);
			uint32_t last_tpos = (uint32_t)(read->a[anchor_offset + len - 1].x);

			cJSON_AddItemToArray(chains_arr, json_chain(score, len, first_qpos, first_tpos, last_qpos, last_tpos));
		}
		anchor_offset += len;
	}

	cJSON_AddItemToObject(query_item, "chains", chains_arr);
	cJSON_AddNumberToObject(query_item, "n_chains", read->n_u);
}

/* ============================================================================
 * JSON Input/Output Helpers (for seed types - mm128_t)
 * ============================================================================ */

/** Load seeds from JSON query object into kalloc memory. Returns number of seeds. */
static inline int64_t load_seeds_from_json(cJSON* query_json, mm128_t** anchors, void* km)
{
	if (!query_json) {
		*anchors = NULL;
		return 0;
	}

	cJSON* seeds = cJSON_GetObjectItem(query_json, "seeds");
	if (!seeds || !cJSON_IsArray(seeds)) {
		*anchors = NULL;
		return 0;
	}

	int n_seeds = cJSON_GetArraySize(seeds);
	if (n_seeds == 0) {
		*anchors = NULL;
		return 0;
	}

	*anchors = (mm128_t*)kmalloc(km, n_seeds * sizeof(mm128_t));
	if (!*anchors) {
		mm_log_error("Memory allocation failed for seeds");
		return 0;
	}

	int idx = 0;
	cJSON* seed;
	cJSON_ArrayForEach(seed, seeds)
	{
		(*anchors)[idx].x = json_get_uint64(cJSON_GetObjectItem(seed, "x"));
		(*anchors)[idx].y = json_get_uint64(cJSON_GetObjectItem(seed, "y"));
		idx++;
	}
	return idx;
}

/** Load chains from JSON query object. Returns number of chains. */
static inline int load_chains_from_json(cJSON* query_json, uint64_t** u, void* km)
{
	if (!query_json) {
		*u = NULL;
		return 0;
	}

	cJSON* chains = cJSON_GetObjectItem(query_json, "chains");
	if (!chains || !cJSON_IsArray(chains)) {
		*u = NULL;
		return 0;
	}

	int n_chains = cJSON_GetArraySize(chains);
	if (n_chains == 0) {
		*u = NULL;
		return 0;
	}

	*u = (uint64_t*)kmalloc(km, n_chains * sizeof(uint64_t));
	if (!*u) {
		mm_log_error("Memory allocation failed for chains");
		return 0;
	}

	int idx = 0;
	cJSON* chain;
	cJSON_ArrayForEach(chain, chains)
	{
		int32_t score = json_get_int(cJSON_GetObjectItem(chain, "score"), 0);
		int32_t length = json_get_int(cJSON_GetObjectItem(chain, "length"), 0);
		(*u)[idx++] = ((uint64_t)score << 32) | (uint32_t)length;
	}
	return n_chains;
}

/** Add seeds array to query item from anchors (mm128_t). */
static inline void add_seeds_to_query(cJSON* query_item, mm128_t* anchors, int64_t n)
{
	cJSON* seeds_arr = cJSON_CreateArray();
	for (int64_t j = 0; j < n; j++) {
		cJSON_AddItemToArray(seeds_arr, json_anchor(anchors[j].x, anchors[j].y));
	}
	cJSON_AddItemToObject(query_item, "seeds", seeds_arr);
	cJSON_AddNumberToObject(query_item, "n_seeds", (int)n);
}

/* ============================================================================
 * Resource Management Context (RAII)
 * ============================================================================ */

/** Context for managing minimap2 resources with automatic cleanup. */
struct mm2_context {
	mm_idx_t* mi;
	mm_bseq1_t* seqs;
	int n_seqs;
	mm_mapopt_t opt;
	cJSON* input_json;
	cJSON* output_json;

	mm2_context() : mi(NULL), seqs(NULL), n_seqs(0),
			input_json(NULL), output_json(NULL)
	{
		mm_mapopt_init(&opt);
	}

	~mm2_context() { cleanup(); }

	// No copy
	mm2_context(const mm2_context&) = delete;
	mm2_context& operator=(const mm2_context&) = delete;

	void cleanup()
	{
		if (seqs) {
			free_queries(seqs, n_seqs);
			seqs = NULL;
			n_seqs = 0;
		}
		if (mi) {
			mm_idx_destroy(mi);
			mi = NULL;
		}
		if (input_json) {
			cJSON_Delete(input_json);
			input_json = NULL;
		}
		if (output_json) {
			cJSON_Delete(output_json);
			output_json = NULL;
		}
	}

	bool init(const char* ref_path, const char* query_path, int k, int w)
	{
		mm_idxopt_t iopt;
		mm_idxopt_init(&iopt);
		iopt.k = k;
		iopt.w = w;

		mi = load_index(ref_path, &iopt);
		if (!mi) return false;

		mm_mapopt_update(&opt, mi);
		mm_idx_index_name(mi);

		seqs = load_queries(query_path, &n_seqs);
		if (!seqs) return false;

		return true;
	}

	bool load_input(const char* path)
	{
		input_json = json_read_file(path);
		return input_json != NULL;
	}

	void create_output(const char* name, const char* desc,
	    const char* ref_path, const char* query_path, int k, int w)
	{
		output_json = create_output_json(name, desc, ref_path, query_path, k, w);
	}

	int write_output(const char* path)
	{
		return json_write_file(path, output_json);
	}

	cJSON* get_input_queries()
	{
		return input_json ? cJSON_GetObjectItem(input_json, "queries") : NULL;
	}
};

#endif /* CLI_COMMON_H */
