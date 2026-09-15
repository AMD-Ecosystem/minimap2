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

#include <string>

#include "mm_dump.h"
#include "mm_dump_util.h"
#include "mm_json_dump.h"
#include "mm_flatbuf_dump.h"
#include "chain_read.h"
#include "mm_log.h"
#include "cJSON.h"

extern "C" {

int mm_dump_init(mm_core_ctx_t* ctx)
{
	if (!ctx) return 0;
	mm_dump_cfg_t* cfg = &ctx->dump_cfg;
	if (!cfg->enabled) return 0;

	switch (cfg->format) {
	case MM_DUMP_JSON: {
		mm_json_state_t* st = mm_json_state_create(cfg->dir);
		if (!st) return -1;
		cfg->state = st;
		return mm_json_dump_init(st);
	}
	case MM_DUMP_FLATBUF: {
		mm_flatbuf_state_t* st = mm_flatbuf_state_create(cfg->dir);
		if (!st) return -1;
		mm_flatbuf_state_set_dump_chained_seeds(st, cfg->dump_chained_seeds);
		cfg->state = st;
		return mm_flatbuf_dump_init(st);
	}
	default:
		return 0;
	}
}

void mm_dump_cleanup(mm_core_ctx_t* ctx)
{
	if (!ctx) return;
	mm_dump_cfg_t* cfg = &ctx->dump_cfg;
	if (!cfg->enabled || !cfg->state) return;

	switch (cfg->format) {
	case MM_DUMP_JSON:
		mm_json_state_destroy((mm_json_state_t*)cfg->state);
		break;
	case MM_DUMP_FLATBUF:
		mm_flatbuf_state_destroy((mm_flatbuf_state_t*)cfg->state);
		break;
	default:
		break;
	}
	cfg->state = NULL;
}

int mm_dump_seeds(const mm_core_ctx_t* ctx, const char* qname,
    const chain_read_t* read, const mm_idx_t* mi)
{
	if (!ctx) return 0;
	const mm_dump_cfg_t* cfg = &ctx->dump_cfg;
	if (!cfg->enabled || !cfg->state) return 0;

	switch (cfg->format) {
	case MM_DUMP_JSON:
		return mm_json_dump_seeds((mm_json_state_t*)cfg->state, qname, read, mi);
	case MM_DUMP_FLATBUF:
		return mm_flatbuf_dump_seeds((mm_flatbuf_state_t*)cfg->state, qname, read, mi);
	default:
		return 0;
	}
}

int mm_dump_chains(const mm_core_ctx_t* ctx, const char* qname,
    const chain_read_t* read, const mm_idx_t* mi)
{
	if (!ctx) return 0;
	const mm_dump_cfg_t* cfg = &ctx->dump_cfg;
	if (!cfg->enabled || !cfg->state) return 0;

	switch (cfg->format) {
	case MM_DUMP_JSON:
		return mm_json_dump_chains((mm_json_state_t*)cfg->state, qname, read, mi);
	case MM_DUMP_FLATBUF:
		return mm_flatbuf_dump_chains((mm_flatbuf_state_t*)cfg->state, qname, read, mi);
	default:
		return 0;
	}
}

int mm_dump_alignments(const mm_core_ctx_t* ctx, const char* qname,
    int n_segs, int qlen_sum, const int* n_regs, mm_reg1_t** regs,
    const mm_idx_t* mi)
{
	if (!ctx) return 0;
	const mm_dump_cfg_t* cfg = &ctx->dump_cfg;
	if (!cfg->enabled || !cfg->state) return 0;

	switch (cfg->format) {
	case MM_DUMP_JSON:
		return mm_json_dump_alignments((mm_json_state_t*)cfg->state, qname, n_segs, qlen_sum, n_regs, regs, mi);
	case MM_DUMP_FLATBUF:
		return mm_flatbuf_dump_alignments((mm_flatbuf_state_t*)cfg->state, qname, n_segs, qlen_sum, n_regs, regs, mi, cfg->dump_chained_seeds);
	default:
		return 0;
	}
}

int mm_dump_write_timer_json(const char* dir, const mm_timer_stats_t* ts,
    double wall_time, double cpu_time, double peak_rss_gb, int n_threads)
{
	if (!dir || !ts) return 0;

	std::string stats_dir = std::string(dir) + "/stats";
	if (mm_dump_mkdir_p(stats_dir) != 0) {
		mm_log_warn("Failed to create stats directory: {}", stats_dir.c_str());
		return -1;
	}

	cJSON* root = cJSON_CreateObject();
	if (!root) return -1;

	const double total_sum = ts->seed_sum + ts->chain_sum + ts->align_sum;
	const double inv = n_threads > 0 ? (1.0 / (double)n_threads) : 0.0;
	const double stage_per_thread_equiv = total_sum * inv;
	const double overhead = wall_time - stage_per_thread_equiv;
	const double per_batch_thread = (ts->batch_count > 0 && n_threads > 0)
	    ? (total_sum / ((double)n_threads * (double)ts->batch_count))
	    : 0.0;
	const double per_query = (ts->query_count > 0)
	    ? (total_sum / (double)ts->query_count)
	    : 0.0;

	cJSON_AddNumberToObject(root, "n_threads", n_threads);
	cJSON_AddNumberToObject(root, "wall_time_sec", wall_time);
	cJSON_AddNumberToObject(root, "cpu_time_sec", cpu_time);
	cJSON_AddNumberToObject(root, "peak_rss_gb", peak_rss_gb);

	mm_dump_json_add_uint64(root, "batch_count", ts->batch_count);
	mm_dump_json_add_uint64(root, "query_count", ts->query_count);

	cJSON* stages = cJSON_CreateObject();
	if (!stages) {
		cJSON_Delete(root);
		return -1;
	}

	cJSON* seed = cJSON_CreateObject();
	cJSON_AddNumberToObject(seed, "sum_sec", ts->seed_sum);
	cJSON_AddNumberToObject(seed, "total_per_thread_sec", ts->seed_sum * inv);
	cJSON_AddNumberToObject(seed, "min_thread_batch_sec", ts->seed_min);
	cJSON_AddNumberToObject(seed, "max_thread_batch_sec", ts->seed_max);
	cJSON_AddNumberToObject(seed, "avg_thread_batch_sec", ts->seed_avg);
	cJSON_AddNumberToObject(seed, "call_min_sec", ts->seed_call_min);
	cJSON_AddNumberToObject(seed, "call_max_sec", ts->seed_call_max);
	cJSON_AddNumberToObject(seed, "call_avg_sec", ts->seed_call_avg);
	mm_dump_json_add_uint64(seed, "call_count", ts->seed_call_count);
	cJSON_AddItemToObject(stages, "seed", seed);

	cJSON* chain = cJSON_CreateObject();
	cJSON_AddNumberToObject(chain, "sum_sec", ts->chain_sum);
	cJSON_AddNumberToObject(chain, "total_per_thread_sec", ts->chain_sum * inv);
	cJSON_AddNumberToObject(chain, "min_thread_batch_sec", ts->chain_min);
	cJSON_AddNumberToObject(chain, "max_thread_batch_sec", ts->chain_max);
	cJSON_AddNumberToObject(chain, "avg_thread_batch_sec", ts->chain_avg);
	cJSON_AddNumberToObject(chain, "call_min_sec", ts->chain_call_min);
	cJSON_AddNumberToObject(chain, "call_max_sec", ts->chain_call_max);
	cJSON_AddNumberToObject(chain, "call_avg_sec", ts->chain_call_avg);
	mm_dump_json_add_uint64(chain, "call_count", ts->chain_call_count);
	cJSON_AddItemToObject(stages, "chain", chain);

	cJSON* align = cJSON_CreateObject();
	cJSON_AddNumberToObject(align, "sum_sec", ts->align_sum);
	cJSON_AddNumberToObject(align, "total_per_thread_sec", ts->align_sum * inv);
	cJSON_AddNumberToObject(align, "min_thread_batch_sec", ts->align_min);
	cJSON_AddNumberToObject(align, "max_thread_batch_sec", ts->align_max);
	cJSON_AddNumberToObject(align, "avg_thread_batch_sec", ts->align_avg);
	cJSON_AddNumberToObject(align, "call_min_sec", ts->align_call_min);
	cJSON_AddNumberToObject(align, "call_max_sec", ts->align_call_max);
	cJSON_AddNumberToObject(align, "call_avg_sec", ts->align_call_avg);
	mm_dump_json_add_uint64(align, "call_count", ts->align_call_count);
	cJSON_AddItemToObject(stages, "align", align);

	cJSON_AddItemToObject(root, "stages", stages);
	cJSON_AddNumberToObject(root, "stage_total_sum_sec", total_sum);
	cJSON_AddNumberToObject(root, "stage_per_thread_equiv_sec", stage_per_thread_equiv);
	cJSON_AddNumberToObject(root, "avg_total_per_thread_per_batch_sec", per_batch_thread);
	cJSON_AddNumberToObject(root, "avg_total_per_query_sec", per_query);
	cJSON_AddNumberToObject(root, "approx_non_stage_overhead_sec", overhead);
	mm_dump_json_add_uint64(root, "pool_reset_count", ts->pool_reset_count);

	std::string filepath = stats_dir + "/timer_summary.json";
	int ret = mm_dump_write_json_file(filepath, root);
	cJSON_Delete(root);
	return ret;
}

} // extern "C"
