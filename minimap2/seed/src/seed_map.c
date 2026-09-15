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
 * @file seed_map.c
 * @brief Main mm_map_seed implementation for the seeding stage
 *
 * This file contains mm_map_seed, the unified API for Stage 1 (seeding)
 * of the minimap2 mapping pipeline. It operates on chain_read_t structures
 * and is called from map.c, CLI tools, and tests.
 */

#include <stdio.h>
#include <string.h>
#include "seed_priv.h"
#include "chain_read.h"
#include "kalloc.h"
#include "misc.h"
#include "mm_log.h"
#include "mm_profiler.h"
#include "mm_timer.h"
#include "mm_dump.h"
#include "core_ctx.h"

/**
 * mm_map_seed: STAGE 1 - Find k-mer matches between query and index
 *
 * First stage of the mapping pipeline:
 * 1. Collect k-mer minimizers from query using mm_sketch()
 * 2. Filter minimizers by seed quality/occupancy
 * 3. Look up minimizers in the index to find candidate hits
 * 4. Collect and return matched anchors (seed hits)
 *
 * @param mi      Minimap2 index
 * @param opt     Mapping options
 * @param read_   Query read structure (input/output)
 * @param t_seed  Timer for seeding stage (can be NULL)
 * @param km      Memory pool for allocation
 *
 * Input from read_:
 *   - n_seg: Number of query segments
 *   - qlens: Array of query lengths
 *   - qseqs: Array of query sequences
 *   - seq.name: Query name
 *
 * Output to read_:
 *   - a: Array of seed anchors
 *   - n: Number of anchors
 *   - rep_len: Repetitive length
 *   - n_mini_pos: Number of minimizer positions
 *   - mini_pos: Minimizer positions array
 *   - seq.qlen_sum: Sum of query lengths
 */
void mm_map_seed(const mm_idx_t* mi, const mm_mapopt_t* opt,
    chain_read_t* read_, mm_stage_timer_t* t_seed, void* km,
    const mm_core_ctx_t* ctx)
{
	MM_PROFILE_SEED("mm_map_seed");

	int n_segs = read_->n_seg;
	const int* qlens = read_->qlens;
	const char** seqs = read_->qseqs;
	const char* qname = read_->seq.name;
	int* rep_len = &read_->rep_len;
	int* qlen_sum = &read_->seq.qlen_sum;
	int* n_mini_pos = &read_->n_mini_pos;
	uint64_t** mini_pos = &read_->mini_pos;
	int64_t* n_a = &read_->n;
	mm128_t** a = &read_->a;

	int i;
	double t1 = t_seed ? realtime() : 0.0;
	mm128_v mv = {0, 0, 0};

#ifdef MM_DEBUG_SEED_VERBOSE
	// DEBUG: Print input values
	mm_log_debug("=== mm_map_seed INPUT ===");
	mm_log_debug("qname: %s", qname);
	mm_log_debug("n_segs: %d", n_segs);
	for (i = 0; i < n_segs && i < 3; ++i) {
		mm_log_debug("qlens[%d]: %d", i, qlens[i]);
		if (seqs[i]) {
			mm_log_debug("seq[%d] (first 50bp): %.50s...", i, seqs[i]);
		}
	}
	mm_log_debug("Initial n_a: %lld", (long long)*n_a);
	mm_log_debug("Initial rep_len: %d", *rep_len);
#endif

	/* Compute qlen_sum */
	for (i = 0, *qlen_sum = 0; i < n_segs; ++i) *qlen_sum += qlens[i];

	/* Early exit for invalid input */
	if (*qlen_sum == 0 || n_segs <= 0 || n_segs > MM_MAX_SEG) {
		*n_a = 0;
		*rep_len = 0;
		*n_mini_pos = 0;
		*mini_pos = NULL;
		*a = NULL;
		MM_PROFILE_RANGE_POP();
		return;
	}
	if (opt->max_qlen > 0 && *qlen_sum > opt->max_qlen) {
		*n_a = 0;
		*rep_len = 0;
		*n_mini_pos = 0;
		*mini_pos = NULL;
		*a = NULL;
		MM_PROFILE_RANGE_POP();
		return;
	}

	/* Stage 1a: Collect minimizers from all query segments */
	mm_collect_minimizers(km, mi, n_segs, qlens, seqs, opt->sdust_thres, &mv);

	/* Stage 1b: Filter high-frequency minimizers */
	if (opt->q_occ_frac > 0.0f)
		mm_seed_mz_flt(km, &mv, opt->mid_occ, opt->q_occ_frac);

	/* Stage 1c: Collect seed hits using heap or radix sort */
	if (opt->flag & MM_F_HEAP_SORT)
		*a = collect_seed_hits_heap(km, opt, opt->mid_occ, mi, qname, &mv,
		    *qlen_sum, n_a, rep_len, n_mini_pos, mini_pos);
	else
		*a = collect_seed_hits(km, opt, opt->mid_occ, mi, qname, &mv,
		    *qlen_sum, n_a, rep_len, n_mini_pos, mini_pos);

	/* Debug output: print seeds when --print-seed is used */
	if (mm_core_ctx_dbg_flag(ctx) & MM_DBG_PRINT_SEED) {
		mm_log_debug("RS\t%d", *rep_len);
		for (i = 0; i < *n_a; ++i)
			mm_log_debug("SD\t%s\t%d\t%c\t%d\t%d\t%d", mi->seq[(*a)[i].x << 1 >> 33].name, (int32_t)(*a)[i].x, "+-"[(*a)[i].x >> 63], (int32_t)(*a)[i].y, (int32_t)((*a)[i].y >> 32 & 0xff),
			    i == 0 ? 0 : ((int32_t)(*a)[i].y - (int32_t)(*a)[i - 1].y) - ((int32_t)(*a)[i].x - (int32_t)(*a)[i - 1].x));
	}

	read_->n_minimizers = (int)mv.n;

#ifdef MM_DEBUG_SEED_VERBOSE
	// DEBUG: Print output values
	mm_log_debug("=== mm_map_seed OUTPUT ===");
	mm_log_debug("qlen_sum: %d", *qlen_sum);
	mm_log_debug("n_a (number of anchors): %lld", (long long)*n_a);
	mm_log_debug("rep_len: %d", *rep_len);
	mm_log_debug("n_mini_pos: %d", *n_mini_pos);
	mm_log_debug("mv.n (minimizers): %lld", (long long)mv.n);
	if (*n_a > 0) {
		// Print all anchors for test data generation
		mm_log_debug("All %lld anchors:", (long long)*n_a);
		for (i = 0; i < *n_a; ++i) {
			mm_log_debug("  a[%d]: x=%llu, y=%u",
			    i, (unsigned long long)(*a)[i].x, (uint32_t)(*a)[i].y);
		}
	}
	mm_log_debug("=========================");
#endif

	mm_dump_seeds(ctx, qname, read_, mi);

	// Keep minimizers only for potential re-chaining; otherwise free now.
	if (opt->max_occ > opt->mid_occ && *rep_len > 0 && !(opt->flag & MM_F_RMQ)) {
		read_->minimizers_a = mv.a;
		read_->minimizers_n = (int64_t)mv.n;
		mv.a = NULL;
	}
	kfree(km, mv.a);

	/* Update timer */
	mm_stage_timer_record(t_seed, realtime() - t1);

	MM_PROFILE_RANGE_POP();
}
