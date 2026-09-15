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
 * @file chain_map.c
 * @brief Implementation of mm_map_chain - the chaining stage entry point
 *
 * This file implements mm_map_chain(), the second stage of minimap2's mapping
 * pipeline. It links individual seed anchors into colinear chains.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "chain_priv.h"
#include "kalloc.h"
#include "misc.h" /* For radix_sort_128x, realtime() */
#include "mm_log.h"
#include "mm_profiler.h"
#include "mm_timer.h"
#include "mm_dump.h"

/**
 * mm_map_chain: STAGE 2 - Link seed hits into chains
 * @param mi minimap2 index (used for k-mer size and re-seeding)
 * @param opt mapping options
 * @param read_ query read structure (containing seed anchors from Stage 1)
 * @param t_chain pointer to chain timer (can be NULL to skip timing)
 * @param km memory pool for allocation
 * 
 * Second stage of the mapping pipeline:
 * 1. Calculates chaining gap parameters based on query/reference spans
 * 2. Calls chaining algorithm (mg_lchain_dp or mg_lchain_rmq)
 * 3. Handles long-range chaining if bw_long mode is enabled
 * 4. Re-chains with higher max_occ threshold if chains incomplete (short reads)
 * 
 * Links individual seed anchors into chains that represent colinear matches.
 * Output stored in read_->u (chain info) and chains point back into read_->a.
 */
void mm_map_chain(const mm_idx_t* mi, const mm_mapopt_t* opt,
    chain_read_t* read_, mm_stage_timer_t* t_chain, void* km, const mm_core_ctx_t* ctx)
{
	MM_PROFILE_CHAIN("mm_map_chain");

	int n_segs = read_->n_seg;
	const char* qname = read_->seq.name;
	int* rep_len = &read_->rep_len;
	int* frag_gap = &read_->frag_gap;
	int* qlen_sum = &read_->seq.qlen_sum;
	int* n_regs0 = &read_->n_u;
	int* n_mini_pos = &read_->n_mini_pos;
	uint64_t** mini_pos = &read_->mini_pos;
	int64_t* n_a = &read_->n;
	uint64_t** u = &read_->u;
	mm128_t** a = &read_->a;

#ifdef MM_DEBUG_CHAIN_VERBOSE
	// Debug: print input anchors before chaining (limit to first/last 10 for large sets)
	mm_log_debug("DEBUG_CHAIN_INPUT: qname=%s n_seg=%d qlen_sum=%d n_anchors=%lld rep_len=%d n_mini_pos=%d",
	    qname ? qname : "(null)", n_segs, *qlen_sum, (long long)*n_a, *rep_len, *n_mini_pos);
	if (*n_a > 0) {
		// Print all anchors for test data generation
		mm_log_debug("All %lld anchors:", (long long)*n_a);
		for (int64_t di = 0; di < *n_a; di++) {
			mm_log_debug("DEBUG_CHAIN_ANCHOR_IN: %lld x=0x%llx y=0x%llx",
			    (long long)di, (unsigned long long)(*a)[di].x, (unsigned long long)(*a)[di].y);
		}
	}
#endif

	int i;
	int max_chain_gap_qry, max_chain_gap_ref;
	int is_splice = !!(opt->flag & MM_F_SPLICE);
	int is_sr = !!(opt->flag & MM_F_SR);
	mm128_v mv = {0, 0, 0};
	int regenerated_mv = 0;
	float chn_pen_gap, chn_pen_skip;
	double t1 = realtime();

	if (read_->minimizers_a && read_->minimizers_n > 0) {
		mv.a = read_->minimizers_a;
		mv.n = mv.m = (size_t)read_->minimizers_n;
	}

	// set max chaining gap on the query and the reference sequence
	if (is_sr)
		max_chain_gap_qry = *qlen_sum > opt->max_gap ? *qlen_sum : opt->max_gap;
	else
		max_chain_gap_qry = opt->max_gap;
	if (opt->max_gap_ref > 0) {
		max_chain_gap_ref = opt->max_gap_ref; // always honor mm_mapopt_t::max_gap_ref if set
	} else if (opt->max_frag_len > 0) {
		max_chain_gap_ref = opt->max_frag_len - *qlen_sum;
		if (max_chain_gap_ref < opt->max_gap) max_chain_gap_ref = opt->max_gap;
	} else
		max_chain_gap_ref = opt->max_gap;

	chn_pen_gap = opt->chain_gap_scale * 0.01 * mi->k;
	chn_pen_skip = opt->chain_skip_scale * 0.01 * mi->k;

	if (opt->flag & MM_F_RMQ) {
		*a = mg_lchain_rmq(opt->max_gap, opt->rmq_inner_dist, opt->bw, opt->max_chain_skip, opt->rmq_size_cap, opt->min_cnt, opt->min_chain_score,
		    chn_pen_gap, chn_pen_skip, *n_a, *a, n_regs0, u, km);
	} else {
		*a = mg_lchain_dp(max_chain_gap_ref, max_chain_gap_qry, opt->bw, opt->max_chain_skip, opt->max_chain_iter, opt->min_cnt, opt->min_chain_score,
		    chn_pen_gap, chn_pen_skip, is_splice, n_segs, *n_a, *a, n_regs0, u, km);
	}

	// Long-join re-chaining for long sequences (uses RMQ with wider bandwidth)
	if (opt->bw_long > opt->bw && (opt->flag & (MM_F_SPLICE | MM_F_SR | MM_F_NO_LJOIN)) == 0 && n_segs == 1 && *n_regs0 > 1) {
		int32_t st = (int32_t)(*a)[0].y, en = (int32_t)(*a)[(int32_t)(*u)[0] - 1].y;
		if (*qlen_sum - (en - st) > opt->rmq_rescue_size || en - st > *qlen_sum * opt->rmq_rescue_ratio) {
			int32_t i;
			for (i = 0, *n_a = 0; i < *n_regs0; ++i) *n_a += (int32_t)(*u)[i];
			kfree(km, *u);
			radix_sort_128x(*a, (*a) + *n_a);
			*a = mg_lchain_rmq(opt->max_gap, opt->rmq_inner_dist, opt->bw_long, opt->max_chain_skip, opt->rmq_size_cap, opt->min_cnt, opt->min_chain_score,
			    chn_pen_gap, chn_pen_skip, *n_a, *a, n_regs0, u, km);
		}
	} else if (opt->max_occ > opt->mid_occ && *rep_len > 0 && !(opt->flag & MM_F_RMQ)) { // re-chain, mostly for short reads
		int rechain = 0;
		if (*n_regs0 > 0) { // test if the best chain has all the segments
			int n_chained_segs = 1, max = 0, max_i = -1, max_off = -1, off = 0;
			for (i = 0; i < *n_regs0; ++i) { // find the best chain
				if (max < (int)((*u)[i] >> 32)) max = (*u)[i] >> 32, max_i = i, max_off = off;
				off += (uint32_t)(*u)[i];
			}
			for (i = 1; i < (int32_t)(*u)[max_i]; ++i) // count the number of segments in the best chain
				if (((*a)[max_off + i].y & MM_SEED_SEG_MASK) != ((*a)[max_off + i - 1].y & MM_SEED_SEG_MASK))
					++n_chained_segs;
			if (n_chained_segs < n_segs)
				rechain = 1;
		} else
			rechain = 1;
		if (rechain) { // redo chaining with a higher max_occ threshold
			kfree(km, *a);
			kfree(km, *u);
			kfree(km, *mini_pos);
			if (mv.a == NULL) {
				// Fallback for paths that don't cache minimizers from seed stage.
				mm_collect_minimizers(km, mi, n_segs, read_->qlens, read_->qseqs, opt->sdust_thres, &mv);
				if (opt->q_occ_frac > 0.0f) mm_seed_mz_flt(km, &mv, opt->mid_occ, opt->q_occ_frac);
				regenerated_mv = 1;
			}
			if (opt->flag & MM_F_HEAP_SORT)
				*a = collect_seed_hits_heap(km, opt, opt->max_occ, mi, qname, &mv, *qlen_sum, n_a, rep_len, n_mini_pos, mini_pos);
			else
				*a = collect_seed_hits(km, opt, opt->max_occ, mi, qname, &mv, *qlen_sum, n_a, rep_len, n_mini_pos, mini_pos);
			*a = mg_lchain_dp(max_chain_gap_ref, max_chain_gap_qry, opt->bw, opt->max_chain_skip, opt->max_chain_iter, opt->min_cnt, opt->min_chain_score,
			    chn_pen_gap, chn_pen_skip, is_splice, n_segs, *n_a, *a, n_regs0, u, km);
		}
	}
	if (mv.a) {
		kfree(km, mv.a);
		if (!regenerated_mv) {
			read_->minimizers_a = NULL;
			read_->minimizers_n = 0;
		}
	}
	*frag_gap = max_chain_gap_ref;

	mm_stage_timer_record(t_chain, realtime() - t1);

#ifdef MM_DEBUG_CHAIN_VERBOSE
	// Debug: print chains after chaining
	mm_log_debug("=== mm_map_chain Results ===");
	mm_log_debug("Query: %s (length: %d)", qname ? qname : "(null)", *qlen_sum);
	mm_log_debug("Number of chains: %d", *n_regs0);
	mm_log_debug("Fragment gap: %d", *frag_gap);

	int64_t anchor_offset = 0;
	for (i = 0; i < *n_regs0; i++) {
		int32_t chain_len = (int32_t)((*u)[i]);
		int32_t chain_score = (int32_t)((*u)[i] >> 32);
		mm_log_debug("Chain %d: score=%d, anchors=%d", i, chain_score, chain_len);

		if (chain_len > 0) {
			uint32_t first_qpos = (uint32_t)((*a)[anchor_offset].y & 0xFFFFFFFF);
			uint32_t first_tpos = (uint32_t)((*a)[anchor_offset].x);
			uint32_t last_qpos = (uint32_t)((*a)[anchor_offset + chain_len - 1].y & 0xFFFFFFFF);
			uint32_t last_tpos = (uint32_t)((*a)[anchor_offset + chain_len - 1].x);
			mm_log_debug("  First anchor: qpos=%u, tpos=%u", first_qpos, first_tpos);
			mm_log_debug("  Last anchor:  qpos=%u, tpos=%u", last_qpos, last_tpos);
			mm_log_debug("  Span: query=%u, target=%u", last_qpos - first_qpos, last_tpos - first_tpos);
		}
		anchor_offset += chain_len;
	}
	mm_log_debug("============================");
#endif

	/* Dump chain intermediates (--dump-intermediates) */
	mm_dump_chains(ctx, qname, read_, mi);

	MM_PROFILE_RANGE_POP();
}
