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

#ifndef _PLUTILS_H_
#define _PLUTILS_H_

#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kalloc.h"
#include "minimap.h"
#include "mm2_seed.h" /* For collect_seed_hits, collect_seed_hits_heap wrappers */
#include "misc.h" /* For radix_sort_128x */
#include "chain_read.h"
#include "plchain.h" /* gpu_chain_ctx_t, lifecycle, getters, config types */

/* Chaining Options */

typedef struct {
	int max_iter, max_dist_x, max_dist_y, max_skip, bw, min_cnt, min_score,
	    is_cdna, n_seg;
	float chn_pen_gap, chn_pen_skip;
} Misc;

typedef struct seg_t {
	size_t start_idx;
	size_t end_idx;
//DEBUG: used for debug plchain_cal_long_seg_range_dis LONG_SEG_RANGE_DIS
#ifdef DEBUG_VERBOSE
	size_t start_segid;
	size_t end_segid;
#endif // DEBUG_VERBOSE
} seg_t;

// Build Misc structure with chaining parameters
static inline Misc build_misc(const mm_idx_t* mi, const mm_mapopt_t* opt, const int64_t qlen_sum, const int n_seg)
{
	int max_chain_gap_qry, max_chain_gap_ref, is_splice = !!(opt->flag & MM_F_SPLICE), is_sr = !!(opt->flag & MM_F_SR);
	float chn_pen_gap, chn_pen_skip;

	// set max chaining gap on the query and the reference sequence
	if (is_sr)
		max_chain_gap_qry = qlen_sum > opt->max_gap ? qlen_sum : opt->max_gap;
	else
		max_chain_gap_qry = opt->max_gap;
	if (opt->max_gap_ref > 0) {
		max_chain_gap_ref = opt->max_gap_ref; // always honor mm_mapopt_t::max_gap_ref if set
	} else if (opt->max_frag_len > 0) {
		max_chain_gap_ref = opt->max_frag_len - qlen_sum;
		if (max_chain_gap_ref < opt->max_gap) max_chain_gap_ref = opt->max_gap;
	} else
		max_chain_gap_ref = opt->max_gap;

	chn_pen_gap = opt->chain_gap_scale * 0.01 * mi->k;
	chn_pen_skip = opt->chain_skip_scale * 0.01 * mi->k;

	Misc misc;
	misc.max_iter = opt->max_chain_iter;
	misc.max_dist_y = max_chain_gap_qry;
	misc.max_dist_x = max_chain_gap_ref;
	misc.max_skip = opt->max_chain_skip;
	misc.bw = opt->bw;
	misc.min_cnt = opt->min_cnt;
	misc.min_score = opt->min_chain_score;
	misc.is_cdna = is_splice;
	misc.n_seg = n_seg;
	misc.chn_pen_gap = chn_pen_gap;
	misc.chn_pen_skip = chn_pen_skip;

	return misc;
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/* <lchain.c> Chaining backtracking methods */
uint64_t* mg_chain_backtrack(void* km, int64_t n, const int32_t* f,
    const int64_t* p, int32_t* v, int32_t* t,
    int32_t min_cnt, int32_t min_sc, int32_t max_drop,
    int32_t* n_u_, int32_t* n_v_);
mm128_t* compact_a(void* km, int32_t n_u, uint64_t* u, int32_t n_v, int32_t* v, mm128_t* a);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // _PLUTILS_H_
