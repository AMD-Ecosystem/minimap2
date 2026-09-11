/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2018-2022 Heng Li; Dana-Farber Cancer Institute; Broad Institute
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
 * @file esterr.c
 * @brief Error rate estimation for sequence alignments
 *
 * Estimates sequence divergence between query and reference based on
 * minimizer positions. Used by the alignment stage to compute div field
 * in mm_reg1_t which influences MAPQ calculation.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "minimap.h"
#include "misc.h" /* for mm_verbose */
#include "mm_log.h"
#include "mm_profiler.h"
#include "core_ctx.h"

static inline int32_t get_for_qpos(int32_t qlen, const mm128_t* a)
{
	int32_t x = (int32_t)a->y;
	int32_t q_span = a->y >> 32 & 0xff;
	if (a->x >> 63)
		x = qlen - 1 - (x + 1 - q_span); // revert the position to the forward strand of query
	return x;
}

static int get_mini_idx(int qlen, const mm128_t* a, int32_t n, const uint64_t* mini_pos)
{
	int32_t x, L = 0, R = n - 1;
	x = get_for_qpos(qlen, a);
	while (L <= R) { // binary search
		int32_t m = ((uint64_t)L + R) >> 1;
		int32_t y = (int32_t)mini_pos[m];
		if (y < x)
			L = m + 1;
		else if (y > x)
			R = m - 1;
		else
			return m;
	}
	return -1;
}

void mm_est_err(const mm_idx_t* mi, int qlen, int n_regs, mm_reg1_t* regs, const mm128_t* a, int32_t n, const uint64_t* mini_pos,
    const mm_core_ctx_t* ctx)
{
	MM_PROFILE_ALIGN("est_err");
	int i;
	uint64_t sum_k = 0;
	float avg_k;

	if (n == 0) {
		MM_PROFILE_RANGE_POP();
		return;
	}
	for (i = 0; i < n; ++i)
		sum_k += mini_pos[i] >> 32 & 0xff;
	avg_k = (float)sum_k / n;

	for (i = 0; i < n_regs; ++i) {
		mm_reg1_t* r = &regs[i];
		int32_t st, en, j, k, n_match, n_tot, l_ref;
		r->div = -1.0f;
		if (r->cnt == 0) continue;
		st = en = get_mini_idx(qlen, r->rev ? &a[r->as + r->cnt - 1] : &a[r->as], n, mini_pos);
		if (st < 0) {
			if (mm_core_ctx_verbose(ctx) >= 2)
				mm_log("[WARNING] logic inconsistency in mm_est_err(). Please contact the developer.");
			continue;
		}
		l_ref = mi->seq[r->rid].len;
		for (k = 1, j = st + 1, n_match = 1; j < n && k < r->cnt; ++j) {
			int32_t x;
			x = get_for_qpos(qlen, r->rev ? &a[r->as + r->cnt - 1 - k] : &a[r->as + k]);
			if (x == (int32_t)mini_pos[j])
				++k, en = j, ++n_match;
		}
		n_tot = en - st + 1;
		if (r->qs > avg_k && r->rs > avg_k) ++n_tot;
		if (qlen - r->qs > avg_k && l_ref - r->re > avg_k) ++n_tot;
		r->div = n_match >= n_tot ? 0.0f : (float)(1.0 - pow((double)n_match / n_tot, 1.0 / avg_k));
	}
	MM_PROFILE_RANGE_POP();
}
