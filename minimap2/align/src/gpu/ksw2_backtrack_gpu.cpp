/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
/**
 * @file ksw2_backtrack_gpu.cpp
 * @brief GPU kernel for batched CIGAR backtracking
 *
 * One thread per alignment. Each thread traces back through the DP
 * backtrack matrix (p) and generates RLE-compressed CIGAR ops.
 * Data is already on GPU from the DP kernel — no extra D2H needed.
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include "ksw2.h"

/**
 * GPU backtracking kernel — one thread per alignment.
 *
 * Reads p/off/off_end data already resident on GPU from the DP kernel.
 * Produces RLE-compressed CIGAR into a pre-allocated contiguous buffer.
 *
 * Parameters:
 *   p          - contiguous backtrack matrix (all alignments, indexed via p_ps)
 *   off        - contiguous off array (indexed via off_ps)
 *   off_end    - contiguous off_end array (indexed via off_ps)
 *   p_ps       - prefix sums into p for each alignment
 *   off_ps     - prefix sums into off/off_end for each alignment
 *   ez         - DP kernel output (scores, max positions, zdrop status)
 *   qlen       - per-alignment query length
 *   tlen       - per-alignment target length
 *   w          - bandwidth parameter
 *   flag       - unified alignment flags for this batch
 *   end_bonus  - unified end bonus for this batch
 *   cigar_buf  - output: contiguous CIGAR buffer (pre-allocated)
 *   cigar_off  - per-alignment offsets into cigar_buf
 *   n_cigar_out- output: actual number of CIGAR ops per alignment
 *   reach_end_out - output: reach_end flag (updated for case2 alignments)
 *   num_align  - total number of alignments
 */
__global__ void ksw_backtrack_batch_kernel(
    const uint8_t* p,
    const int* off,
    const int* off_end,
    const uint64_t* p_ps,
    const uint32_t* off_ps,
    const ksw_extz_t* ez,
    const int* qlen,
    const int* tlen,
    const int* w,
    int flag,
    int end_bonus,
    uint32_t* cigar_buf,
    const uint32_t* cigar_off,
    int* n_cigar_out,
    int* reach_end_out,
    int num_align)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num_align) return;

	/* Skip SCORE_ONLY alignments */
	if (flag & KSW_EZ_SCORE_ONLY) {
		n_cigar_out[tid] = 0;
		reach_end_out[tid] = ez[tid].reach_end;
		return;
	}

	/* Read ez results for this alignment */
	int is_zdropped = ez[tid].zdropped;
	int max_t = ez[tid].max_t;
	int max_q = ez[tid].max_q;
	int mqe = ez[tid].mqe;
	int mqe_t = ez[tid].mqe_t;
	uint32_t max_score = ez[tid].max;
	int reach_end = ez[tid].reach_end;

	/* Determine start position (i0, j0) for backtracking */
	int i0, j0;
	int do_backtrack = 1;

	if (!is_zdropped && !(flag & KSW_EZ_EXTZ_ONLY)) {
		/* Case 1: full alignment, backtrack from end */
		i0 = tlen[tid] - 1;
		j0 = qlen[tid] - 1;
	} else if (!is_zdropped && (flag & KSW_EZ_EXTZ_ONLY) &&
	    mqe + end_bonus > (int)max_score) {
		/* Case 2: extension-only, query reached end */
		reach_end = 1;
		i0 = mqe_t;
		j0 = qlen[tid] - 1;
	} else if (max_t >= 0 && max_q >= 0) {
		/* Case 3: backtrack from max position (zdropped or extension) */
		i0 = max_t;
		j0 = max_q;
	} else {
		/* No valid backtrack position */
		do_backtrack = 0;
	}

	reach_end_out[tid] = reach_end;

	if (!do_backtrack) {
		n_cigar_out[tid] = 0;
		return;
	}

	/* Compute n_col for this alignment (matches host computation) */
	int ql = qlen[tid], tl = tlen[tid];
	int w_tmp = w[tid];
	if (w_tmp < 0) w_tmp = (tl > ql) ? tl : ql;
	int n_col_ = (ql < tl) ? ql : tl;
	n_col_ = ((n_col_ < w_tmp + 1 ? n_col_ : w_tmp + 1) + 15) / 16 + 1;
	int n_col = n_col_ * 16;

	/* Get pointers into contiguous data */
	const uint8_t* my_p = p + p_ps[tid];
	const int* my_off = off + off_ps[tid];
	const int* my_off_end = off_end + off_ps[tid];

	/* Output CIGAR buffer for this alignment */
	uint32_t* my_cigar = cigar_buf + cigar_off[tid];
	int n_cigar = 0;
	int max_cigar = (int)(cigar_off[tid + 1] - cigar_off[tid]); /* available space */

	/* Backtracking loop (is_rot=1, min_intron_len=0) */
	int i = i0, j = j0;
	int state = 0;

	while (i >= 0 && j >= 0) {
		int force_state = -1;
		int r = i + j;

		if (i < my_off[r]) force_state = 2;
		if (i > my_off_end[r]) force_state = 1;

		uint8_t tmp = (force_state < 0) ? my_p[(size_t)r * n_col + i - my_off[r]] : 0;

		if (state == 0)
			state = tmp & 7;
		else if (!(tmp >> (state + 2) & 1))
			state = 0;
		if (state == 0) state = tmp & 7;
		if (force_state >= 0) state = force_state;

		/* Determine CIGAR op */
		uint32_t op;
		if (state == 0) {
			op = KSW_CIGAR_MATCH;
			--i;
			--j;
		} else if (state == 1 || state == 3) {
			op = KSW_CIGAR_DEL; /* min_intron_len=0, so state 3 is also DEL */
			--i;
		} else {
			op = KSW_CIGAR_INS;
			--j;
		}

		/* RLE push: merge with previous if same op */
		if (n_cigar > 0 && op == (my_cigar[n_cigar - 1] & 0xf)) {
			my_cigar[n_cigar - 1] += (1 << 4);
		} else {
			if (n_cigar < max_cigar) {
				my_cigar[n_cigar++] = (1 << 4) | op;
			}
		}
	}

	/* Handle remaining i or j */
	if (i >= 0) {
		uint32_t op = KSW_CIGAR_DEL;
		int len = i + 1;
		if (n_cigar > 0 && op == (my_cigar[n_cigar - 1] & 0xf)) {
			my_cigar[n_cigar - 1] += (len << 4);
		} else if (n_cigar < max_cigar) {
			my_cigar[n_cigar++] = (len << 4) | op;
		}
	}
	if (j >= 0) {
		uint32_t op = KSW_CIGAR_INS;
		int len = j + 1;
		if (n_cigar > 0 && op == (my_cigar[n_cigar - 1] & 0xf)) {
			my_cigar[n_cigar - 1] += (len << 4);
		} else if (n_cigar < max_cigar) {
			my_cigar[n_cigar++] = (len << 4) | op;
		}
	}

	/* Reverse CIGAR if needed (is_rev determined by REV_CIGAR flag) */
	int is_rev = !!(flag & KSW_EZ_REV_CIGAR);
	if (!is_rev) {
		/* Reverse in place */
		for (int k = 0; k < n_cigar / 2; ++k) {
			uint32_t t = my_cigar[k];
			my_cigar[k] = my_cigar[n_cigar - 1 - k];
			my_cigar[n_cigar - 1 - k] = t;
		}
	}

	n_cigar_out[tid] = n_cigar;
}
