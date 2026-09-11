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

#include <hip/hip_runtime.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "ksw2.h"

#define NUM_THREADS 64

#ifndef OLD

__device__ __inline__ int ksw_apply_zdrop_gpu(ksw_extz_t* const ez, const int is_rot, const int32_t H, const int a, const int b, const int zdrop, const int8_t e)
{
	int r, t;
	if (is_rot)
		r = a, t = b;
	else
		r = a + b, t = a;
	if (H > (int32_t)ez->max) {
		ez->max = H, ez->max_t = t, ez->max_q = r - t;
	} else if (t >= ez->max_t && r - t >= ez->max_q) {
		int tl = t - ez->max_t, ql = (r - t) - ez->max_q, l;
		l = tl > ql ? tl - ql : ql - tl;
		if (zdrop >= 0 && ez->max - H > zdrop + l * e) {
			ez->zdropped = 1;
			return 1;
		}
	}
	return 0;
}


__global__ void ksw_extd2_gpu( // void *km,
    int8_t* __restrict__ const u, int8_t* __restrict__ const v, int8_t* __restrict__ const x, int8_t* __restrict__ const y, int8_t* __restrict__ const x2,
    int8_t* __restrict__ const y2, int8_t* __restrict__ const s, int8_t* __restrict__ const p, int32_t* __restrict__ const H, int* __restrict__ const off, int* __restrict__ const off_end,
    const int qlen, const uint8_t* __restrict__ const query, const int tlen, const uint8_t* __restrict__ const target, int8_t m, const int8_t* __restrict__ const mat,
    int8_t q, int8_t e, int8_t q2, int8_t e2, int w, const int zdrop, const int end_bonus, const int flag, ksw_extz_t* __restrict__ const ez)
{
	// printf("in extd2_gpu kernel\n");
	if (m <= 1 || qlen <= 0 || tlen <= 0) return; // end due to invalid alignment

	int r, t, n_col_, last_st, last_en, max_sc, min_sc, long_thres;
	const int with_cigar = !(flag & KSW_EZ_SCORE_ONLY), approx_max = !!(flag & KSW_EZ_APPROX_MAX);

	int32_t H0 = 0, last_H0_t = 0;

	ez->max_q = ez->max_t = ez->mqe_t = ez->mte_q = -1;
	ez->max = 0, ez->score = ez->mqe = ez->mte = KSW_NEG_INF;
	ez->n_cigar = 0, ez->zdropped = 0, ez->reach_end = 0;

	if (q2 + e2 < q + e) t = q, q = q2, q2 = t, t = e, e = e2, e2 = t; // make sure q+e no larger than q2+e2

	const int8_t qe = q + e;
	const int8_t qe2 = q2 + e2;
	const int8_t sc_mch = mat[0];
	const int8_t sc_mis = mat[1];
	const int8_t sc_N = mat[m * m - 1] == 0 ? -e2 : mat[m * m - 1];
	const int8_t m1 = m - 1;

	if (w < 0) w = tlen > qlen ? tlen : qlen; // set bandwidth in case it is negative (non-banded alignment)
	const int wl = w;
	const int wr = w;
	const int tlen_ = (tlen + 15) / 16; // round to multiple of 16
	n_col_ = qlen < tlen ? qlen : tlen;
	n_col_ = ((n_col_ < w + 1 ? n_col_ : w + 1) + 15) / 16 + 1; // round to multiple of 16
	const int qlen_ = (qlen + 15) / 16; // round to multiple of 16

	for (t = 1, max_sc = mat[0], min_sc = mat[1]; t < m * m; ++t) { // set max_sc to maximum score in mat, and min_sc to minimum score in mat
		max_sc = max_sc > mat[t] ? max_sc : mat[t];
		min_sc = min_sc < mat[t] ? min_sc : mat[t];
	}

	if (-min_sc > 2 * (q + e)) return; /* otherwise, we won't see any mismatches */

	long_thres = e != e2 ? (q2 - q) / (e - e2) - 1 : 0; // if double affine, set the threshold for long gaps(?) to ((q2 - q) / (e - e2) - 1), otherwise 0
	if (q2 + e2 + long_thres * e2 > q + e + long_thres * e)
		++long_thres;
	const int long_diff = long_thres * (e - e2) - (q2 - q) - e2;

	for (int i = 0; i < tlen_ * 16; i += blockDim.x)
		if (threadIdx.x + i < tlen_ * 16)
			u[threadIdx.x + i] = -q - e;
	__syncthreads();

	for (int i = 0; i < tlen_ * 16; i += blockDim.x)
		if (threadIdx.x + i < tlen_ * 16)
			v[threadIdx.x + i] = -q - e;
	__syncthreads();

	for (int i = 0; i < tlen_ * 16; i += blockDim.x)
		if (threadIdx.x + i < tlen_ * 16)
			x[threadIdx.x + i] = -q - e;
	__syncthreads();

	for (int i = 0; i < tlen_ * 16; i += blockDim.x)
		if (threadIdx.x + i < tlen_ * 16)
			y[threadIdx.x + i] = -q - e;
	__syncthreads();

	for (int i = 0; i < tlen_ * 16; i += blockDim.x)
		if (threadIdx.x + i < tlen_ * 16)
			x2[threadIdx.x + i] = -q2 - e2;
	__syncthreads();

	for (int i = 0; i < tlen_ * 16; i += blockDim.x)
		if (threadIdx.x + i < tlen_ * 16)
			y2[threadIdx.x + i] = -q2 - e2;
	__syncthreads();

	if (!approx_max) { // if !approximated alignment
		for (int t = 0; t < tlen_ * 16; t += blockDim.x) {
			const int tid = threadIdx.x + t;
			if (tid < tlen_ * 16)
				H[tid] = KSW_NEG_INF; // initialize to -inf (gmem)
		}
		__syncthreads();
	}

	// alignment main loop
	for (r = 0, last_st = last_en = -1; r < qlen + tlen - 1; ++r) {

		int st = 0, en = tlen - 1, st0, en0, st_, en_;
		int8_t x1, x21, v1;
		// const uint8_t* const qrr = query + (qlen - 1 - r);

		/* find the boundaries */
		if (st < r - qlen + 1) st = r - qlen + 1;
		if (en > r) en = r;
		if (st < (r - wr + 1) >> 1) st = (r - wr + 1) >> 1; /* take the ceil */
		if (en > (r + wl) >> 1) en = (r + wl) >> 1; /* take the floor */
		if (st > en) {
			ez->zdropped = 1;
			break;
		}

		st0 = st, en0 = en;
		st = st / 16 * 16, en = (en + 16) / 16 * 16 - 1;

		/* set boundary conditions */
		if (st > 0) {
			if (st - 1 >= last_st && st - 1 <= last_en) {
				x1 = x[st - 1], x21 = x2[st - 1], v1 = v[st - 1]; /* (r-1,s-1) calculated in the last round */
			} else {
				x1 = -q - e, x21 = -q2 - e2;
				v1 = -q - e;
			}
		} else {
			x1 = -q - e, x21 = -q2 - e2;
			v1 = r == 0 ? -q - e : r < long_thres ? -e
			    : r == long_thres		      ? long_diff
							      : -e2;
		}

		if (en >= r) {
			y[r] = -q - e, y2[r] = -q2 - e2; // ((int8_t*)y)[r] = -q - e, ((int8_t*)y2)[r] = -q2 - e2;
			u[r] = r == 0 ? -q - e : r < long_thres ? -e
			    : r == long_thres			? long_diff
								: -e2;
		}

		/* loop fission: set scores first */
		if (!(flag & KSW_EZ_GENERIC_SC)) {

			const int iter = ((en0 - st0) / 16 + 1) * 16 + st0;

			for (int t = st0; t < iter; t += blockDim.x) {
				const int tid = t + threadIdx.x;
				if (tid < iter) {
					const uint8_t tmp = target[tid] == query[qlen - 1 - r + tid] ? sc_mch : sc_mis;
					/* CPU uses OR: sc_N if either base is the wildcard (m1). */
					uint8_t mask = (target[tid] == m1) | (query[qlen - 1 - r + tid] == m1);
					s[tid] = mask ? sc_N : tmp;
				}
			}
			__syncthreads();

		} else {

			for (int t = st0; t <= en0; t += blockDim.x) {
				const int tid = threadIdx.x + t;
				if (tid <= en0) {
					s[tid] = mat[target[tid] * m + query[qlen - 1 - r + tid]]; // ((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
				}
			}
			__syncthreads();
		}


		st_ = st / 16, en_ = en / 16;

		assert(en_ - st_ + 1 <= n_col_);

		if (!with_cigar) { // score only

			int8_t x1_local = x1, x21_local = x21, v1_local = v1;

			for (int t = st; t <= en; t += blockDim.x) {

				const int tid = threadIdx.x + t;
				const int last_lane = (en - t < blockDim.x - 1) ? (en - t) : (blockDim.x - 1);

				if (tid <= en) {

					/* Read prior-diagonal boundary values BEFORE any writes.
					 * Thread 0 of the next chunk needs x[r-1][t+last_lane],
					 * x2[r-1][t+last_lane], v[r-1][t+last_lane] — the values
					 * that will be overwritten below. */
					int8_t x_bnd = __shfl(x[tid], last_lane);
					int8_t x2_bnd = __shfl(x2[tid], last_lane);
					int8_t v_bnd = __shfl(v[tid], last_lane);

					int8_t xt1 = threadIdx.x == 0 ? x1_local : x[tid - 1];
					int8_t x2t1 = threadIdx.x == 0 ? x21_local : x2[tid - 1];
					int8_t vt1 = threadIdx.x == 0 ? v1_local : v[tid - 1];

					int8_t a = xt1 + vt1;
					int8_t ut = u[tid];
					int8_t b = y[tid] + ut;

					int8_t z = s[tid];

					int8_t a2 = x2t1 + vt1;
					int8_t b2 = y2[tid] + ut;
					z = z > a ? z : a;
					z = z > b ? z : b;
					z = z > a2 ? z : a2;
					z = z > b2 ? z : b2;
					z = z < sc_mch ? z : sc_mch;
					u[tid] = z - vt1;
					int8_t v_new = z - ut;
					v[tid] = v_new;
					a = a - (z - q);
					b = b - (z - q);
					a2 = a2 - (z - q2);
					b2 = b2 - (z - q2);
					int8_t x_new = (a > 0 ? a : 0) - qe;
					int8_t y_new = (b > 0 ? b : 0) - qe;
					int8_t x2_new = (a2 > 0 ? a2 : 0) - qe2;
					x[tid] = x_new;
					y[tid] = y_new;
					x2[tid] = x2_new;
					y2[tid] = (b2 > 0 ? b2 : 0) - qe2;

					/* Carry prior-diagonal boundary values for thread 0 of next chunk. */
					x1_local = x_bnd;
					x21_local = x2_bnd;
					v1_local = v_bnd;
				}
			}

			__syncthreads();

		} else if (!(flag & KSW_EZ_RIGHT)) { /* gap left-alignment */
			const size_t size = (size_t)r * n_col_ * 16 - st;
			off[r] = st, off_end[r] = en;

			int8_t x1_local = x1, x21_local = x21, v1_local = v1;

			for (int t = st; t <= en; t += blockDim.x) {

				const int tid = threadIdx.x + t;
				const int last_lane = (en - t < blockDim.x - 1) ? (en - t) : (blockDim.x - 1);

				if (tid <= en) {

					/* Read prior-diagonal boundary values BEFORE any writes. */
					int8_t x_bnd = __shfl(x[tid], last_lane);
					int8_t x2_bnd = __shfl(x2[tid], last_lane);
					int8_t v_bnd = __shfl(v[tid], last_lane);

					int8_t xt1 = threadIdx.x == 0 ? x1_local : x[tid - 1];
					int8_t x2t1 = threadIdx.x == 0 ? x21_local : x2[tid - 1];
					int8_t vt1 = threadIdx.x == 0 ? v1_local : v[tid - 1];

					int8_t a = xt1 + vt1;
					int8_t ut = u[tid];
					int8_t b = y[tid] + ut;

					int8_t z = s[tid];

					int8_t a2 = x2t1 + vt1;
					int8_t b2 = y2[tid] + ut;

					int8_t d = a > z ? 1 : 0;

					z = z > a ? z : a;
					d = b > z ? 2 : d;
					z = z > b ? z : b;
					d = a2 > z ? 3 : d;
					z = z > a2 ? z : a2;
					d = b2 > z ? 4 : d;
					z = z > b2 ? z : b2;
					z = z < sc_mch ? z : sc_mch;

					u[tid] = z - vt1;
					int8_t v_new = z - ut;
					v[tid] = v_new;

					a = a - (z - q);
					b = b - (z - q);
					a2 = a2 - (z - q2);
					b2 = b2 - (z - q2);

					int8_t x_new = (a > 0 ? a : 0) - qe;
					int8_t x2_new = (a2 > 0 ? a2 : 0) - qe2;

					x[tid] = x_new;
					d |= a > 0 ? 0x08 : 0;
					y[tid] = (b > 0 ? b : 0) - qe;
					d |= b > 0 ? 0x10 : 0;
					x2[tid] = x2_new;
					d |= a2 > 0 ? 0x20 : 0;
					y2[tid] = (b2 > 0 ? b2 : 0) - qe2;
					d |= b2 > 0 ? 0x40 : 0;

					p[size + tid] = d;

					/* Carry prior-diagonal boundary values for thread 0 of next chunk. */
					x1_local = x_bnd;
					x21_local = x2_bnd;
					v1_local = v_bnd;
				}
			}

			__syncthreads();

		} else { /* gap right-alignment */
			const size_t size = (size_t)r * (n_col_ * 16) - st;
			off[r] = st, off_end[r] = en;

			int8_t x1_local = x1, x21_local = x21, v1_local = v1;

			for (int t = st; t <= en; t += blockDim.x) {

				const int tid = threadIdx.x + t;
				const int last_lane = (en - t < blockDim.x - 1) ? (en - t) : (blockDim.x - 1);

				if (tid <= en) {

					/* Read prior-diagonal boundary values BEFORE any writes. */
					int8_t x_bnd = __shfl(x[tid], last_lane);
					int8_t x2_bnd = __shfl(x2[tid], last_lane);
					int8_t v_bnd = __shfl(v[tid], last_lane);

					int8_t xt1 = threadIdx.x == 0 ? x1_local : x[tid - 1];
					int8_t x2t1 = threadIdx.x == 0 ? x21_local : x2[tid - 1];
					int8_t vt1 = threadIdx.x == 0 ? v1_local : v[tid - 1];

					int8_t a = xt1 + vt1;
					int8_t ut = u[tid];
					int8_t b = y[tid] + ut;

					int8_t z = s[tid];

					int8_t a2 = x2t1 + vt1;
					int8_t b2 = y2[tid] + ut;

					int8_t d;

					d = z > a ? 0 : 1;
					z = z > a ? z : a;
					d = z > b ? d : 2;
					z = z > b ? z : b;
					d = z > a2 ? d : 3;
					z = z > a2 ? z : a2;
					d = z > b2 ? d : 4;
					z = z > b2 ? z : b2;
					z = z < sc_mch ? z : sc_mch;

					u[tid] = z - vt1;
					int8_t v_new = z - ut;
					v[tid] = v_new;

					a = a - (z - q);
					b = b - (z - q);
					a2 = a2 - (z - q2);
					b2 = b2 - (z - q2);

					int8_t x_new = (a >= 0 ? a : 0) - qe;
					int8_t x2_new = (a2 >= 0 ? a2 : 0) - qe2;

					x[tid] = x_new;
					d |= a >= 0 ? 0x08 : 0;
					y[tid] = (b >= 0 ? b : 0) - qe;
					d |= b >= 0 ? 0x10 : 0;
					x2[tid] = x2_new;
					d |= a2 >= 0 ? 0x20 : 0;
					y2[tid] = (b2 >= 0 ? b2 : 0) - qe2;
					d |= b2 >= 0 ? 0x40 : 0;

					p[size + tid] = d;

					/* Carry prior-diagonal boundary values for thread 0 of next chunk. */
					x1_local = x_bnd;
					x21_local = x2_bnd;
					v1_local = v_bnd;
				}
			}

			__syncthreads();
		}

		if (!approx_max) { /* find the exact max with a 32-bit score array */

			int32_t max_H, max_t;
			/* compute H[], max_H and max_t */
			if (r > 0) {

				int32_t max_H_local, max_t_local;

				/* special casing the last element - this is the initial max candidate */
				int32_t H_en0 = en0 > 0 ? H[en0 - 1] + u[en0] : H[en0] + v[en0];
				H[en0] = H_en0;
				max_H = H_en0;
				max_t = en0;

				/* Initialize local max to invalid values - we'll compare against global max later */
				max_H_local = INT32_MIN;
				max_t_local = -1;

				for (t = st0; t < en0; t += blockDim.x) { /* this implements: H[t]+=v8[t]-qe; if(H[t]>max_H) max_H=H[t],max_t=t; */
					const int tid = threadIdx.x + t;
					if (tid < en0) {
						H[tid] += v[tid];
						if (H[tid] > max_H_local) {
							max_H_local = H[tid];
							max_t_local = tid;
						}
					}
				}
				__syncthreads();

				/* Parallel reduction to find max across threads */
				for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
					int tmp_H = __shfl_down(max_H_local, offset);
					int tmp_t = __shfl_down(max_t_local, offset);

					if (tmp_H > max_H_local) {
						max_H_local = tmp_H;
						max_t_local = tmp_t;
					}
				}

				/* Thread 0 has the max from parallel portion, now compare with H[en0] */
				if (threadIdx.x == 0) {
					int32_t reduced_max_H = max_H_local;
					int32_t reduced_max_t = max_t_local;

					/* Compare reduced max with the initial H[en0] value */
					if (reduced_max_H > max_H) {
						max_H = reduced_max_H;
						max_t = reduced_max_t;
					}
				}

				/* Broadcast final max_H and max_t to all threads */
				max_H = __shfl(max_H, 0);
				max_t = __shfl(max_t, 0);

				__syncthreads();

				/*for (t = en1; t < en0; ++t) {  // for the rest of values that haven't been computed with SSE 
					H[t] += (int32_t)v[t];
					if (H[t] > max_H)
						max_H = H[t], max_t = t;
				}*/

			} else
				H[0] = v[0] - qe, max_H = H[0], max_t = 0; /* special casing r==0 */

			/* update ez */
			if (en0 == tlen - 1 && H[en0] > ez->mte)
				ez->mte = H[en0], ez->mte_q = r - en0;
			if (r - st0 == qlen - 1 && H[st0] > ez->mqe)
				ez->mqe = H[st0], ez->mqe_t = st0;
			if (ksw_apply_zdrop_gpu(ez, 1, max_H, r, max_t, zdrop, e2)) break; // if score drops too fast, interrupt alignment
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez->score = H[tlen - 1];

		} else { /* find approximate max; Z-drop might be inaccurate, too. */

			if (r > 0) {
				if (last_H0_t >= st0 && last_H0_t <= en0 && last_H0_t + 1 >= st0 && last_H0_t + 1 <= en0) {
					int32_t d0 = v[last_H0_t];
					int32_t d1 = u[last_H0_t + 1];
					if (d0 > d1)
						H0 += d0;
					else
						H0 += d1, ++last_H0_t;
				} else if (last_H0_t >= st0 && last_H0_t <= en0) {
					H0 += v[last_H0_t];
				} else {
					++last_H0_t, H0 += u[last_H0_t];
				}
			} else
				H0 = v[0] - qe, last_H0_t = 0;
			if ((flag & KSW_EZ_APPROX_DROP) && ksw_apply_zdrop_gpu(ez, 1, H0, r, last_H0_t, zdrop, e2)) break;
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez->score = H0;
		}
		last_st = st, last_en = en;
	}

	// kfree(km, mem);
	// if (!approx_max) kfree(km, H);
	// if (with_cigar) { // backtrack
	// 	int rev_cigar = !!(flag & KSW_EZ_REV_CIGAR);
	// 	if (!ez->zdropped && !(flag&KSW_EZ_EXTZ_ONLY)) {
	// 		ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*16, tlen-1, qlen-1, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
	// 	} else if (!ez->zdropped && (flag&KSW_EZ_EXTZ_ONLY) && ez->mqe + end_bonus > (int)ez->max) {
	// 		ez->reach_end = 1;
	// 		ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*16, ez->mqe_t, qlen-1, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
	// 	} else if (ez->max_t >= 0 && ez->max_q >= 0) {
	// 		ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*16, ez->max_t, ez->max_q, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
	// 	}
	//
	// 	// // @todo: check whether we can remove it
	// 	// if (flag & KSW_EZ_EQX) {
	// 	// 	int32_t nc0 = ez->n_cigar;
	// 	// 	uint32_t *ci0;
	// 	// 	ci0 = (uint32_t*)kmalloc(km, nc0 * sizeof(uint32_t));
	// 	// 	memcpy(ci0, ez->cigar, nc0 * sizeof(uint32_t));
	// 	// 	ksw_cigar2eqx(km, query, target, nc0, ci0, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
	// 	// 	kfree(km, ci0);
	// 	// }
	// 	kfree(km, mem2); kfree(km, off);
	// }
}

#else

__device__ __inline__ int ksw_apply_zdrop_gpu(ksw_extz_t* const ez, const int is_rot, const int32_t H, const int a, const int b, const int zdrop, const int8_t e)
{
	int r, t;
	if (is_rot)
		r = a, t = b;
	else
		r = a + b, t = a;
	if (H > (int32_t)ez->max) {
		ez->max = H, ez->max_t = t, ez->max_q = r - t;
	} else if (t >= ez->max_t && r - t >= ez->max_q) {
		int tl = t - ez->max_t, ql = (r - t) - ez->max_q, l;
		l = tl > ql ? tl - ql : ql - tl;
		if (zdrop >= 0 && ez->max - H > zdrop + l * e) {
			ez->zdropped = 1;
			return 1;
		}
	}
	return 0;
}

__global__ void ksw_extd2_gpu( // void *km,
    int8_t* __restrict__ const u, int8_t* __restrict__ const v, int8_t* __restrict__ const x, int8_t* __restrict__ const y, int8_t* __restrict__ const x2,
    int8_t* __restrict__ const y2, int8_t* __restrict__ const s, int8_t* __restrict__ const p, int32_t* __restrict__ const H, int* __restrict__ const off, int* __restrict__ const off_end,
    const int qlen, const uint8_t* __restrict__ const query, const int tlen, const uint8_t* __restrict__ const target, int8_t m, const int8_t* __restrict__ const mat,
    int8_t q, int8_t e, int8_t q2, int8_t e2, int w, const int zdrop, const int end_bonus, const int flag, ksw_extz_t* __restrict__ const ez)
{
	if (m <= 1 || qlen <= 0 || tlen <= 0) return; // end due to invalid alignment

	int r, t, n_col_, /* *off_end = 0, tlen_, qlen_,*/ last_st, last_en, /* wl, wr, */ max_sc, min_sc, long_thres;
	const int with_cigar = !(flag & KSW_EZ_SCORE_ONLY), approx_max = !!(flag & KSW_EZ_APPROX_MAX);

	int32_t H0 = 0, last_H0_t = 0;

	ez->max_q = ez->max_t = ez->mqe_t = ez->mte_q = -1;
	ez->max = 0, ez->score = ez->mqe = ez->mte = KSW_NEG_INF;
	ez->n_cigar = 0, ez->zdropped = 0, ez->reach_end = 0;

	if (q2 + e2 < q + e) t = q, q = q2, q2 = t, t = e, e = e2, e2 = t; // make sure q+e no larger than q2+e2

	const int8_t qe = q + e;
	const int8_t qe2 = q2 + e2;
	const int8_t sc_mch = mat[0];
	const int8_t sc_mis = mat[1];
	const int8_t sc_N = mat[m * m - 1] == 0 ? -e2 : mat[m * m - 1];
	const int8_t m1 = m - 1;

	if (w < 0) w = tlen > qlen ? tlen : qlen; // set bandwidth in case it is negative (non-banded alignment)
	const int wl = w;
	const int wr = w;
	const int tlen_ = (tlen + 15) / 16; // round to multiple of 16
	n_col_ = qlen < tlen ? qlen : tlen;
	n_col_ = ((n_col_ < w + 1 ? n_col_ : w + 1) + 15) / 16 + 1; // round to multiple of 16
	const int qlen_ = (qlen + 15) / 16; // round to multiple of 16

	for (t = 1, max_sc = mat[0], min_sc = mat[1]; t < m * m; ++t) { // set max_sc to maximum score in mat, and min_sc to minimum score in mat
		max_sc = max_sc > mat[t] ? max_sc : mat[t];
		min_sc = min_sc < mat[t] ? min_sc : mat[t];
	}

	if (-min_sc > 2 * (q + e)) return; /* otherwise, we won't see any mismatches */

	long_thres = e != e2 ? (q2 - q) / (e - e2) - 1 : 0; // if double affine, set the threshold for long gaps(?) to ((q2 - q) / (e - e2) - 1), otherwise 0
	if (q2 + e2 + long_thres * e2 > q + e + long_thres * e)
		++long_thres;
	const int long_diff = long_thres * (e - e2) - (q2 - q) - e2;

	for (int i = 0; i < tlen_ * 16; i += blockDim.x) {
		if (threadIdx.x + i < tlen_ * 16) {
			u[threadIdx.x + i] = -q - e;
		}
	}
	__syncthreads();

	for (int i = 0; i < tlen_ * 16; i += blockDim.x)
		if (threadIdx.x + i < tlen_ * 16)
			v[threadIdx.x + i] = -q - e;
	__syncthreads();

	for (int i = 0; i < tlen_ * 16; i += blockDim.x)
		if (threadIdx.x + i < tlen_ * 16)
			x[threadIdx.x + i] = -q - e;
	__syncthreads();

	for (int i = 0; i < tlen_ * 16; i += blockDim.x)
		if (threadIdx.x + i < tlen_ * 16)
			y[threadIdx.x + i] = -q - e;
	__syncthreads();

	for (int i = 0; i < tlen_ * 16; i += blockDim.x)
		if (threadIdx.x + i < tlen_ * 16)
			x2[threadIdx.x + i] = -q2 - e2;
	__syncthreads();

	for (int i = 0; i < tlen_ * 16; i += blockDim.x)
		if (threadIdx.x + i < tlen_ * 16)
			y2[threadIdx.x + i] = -q2 - e2;
	__syncthreads();

	if (!approx_max) { // if !approximated alignment
		for (int t = 0; t < tlen_ * 16; t += blockDim.x) {
			const int tid = threadIdx.x + t;
			if (tid < tlen_ * 16)
				H[tid] = KSW_NEG_INF; // initialize to -inf (gmem)
		}
		__syncthreads();
	}

	// alignment main loop
	for (r = 0, last_st = last_en = -1; r < qlen + tlen - 1; ++r) {

		int st = 0, en = tlen - 1, st0, en0, st_, en_;
		int8_t x1, x21, v1;
		// const uint8_t* const qrr = query + (qlen - 1 - r);

		/* find the boundaries */
		if (st < r - qlen + 1) st = r - qlen + 1;
		if (en > r) en = r;
		if (st < (r - wr + 1) >> 1) st = (r - wr + 1) >> 1; /* take the ceil */
		if (en > (r + wl) >> 1) en = (r + wl) >> 1; /* take the floor */
		if (st > en) {
			ez->zdropped = 1;
			break;
		}

		st0 = st, en0 = en;
		st = st / 16 * 16, en = (en + 16) / 16 * 16 - 1;

		/* set boundary conditions */
		if (st > 0) {
			if (st - 1 >= last_st && st - 1 <= last_en) {
				x1 = x[st - 1], x21 = x2[st - 1], v1 = v[st - 1]; /* (r-1,s-1) calculated in the last round */
			} else {
				x1 = -q - e, x21 = -q2 - e2;
				v1 = -q - e;
			}
		} else {
			x1 = -q - e, x21 = -q2 - e2;
			v1 = r == 0 ? -q - e : r < long_thres ? -e
			    : r == long_thres		      ? long_diff
							      : -e2;
		}

		if (en >= r) {
			y[r] = -q - e, y2[r] = -q2 - e2; // ((int8_t*)y)[r] = -q - e, ((int8_t*)y2)[r] = -q2 - e2;
			u[r] = r == 0 ? -q - e : r < long_thres ? -e
			    : r == long_thres			? long_diff
								: -e2;
		}

		/* loop fission: set scores first */
		if (!(flag & KSW_EZ_GENERIC_SC)) {

			const int iter = ((en0 - st0) / 16 + 1) * 16 + st0;

			for (int t = st0; t < iter; t += blockDim.x) {
				const int tid = t + threadIdx.x;
				if (tid < iter) {
					const uint8_t tmp = target[tid] == query[qlen - 1 - r + tid] ? sc_mch : sc_mis;
					/* CPU uses OR: sc_N if either base is the wildcard (m1). */
					uint8_t mask = (target[tid] == m1) | (query[qlen - 1 - r + tid] == m1);
					s[tid] = mask ? sc_N : tmp;
				}
			}
			__syncthreads();

		} else {

			for (int t = st0; t <= en0; t += blockDim.x) {
				const int tid = threadIdx.x + t;
				if (tid <= en0) {
					s[tid] = mat[target[tid] * m + query[qlen - 1 - r + tid]]; // ((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
				}
			}
			__syncthreads();
		}


		st_ = st / 16, en_ = en / 16;

		assert(en_ - st_ + 1 <= n_col_);

		if (!with_cigar) { // score only

			int8_t x1_local = x1, x21_local = x21, v1_local = v1;

			for (int t = st; t <= en; t += blockDim.x) {

				const int tid = threadIdx.x + t;

				if (tid <= en) {

					int8_t vt1, x2t1, xt1;
					const bool coeff = threadIdx.x == 0;

					xt1 = coeff * x1_local + !(coeff)*x[tid - 1];
					x2t1 = coeff * x21_local + !(coeff)*x2[tid - 1];
					vt1 = coeff * v1_local + !(coeff)*v[tid - 1];
					x1_local = x[tid + blockDim.x - 1];
					x21_local = x2[tid + blockDim.x - 1];
					v1_local = v[tid + blockDim.x - 1];

					// xt1 = x1_local;
					// x1_local  = x[tid + blockDim.x - 1];
					// x2t1 = x21_local;
					// x21_local = x2[tid + blockDim.x - 1];
					// vt1 = v1_local;
					// v1_local = v[tid + blockDim.x - 1];
					//
					// if (threadIdx.x!=0){
					// 	xt1 = x[tid - 1];
					// 	vt1 = v[tid - 1];
					// 	x2t1 = x2[tid - 1];
					// }

					int8_t a = xt1 + vt1;
					int8_t ut = u[tid];
					int8_t b = y[tid] + ut;

					int8_t z = s[tid];

					int8_t a2 = x2t1 + vt1;
					int8_t b2 = y2[tid] + ut;
					z = z > a ? z : a;
					z = z > b ? z : b;
					z = z > a2 ? z : a2;
					z = z > b2 ? z : b2;
					z = z < sc_mch ? z : sc_mch;
					u[tid] = z - vt1;
					v[tid] = z - ut;
					a = a - (z - q);
					b = b - (z - q);
					a2 = a2 - (z - q2);
					b2 = b2 - (z - q2);
					x[tid] = (a > 0 ? a : 0) - qe;
					y[tid] = (b > 0 ? b : 0) - qe;
					x2[tid] = (a2 > 0 ? a2 : 0) - qe2;
					y2[tid] = (b2 > 0 ? b2 : 0) - qe2;
				}
			}

			__syncthreads();

		} else if (!(flag & KSW_EZ_RIGHT)) { /* gap left-alignment */
			// std::cout<<"Gap left-alignment"<<std::endl;
			// printf("Gap left-alignment\n");

			// int8_t* const pr = p + ((size_t)r * n_col_ - st_)*16; // __m128i *pr = p + (size_t)r * n_col_ - st_;
			const size_t size = (size_t)r * n_col_ * 16 - st;
			off[r] = st, off_end[r] = en;

			int8_t x1_local = x1, x21_local = x21, v1_local = v1;

			for (int t = st; t <= en; t += blockDim.x) {

				const int tid = threadIdx.x + t;

				if (tid <= en) {

					int8_t vt1, x2t1, xt1;

					const bool coeff = threadIdx.x == 0;

					xt1 = coeff * x1_local + !(coeff)*x[tid - 1];
					x2t1 = coeff * x21_local + !(coeff)*x2[tid - 1];
					vt1 = coeff * v1_local + !(coeff)*v[tid - 1];
					x1_local = x[tid + blockDim.x - 1];
					x21_local = x2[tid + blockDim.x - 1];
					v1_local = v[tid + blockDim.x - 1];

					// xt1 = x1_local;
					// x1_local  = x[tid + blockDim.x - 1];
					// x2t1 = x21_local;
					// x21_local = x2[tid + blockDim.x - 1];
					// vt1 = v1_local;
					// v1_local = v[tid + blockDim.x - 1];
					// if (threadIdx.x != 0){
					// 	xt1 = x[tid - 1];
					// 	vt1 = v[tid - 1];
					// 	x2t1 = x2[tid - 1];
					// }

					int8_t a = xt1 + vt1;
					int8_t ut = u[tid];
					int8_t b = y[tid] + ut;

					int8_t z = s[tid];

					int8_t a2 = x2t1 + vt1;
					int8_t b2 = y2[tid] + ut;

					int8_t d = a > z ? 1 : 0;

					z = z > a ? z : a;
					d = b > z ? 2 : d;
					z = z > b ? z : b;
					d = a2 > z ? 3 : d;
					z = z > a2 ? z : a2;
					d = b2 > z ? 4 : d;
					z = z > b2 ? z : b2;
					z = z < sc_mch ? z : sc_mch;

					u[tid] = z - vt1;
					v[tid] = z - ut;

					a = a - (z - q);
					b = b - (z - q);
					a2 = a2 - (z - q2);
					b2 = b2 - (z - q2);

					x[tid] = (a > 0 ? a : 0) - qe;
					d |= a > 0 ? 0x08 : 0;
					y[tid] = (b > 0 ? b : 0) - qe;
					d |= b > 0 ? 0x10 : 0;
					x2[tid] = (a2 > 0 ? a2 : 0) - qe2;
					d |= a2 > 0 ? 0x20 : 0;
					y2[tid] = (b2 > 0 ? b2 : 0) - qe2;
					d |= b2 > 0 ? 0x40 : 0;

					p[size + tid] = d;
				}
			}

			__syncthreads();

		} else { /* gap right-alignment */
			// std::cout<<"Gap right-alignment"<<std::endl;
			// printf("Gap right-alignment\n");
			// int8_t* const pr = p + ((size_t)r * n_col_ - st_)*16; // __m128i *pr = p + (size_t)r * n_col_ - st_;
			const size_t size = (size_t)r * (n_col_ * 16) - st;
			off[r] = st, off_end[r] = en;

			int8_t x1_local = x1, x21_local = x21, v1_local = v1;

			for (int t = st; t <= en; t += blockDim.x) {

				const int tid = threadIdx.x + t;

				if (tid <= en) {

					int8_t vt1;
					int8_t x2t1;
					int8_t xt1;

					const bool coeff = threadIdx.x == 0;

					xt1 = coeff * x1_local + !(coeff)*x[tid - 1];
					x2t1 = coeff * x21_local + !(coeff)*x2[tid - 1];
					vt1 = coeff * v1_local + !(coeff)*v[tid - 1];
					x1_local = x[tid + blockDim.x - 1];
					x21_local = x2[tid + blockDim.x - 1];
					v1_local = v[tid + blockDim.x - 1];

					// xt1  = x1_local;
					// x1_local  = x[tid + blockDim.x - 1];
					// x2t1 = x21_local;
					// x21_local = x2[tid + blockDim.x - 1];
					// vt1 = v1_local;
					// v1_local = v[tid + blockDim.x - 1];
					// if (threadIdx.x!=0){
					// 	xt1 = x[tid - 1];
					// 	vt1 = v[tid - 1];
					// 	x2t1 = x2[tid - 1];
					// }

					int8_t a = xt1 + vt1;
					int8_t ut = u[tid];
					int8_t b = y[tid] + ut;

					int8_t z = s[tid];

					int8_t a2 = x2t1 + vt1;
					int8_t b2 = y2[tid] + ut;

					int8_t d;

					d = z > a ? 0 : 1;
					z = z > a ? z : a;
					d = z > b ? d : 2;
					z = z > b ? z : b;
					d = z > a2 ? d : 3;
					z = z > a2 ? z : a2;
					d = z > b2 ? d : 4;
					z = z > b2 ? z : b2;
					z = z < sc_mch ? z : sc_mch;

					u[tid] = z - vt1;
					v[tid] = z - ut;

					a = a - (z - q);
					b = b - (z - q);
					a2 = a2 - (z - q2);
					b2 = b2 - (z - q2);

					x[tid] = (a >= 0 ? a : 0) - qe;
					d |= a >= 0 ? 0x08 : 0; // gap-right: >= 0
					y[tid] = (b >= 0 ? b : 0) - qe;
					d |= b >= 0 ? 0x10 : 0; // gap-right: >= 0
					x2[tid] = (a2 >= 0 ? a2 : 0) - qe2;
					d |= a2 >= 0 ? 0x20 : 0; // gap-right: >= 0
					y2[tid] = (b2 >= 0 ? b2 : 0) - qe2;
					d |= b2 >= 0 ? 0x40 : 0; // gap-right: >= 0

					p[size + tid] = d;
				}
			}

			__syncthreads();
		}

		if (!approx_max) { /* find the exact max with a 32-bit score array */

			int32_t max_H, max_t;
			/* compute H[], max_H and max_t */
			if (r > 0) {

				const int32_t en1 = st0 + (en0 - st0) / 4 * 4;
				__shared__ int32_t HH[NUM_THREADS], tt[NUM_THREADS];
				__shared__ int32_t max_H_[NUM_THREADS], max_t_[NUM_THREADS];
				max_H = H[en0] = en0 > 0 ? H[en0 - 1] + u[en0] : H[en0] + v[en0]; /* special casing the last element */
				max_t = en0;

				for (int i = 0; i < NUM_THREADS; i += blockDim.x) {
					const int tid = threadIdx.x + i;
					if (tid < NUM_THREADS)
						max_H_[tid] = max_H;
				}
				__syncthreads();

				for (int i = 0; i < NUM_THREADS; i += blockDim.x) {
					const int tid = threadIdx.x + i;
					if (tid < NUM_THREADS)
						max_t_[tid] = max_t;
				}
				__syncthreads();

				for (t = st0; t < en1; t += blockDim.x) { /* this implements: H[t]+=v8[t]-qe; if(H[t]>max_H) max_H=H[t],max_t=t; */
					const int tid = threadIdx.x + t;
					if (tid < en1) {
						int32_t H1 = H[tid];
						int32_t t_ = v[tid];
						H1 = H1 + t_;
						H[tid] = H1;
						t_ = t + (threadIdx.x % 4);
						int32_t tmp = H1 > max_H_[threadIdx.x] ? 0xFFFFFFFF : 0;
						max_H_[threadIdx.x] = (tmp & H1) | ((~tmp) & max_H_[threadIdx.x]);
						max_t_[threadIdx.x] = (tmp & t_) | ((~tmp) & max_t_[threadIdx.x]);
					}
				}
				__syncthreads();

				for (int i = 0; i < NUM_THREADS; i += blockDim.x) {
					const int tid = threadIdx.x + i;
					if (tid < NUM_THREADS)
						HH[tid] = max_H_[tid];
				}
				__syncthreads();

				for (int i = 0; i < NUM_THREADS; i += blockDim.x) {
					const int tid = threadIdx.x + i;
					if (tid < NUM_THREADS)
						tt[tid] = max_t_[tid];
				}
				__syncthreads();

				for (int i = 0; i < NUM_THREADS; ++i)
					if (max_H < HH[i]) max_H = HH[i], max_t = tt[i] + i;

				for (t = en1; t < en0; ++t) { /* for the rest of values that haven't been computed with SSE */
					H[t] += (int32_t)v[t];
					if (H[t] > max_H)
						max_H = H[t], max_t = t;
				}


			} else
				H[0] = v[0] - qe, max_H = H[0], max_t = 0; /* special casing r==0 */
			/* update ez */
			if (en0 == tlen - 1 && H[en0] > ez->mte)
				ez->mte = H[en0], ez->mte_q = r - en0;
			if (r - st0 == qlen - 1 && H[st0] > ez->mqe)
				ez->mqe = H[st0], ez->mqe_t = st0;
			if (ksw_apply_zdrop_gpu(ez, 1, max_H, r, max_t, zdrop, e2)) break; // if score drops too fast, interrupt alignment
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez->score = H[tlen - 1];
		} else { /* find approximate max; Z-drop might be inaccurate, too. */
			if (r > 0) {
				if (last_H0_t >= st0 && last_H0_t <= en0 && last_H0_t + 1 >= st0 && last_H0_t + 1 <= en0) {
					int32_t d0 = v[last_H0_t];
					int32_t d1 = u[last_H0_t + 1];
					if (d0 > d1)
						H0 += d0;
					else
						H0 += d1, ++last_H0_t;
				} else if (last_H0_t >= st0 && last_H0_t <= en0) {
					H0 += v[last_H0_t];
				} else {
					++last_H0_t, H0 += u[last_H0_t];
				}
			} else
				H0 = v[0] - qe, last_H0_t = 0;
			if ((flag & KSW_EZ_APPROX_DROP) && ksw_apply_zdrop_gpu(ez, 1, H0, r, last_H0_t, zdrop, e2)) break;
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez->score = H0;
		}
		last_st = st, last_en = en;
		//for (t = st0; t <= en0; ++t) printf("(%d,%d)\t(%d,%d,%d,%d)\t%d\n", r, t, ((int8_t*)u)[t], ((int8_t*)v)[t], ((int8_t*)x)[t], ((int8_t*)y)[t], H[t]); // for debugging
	}

	// kfree(km, mem);
	// if (!approx_max) kfree(km, H);
	// if (with_cigar) { // backtrack
	// 	int rev_cigar = !!(flag & KSW_EZ_REV_CIGAR);
	// 	if (!ez->zdropped && !(flag&KSW_EZ_EXTZ_ONLY)) {
	// 		ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*16, tlen-1, qlen-1, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
	// 	} else if (!ez->zdropped && (flag&KSW_EZ_EXTZ_ONLY) && ez->mqe + end_bonus > (int)ez->max) {
	// 		ez->reach_end = 1;
	// 		ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*16, ez->mqe_t, qlen-1, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
	// 	} else if (ez->max_t >= 0 && ez->max_q >= 0) {
	// 		ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*16, ez->max_t, ez->max_q, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
	// 	}
	//
	// 	// // @todo: check whether we can remove it
	// 	// if (flag & KSW_EZ_EQX) {
	// 	// 	int32_t nc0 = ez->n_cigar;
	// 	// 	uint32_t *ci0;
	// 	// 	ci0 = (uint32_t*)kmalloc(km, nc0 * sizeof(uint32_t));
	// 	// 	memcpy(ci0, ez->cigar, nc0 * sizeof(uint32_t));
	// 	// 	ksw_cigar2eqx(km, query, target, nc0, ci0, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
	// 	// 	kfree(km, ci0);
	// 	// }
	// 	kfree(km, mem2); kfree(km, off);
	// }
}

#endif
