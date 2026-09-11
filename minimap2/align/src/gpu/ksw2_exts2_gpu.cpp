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
#include "misc.h"

#include <hip/hip_runtime.h>

#define NUM_THREADS 64

__device__ __inline__ int ksw_apply_zdrop_gpu(ksw_extz_t* ez, int is_rot, int32_t H, int a, int b, int zdrop, int8_t e)
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

__global__ void ksw_exts2_gpu( // void *km,
    int8_t* const u, int8_t* const v, int8_t* const x, int8_t* const y, int8_t* const x2, int8_t* const s, int8_t* const p, int32_t* const H, int32_t* const off, int32_t* const off_end,
    int8_t* const acceptor, int8_t* const donor, const int qlen, const uint8_t* query, const int tlen, const uint8_t* target, const int8_t m, const int8_t* mat,
    const int8_t q, const int8_t e, const int8_t q2, const int8_t noncan, const int zdrop, const int8_t junc_bonus, const int8_t junc_pen, const int flag, const uint8_t* junc, ksw_extz_t* ez)
// ~KSW_CPU_DISPATCH
{
	if (m <= 1 || qlen <= 0 || tlen <= 0 || q2 <= q + e) return;

	assert((flag & KSW_EZ_SPLICE_FOR) == 0 || (flag & KSW_EZ_SPLICE_REV) == 0); // can't be both set

	int r, t, last_st, last_en, max_sc, min_sc, long_thres, long_diff;
	int with_cigar = !(flag & KSW_EZ_SCORE_ONLY), approx_max = !!(flag & KSW_EZ_APPROX_MAX);
	int32_t H0 = 0, last_H0_t = 0;

	ez->max_q = ez->max_t = ez->mqe_t = ez->mte_q = -1;
	ez->max = 0, ez->score = ez->mqe = ez->mte = KSW_NEG_INF;
	ez->n_cigar = 0, ez->zdropped = 0, ez->reach_end = 0;

	const int8_t qe = q + e;
	const int8_t sc_mch = mat[0];
	const int8_t sc_mis = mat[1];
	const int8_t sc_N = mat[m * m - 1] == 0 ? -e : mat[m * m - 1];
	const int8_t m1 = m - 1;

	const int tlen_ = (tlen + 15) / 16;
	const int n_col_ = ((qlen < tlen ? qlen : tlen) + 15) / 16 + 1;
	const int qlen_ = (qlen + 15) / 16;

	for (t = 1, max_sc = mat[0], min_sc = mat[1]; t < m * m; ++t) {
		max_sc = max_sc > mat[t] ? max_sc : mat[t];
		min_sc = min_sc < mat[t] ? min_sc : mat[t];
	}

	if (-min_sc > 2 * (q + e)) return; // otherwise, we won't see any mismatches

	long_thres = (q2 - q) / e - 1;
	if (q2 > q + e + long_thres * e)
		++long_thres;
	long_diff = long_thres * e - (q2 - q);

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
			x2[threadIdx.x + i] = -q2;
	__syncthreads();

	if (!approx_max) { // if !approximated alignment
		for (t = 0; t < tlen_ * 16; t += blockDim.x) {
			const int tid = threadIdx.x + t;
			if (tid < tlen_ * 16)
				H[tid] = KSW_NEG_INF; // initialize to -inf (gmem)
		}
		__syncthreads();
	}

	// set the donor and acceptor arrays. TODO: this assumes 0/1/2/3 encoding!
	if (flag & (KSW_EZ_SPLICE_FOR | KSW_EZ_SPLICE_REV)) {

		const int sp0[4] = {8, 15, 21, 30};
		int sp[4];
		if (flag & KSW_EZ_SPLICE_CMPLX) {
			for (t = 0; t < 4; ++t)
				sp[t] = (int)((double)sp0[t] / 3. + .499);
		} else {
			sp[0] = flag & KSW_EZ_SPLICE_FLANK ? noncan / 2 : 0;
			sp[1] = sp[2] = sp[3] = noncan;
		}

		for (int i = 0; i < tlen_ * 16; i += blockDim.x)
			if (threadIdx.x + i < tlen_ * 16)
				donor[threadIdx.x + i] = -sp[3];
		__syncthreads();

		for (int i = 0; i < tlen_ * 16; i += blockDim.x)
			if (threadIdx.x + i < tlen_ * 16)
				acceptor[threadIdx.x + i] = -sp[3];
		__syncthreads();

		if (!(flag & KSW_EZ_REV_CIGAR)) {

			if (flag & KSW_EZ_SPLICE_FOR) {

				for (int t = 0; t < tlen - 4; t += blockDim.x) {
					const int tid = threadIdx.x + t;
					if (tid < tlen - 4) {

						int z = 3;
						const int8_t t1 = target[tid + 1];
						const int8_t t2 = target[tid + 2];
						const int8_t t3 = target[tid + 3];

						z = ((t1 == 2) && (t2 == 3)) ? 0 : z; // if (target[tid+1] == 2 && target[tid+2] == 3)             // |GT.
						z = ((t1 == 2) && (t2 == 3) && (t3 == 0)) ? -1 : z; //     z = target[tid+3] == 0 || target[tid+3] == 2? -1 : 0; // |GTr or not
						z = ((t1 == 2) && (t2 == 3) && (t3 == 2)) ? -1 : z; // else if (target[tid+1] == 2 && target[tid+2] == 1) z = 1; // |GC.
						z = ((t1 == 2) && (t2 == 1)) ? 1 : z; // else if (target[tid+1] == 0 && target[tid+2] == 3) z = 2; // |AT.
						z = ((t1 == 0) && (t2 == 3)) ? 2 : z;

						donor[tid] = z < 0 ? 0 : -sp[z];
					}
				}
				__syncthreads();

				for (t = 2; t < tlen; t += blockDim.x) {
					const int tid = threadIdx.x + t;
					if (tid < tlen) {
						int z = 3;
						const int8_t tt = target[tid];
						const int8_t t1 = target[tid - 1];
						const int8_t t2 = target[tid - 2];

						z = ((t1 == 0) && (tt == 2)) ? 0 : z; // if (target[tid-1] == 0 && target[tid] == 2)               // .AG|
						z = ((t1 == 0) && (tt == 2) && (t2 == 1)) ? -1 : z; // 	   z = target[tid-2] == 1 || target[tid-2] == 3? -1 : 0; // yAG| or not
						z = ((t1 == 0) && (tt == 2) && (t2 == 3)) ? -1 : z; // else if (target[tid-1] == 0 && target[tid] == 1) z = 2;   // .AC|
						z = ((t1 == 0) && (tt == 1)) ? 2 : z;

						acceptor[tid] = z < 0 ? 0 : -sp[z];
					}
				}
				__syncthreads();

			} else if (flag & KSW_EZ_SPLICE_REV) {

				for (int t = 0; t < tlen - 4; t += blockDim.x) {
					const int tid = threadIdx.x + t;
					if (tid < tlen - 4) {
						int z = 3;

						const int t1 = target[tid + 1];
						const int t2 = target[tid + 2];
						const int t3 = target[tid + 3];

						z = ((t1 == 1) && (t2 == 3)) ? 0 : z;
						z = ((t1 == 1) && (t2 == 3) && (t3 == 0)) ? -1 : z;
						z = ((t1 == 1) && (t2 == 3) && (t3 == 2)) ? -1 : z;
						z = ((t1 == 2) && (t2 == 3)) ? 2 : z;

						donor[tid] = z < 0 ? 0 : -sp[z];
					}
				}
				__syncthreads();

				for (t = 2; t < tlen; t += blockDim.x) {
					const int tid = threadIdx.x + t;
					if (tid < tlen) {
						int z = 3;
						const int8_t tt = target[tid];
						const int8_t t1 = target[tid - 1];
						const int8_t t2 = target[tid - 2];

						z = ((t1 == 0) && (tt == 1)) ? 0 : z;
						z = ((t1 == 0) && (tt == 1) && (t2 == 1)) ? -1 : z;
						z = ((t1 == 0) && (tt == 1) && (t2 == 3)) ? -1 : z;
						z = ((t1 == 2) && (tt == 1)) ? 1 : z;
						z = ((t1 == 0) && (tt == 3)) ? 2 : z;

						acceptor[tid] = z < 0 ? 0 : -sp[z];
					}
				}
				__syncthreads();
			}

		} else {

			if (flag & KSW_EZ_SPLICE_FOR) {

				for (t = 0; t < tlen - 4; t += blockDim.x) {
					const int tid = threadIdx.x + t;
					if (tid < tlen - 4) {
						int z = 3;
						const int t1 = target[tid + 1];
						const int t2 = target[tid + 2];
						const int t3 = target[tid + 3];

						z = (t1 == 2) && (t2 == 0) ? 0 : z; // |GA. (rev of .AG|)
						z = (t1 == 2) && (t2 == 0) && (t3 == 1) ? -1 : z;
						z = (t1 == 2) && (t2 == 0) && (t3 == 3) ? -1 : z;
						z = (t1 == 1) && (t2 == 0) ? 2 : z; // |CA. (rev of .AC|)

						donor[tid] = z < 0 ? 0 : -sp[z];
					}
				}
				__syncthreads();

				for (t = 2; t < tlen; t += blockDim.x) {
					const int tid = threadIdx.x + t;
					if (tid < tlen) {
						int z = 3;
						const int tt = target[tid];
						const int t1 = target[tid - 1];
						const int t2 = target[tid - 2];

						z = (t1 == 3) && (tt == 2) ? 0 : z; // .TG| (rev of |GT.)
						z = (t1 == 3) && (tt == 2) && (t2 == 0) ? -1 : z;
						z = (t1 == 3) && (tt == 2) && (t2 == 2) ? -1 : z;
						z = (t1 == 1) && (tt == 2) ? 1 : z; // .CG| (rev of |GC.)
						z = (t1 == 3) && (tt == 0) ? 2 : z; // .TA| (rev of |AT.)

						acceptor[tid] = z < 0 ? 0 : -sp[z];
					}
				}
				__syncthreads();

			} else if (flag & KSW_EZ_SPLICE_REV) {

				for (t = 0; t < tlen - 4; t += blockDim.x) {
					const int tid = threadIdx.x + t;
					if (tid < tlen - 4) {
						int z = 3;
						const int t1 = target[tid + 1];
						const int t2 = target[tid + 2];
						const int t3 = target[tid + 3];

						z = ((t1 == 1) && (t2 == 0)) ? 0 : z; // |CA. (comp of |GT.)
						z = ((t1 == 1) && (t2 == 0) && (t3 == 1)) ? -1 : z;
						z = ((t1 == 1) && (t2 == 0) && (t3 == 3)) ? -1 : z;
						z = ((t1 == 1) && (t2 == 2)) ? 1 : z; // |CG. (comp of |GC.)
						z = ((t1 == 3) && (t2 == 0)) ? 2 : z; // |TA. (comp of |AT.)

						donor[tid] = z < 0 ? 0 : -sp[z];
					}
				}
				__syncthreads();

				for (t = 2; t < tlen; t += blockDim.x) {
					const int tid = threadIdx.x + t;
					if (tid < tlen) {
						int z = 3;
						const int tt = target[tid];
						const int t1 = target[tid - 1];
						const int t2 = target[tid - 2];

						z = (t1 == 3) && (tt == 1) ? 0 : z; // .TC| (comp of .AG|)
						z = (t1 == 3) && (tt == 1) && (t2 == 0) ? -1 : z;
						z = (t1 == 3) && (tt == 1) && (t2 == 2) ? -1 : z;
						z = (t1 == 3) && (tt == 2) ? 2 : z; // .TG| (comp of .AC|)

						acceptor[tid] = z < 0 ? 0 : -sp[z];
					}
				}
				__syncthreads();
			}
		}
	}

	if (junc && (flag & KSW_EZ_SPLICE_SCORE)) { // junc[] keeps the donor score

		uint8_t donor_val = !!(flag & KSW_EZ_SPLICE_FOR) == !(flag & KSW_EZ_REV_CIGAR) ? 0 : 1;

		for (t = 0; t < tlen - 1; t += blockDim.x) {
			const int tid = threadIdx.x + t;
			if (tid < tlen - 1)
				donor[tid] += junc[tid + 1] == 0xff || (junc[tid + 1] & 1) != donor_val ? -junc_pen : (int8_t)(junc[tid + 1] >> 1) - (int8_t)KSW_SPSC_OFFSET;
		}
		__syncthreads();

		for (t = 0; t < tlen - 1; t += blockDim.x) {
			const int tid = threadIdx.x + t;
			if (tid < tlen - 1)
				acceptor[tid] += junc[tid + 1] == 0xff || (junc[tid + 1] & 1) != !donor_val ? -junc_pen : (int8_t)(junc[tid + 1] >> 1) - (int8_t)KSW_SPSC_OFFSET;
		}
		__syncthreads();


		//for (t = 0; t < tlen - 1; ++t) if (junc[t+1] != 0xff) fprintf(stderr, "Y2\t%d\t%d\t%c\t%d\n", ((int8_t*)donor)[t], ((int8_t*)acceptor)[t], "DA"[junc[t+1]&1], (int8_t)(junc[t+1]>>1) - (int8_t)KSW_SPSC_OFFSET);

	} else if (junc) { // junc[] keeps the splice sites
		if (!(flag & KSW_EZ_REV_CIGAR)) {

			for (t = 0; t < tlen - 1; t += blockDim.x) {
				const int tid = threadIdx.x + t;
				if (tid < tlen - 1) {
					if (((flag & KSW_EZ_SPLICE_FOR) && (junc[tid + 1] & 1)) || ((flag & KSW_EZ_SPLICE_REV) && (junc[tid + 1] & 8)))
						donor[tid] += junc_bonus;
				}
			}
			__syncthreads();

			for (t = 0; t < tlen; t += blockDim.x) {
				const int tid = threadIdx.x + t;
				if (tid < tlen) {
					if (((flag & KSW_EZ_SPLICE_FOR) && (junc[tid] & 2)) || ((flag & KSW_EZ_SPLICE_REV) && (junc[tid] & 4)))
						acceptor[tid] += junc_bonus;
				}
			}
			__syncthreads();

		} else {

			for (t = 0; t < tlen - 1; t += blockDim.x) {
				const int tid = threadIdx.x + t;
				if (tid < tlen - 1) {
					if (((flag & KSW_EZ_SPLICE_FOR) && (junc[tid + 1] & 2)) || ((flag & KSW_EZ_SPLICE_REV) && (junc[tid + 1] & 4)))
						donor[tid] += junc_bonus;
				}
			}
			__syncthreads();

			for (t = 0; t < tlen; t += blockDim.x) {
				const int tid = threadIdx.x + t;
				if (tid < tlen) {
					if (((flag & KSW_EZ_SPLICE_FOR) && (junc[tid] & 1)) || ((flag & KSW_EZ_SPLICE_REV) && (junc[tid] & 8)))
						acceptor[tid] += junc_bonus;
				}
			}
			__syncthreads();
		}
	}

	for (r = 0, last_st = last_en = -1; r < qlen + tlen - 1; ++r) {
		int st = 0, en = tlen - 1, st0, en0, st_, en_;
		int8_t x1, x21, v1;
		const uint8_t* qrr = query + (qlen - 1 - r);

		// find the boundaries
		if (st < r - qlen + 1) st = r - qlen + 1;
		if (en > r) en = r;
		st0 = st, en0 = en;
		st = st / 16 * 16, en = (en + 16) / 16 * 16 - 1;

		// set boundary conditions
		if (st > 0) {
			if (st - 1 >= last_st && st - 1 <= last_en)
				x1 = x[st - 1], x21 = x2[st - 1], v1 = v[st - 1]; // (r-1,s-1) calculated in the last round
			else
				x1 = -q - e, x21 = -q2, v1 = -q - e;
		} else {
			x1 = -q - e, x21 = -q2;
			v1 = r == 0 ? -q - e : r < long_thres ? -e
			    : r == long_thres		      ? long_diff
							      : 0;
		}

		if (en >= r) {
			y[r] = -q - e;
			u[r] = r == 0 ? -q - e : r < long_thres ? -e
			    : r == long_thres			? long_diff
								: 0;
		}

		// loop fission: set scores first
		if (!(flag & KSW_EZ_GENERIC_SC)) {

			int iter = ((en0 - st0) / 16 + 1) * 16 + st0;

			for (t = st0; t < iter; t += blockDim.x) {
				const int tid = t + threadIdx.x;
				if (tid < iter) {
					const uint8_t tmp = target[tid] == qrr[tid] ? sc_mch : sc_mis;
					uint8_t mask = target[tid] == m1 ? 1 : 0;
					mask *= (qrr[tid] == m1 ? 1 : 0);
					s[tid] = mask ? sc_N : tmp;
				}
			}
			__syncthreads();

		} else {

			for (t = st0; t <= en0; t += blockDim.x) {
				const int tid = threadIdx.x + t;
				if (tid <= en0) {
					s[tid] = mat[target[tid] * m + qrr[tid]]; // ((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
				}
			}
			__syncthreads();
		}

		st_ = st / 16, en_ = en / 16;
		assert(en_ - st_ + 1 <= n_col_);

		if (!with_cigar) { // score only

			int8_t x1_local = x1, x21_local = x21, v1_local = v1;

			for (t = st; t <= en; t += blockDim.x) {

				const int tid = threadIdx.x + t;

				if (tid <= en) {

					const int8_t x_val = x[tid];
					const int8_t x2_val = x2[tid];
					const int8_t v_val = v[tid];

					int8_t xt1 = __shfl_up(x_val, 1);
					int8_t vt1 = __shfl_up(v_val, 1);
					int8_t x2t1 = __shfl_up(x2_val, 1);

					if (threadIdx.x == 0) {
						xt1 = x1_local;
						x1_local = x[tid + blockDim.x - 1];
						x2t1 = x21_local;
						x21_local = x2[tid + blockDim.x - 1];
						vt1 = v1_local;
						v1_local = v[tid + blockDim.x - 1];
					}

					int8_t a = xt1 + vt1;
					int8_t ut = u[tid];
					int8_t b = y[tid] + ut;
					int8_t z = s[tid];
					int8_t a2 = x2t1 + vt1;
					int8_t a2a = a2 + acceptor[tid];

					z = z > a ? z : a;
					z = z > b ? z : b;
					z = z > a2a ? z : a2a;

					u[tid] = z - vt1;
					v[tid] = z - ut;
					a = a - (z - q);
					b = b - (z - q);
					a2 = a2 - (z - q2);

					x[tid] = (a > 0 ? a : 0) - qe;
					y[tid] = (b > 0 ? b : 0) - qe;
					x2[tid] = (a2 > donor[tid] ? a2 : donor[tid]) - q2;
				}
			}

			__syncthreads();

		} else if (!(flag & KSW_EZ_RIGHT)) { // gap left-alignment

			int8_t* pr = p + ((size_t)r * n_col_ - st_) * 16;
			off[r] = st, off_end[r] = en;

			int8_t x1_local = x1, x21_local = x21, v1_local = v1;

			for (t = st; t <= en; t += blockDim.x) {

				const int tid = threadIdx.x + t;

				if (tid <= en) {

					const int8_t x_val = x[tid];
					const int8_t x2_val = x2[tid];
					const int8_t v_val = v[tid];

					int8_t xt1 = __shfl_up(x_val, 1);
					int8_t vt1 = __shfl_up(v_val, 1);
					int8_t x2t1 = __shfl_up(x2_val, 1);

					if (threadIdx.x == 0) {
						xt1 = x1_local;
						x1_local = x[tid + blockDim.x - 1];
						x2t1 = x21_local;
						x21_local = x2[tid + blockDim.x - 1];
						vt1 = v1_local;
						v1_local = v[tid + blockDim.x - 1];
					}

					int8_t a = xt1 + vt1;
					int8_t ut = u[tid];
					int8_t b = y[tid] + ut;
					int8_t z = s[tid];
					int8_t a2 = x2t1 + vt1;
					int8_t a2a = a2 + acceptor[tid];

					// Tie-breaking must match SSE implementation: only switch
					// state when the new candidate is strictly greater.
					int8_t d = 0; // start from H state
					if (a > z) {
						z = a;
						d = 1;
					}
					if (b > z) {
						z = b;
						d = 2;
					}
					if (a2a > z) {
						z = a2a;
						d = 3;
					}
					u[tid] = z - vt1;
					v[tid] = z - ut;

					a = a - (z - q);
					b = b - (z - q);
					a2 = a2 - (z - q2);

					x[tid] = (a > 0 ? a : 0) - qe;
					d |= a > 0 ? 0x08 : 0;
					y[tid] = (b > 0 ? b : 0) - qe;
					d |= b > 0 ? 0x10 : 0;

					int8_t tmp = a2 > donor[tid] ? 0xFF : 0;
					int8_t tmp2 = a2 > donor[tid] ? a2 : donor[tid];

					x2[tid] = tmp2 - q2;
					d |= tmp & 0x20;
					pr[tid] = d;
				}
			}

			__syncthreads();

		} else { // gap right-alignment

			int8_t* pr = p + ((size_t)r * n_col_ - st_) * 16;
			off[r] = st, off_end[r] = en;

			int8_t x1_local = x1, x21_local = x21, v1_local = v1;

			for (t = st; t <= en; t += blockDim.x) {

				const int tid = threadIdx.x + t;

				if (tid <= en) {

					const int8_t x_val = x[tid];
					const int8_t x2_val = x2[tid];
					const int8_t v_val = v[tid];

					int8_t xt1 = __shfl_up(x_val, 1);
					int8_t vt1 = __shfl_up(v_val, 1);
					int8_t x2t1 = __shfl_up(x2_val, 1);

					if (threadIdx.x == 0) {
						xt1 = x1_local;
						x1_local = x[tid + blockDim.x - 1];
						x2t1 = x21_local;
						x21_local = x2[tid + blockDim.x - 1];
						vt1 = v1_local;
						v1_local = v[tid + blockDim.x - 1];
					}

					int8_t a = xt1 + vt1;
					int8_t ut = u[tid];
					int8_t b = y[tid] + ut;
					int8_t z = s[tid];
					int8_t a2 = x2t1 + vt1;
					int8_t a2a = a2 + acceptor[tid];

					int8_t d;

					d = z > a ? 0 : 1;
					z = z > a ? z : a;
					d = z > b ? d : 2;
					z = z > b ? z : b;
					d = z > a2a ? d : 3;
					z = z > a2a ? z : a2a;

					u[tid] = z - vt1;
					v[tid] = z - ut;

					a = a - (z - q);
					b = b - (z - q);
					a2 = a2 - (z - q2);

					x[tid] = (a > 0 ? a : 0) - qe;
					d |= a > 0 ? 0x08 : 0;
					y[tid] = (b > 0 ? b : 0) - qe;
					d |= b > 0 ? 0x10 : 0;

					int8_t tmp = a2 > donor[tid] ? 0xFF : 0;
					int8_t tmp2 = a2 > donor[tid] ? a2 : donor[tid];

					x2[tid] = tmp2 - q2;
					d |= tmp & 0x20;
					pr[tid] = d;
				}
			}

			__syncthreads();
		}

		if (!approx_max) { // find the exact max with a 32-bit score array
			int32_t max_H, max_t;
			// compute H[], max_H and max_t
			if (r > 0) {

				int32_t max_H_local, max_t_local;

				max_H = max_H_local = en0 > 0 ? H[en0 - 1] + u[en0] : H[en0] + v[en0]; /* special casing the last element */
				H[en0] = en0 > 0 ? H[en0 - 1] + u[en0] : H[en0] + v[en0];
				max_t = max_t_local = en0;

				for (t = st0; t < en0; t += blockDim.x) { /* this implements: H[t]+=v8[t]-qe; if(H[t]>max_H) max_H=H[t],max_t=t; */
					const int tid = threadIdx.x + t;
					if (tid < en0) {
						H[tid] += v[tid];
						const int32_t tmp = H[tid] > max_H_local ? 1 : 0;
						max_H_local = tmp ? H[tid] : max_H_local;
						max_t_local = tmp ? tid : max_t_local;
					}
				}
				__syncthreads();

				if (max_H_local <= max_H) max_H_local = INT32_MIN, max_t_local = -1;
				for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
					int tmp_H = __shfl_down(max_H_local, offset);
					int tmp_t = __shfl_down(max_t_local, offset);

					if (tmp_H > max_H_local) max_H_local = tmp_H, max_t_local = tmp_t;
				}

				max_H = __shfl(max_H_local, 0);
				max_t = __shfl(max_t_local, 0);

				__syncthreads();

				/*for (t = en1; t < en0; ++t) {  // for the rest of values that haven't been computed with SSE 
					H[t] += (int32_t)v[t];
					if (H[t] > max_H)
						max_H = H[t], max_t = t;
				}*/

			} else
				H[0] = v[0] - qe, max_H = H[0], max_t = 0; // special casing r==0
			// update ez
			if (en0 == tlen - 1 && H[en0] > ez->mte)
				ez->mte = H[en0], ez->mte_q = r - en0;
			if (r - st0 == qlen - 1 && H[st0] > ez->mqe)
				ez->mqe = H[st0], ez->mqe_t = st0;
			if (ksw_apply_zdrop_gpu(ez, 1, max_H, r, max_t, zdrop, 0)) break;
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez->score = H[tlen - 1];
		} else { // find approximate max; Z-drop might be inaccurate, too.
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
			if ((flag & KSW_EZ_APPROX_DROP) && ksw_apply_zdrop_gpu(ez, 1, H0, r, last_H0_t, zdrop, 0)) break;
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez->score = H0;
		}
		last_st = st, last_en = en;
		//for (t = st0; t <= en0; ++t) printf("(%d,%d)\t(%d,%d,%d,%d)\t%d\n", r, t, ((int8_t*)u)[t], ((int8_t*)v)[t], ((int8_t*)x)[t], ((int8_t*)y)[t], H[t]); // for debugging
	}

	// printf ("%d\n", ez->score);

	// kfree(km, mem);
	// if (!approx_max) kfree(km, H);
	// if (with_cigar) { // backtrack
	// 	int rev_cigar = !!(flag & KSW_EZ_REV_CIGAR);
	// 	if (!ez->zdropped && !(flag&KSW_EZ_EXTZ_ONLY))
	// 		ksw_backtrack(km, 1, rev_cigar, long_thres, (uint8_t*)p, off, off_end, n_col_*16, tlen-1, qlen-1, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
	// 	else if (ez->max_t >= 0 && ez->max_q >= 0)
	// 		ksw_backtrack(km, 1, rev_cigar, long_thres, (uint8_t*)p, off, off_end, n_col_*16, ez->max_t, ez->max_q, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
	// 	kfree(km, mem2); kfree(km, off);
	// }
}

// #ifdef OLD
//
// __global__ void ksw_exts2_gpu( // void *km,
// 					int8_t* const u, int8_t* const v, int8_t* const x, int8_t* const y, int8_t* const x2, int8_t* const s, int8_t* const p, int32_t* const H, int32_t* const off, int32_t* const off_end,
// 					int8_t* const acceptor, int8_t* const donor, const int qlen, const uint8_t *query, const int tlen, const uint8_t *target, const int8_t m, const int8_t *mat,
// 				   	const int8_t q, const int8_t e, const int8_t q2, const int8_t noncan, const int zdrop, const int8_t junc_bonus, const int8_t junc_pen, const int flag, const uint8_t *junc, ksw_extz_t *ez)
// // ~KSW_CPU_DISPATCH
// {
// 	if (m <= 1 || qlen <= 0 || tlen <= 0 || q2 <= q + e) return;
// 	assert((flag & KSW_EZ_SPLICE_FOR) == 0 || (flag & KSW_EZ_SPLICE_REV) == 0); // can't be both set
//
// 	int r, t, last_st, last_en, max_sc, min_sc, long_thres, long_diff;
// 	int with_cigar = !(flag&KSW_EZ_SCORE_ONLY), approx_max = !!(flag&KSW_EZ_APPROX_MAX);
// 	int32_t H0 = 0, last_H0_t = 0;
//
// 	ez->max_q = ez->max_t = ez->mqe_t = ez->mte_q = -1;
// 	ez->max = 0, ez->score = ez->mqe = ez->mte = KSW_NEG_INF;
// 	ez->n_cigar = 0, ez->zdropped = 0, ez->reach_end = 0;
//
// 	const int8_t qe = q + e;
// 	const int8_t sc_mch = mat[0];
// 	const int8_t sc_mis = mat[1];
// 	const int8_t sc_N = mat[m*m-1] == 0? -e : mat[m*m-1];
// 	const int8_t m1 = m-1;
//
// 	const int tlen_ = (tlen + 15) / 16;
// 	const int n_col_ = ((qlen < tlen? qlen : tlen) + 15) / 16 + 1;
// 	const int qlen_ = (qlen + 15) / 16;
//
// 	for (t = 1, max_sc = mat[0], min_sc = mat[1]; t < m * m; ++t) {
// 		max_sc = max_sc > mat[t]? max_sc : mat[t];
// 		min_sc = min_sc < mat[t]? min_sc : mat[t];
// 	}
//
// 	if (-min_sc > 2 * (q + e)) return; // otherwise, we won't see any mismatches
//
// 	long_thres = (q2 - q) / e - 1;
// 	if (q2 > q + e + long_thres * e)
// 		++long_thres;
// 	long_diff = long_thres * e - (q2 - q);
//
// 	for(int i = 0; i < tlen_*16; i += blockDim.x)
// 		if(threadIdx.x + i < tlen_*16)
// 			u[threadIdx.x + i] = -q - e;
// 	__syncthreads();
//
// 	for(int i = 0; i < tlen_*16; i += blockDim.x)
// 		if(threadIdx.x + i < tlen_*16)
// 			v[threadIdx.x + i] = -q - e;
// 	__syncthreads();
//
// 	for(int i = 0; i < tlen_*16; i += blockDim.x)
// 		if(threadIdx.x + i < tlen_*16)
// 			x[threadIdx.x + i] = -q - e;
// 	__syncthreads();
//
// 	for(int i = 0; i < tlen_*16; i += blockDim.x)
// 		if(threadIdx.x + i < tlen_*16)
// 			y[threadIdx.x + i] = -q - e;
// 	__syncthreads();
//
// 	for(int i = 0; i < tlen_*16; i += blockDim.x)
// 		if(threadIdx.x + i < tlen_*16)
// 			x2[threadIdx.x + i] = -q2;
// 	__syncthreads();
//
// 	if (!approx_max) { // if !approximated alignment
// 		for (t = 0; t < tlen_*16; t += blockDim.x) {
// 			const int tid = threadIdx.x + t;
// 			if (tid < tlen_*16)
// 				H[tid] = KSW_NEG_INF; // initialize to -inf (gmem)
// 		}
// 		__syncthreads();
// 	}
//
// 	// set the donor and acceptor arrays. TODO: this assumes 0/1/2/3 encoding!
// 	if (flag & (KSW_EZ_SPLICE_FOR|KSW_EZ_SPLICE_REV)) {
//
// 		const int sp0[4] = { 8, 15, 21, 30 };
// 		int sp[4];
// 		if (flag & KSW_EZ_SPLICE_CMPLX) {
// 			for (t = 0; t < 4; ++t)
// 				sp[t] = (int)((double)sp0[t] / 3. + .499);
// 		} else {
// 			sp[0] = flag&KSW_EZ_SPLICE_FLANK? noncan / 2 : 0;
// 			sp[1] = sp[2] = sp[3] = noncan;
// 		}
//
// 		for(int i = 0; i < tlen_*16; i += blockDim.x)
// 			if(threadIdx.x + i < tlen_*16)
// 				donor[threadIdx.x + i] = -sp[3];
// 		__syncthreads();
//
// 		for(int i = 0; i < tlen_*16; i += blockDim.x)
// 			if(threadIdx.x + i < tlen_*16)
// 				acceptor[threadIdx.x + i] = -sp[3];
// 		__syncthreads();
//
// 		if (!(flag & KSW_EZ_REV_CIGAR)) {
//
// 			for (t = 0; t < tlen - 4; ++t) {
// 				int z = 3;
// 				if (flag & KSW_EZ_SPLICE_FOR) {
// 					if (target[t+1] == 2 && target[t+2] == 3)             // |GT.
// 						z = target[t+3] == 0 || target[t+3] == 2? -1 : 0; // |GTr or not
// 					else if (target[t+1] == 2 && target[t+2] == 1) z = 1; // |GC.
// 					else if (target[t+1] == 0 && target[t+2] == 3) z = 2; // |AT.
// 				} else if (flag & KSW_EZ_SPLICE_REV) {
// 					if (target[t+1] == 1 && target[t+2] == 3)             // |CT. (revcomp of .AG|)
// 						z = target[t+3] == 0 || target[t+3] == 2? -1 : 0;
// 					else if (target[t+1] == 2 && target[t+2] == 3) z = 2; // |GT. (revcomp of .AC|)
// 				}
// 				((int8_t*)donor)[t] = z < 0? 0 : -sp[z];
// 			}
//
// 			for (t = 2; t < tlen; ++t) {
// 				int z = 3;
// 				if (flag & KSW_EZ_SPLICE_FOR) {
// 					if (target[t-1] == 0 && target[t] == 2)               // .AG|
// 						z = target[t-2] == 1 || target[t-2] == 3? -1 : 0; // yAG| or not
// 					else if (target[t-1] == 0 && target[t] == 1) z = 2;   // .AC|
// 				} else if (flag & KSW_EZ_SPLICE_REV) {
// 					if (target[t-1] == 0 && target[t] == 1)               // .AC| (revcomp of |GT.)
// 						z = target[t-2] == 1 || target[t-2] == 3? -1 : 0; // yAC| or not
// 					else if (target[t-1] == 2 && target[t] == 1) z = 1;   // .GC| (revcomp of |GC.)
// 					else if (target[t-1] == 0 && target[t] == 3) z = 2;   // .AT| (revcomp of |AT.)
// 				}
// 				((int8_t*)acceptor)[t] = z < 0? 0 : -sp[z];
// 			}
// 		} else {
// 			for (t = 0; t < tlen - 4; ++t) {
// 				int z = 3;
// 				if (flag & KSW_EZ_SPLICE_FOR) {
// 					if (target[t+1] == 2 && target[t+2] == 0)             // |GA. (rev of .AG|)
// 						z = target[t+3] == 1 || target[t+3] == 3? -1 : 0;
// 					else if (target[t+1] == 1 && target[t+2] == 0) z = 2; // |CA. (rev of .AC|)
// 				} else if (flag & KSW_EZ_SPLICE_REV) {
// 					if (target[t+1] == 1 && target[t+2] == 0)             // |CA. (comp of |GT.)
// 						z = target[t+3] == 1 || target[t+3] == 3? -1 : 0;
// 					else if (target[t+1] == 1 && target[t+2] == 2) z = 1; // |CG. (comp of |GC.)
// 					else if (target[t+1] == 3 && target[t+2] == 0) z = 2; // |TA. (comp of |AT.)
// 				}
// 				((int8_t*)donor)[t] = z < 0? 0 : -sp[z];
// 			}
// 			for (t = 2; t < tlen; ++t) {
// 				int z = 3;
// 				if (flag & KSW_EZ_SPLICE_FOR) {
// 					if (target[t-1] == 3 && target[t] == 2)               // .TG| (rev of |GT.)
// 						z = target[t-2] == 0 || target[t-2] == 2? -1 : 0;
// 					else if (target[t-1] == 1 && target[t] == 2) z = 1;   // .CG| (rev of |GC.)
// 					else if (target[t-1] == 3 && target[t] == 0) z = 2;   // .TA| (rev of |AT.)
// 				} else if (flag & KSW_EZ_SPLICE_REV) {
// 					if (target[t-1] == 3 && target[t] == 1)               // .TC| (comp of .AG|)
// 						z = target[t-2] == 0 || target[t-2] == 2? -1 : 0;
// 					else if (target[t-1] == 3 && target[t] == 2) z = 2;   // .TG| (comp of .AC|)
// 				}
// 				((int8_t*)acceptor)[t] = z < 0? 0 : -sp[z];
// 			}
// 		}
// 	}
//
// 	if (junc && (flag & KSW_EZ_SPLICE_SCORE)) { // junc[] keeps the donor score
// 		uint8_t donor_val = !!(flag & KSW_EZ_SPLICE_FOR) == !(flag & KSW_EZ_REV_CIGAR)? 0 : 1;
// 		for (t = 0; t < tlen - 1; ++t)
// 			((int8_t*)donor)[t]    += junc[t+1] == 0xff || (junc[t+1]&1) !=  donor_val? -junc_pen : (int8_t)(junc[t+1]>>1) - (int8_t)KSW_SPSC_OFFSET;
// 		for (t = 0; t < tlen - 1; ++t)
// 			((int8_t*)acceptor)[t] += junc[t+1] == 0xff || (junc[t+1]&1) != !donor_val? -junc_pen : (int8_t)(junc[t+1]>>1) - (int8_t)KSW_SPSC_OFFSET;
// 		//for (t = 0; t < tlen - 1; ++t) if (junc[t+1] != 0xff) fprintf(stderr, "Y2\t%d\t%d\t%c\t%d\n", ((int8_t*)donor)[t], ((int8_t*)acceptor)[t], "DA"[junc[t+1]&1], (int8_t)(junc[t+1]>>1) - (int8_t)KSW_SPSC_OFFSET);
// 	} else if (junc) { // junc[] keeps the splice sites
// 		if (!(flag & KSW_EZ_REV_CIGAR)) {
// 			for (t = 0; t < tlen - 1; ++t)
// 				if (((flag & KSW_EZ_SPLICE_FOR) && (junc[t+1]&1)) || ((flag & KSW_EZ_SPLICE_REV) && (junc[t+1]&8)))
// 					((int8_t*)donor)[t] += junc_bonus;
// 			for (t = 0; t < tlen; ++t)
// 				if (((flag & KSW_EZ_SPLICE_FOR) && (junc[t]&2)) || ((flag & KSW_EZ_SPLICE_REV) && (junc[t]&4)))
// 					((int8_t*)acceptor)[t] += junc_bonus;
// 		} else {
// 			for (t = 0; t < tlen - 1; ++t)
// 				if (((flag & KSW_EZ_SPLICE_FOR) && (junc[t+1]&2)) || ((flag & KSW_EZ_SPLICE_REV) && (junc[t+1]&4)))
// 					((int8_t*)donor)[t] += junc_bonus;
// 			for (t = 0; t < tlen; ++t)
// 				if (((flag & KSW_EZ_SPLICE_FOR) && (junc[t]&1)) || ((flag & KSW_EZ_SPLICE_REV) && (junc[t]&8)))
// 					((int8_t*)acceptor)[t] += junc_bonus;
// 		}
// 	}
//
// 	for (r = 0, last_st = last_en = -1; r < qlen + tlen - 1; ++r) {
// 		int st = 0, en = tlen - 1, st0, en0, st_, en_;
// 		int8_t x1, x21, v1, *u8 = (int8_t*)u, *v8 = (int8_t*)v;
// 		const uint8_t* qrr = query + (qlen - 1 - r);
//
// 		// find the boundaries
// 		if (st < r - qlen + 1) st = r - qlen + 1;
// 		if (en > r) en = r;
// 		st0 = st, en0 = en;
// 		st = st / 16 * 16, en = (en + 16) / 16 * 16 - 1;
//
// 		// set boundary conditions
// 		if (st > 0) {
// 			if (st - 1 >= last_st && st - 1 <= last_en)
// 				x1 = x[st - 1], x21 = x2[st - 1], v1 = v8[st - 1]; // (r-1,s-1) calculated in the last round
// 			else x1 = -q - e, x21 = -q2, v1 = -q - e;
// 		} else {
// 			x1 = -q - e, x21 = -q2;
// 			v1 = r == 0? -q - e : r < long_thres? -e : r == long_thres? long_diff : 0;
// 		}
//
// 		if (en >= r) {
// 			y[r] = -q - e;
// 			u8[r] = r == 0? -q - e : r < long_thres? -e : r == long_thres? long_diff : 0;
// 		}
//
// 		// loop fission: set scores first
// 		if (!(flag & KSW_EZ_GENERIC_SC)) {
//
// 			int iter = ((en0 - st0)/16 + 1)*16 + st0;
//
// 			for (t = st0; t < iter; t += blockDim.x) {
// 				const int tid = t + threadIdx.x;
// 				if (tid < iter){
// 					const uint8_t tmp = target[tid] == qrr[tid] ? sc_mch : sc_mis;
// 					uint8_t mask = target[tid] == m1 ? 1 : 0;
// 					mask *= (qrr[tid] == m1 ? 1 : 0);
// 					s[tid] = mask ? sc_N : tmp;
// 				}
//
// 			}
// 			__syncthreads();
//
// 		} else {
//
// 			for (t = st0; t <= en0; t += blockDim.x){
// 				const int tid = threadIdx.x + t;
// 				if (tid <= en0){
// 					s[tid] = mat[target[tid] * m + qrr[tid]]; // ((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
// 				}
// 			}
// 			__syncthreads();
//
// 		}
//
// 		st_ = st / 16, en_ = en / 16;
// 		assert(en_ - st_ + 1 <= n_col_);
//
// 		if (!with_cigar) { // score only
//
// 			int8_t x1_local = x1, x21_local = x21, v1_local = v1;
//
// 			for (t = st; t <= en; t += blockDim.x) {
//
// 				const int tid = threadIdx.x + t;
//
// 				if (tid <= en){
//
// 					int8_t vt1;
// 					int8_t x2t1;
// 					int8_t xt1;
//
// 					if((threadIdx.x)==0){
// 						xt1  = x1_local;
// 						x1_local  = x[tid + blockDim.x - 1];
// 						x2t1 = x21_local;
// 						x21_local = x2[tid + blockDim.x - 1];
// 						vt1  = v1_local;
// 						v1_local  = v[tid + blockDim.x - 1];
// 					} else {
// 						xt1 = x[tid - 1];
// 						vt1 = v[tid - 1];
// 						x2t1 = x2[tid - 1];
// 					}
//
// 					int8_t a = xt1 + vt1;
// 					int8_t ut = u[tid];
// 					int8_t b = y[tid] + ut;
// 					int8_t z = s[tid];
// 					int8_t a2 = x2t1 + vt1;
// 					int8_t a2a = a2 + acceptor[tid];
//
// 					z = z > a ? z : a;
// 					z = z > b ? z : b;
// 					z = z > a2a ? z : a2a;
//
// 					u[tid] = z - vt1;
// 					v[tid] = z - ut;
// 					a = a - (z - q);
// 					b = b - (z - q);
// 					a2 = a2 - (z - q2);
//
// 					x[tid]  = (a > 0 ? a : 0) - qe;
// 					y[tid]  = (b > 0 ? b : 0) - qe;
// 					x2[tid] = (a2 > donor[tid] ? a2 : donor[tid]) - q2;
//
// 				}
//
// 			}
//
// 			__syncthreads();
//
// 		} else if (!(flag&KSW_EZ_RIGHT)) { // gap left-alignment
//
// 			int8_t *pr = p + ((size_t)r * n_col_ - st_)*16;
// 			off[r] = st, off_end[r] = en;
//
// 			int8_t x1_local = x1, x21_local = x21, v1_local = v1;
//
// 			for (t = st; t <= en; t += blockDim.x) {
//
// 				const int tid = threadIdx.x + t;
//
// 				if (tid <= en){
//
// 					int8_t vt1;
// 					int8_t x2t1;
// 					int8_t xt1;
//
// 					if((threadIdx.x)==0){
// 						xt1  = x1_local;
// 						x1_local = x[tid + blockDim.x - 1];
// 						x2t1 = x21_local;
// 						x21_local = x2[tid + blockDim.x - 1];
// 						vt1  = v1_local;
// 						v1_local = v[tid + blockDim.x - 1];
// 					} else {
// 						xt1 = x[tid - 1];
// 						vt1 = v[tid - 1];
// 						x2t1 = x2[tid - 1];
// 					}
//
// 					int8_t a = xt1 + vt1;
// 					int8_t ut = u[tid];
// 					int8_t b = y[tid] + ut;
// 					int8_t z = s[tid];
// 					int8_t a2 = x2t1 + vt1;
// 					int8_t a2a = a2 + acceptor[tid];
//
// 					int8_t d;
// 					d = a > z ? 1 : 0;
// 					z = z > a ? z : a;
// 					d = b > z ? 2 : d;
// 					z = z > b ? z : b;
// 					d = a2a > z ? 3 : d;
// 					z = z > a2a ? z : a2a;
//
// 					u[tid] = z - vt1;
// 					v[tid] = z - ut;
//
// 					a = a - (z - q);
// 					b = b - (z - q);
// 					a2 = a2 - (z - q2);
//
// 					x[tid] = (a > 0 ? a : 0)  - qe;
// 					d = a > 0 ? 0x08 : 0;
// 					y[tid] = (b > 0 ? b : 0) - qe;
// 					d = b > 0 ? 0x10 : 0;
//
// 					int8_t tmp = a2 > donor[tid] ? 0xFF : 0;
// 					int8_t tmp2 = a2 > donor[tid] ? a2 : donor[tid];
//
// 					x2[tid] = tmp2 - q2;
// 					d = tmp & 0x20;
//
// 					pr[tid] = d;
// 				}
//
// 			}
//
// 			__syncthreads();
//
// 		} else { // gap right-alignment
//
// 			int8_t *pr = p + ((size_t)r * n_col_ - st_)*16;
// 			off[r] = st, off_end[r] = en;
//
// 			int8_t x1_local = x1, x21_local = x21, v1_local = v1;
//
// 			for (t = st; t <= en; t += blockDim.x) {
//
// 				const int tid = threadIdx.x + t;
//
// 				if (tid <= en){
//
// 					int8_t vt1;
// 					int8_t x2t1;
// 					int8_t xt1;
//
// 					if (threadIdx.x==0){
// 						xt1  = x1_local;
// 						x1_local  = x[tid + blockDim.x - 1];
// 						x2t1 = x21_local;
// 						x21_local = x2[tid + blockDim.x - 1];
// 						vt1 = v1_local;
// 						v1_local = v[tid + blockDim.x - 1];
// 					} else {
// 						xt1 = x[tid - 1];
// 						vt1 = v[tid - 1];
// 						x2t1 = x2[tid - 1];
// 					}
//
// 					int8_t a = xt1 + vt1;
// 					int8_t ut = u[tid];
// 					int8_t b = y[tid] + ut;
// 					int8_t z = s[tid];
// 					int8_t a2 = x2t1 + vt1;
// 					int8_t a2a = a2 + acceptor[tid];
//
// 					int8_t d;
//
// 					d = z > a ? 0 : 1;
// 					z = z > a ? z : a;
// 					d = z > b ? d : 2;
// 					z = z > b ? z : b;
// 					d = z > a2a ? d : 3;
// 					z = z > a2a ? z : a2a;
//
// 					u[tid] = z - vt1;
// 					v[tid] = z - ut;
//
// 					a = a - (z - q);
// 					b = b - (z - q);
// 					a2 = a2 - (z - q2);
//
// 					x[tid] = (a > 0 ? a : 0)  - qe;
// 					d = a > 0 ? 0x08 : 0;
// 					y[tid] = (b > 0 ? b : 0) - qe;
// 					d = b > 0 ? 0x10 : 0;
//
// 					int8_t tmp = a2 > donor[tid] ? 0xFF : 0;
// 					int8_t tmp2 = a2 > donor[tid] ? a2 : donor[tid];
//
// 					x2[tid] = tmp2 - q2;
// 					d = a > 0 ? 0x20 : 0;
//
// 					pr[tid] = d;
// 				}
// 			}
//
// 			__syncthreads();
// 		}
// 		if (!approx_max) { // find the exact max with a 32-bit score array
// 			int32_t max_H, max_t;
// 			// compute H[], max_H and max_t
// 			if (r > 0) {
// 				const int32_t en1 = st0 + (en0 - st0) / blockDim.x * blockDim.x;
// 				__shared__ int32_t HH[NUM_THREADS], tt[NUM_THREADS];
// 				int32_t max_H_, max_t_;
// 				max_H = H[en0] = en0 > 0? H[en0-1] + u8[en0] : H[en0] + v8[en0]; /* special casing the last element */
// 				max_t = en0;
//
// 				max_H_ = max_H;
// 				max_t_ = max_t;
//
// 				for (t = st0; t < en1; t+=blockDim.x) { /* this implements: H[t]+=v8[t]-qe; if(H[t]>max_H) max_H=H[t],max_t=t; */
// 					const int tid = threadIdx.x + t;
// 					if (tid < en1) {
// 						int32_t H1 = H[tid];
// 						int32_t t_ = v8[tid];
// 						H1 = H1 + t_;
// 						H[tid] = H1;
// 						t_ = t;
// 						int32_t tmp = H1 > max_H_ ? 0xFFFFFFFF : 0;
// 						max_H_ = (tmp & H1) | ((~tmp) & max_H_);
// 						max_t_ = (tmp & t_) | ((~tmp) & max_t_);
// 					}
// 				}
// 				__syncthreads();
//
// 				HH[threadIdx.x] = max_H_;
// 				tt[threadIdx.x] = max_t_;
//
// 				for (int i = 0; i < 4; ++i)
// 					if (max_H < HH[i]) max_H = HH[i], max_t = tt[i] + i;
//
// 				for (; t < en0; ++t) { // for the rest of values that haven't been computed with SSE
// 					H[t] += (int32_t)v8[t];
// 					if (H[t] > max_H)
// 						max_H = H[t], max_t = t;
// 				}
// 			} else H[0] = v8[0] - qe, max_H = H[0], max_t = 0; // special casing r==0
// 			// update ez
// 			if (en0 == tlen - 1 && H[en0] > ez->mte)
// 				ez->mte = H[en0], ez->mte_q = r - en0;
// 			if (r - st0 == qlen - 1 && H[st0] > ez->mqe)
// 				ez->mqe = H[st0], ez->mqe_t = st0;
// 			if (ksw_apply_zdrop_gpu(ez, 1, max_H, r, max_t, zdrop, 0)) break;
// 			if (r == qlen + tlen - 2 && en0 == tlen - 1)
// 				ez->score = H[tlen - 1];
// 		} else { // find approximate max; Z-drop might be inaccurate, too.
// 			if (r > 0) {
// 				if (last_H0_t >= st0 && last_H0_t <= en0 && last_H0_t + 1 >= st0 && last_H0_t + 1 <= en0) {
// 					int32_t d0 = v8[last_H0_t];
// 					int32_t d1 = u8[last_H0_t + 1];
// 					if (d0 > d1) H0 += d0;
// 					else H0 += d1, ++last_H0_t;
// 				} else if (last_H0_t >= st0 && last_H0_t <= en0) {
// 					H0 += v8[last_H0_t];
// 				} else {
// 					++last_H0_t, H0 += u8[last_H0_t];
// 				}
// 			} else H0 = v8[0] - qe, last_H0_t = 0;
// 			if ((flag & KSW_EZ_APPROX_DROP) && ksw_apply_zdrop_gpu(ez, 1, H0, r, last_H0_t, zdrop, 0)) break;
// 			if (r == qlen + tlen - 2 && en0 == tlen - 1)
// 				ez->score = H0;
// 		}
// 		last_st = st, last_en = en;
// 		//for (t = st0; t <= en0; ++t) printf("(%d,%d)\t(%d,%d,%d,%d)\t%d\n", r, t, ((int8_t*)u)[t], ((int8_t*)v)[t], ((int8_t*)x)[t], ((int8_t*)y)[t], H[t]); // for debugging
// 	}
//
// 	// printf ("%d\n", ez->score);
//
// 	// kfree(km, mem);
// 	// if (!approx_max) kfree(km, H);
// 	// if (with_cigar) { // backtrack
// 	// 	int rev_cigar = !!(flag & KSW_EZ_REV_CIGAR);
// 	// 	if (!ez->zdropped && !(flag&KSW_EZ_EXTZ_ONLY))
// 	// 		ksw_backtrack(km, 1, rev_cigar, long_thres, (uint8_t*)p, off, off_end, n_col_*16, tlen-1, qlen-1, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
// 	// 	else if (ez->max_t >= 0 && ez->max_q >= 0)
// 	// 		ksw_backtrack(km, 1, rev_cigar, long_thres, (uint8_t*)p, off, off_end, n_col_*16, ez->max_t, ez->max_q, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
// 	// 	kfree(km, mem2); kfree(km, off);
// 	// }
// }
//
// #endif
