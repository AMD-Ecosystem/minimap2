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

__global__ void ksw_extz2_gpu_multialign( //void *km,
    int8_t* __restrict__ const u_d,
    int8_t* __restrict__ const v_d,
    int8_t* __restrict__ const x_d,
    int8_t* __restrict__ const y_d,
    int8_t* __restrict__ const s_d,
    uint8_t* __restrict__ const p_d,
    int32_t* __restrict__ const H_d,
    int* __restrict__ const off_d,
    int* __restrict__ const off_end_d,
    const int* __restrict__ const qlen_d,
    const uint8_t* __restrict__ const query_d,
    const int* __restrict__ const tlen_d,
    const uint8_t* __restrict__ const target_d,
    const int8_t m,
    const int8_t* __restrict__ const mat,
    int8_t q,
    int8_t e,
    const int w_d,
    const int zdrop,
    const int end_bonus,
    const int flag,
    ksw_extz_t* __restrict__ const ez_d,
    const uint64_t* __restrict__ const tlen_ps_d,
    const uint64_t* __restrict__ const qlen_ps_d,
    const uint64_t* __restrict__ const p_ps_d,
    const uint64_t* __restrict__ const off_ps_d)
// ~KSW_CPU_DISPATCH
{
	const int qlen = qlen_d[blockIdx.x];
	const int tlen = tlen_d[blockIdx.x];
	const uint8_t* const target = target_d + tlen_ps_d[blockIdx.x];
	const uint8_t* const query = query_d + qlen_ps_d[blockIdx.x];
	uint8_t* const p = p_d + p_ps_d[blockIdx.x];
	int8_t* const u = u_d + tlen_ps_d[blockIdx.x];
	int8_t* const v = v_d + tlen_ps_d[blockIdx.x];
	int8_t* const x = x_d + tlen_ps_d[blockIdx.x];
	int8_t* const y = y_d + tlen_ps_d[blockIdx.x];
	int8_t* const s = s_d + tlen_ps_d[blockIdx.x];
	int32_t* const H = H_d + tlen_ps_d[blockIdx.x];

	if (m <= 0 || qlen <= 0 || tlen <= 0) return;

	int* const off = !(flag & KSW_EZ_SCORE_ONLY) ? off_d + off_ps_d[blockIdx.x] : 0;
	int* const off_end = !(flag & KSW_EZ_SCORE_ONLY) ? off_end_d + off_ps_d[blockIdx.x] : 0;

	ksw_extz_t ez_local;

	int r, t, qe = q + e, n_col_, tlen_, qlen_, last_st, last_en, wl, wr, max_sc, min_sc;
	const int with_cigar = !(flag & KSW_EZ_SCORE_ONLY), approx_max = !!(flag & KSW_EZ_APPROX_MAX);
	int32_t H0 = 0, last_H0_t = 0;

	ez_local.max_q = ez_local.max_t = ez_local.mqe_t = ez_local.mte_q = -1;
	ez_local.max = 0, ez_local.score = ez_local.mqe = ez_local.mte = KSW_NEG_INF;
	ez_local.n_cigar = 0, ez_local.zdropped = 0, ez_local.reach_end = 0;

	const int8_t sc_mch_ = mat[0];
	const int8_t sc_mis_ = mat[1];
	const int8_t sc_N_ = mat[m * m - 1] == 0 ? -e : mat[m * m - 1];
	const int8_t m1_ = m - 1;
	const int8_t max_sc_ = mat[0] + (q + e) * 2; // max_sc_ = _mm_set1_epi8(mat[0] + (q + e) * 2);

	int w = w_d;
	if (w < 0) w = tlen > qlen ? tlen : qlen;
	wl = wr = w;
	tlen_ = (tlen + 15) / 16; // round to multiple of 16
	n_col_ = qlen < tlen ? qlen : tlen;
	n_col_ = ((n_col_ < w + 1 ? n_col_ : w + 1) + 15) / 16 + 1; // round to multiple of 16
	qlen_ = (qlen + 15) / 16; // round to multiple of 16

	for (t = 1, max_sc = mat[0], min_sc = mat[1]; t < m * m; ++t) {
		max_sc = max_sc > mat[t] ? max_sc : mat[t];
		min_sc = min_sc < mat[t] ? min_sc : mat[t];
	}

	for (int i = 0; i < tlen_ * 16; i += blockDim.x)
		if (threadIdx.x + i < tlen_ * 16)
			u[threadIdx.x + i] = 0;
	__syncthreads();

	for (int i = 0; i < tlen_ * 16; i += blockDim.x)
		if (threadIdx.x + i < tlen_ * 16)
			v[threadIdx.x + i] = 0;
	__syncthreads();

	for (int i = 0; i < tlen_ * 16; i += blockDim.x)
		if (threadIdx.x + i < tlen_ * 16)
			x[threadIdx.x + i] = 0;
	__syncthreads();

	for (int i = 0; i < tlen_ * 16; i += blockDim.x)
		if (threadIdx.x + i < tlen_ * 16)
			y[threadIdx.x + i] = 0;
	__syncthreads();

	if (-min_sc > 2 * (q + e)) return; // otherwise, we won't see any mismatches

	if (!approx_max) { // if !approximated alignment
		for (t = 0; t < tlen_ * 16; t += blockDim.x) {
			const int tid = threadIdx.x + t;
			if (tid < tlen_ * 16)
				H[tid] = KSW_NEG_INF; // initialize to -inf (gmem)
		}
		__syncthreads();
	}

	for (r = 0, last_st = last_en = -1; r < qlen + tlen - 1; ++r) {
		int st = 0, en = tlen - 1, st0, en0, st_, en_;
		int8_t x1, v1;
		const uint8_t* qrr = query + (qlen - 1 - r);
		int8_t *u8 = u, *v8 = v;

		/* find the boundaries */
		if (st < r - qlen + 1) st = r - qlen + 1;
		if (en > r) en = r;
		if (st < (r - wr + 1) >> 1) st = (r - wr + 1) >> 1; /* take the ceil */
		if (en > (r + wl) >> 1) en = (r + wl) >> 1; /* take the floor */
		if (st > en) {
			ez_local.zdropped = 1;
			break;
		}

		st0 = st, en0 = en;
		st = st / 16 * 16, en = (en + 16) / 16 * 16 - 1;

		/* set boundary conditions */
		if (st > 0) {
			if (st - 1 >= last_st && st - 1 <= last_en)
				x1 = x[st - 1], v1 = v8[st - 1]; // (r-1,s-1) calculated in the last round
			else
				x1 = v1 = 0; // not calculated; set to zeros
		} else
			x1 = 0, v1 = r ? q : 0;

		if (en >= r) y[r] = 0, u8[r] = r ? q : 0;

		// loop fission: set scores first
		if (!(flag & KSW_EZ_GENERIC_SC)) {

			int iter = ((en0 - st0) / 16 + 1) * 16 + st0;

			for (t = st0; t < iter; t += blockDim.x) {
				const int tid = t + threadIdx.x;
				if (tid < iter) {
					const uint8_t tmp = target[tid] == qrr[tid] ? sc_mch_ : sc_mis_;
					uint8_t mask = target[tid] == m1_ ? 1 : 0;
					mask *= (qrr[tid] == m1_ ? 1 : 0);
					s[tid] = mask ? sc_N_ : tmp;
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

			int8_t x1_local = x1, v1_local = v1;

			for (t = st; t <= en; t += blockDim.x) {

				const int tid = threadIdx.x + t;

				if (tid <= en) {

					int8_t x_val = x[tid];
					int8_t v_val = v[tid];

					int8_t xt1 = __shfl_up(x_val, 1);
					int8_t vt1 = __shfl_up(v_val, 1);

					if (threadIdx.x == 0) {
						xt1 = x1_local;
						x1_local = x[tid + blockDim.x - 1];
						vt1 = v1_local;
						v1_local = v[tid + blockDim.x - 1];
					}

					int8_t a = xt1 + vt1;
					int8_t ut = u[tid];
					int8_t b = y[tid] + ut;

					int8_t z = s[tid] + ((q + e) * 2);

					z = z > a ? z : a;
					z = z > b ? z : b;
					z = z < max_sc_ ? z : max_sc_;
					u[tid] = z - vt1;
					v[tid] = z - ut;
					z = z - q;
					a = a - z;
					b = b - z;
					x[tid] = a > 0 ? a : 0;
					y[tid] = b > 0 ? b : 0;
				}
			}

			__syncthreads();

		} else if (!(flag & KSW_EZ_RIGHT)) { /* gap left-alignment */

			// int8_t *pr = p + ((size_t)r * n_col_ - st_)*16; // __m128i *pr = p + (size_t)r * n_col_ - st_;
			const int size = ((size_t)r * n_col_ - st_) * 16;
			off[r] = st, off_end[r] = en;

			int8_t x1_local = x1, v1_local = v1;

			for (t = st; t <= en; t += blockDim.x) {

				const int tid = threadIdx.x + t;

				if (tid <= en) {

					int8_t x_val = x[tid];
					int8_t v_val = v[tid];

					int8_t xt1 = __shfl_up(x_val, 1);
					int8_t vt1 = __shfl_up(v_val, 1);

					if (threadIdx.x == 0) {
						xt1 = x1_local;
						x1_local = x[tid + blockDim.x - 1];
						vt1 = v1_local;
						v1_local = v[tid + blockDim.x - 1];
					}

					int8_t a = xt1 + vt1;
					int8_t ut = u[tid];
					int8_t b = y[tid] + ut;

					int8_t z = s[tid] + ((q + e) * 2);

					int8_t d = a > z ? 1 : 0;
					z = z > a ? z : a;
					d = b > z ? 2 : d;

					z = z > b ? z : b;
					z = z < max_sc_ ? z : max_sc_;

					u[tid] = z - vt1;
					v[tid] = z - ut;

					z = z - q;
					a = a - z;
					b = b - z;

					x[tid] = a > 0 ? a : 0;
					d = a > 0 ? 0x08 : 0;
					y[tid] = b > 0 ? b : 0;
					d = b > 0 ? 0x10 : 0;

					p[size + tid] = d;
				}
			}

			__syncthreads();

		} else { /* gap right-alignment */

			// int8_t *pr = p + ((size_t)r * n_col_ - st_)*16; // __m128i *pr = p + (size_t)r * n_col_ - st_;
			const int size = ((size_t)r * n_col_ - st_) * 16;
			off[r] = st, off_end[r] = en;

			int8_t x1_local = x1, v1_local = v1;

			for (t = st; t <= en; t += blockDim.x) {

				const int tid = threadIdx.x + t;

				if (tid <= en) {

					int8_t x_val = x[tid];
					int8_t v_val = v[tid];

					int8_t xt1 = __shfl_up(x_val, 1);
					int8_t vt1 = __shfl_up(v_val, 1);

					if (threadIdx.x == 0) {
						xt1 = x1_local;
						x1_local = x[tid + blockDim.x - 1];
						vt1 = v1_local;
						v1_local = v[tid + blockDim.x - 1];
					}

					int8_t a = xt1 + vt1;
					int8_t ut = u[tid];
					int8_t b = y[tid] + ut;

					int8_t z = s[tid] + ((q + e) * 2);

					int8_t d;

					d = z > a ? 0 : 1;
					z = z > a ? z : a;
					d = z > b ? d : 2;
					z = z > b ? z : b;
					z = z < max_sc_ ? z : max_sc_;

					u[tid] = z - vt1;
					v[tid] = z - ut;

					a = a - (z - q);
					b = b - (z - q);

					x[tid] = (a > 0 ? a : 0);
					d = a > 0 ? 0x08 : 0;
					y[tid] = (b > 0 ? b : 0);
					d = b > 0 ? 0x10 : 0;

					p[size + tid] = d;
				}
			}

			__syncthreads();
		}

		if (!approx_max) { /* find the exact max with a 32-bit score array */
			int32_t max_H, max_t;
			/* compute H[], max_H and max_t */
			if (r > 0) {

				int32_t max_H_local, max_t_local;

				max_H = max_H_local = en0 > 0 ? H[en0 - 1] + u[en0] - qe : H[en0] + v[en0] - qe; /* special casing the last element */
				H[en0] = en0 > 0 ? H[en0 - 1] + u[en0] - qe : H[en0] + v[en0] - qe;
				max_t = max_t_local = en0;

				for (t = st0; t < en0; t += blockDim.x) { /* this implements: H[t]+=v8[t]-qe; if(H[t]>max_H) max_H=H[t],max_t=t; */
					const int tid = threadIdx.x + t;
					if (tid < en0) {
						H[tid] += v[tid] - qe;
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

			} else
				H[0] = v8[0] - qe - qe, max_H = H[0], max_t = 0; // special casing r==0

			// update ez_local
			if (en0 == tlen - 1 && H[en0] > ez_local.mte)
				ez_local.mte = H[en0], ez_local.mte_q = r - en0;
			if (r - st0 == qlen - 1 && H[st0] > ez_local.mqe)
				ez_local.mqe = H[st0], ez_local.mqe_t = st0;
			if (ksw_apply_zdrop_gpu(&ez_local, 1, max_H, r, max_t, zdrop, e)) break;
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez_local.score = H[tlen - 1];
		} else { // find approximate max; Z-drop might be inaccurate, too.

			if (r > 0) {
				if (last_H0_t >= st0 && last_H0_t <= en0 && last_H0_t + 1 >= st0 && last_H0_t + 1 <= en0) {
					int32_t d0 = v8[last_H0_t] - qe;
					int32_t d1 = u8[last_H0_t + 1] - qe;
					if (d0 > d1)
						H0 += d0;
					else
						H0 += d1, ++last_H0_t;
				} else if (last_H0_t >= st0 && last_H0_t <= en0) {
					H0 += v8[last_H0_t] - qe;
				} else {
					++last_H0_t, H0 += u8[last_H0_t] - qe;
				}
				if ((flag & KSW_EZ_APPROX_DROP) && ksw_apply_zdrop_gpu(&ez_local, 1, H0, r, last_H0_t, zdrop, e)) break;
			} else
				H0 = v8[0] - qe - qe, last_H0_t = 0;

			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez_local.score = H0;
		}
		last_st = st, last_en = en;
		//for (t = st0; t <= en0; ++t) printf("(%d,%d)\t(%d,%d,%d,%d)\t%d\n", r, t, ((int8_t*)u)[t], ((int8_t*)v)[t], ((int8_t*)x)[t], ((int8_t*)y)[t], H[t]); // for debugging
	}

	ez_d[blockIdx.x] = ez_local;


	// printf("%d\n", ez->score);

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
	// 	kfree(km, mem2); kfree(km, off);
	// }
}
