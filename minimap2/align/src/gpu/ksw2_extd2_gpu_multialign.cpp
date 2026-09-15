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
// #include <assert.h>
#include "ksw2.h"

// __constant__ int8_t mat[25] = {2, -4, -4, -4, 0, -4, 2, -4, -4, 0, -4, -4, 2, -4, 0, -4, -4, -4, 2, 0, 0, 0, 0, 0, 0};

#define m 5
#define m1 4

#define NUM_SEGMENTS 2 // num segments (reduced from 4 to lower register pressure)
#define NUM_ELEMENTS 64 // elements processed simultaneously
#define SEGMENT_SIZE 1024 // segment number of elements to process per iteration (NUM_SEGMENTS * 512)
#define SHMEM_SIZE 2048 // size of shared memory buffer (2 * SEGMENT_SIZE for double-buffering)

#define QUARTER_SEGMENT_SIZE 256 // SEGMENT_SIZE/4 for int32_t bulk copies

#define WARP_SIZE 64
#define MAX_WARPS 8 // supports up to 512 threads per block

__device__ size_t get_clock(void) __attribute__((noinline));
__device__ size_t get_clock(void)
{
	return clock64();
}

__device__ __forceinline__ int ksw_apply_zdrop_gpu(ksw_extz_t* const ez, const int32_t H, const int a, const int b, const int zdrop, const int8_t e)
{
	if (H > (int32_t)ez->max) {
		ez->max = H, ez->max_t = b, ez->max_q = a - b;
	} else if (b >= ez->max_t && a - b >= ez->max_q) {
		int tl = b - ez->max_t, ql = (a - b) - ez->max_q, l;
		l = tl > ql ? tl - ql : ql - tl;
		if (zdrop >= 0 && ez->max - H > zdrop + l * e) {
			ez->zdropped = 1;
			return 1;
		}
	}

	return 0;
}

__device__ __forceinline__ void memcpy_to_shared(int8_t* const u, int8_t* const v, int8_t* const x, int8_t* const y, int8_t* const x2, int8_t* const y2,
    int8_t* const u_shmem, int8_t* const v_shmem, int8_t* const x_shmem, int8_t* const y_shmem, int8_t* const x2_shmem, int8_t* const y2_shmem, const int start)
{

	int32_t* const u_p = (int32_t*)(u + start);
	int32_t* const v_p = (int32_t*)(v + start);
	int32_t* const x_p = (int32_t*)(x + start);
	int32_t* const y_p = (int32_t*)(y + start);
	int32_t* const x2_p = (int32_t*)(x2 + start);
	int32_t* const y2_p = (int32_t*)(y2 + start);

	int32_t* const u_shmem_p = (int32_t*)(u_shmem);
	int32_t* const v_shmem_p = (int32_t*)(v_shmem);
	int32_t* const x_shmem_p = (int32_t*)(x_shmem);
	int32_t* const y_shmem_p = (int32_t*)(y_shmem);
	int32_t* const x2_shmem_p = (int32_t*)(x2_shmem);
	int32_t* const y2_shmem_p = (int32_t*)(y2_shmem);

	int32_t tid = threadIdx.x;

	for (int i = 0; i < QUARTER_SEGMENT_SIZE; i += blockDim.x)
		if (tid + i < QUARTER_SEGMENT_SIZE) {
			u_shmem_p[tid + i] = u_p[i + tid];
			v_shmem_p[tid + i] = v_p[i + tid];
			x_shmem_p[tid + i] = x_p[i + tid];
			y_shmem_p[tid + i] = y_p[i + tid];
			x2_shmem_p[tid + i] = x2_p[i + tid];
			y2_shmem_p[tid + i] = y2_p[i + tid];
		}


	// int32_t tid = threadIdx.x;
	// for(int i = 0; i < SEGMENT_SIZE; i += blockDim.x)
	// 	if(tid + i < SEGMENT_SIZE){
	// 		u_shmem[tid + i] = u[start + i + tid];
	// 		v_shmem[tid + i] = v[start + i + tid];
	// 		x_shmem[tid + i] = x[start + i + tid];
	// 		y_shmem[tid + i] = y[start + i + tid];
	// 		x2_shmem[tid + i] = x2[start + i + tid];
	// 		y2_shmem[tid + i] = y2[start + i + tid];
	// 	}
}

__device__ __forceinline__ void memcpy_to_global(int8_t* const u, int8_t* const v, int8_t* const x, int8_t* const y, int8_t* const x2, int8_t* const y2,
    int8_t* const u_shmem, int8_t* const v_shmem, int8_t* const x_shmem, int8_t* const y_shmem, int8_t* const x2_shmem, int8_t* const y2_shmem, const int start, const int no_elements)
{

	int32_t* const u_p = (int32_t*)(u + start);
	int32_t* const v_p = (int32_t*)(v + start);
	int32_t* const x_p = (int32_t*)(x + start);
	int32_t* const y_p = (int32_t*)(y + start);
	int32_t* const x2_p = (int32_t*)(x2 + start);
	int32_t* const y2_p = (int32_t*)(y2 + start);

	int32_t* const u_shmem_p = (int32_t*)(u_shmem);
	int32_t* const v_shmem_p = (int32_t*)(v_shmem);
	int32_t* const x_shmem_p = (int32_t*)(x_shmem);
	int32_t* const y_shmem_p = (int32_t*)(y_shmem);
	int32_t* const x2_shmem_p = (int32_t*)(x2_shmem);
	int32_t* const y2_shmem_p = (int32_t*)(y2_shmem);

	int32_t tid = threadIdx.x;

	int iters = no_elements >> 2; // divide by 4 for int32_t

	for (int i = 0; i < iters; i += blockDim.x)
		if (tid + i < iters) {
			u_p[tid + i] = u_shmem_p[i + tid];
			v_p[tid + i] = v_shmem_p[i + tid];
			x_p[tid + i] = x_shmem_p[i + tid];
			y_p[tid + i] = y_shmem_p[i + tid];
			x2_p[tid + i] = x2_shmem_p[i + tid];
			y2_p[tid + i] = y2_shmem_p[i + tid];
		}


	// int32_t tid = threadIdx.x;
	// for(int i = 0; i < no_elements; i += blockDim.x)
	// 	if(tid + i < no_elements) {
	// 		u[start + i + tid] = u_shmem[tid + i];
	// 		v[start + i + tid] = v_shmem[tid + i];
	// 		x[start + i + tid] = x_shmem[tid + i];
	// 		y[start + i + tid] = y_shmem[tid + i];
	// 		x2[start + i + tid] = x2_shmem[tid + i];
	// 		y2[start + i + tid] = y2_shmem[tid + i];
	// 	}
}

__device__ __forceinline__ void swap_pointers(int8_t*& src, int8_t*& dest)
{
	int8_t* tmp;
	tmp = src;
	src = dest;
	dest = tmp;
}

// __launch_bounds__(64)
__global__ void ksw_extd2_gpu_multialign( // void *km,
    int8_t* __restrict__ const u_d,
    int8_t* __restrict__ const v_d,
    int8_t* __restrict__ const x_d,
    int8_t* __restrict__ const y_d,
    int8_t* __restrict__ const x2_d,
    int8_t* __restrict__ const y2_d,
    // int8_t* __restrict__ const s_d,
    uint8_t* __restrict__ const p_d,
    int32_t* __restrict__ const H_d,
    int* __restrict__ const off_d,
    int* __restrict__ const off_end_d,
    const int* __restrict__ const qlen_d,
    const uint8_t* __restrict__ const query_d,
    const int* __restrict__ const tlen_d,
    const uint8_t* __restrict__ const target_d,
    // const int8_t m,
    const int8_t* const mat_d,
    const int8_t q, const int8_t e, const int8_t q2, const int8_t e2, const int* __restrict__ w_d, const int zdrop, const int end_bonus, const int flag,
    ksw_extz_t* __restrict__ const ez_d,
    const uint32_t* __restrict__ const tlen_ps_d,
    const uint32_t* __restrict__ const qlen_ps_d,
    const uint64_t* __restrict__ const p_ps_d,
    const uint32_t* __restrict__ const off_ps_d,
    const int8_t min_sc)
{

	const int32_t bid = blockIdx.x;
	const int32_t tid = threadIdx.x;

	const uint32_t offset_t_ps = tlen_ps_d[bid];
	const uint32_t offset_q_ps = qlen_ps_d[bid];
	const uint64_t offset_p_ps = p_ps_d[bid];
	const uint32_t offset_o_ps = !(flag & KSW_EZ_SCORE_ONLY) ? off_ps_d[bid] : 0;

	const int qlen = qlen_d[bid];
	const int tlen = tlen_d[bid];

	const uint8_t* const target = target_d + offset_t_ps;

	int8_t* const u = u_d + offset_t_ps;
	int8_t* const v = v_d + offset_t_ps;
	int8_t* const x = x_d + offset_t_ps;
	int8_t* const y = y_d + offset_t_ps;
	int8_t* const x2 = x2_d + offset_t_ps;
	int8_t* const y2 = y2_d + offset_t_ps;
	int32_t* const H = H_d + offset_t_ps;

	const uint8_t* const query = query_d + offset_q_ps;
	uint8_t* const p = p_d + offset_p_ps;

	int* const off = !(flag & KSW_EZ_SCORE_ONLY) ? off_d + offset_o_ps : 0;
	int* const off_end = !(flag & KSW_EZ_SCORE_ONLY) ? off_end_d + offset_o_ps : 0;

	__shared__ int8_t mat[25];
	if (tid < 25) mat[tid] = mat_d[tid];

	__shared__ int8_t u_shmem[SHMEM_SIZE];
	__shared__ int8_t v_shmem[SHMEM_SIZE];
	__shared__ int8_t x_shmem[SHMEM_SIZE];
	__shared__ int8_t y_shmem[SHMEM_SIZE];
	__shared__ int8_t x2_shmem[SHMEM_SIZE];
	__shared__ int8_t y2_shmem[SHMEM_SIZE];

	__shared__ int8_t bnd_x[MAX_WARPS];
	__shared__ int8_t bnd_v[MAX_WARPS];
	__shared__ int8_t bnd_x2[MAX_WARPS];

	__shared__ int32_t reduce_H[MAX_WARPS];
	__shared__ int32_t reduce_t[MAX_WARPS];

	const int warp_id = tid / WARP_SIZE;
	const int lane_id = tid % WARP_SIZE;

	__shared__ ksw_extz_t ez_local;
	__shared__ int should_break;
	if (tid == 0) {
		ez_local.max_q = ez_local.max_t = ez_local.mqe_t = ez_local.mte_q = -1;
		ez_local.max = 0, ez_local.score = ez_local.mqe = ez_local.mte = KSW_NEG_INF;
		ez_local.n_cigar = 0, ez_local.zdropped = 0, ez_local.reach_end = 0;
		ez_local.m_cigar = 0;
		ez_local.cigar = NULL;
		should_break = 0;
	}
	__syncthreads();

	// ez_d[bid].max_q = ez_d[bid].max_t = ez_d[bid].mqe_t = ez_d[bid].mte_q = -1;
	// ez_d[bid].max = 0, ez_d[bid].score = ez_d[bid].mqe = ez_d[bid].mte = KSW_NEG_INF;
	// ez_d[bid].n_cigar = 0, ez_d[bid].zdropped = 0, ez_d[bid].reach_end = 0;

	/* Mark as zdropped on early exit to prevent backtracking with uninitialized off[] */
	if (qlen <= 0 || tlen <= 0) {
		if (tid == 0) {
			ez_local.zdropped = 1;
			ez_d[bid] = ez_local;
		}
		return;
	}

	int w = w_d[bid];
	if (w < 0) w = tlen > qlen ? tlen : qlen; // set bandwidth in case it is negative (non-banded alignment)
	const int tlen_ = (tlen + 15) >> 4; // / 16;										// round to multiple of 16
	int n_col_ = qlen < tlen ? qlen : tlen;
	n_col_ = (((n_col_ < w + 1 ? n_col_ : w + 1) + 15) >> 4) + 1; // round to multiple of 16
	const int qlen_ = (qlen + 15) >> 4; //  / 16;                                     // round to multiple of 16

	const int8_t qe = q + e;
	const int8_t qe2 = q2 + e2;
	const int8_t sc_mch = mat[0];
	const int8_t sc_mis = mat[1];
	const int8_t sc_N = mat[24] == 0 ? -e2 : mat[24];

	/* Mark as zdropped on early exit to prevent backtracking with uninitialized off[] */
	if (-min_sc > 2 * (qe)) {
		if (tid == 0) {
			ez_local.zdropped = 1;
			ez_d[bid] = ez_local;
		}
		return;
	}

	int long_thres = e != e2 ? (q2 - q) / (e - e2) - 1 : 0; // if double affine, set the threshold for long gaps(?) to ((q2 - q) / (e - e2) - 1), otherwise 0
	if ((qe2) + long_thres * e2 > (qe) + long_thres * e)
		++long_thres;

	// // OLD: byte-wide LDS init (6 ds_write_b8 per element)
	for (int i = 0; i < SHMEM_SIZE; i += blockDim.x) {
		if (tid + i < SHMEM_SIZE) {
			u_shmem[tid + i] = -qe;
			v_shmem[tid + i] = -qe;
			x_shmem[tid + i] = -qe;
			y_shmem[tid + i] = -qe;
			x2_shmem[tid + i] = -(qe2);
			y2_shmem[tid + i] = -(qe2);
		}
	}

	for (int i = 0; i < tlen_ << 4; i += blockDim.x) {
		if (tid + i < tlen_ << 4) {
			u[tid + i] = -qe;
			v[tid + i] = -qe;
			x[tid + i] = -qe;
			y[tid + i] = -qe;
			x2[tid + i] = -(qe2);
			y2[tid + i] = -(qe2);
			if (!(flag & KSW_EZ_APPROX_MAX)) H[tid + i] = KSW_NEG_INF;
		}
	}
	__syncthreads();

	int32_t last_st = -1;
	int32_t last_en = -1;
	int32_t s_local = 0;
	int32_t H0 = 0, last_H0_t = 0;

	// alignment main loop
	for (int r = 0; r < qlen + tlen - 1; ++r) {
		// if (tid == 0) should_break = 0;

		int32_t st = 0;
		int32_t en = tlen - 1;
		int32_t st0;
		int32_t en0;

		/* find the boundaries */
		if (st < r - qlen + 1) st = r - qlen + 1;
		if (en > r) en = r;
		if (st < (r - w + 1) >> 1) st = (r - w + 1) >> 1; /* take the ceil */
		if (en > (r + w) >> 1) en = (r + w) >> 1; /* take the floor */
		if (st > en) {
			ez_local.zdropped = 1;
			break;
		}

		st0 = st, en0 = en;
		st = (st >> 4) << 4, en = (((en + 16) >> 4) << 4) - 1;

		int8_t x1 = -qe, x21 = -qe2;
		int8_t v1 = -qe;

		bool over_shmem = en >= SHMEM_SIZE;

		__syncthreads();

		/* Flush shared memory to global on transition from !over_shmem to over_shmem.
		 * During over_shmem=false rows, computed values live only in shared memory.
		 * When over_shmem becomes true, the kernel loads from global memory, so we
		 * must first sync the shared memory state to global. */
		if (over_shmem && last_en >= 0 && last_en < SHMEM_SIZE) {
			for (int i = 0; i < SHMEM_SIZE; i += blockDim.x) {
				if (tid + i <= last_en) {
					u[tid + i] = u_shmem[tid + i];
					v[tid + i] = v_shmem[tid + i];
					x[tid + i] = x_shmem[tid + i];
					y[tid + i] = y_shmem[tid + i];
					x2[tid + i] = x2_shmem[tid + i];
					y2[tid + i] = y2_shmem[tid + i];
				}
			}
			__syncthreads();
		}

		/* set boundary conditions */

		if (st > 0 && st - 1 >= last_st && st - 1 <= last_en) {
			x1 = !over_shmem ? x_shmem[st - 1] : x[st - 1];
			x21 = !over_shmem ? x2_shmem[st - 1] : x2[st - 1];
			v1 = !over_shmem ? v_shmem[st - 1] : v[st - 1]; /* (r-1,s-1) calculated in the last round */
		}

		int8_t tmp_val = r == 0 ? -qe : r < long_thres ? -e
		    : r == long_thres			       ? (long_thres * (e - e2) + q - qe2)
							       : -e2;

		if (st == 0)
			v1 = tmp_val; // r == 0? -qe : r < long_thres? -e : r == long_thres? (long_thres * (e - e2) + q - qe2) : -e2;

		if (en >= r) {
			if (!over_shmem) {
				y_shmem[r] = -qe;
				y2_shmem[r] = -qe2; // ((int8_t*)y)[r] = -q - e, ((int8_t*)y2)[r] = -q2 - e2;
				u_shmem[r] = tmp_val; // r == 0? -(qe) : r < long_thres? -e : r == long_thres? (long_thres * (e - e2) + q - qe2) : -e2;
			} else {
				y[r] = -qe;
				y2[r] = -qe2; // ((int8_t*)y)[r] = -q - e, ((int8_t*)y2)[r] = -q2 - e2;
				u[r] = tmp_val; // r == 0? -(qe) : r < long_thres? -e : r == long_thres? (long_thres * (e - e2) + q - qe2) : -e2;
			}
		}

		int32_t q_start = qlen - 1 - r;

		int32_t max_H = !(flag & KSW_EZ_APPROX_MAX) ? (en0 > 0 ? H[en0 - 1] : H[en0]) : 0; /* special casing the last element */
		int32_t max_H_local = KSW_NEG_INF;
		int32_t max_t_local = -1;

		int en0_rounded = ((en0 - st0) / 16 + 1) * 16 + st0;

		int8_t q_char_local[NUM_SEGMENTS] = {4};
		int8_t t_char_local[NUM_SEGMENTS] = {4};
		int32_t H_local[NUM_SEGMENTS] = {KSW_NEG_INF};
		int8_t p_local[NUM_SEGMENTS] = {0};

		/* Initial loading with bounds checking to prevent illegal memory access */
		for (int i = 0; i < NUM_SEGMENTS; i++) {
			int id = st + tid + i * blockDim.x;
			int q_idx = q_start + id;
			/* Only load if within valid bounds; use safe defaults otherwise */
			if (id < (tlen_ << 4)) {
				t_char_local[i] = target[id];
				if (!(flag & KSW_EZ_APPROX_MAX)) H_local[i] = H[id];
			}
			if (q_idx >= 0 && q_idx < (qlen_ << 4)) {
				q_char_local[i] = query[q_idx];
			}
		}

		if (over_shmem)
			memcpy_to_shared(u, v, x, y, x2, y2, u_shmem, v_shmem, x_shmem, y_shmem, x2_shmem, y2_shmem, st);

		if (!(flag & KSW_EZ_SCORE_ONLY)) off[r] = st, off_end[r] = en;

		/* Use offset instead of 12 separate pointers to reduce register pressure */
		int curr_off = 0;
		int next_off = SEGMENT_SIZE;

		// //__syncthreads();

		if (flag & KSW_EZ_SCORE_ONLY) { // score only

			for (int t = st; t <= en; t += SEGMENT_SIZE) {

				int shift = !over_shmem ? t : 0;

				// //__syncthreads();

				if (over_shmem) { // Start issue data from global memory for the following portion, number of elements depends on the antidiagonal size
					// int no_elems2 = SEGMENT_SIZE; // t + 2 * SEGMENT_SIZE <= en ? SEGMENT_SIZE : en - (t + SEGMENT_SIZE) + 1;
					memcpy_to_shared(u, v, x, y, x2, y2, u_shmem + next_off, v_shmem + next_off, x_shmem + next_off, y_shmem + next_off, x2_shmem + next_off, y2_shmem + next_off, t + SEGMENT_SIZE);
				}

				for (int i = 0; i < SEGMENT_SIZE; i += blockDim.x) {

					int shmem_idx = tid + i + shift;
					int global_idx = tid + i + t;

					int idx = i / blockDim.x;

					int8_t q_char = q_char_local[idx];
					int8_t t_char = t_char_local[idx];
					int32_t H_val = H_local[idx];

					int8_t z = s_local;

					int cshmem_idx = curr_off + shmem_idx;
					int8_t ut = u_shmem[cshmem_idx];
					int8_t v_val = v_shmem[cshmem_idx];
					int8_t x_val = x_shmem[cshmem_idx];
					int8_t y_val = y_shmem[cshmem_idx];
					int8_t x2_val = x2_shmem[cshmem_idx];
					int8_t y2_val = y2_shmem[cshmem_idx];

					if (global_idx >= st0 && global_idx < en0_rounded) {
						if (!(flag & KSW_EZ_GENERIC_SC)) {
							z = t_char == q_char ? sc_mch : sc_mis;
							z = t_char == m1 || q_char == m1 ? sc_N : z;
							if (t == st) s_local = z;
						} else {
							z = mat[t_char * m + q_char]; // ((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
							if (t == st) s_local = z;
						}
					}

					// Step 1: last lane of each wavefront publishes its OLD value
					if (lane_id == WARP_SIZE - 1) {
						bnd_x[warp_id] = x_val;
						bnd_v[warp_id] = v_val;
						bnd_x2[warp_id] = x2_val;
					}
					__syncthreads();

					// Step 2: intra-wavefront shift
					int8_t xt1 = __shfl_up(x_val, 1);
					int8_t vt1 = __shfl_up(v_val, 1);
					int8_t x2t1 = __shfl_up(x2_val, 1);

					// Step 3: fix lane 0 of each wavefront
					if (lane_id == 0) {
						if (warp_id > 0) {
							xt1 = bnd_x[warp_id - 1];
							vt1 = bnd_v[warp_id - 1];
							x2t1 = bnd_x2[warp_id - 1];
						} else {
							xt1 = x1;
							vt1 = v1;
							x2t1 = x21;
						}
					}

					// Step 4: update carry-over for next SEGMENT chunk
					if (tid == 0) {
						int last_warp = (blockDim.x / WARP_SIZE) - 1;
						x1 = bnd_x[last_warp];
						v1 = bnd_v[last_warp];
						x21 = bnd_x2[last_warp];
					}

					int8_t a = xt1 + vt1;
					int8_t b = y_val + ut;

					int8_t a2 = x2t1 + vt1;
					int8_t b2 = y2_val + ut;

					z = z > a ? z : a;
					z = z > b ? z : b;
					z = z > a2 ? z : a2;
					z = z > b2 ? z : b2;

					z = z < sc_mch ? z : sc_mch;
					// z = min(z, sc_mch);

					v_val = z - ut;
					H_val += v_val;

					if (global_idx >= st0 && global_idx < en0) {
						int32_t tmp = H_val > max_H_local ? 1 : 0;
						max_H_local = tmp ? H_val : max_H_local;
						max_t_local = tmp ? global_idx : max_t_local;
					}
					H_local[idx] = H_val;

					a = a - (z - q);
					b = b - (z - q);
					a2 = a2 - (z - q2);
					b2 = b2 - (z - q2);

					// int8_t u_val = z - vt1;
					x_val = (a > 0 ? a : 0) - (qe);
					y_val = (b > 0 ? b : 0) - (qe);
					x2_val = (a2 > 0 ? a2 : 0) - (qe2);
					y2_val = (b2 > 0 ? b2 : 0) - (qe2);

					if (global_idx <= en) {
						u_shmem[cshmem_idx] = z - vt1;
						v_shmem[cshmem_idx] = v_val;
						x_shmem[cshmem_idx] = x_val;
						y_shmem[cshmem_idx] = y_val;
						x2_shmem[cshmem_idx] = x2_val;
						y2_shmem[cshmem_idx] = y2_val;
					}
				}

				for (int i = 0; i < NUM_SEGMENTS; i++) {
					int id = t + tid + i * blockDim.x;
					if (!(flag & KSW_EZ_APPROX_MAX) && id >= st0 && id < en0_rounded) H[id] = H_local[i];
					/* Prefetch with bounds checking to prevent illegal memory access */
					int next_id = id + SEGMENT_SIZE;
					int next_q_idx = q_start + next_id;
					t_char_local[i] = (next_id < (tlen_ << 4)) ? target[next_id] : 0;
					if (!(flag & KSW_EZ_APPROX_MAX)) H_local[i] = (next_id < (tlen_ << 4)) ? H[next_id] : 0;
					q_char_local[i] = (next_q_idx >= 0 && next_q_idx < (qlen_ << 4)) ? query[next_q_idx] : 0;
				}

				if (over_shmem) {
					int no_elems = t + SEGMENT_SIZE <= en ? SEGMENT_SIZE : en - t + 1;
					memcpy_to_global(u, v, x, y, x2, y2, u_shmem + curr_off, v_shmem + curr_off, x_shmem + curr_off, y_shmem + curr_off, x2_shmem + curr_off, y2_shmem + curr_off, t, no_elems);

					// int tmp_off = curr_off; curr_off = next_off; next_off = tmp_off;
					curr_off = curr_off + next_off; // swap(curr_off, next_off);
					next_off = curr_off - next_off;
					curr_off = curr_off - next_off; // store curr_off in tmp_off for the next iteration's swap
				}
			}

		} else if (!(flag & KSW_EZ_RIGHT)) { /* gap left-alignment */

			int64_t row_offset = (((int64_t)r * n_col_) << 4) - st; // signed 64-bit: offset may be negative and r*n_col_*16 overflows int

			for (int t = st; t <= en; t += SEGMENT_SIZE) {

				int shift = !over_shmem ? t : 0;

				if (over_shmem) { // Start issue data from global memory for the following portion, number of elements depends on the antidiagonal size
					// int no_elems2 = SEGMENT_SIZE; // t + 2 * SEGMENT_SIZE <= en ? SEGMENT_SIZE : en - (t + SEGMENT_SIZE) + 1;
					memcpy_to_shared(u, v, x, y, x2, y2, u_shmem + next_off, v_shmem + next_off, x_shmem + next_off, y_shmem + next_off, x2_shmem + next_off, y2_shmem + next_off, t + SEGMENT_SIZE);
				}

				for (int i = 0; i < SEGMENT_SIZE; i += blockDim.x) {

					int32_t shmem_idx = tid + i + shift;
					int32_t global_idx = tid + i + t;

					int idx = i / blockDim.x;

					int8_t q_char = q_char_local[idx];
					int8_t t_char = t_char_local[idx];
					int32_t H_val = H_local[idx];

					int8_t z = s_local;

					int cshmem_idx = curr_off + shmem_idx;
					int8_t ut = u_shmem[cshmem_idx];
					int8_t v_val = v_shmem[cshmem_idx];
					int8_t x_val = x_shmem[cshmem_idx];
					int8_t y_val = y_shmem[cshmem_idx];
					int8_t x2_val = x2_shmem[cshmem_idx];
					int8_t y2_val = y2_shmem[cshmem_idx];

					if (global_idx >= st0 && global_idx < en0_rounded) {
						if (!(flag & KSW_EZ_GENERIC_SC)) {
							z = t_char == q_char ? sc_mch : sc_mis;
							z = t_char == m1 || q_char == m1 ? sc_N : z;
							if (t == st) s_local = z;
						} else {
							z = mat[t_char * m + q_char]; // ((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
							if (t == st) s_local = z;
						}
					}

					// Step 1: last lane of each wavefront publishes its OLD value
					if (lane_id == WARP_SIZE - 1) {
						bnd_x[warp_id] = x_val;
						bnd_v[warp_id] = v_val;
						bnd_x2[warp_id] = x2_val;
					}
					__syncthreads();

					// Step 2: intra-wavefront shift
					int8_t xt1 = __shfl_up(x_val, 1);
					int8_t vt1 = __shfl_up(v_val, 1);
					int8_t x2t1 = __shfl_up(x2_val, 1);

					// Step 3: fix lane 0 of each wavefront
					if (lane_id == 0) {
						if (warp_id > 0) {
							xt1 = bnd_x[warp_id - 1];
							vt1 = bnd_v[warp_id - 1];
							x2t1 = bnd_x2[warp_id - 1];
						} else {
							xt1 = x1;
							vt1 = v1;
							x2t1 = x21;
						}
					}

					// Step 4: update carry-over for next SEGMENT chunk
					if (tid == 0) {
						int last_warp = (blockDim.x / WARP_SIZE) - 1;
						x1 = bnd_x[last_warp];
						v1 = bnd_v[last_warp];
						x21 = bnd_x2[last_warp];
					}

					int8_t a = xt1 + vt1;
					int8_t b = y_val + ut;

					int8_t a2 = x2t1 + vt1;
					int8_t b2 = y2_val + ut;

					int8_t d = a > z ? 1 : 0;

					z = z > a ? z : a;
					d = b > z ? 2 : d;

					z = z > b ? z : b;
					d = a2 > z ? 3 : d;

					z = z > a2 ? z : a2;
					d = b2 > z ? 4 : d;

					z = z > b2 ? z : b2;
					z = z < sc_mch ? z : sc_mch;

					a = a - z + q;
					b = b - z + q;
					a2 = a2 - z + q2;
					b2 = b2 - z + q2;

					v_val = z - ut;

					H_val += v_val;

					// if(blockIdx.x == 2 && threadIdx.x == 0 && i == 0) printf("%d\n", z);

					if (global_idx >= st0 && global_idx < en0) {
						int32_t tmp = H_val > max_H_local ? 1 : 0;
						max_H_local = tmp ? H_val : max_H_local;
						max_t_local = tmp ? global_idx : max_t_local;
					}
					H_local[idx] = H_val;

					// OR the bits together for traceback (was incorrectly overwriting d)
					d |= (a > 0) ? 0x08 : 0;
					d |= (b > 0) ? 0x10 : 0;
					d |= (a2 > 0) ? 0x20 : 0;
					d |= (b2 > 0) ? 0x40 : 0;

					ut = z - vt1;
					x_val = (a > 0 ? a : 0) - (qe);
					y_val = (b > 0 ? b : 0) - (qe);
					x2_val = (a2 > 0 ? a2 : 0) - (qe2);
					y2_val = (b2 > 0 ? b2 : 0) - (qe2);

					p_local[idx] = (global_idx <= en) ? d : 0;


					if (global_idx <= en) {
						u_shmem[cshmem_idx] = ut;
						v_shmem[cshmem_idx] = v_val;
						x_shmem[cshmem_idx] = x_val;
						y_shmem[cshmem_idx] = y_val;
						x2_shmem[cshmem_idx] = x2_val;
						y2_shmem[cshmem_idx] = y2_val;

						// p_local[idx] = d; // p[row_offset + global_idx] = d;
					}
				}

				for (int i = 0; i < NUM_SEGMENTS; i++) {
					int id = t + tid + i * blockDim.x;
					if (!(flag & KSW_EZ_APPROX_MAX) && id >= st0 && id < en0_rounded) H[id] = H_local[i];
					if (id <= en) p[row_offset + id] = p_local[i];
					/* Prefetch with bounds checking to prevent illegal memory access */
					int next_id = id + SEGMENT_SIZE;
					int next_q_idx = q_start + next_id;
					t_char_local[i] = (next_id < (tlen_ << 4)) ? target[next_id] : 0;
					if (!(flag & KSW_EZ_APPROX_MAX)) H_local[i] = (next_id < (tlen_ << 4)) ? H[next_id] : 0;
					q_char_local[i] = (next_q_idx >= 0 && next_q_idx < (qlen_ << 4)) ? query[next_q_idx] : 0;
				}

				if (over_shmem) {
					int no_elems = t + SEGMENT_SIZE <= en ? SEGMENT_SIZE : en - t + 1;
					memcpy_to_global(u, v, x, y, x2, y2, u_shmem + curr_off, v_shmem + curr_off, x_shmem + curr_off, y_shmem + curr_off, x2_shmem + curr_off, y2_shmem + curr_off, t, no_elems);

					// int tmp_off = curr_off; curr_off = next_off; next_off = tmp_off;
					curr_off = curr_off + next_off; // swap(curr_off, next_off);
					next_off = curr_off - next_off;
					curr_off = curr_off - next_off;
				}
			}

		} else { /* gap right-alignment */

			int64_t row_offset = (((int64_t)r * n_col_) << 4) - st; // signed 64-bit: offset may be negative and r*n_col_*16 overflows int

			for (int t = st; t <= en; t += SEGMENT_SIZE) {

				int shift = !over_shmem ? t : 0;

				if (over_shmem) { // Start issue data from global memory for the following portion, number of elements depends on the antidiagonal size
					// int no_elems2 = SEGMENT_SIZE; //  t + 2 * SEGMENT_SIZE <= en ? SEGMENT_SIZE : en - (t + SEGMENT_SIZE) + 1;
					memcpy_to_shared(u, v, x, y, x2, y2, u_shmem + next_off, v_shmem + next_off, x_shmem + next_off, y_shmem + next_off, x2_shmem + next_off, y2_shmem + next_off, t + SEGMENT_SIZE);
				}

				for (int i = 0; i < SEGMENT_SIZE; i += blockDim.x) {

					int shmem_idx = tid + shift + i;
					int global_idx = tid + t + i;

					int idx = i / blockDim.x;

					int8_t q_char = q_char_local[idx];
					int8_t t_char = t_char_local[idx];
					int32_t H_val = H_local[idx];

					int8_t z = s_local;

					int cshmem_idx = curr_off + shmem_idx;
					int8_t x_val = x_shmem[cshmem_idx];
					int8_t x2_val = x2_shmem[cshmem_idx];
					int8_t v_val = v_shmem[cshmem_idx];
					int8_t y_val = y_shmem[cshmem_idx];
					int8_t y2_val = y2_shmem[cshmem_idx];
					int8_t ut = u_shmem[cshmem_idx];

					if (global_idx >= st0 && global_idx < en0_rounded) {
						if (!(flag & KSW_EZ_GENERIC_SC)) {
							z = t_char == q_char ? sc_mch : sc_mis;
							z = t_char == m1 || q_char == m1 ? sc_N : z;
							if (t == st) s_local = z;
						} else {
							z = mat[t_char * m + q_char]; // ((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
							if (t == st) s_local = z;
						}
					}

					// Step 1: last lane of each wavefront publishes its OLD value
					if (lane_id == WARP_SIZE - 1) {
						bnd_x[warp_id] = x_val;
						bnd_v[warp_id] = v_val;
						bnd_x2[warp_id] = x2_val;
					}
					__syncthreads();

					// Step 2: intra-wavefront shift
					int8_t xt1 = __shfl_up(x_val, 1);
					int8_t vt1 = __shfl_up(v_val, 1);
					int8_t x2t1 = __shfl_up(x2_val, 1);

					// Step 3: fix lane 0 of each wavefront
					if (lane_id == 0) {
						if (warp_id > 0) {
							xt1 = bnd_x[warp_id - 1];
							vt1 = bnd_v[warp_id - 1];
							x2t1 = bnd_x2[warp_id - 1];
						} else {
							xt1 = x1;
							vt1 = v1;
							x2t1 = x21;
						}
					}

					// Step 4: update carry-over for next SEGMENT chunk
					if (tid == 0) {
						int last_warp = (blockDim.x / WARP_SIZE) - 1;
						x1 = bnd_x[last_warp];
						v1 = bnd_v[last_warp];
						x21 = bnd_x2[last_warp];
					}

					int8_t a = xt1 + vt1;
					int8_t b = y_val + ut;

					int8_t a2 = x2t1 + vt1;
					int8_t b2 = y2_val + ut;

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

					a = a - (z - q);
					b = b - (z - q);
					a2 = a2 - (z - q2);
					b2 = b2 - (z - q2);

					// OR the bits together for traceback (gap-right: >= 0)
					d |= (a >= 0) ? 0x08 : 0;
					d |= (b >= 0) ? 0x10 : 0;
					d |= (a2 >= 0) ? 0x20 : 0;
					d |= (b2 >= 0) ? 0x40 : 0;

					H_val += z - ut;
					H_local[idx] = H_val;
					if (global_idx >= st0 && global_idx < en0) {
						int32_t tmp = H_val > max_H_local ? 1 : 0;
						max_H_local = tmp ? H_val : max_H_local;
						max_t_local = tmp ? global_idx : max_t_local;
					}

					p_local[idx] = (global_idx <= en) ? d : 0;

					if (global_idx <= en) {
						u_shmem[cshmem_idx] = z - vt1;
						v_shmem[cshmem_idx] = z - ut;
						x_shmem[cshmem_idx] = (a >= 0 ? a : 0) - (qe);
						y_shmem[cshmem_idx] = (b >= 0 ? b : 0) - (qe);
						x2_shmem[cshmem_idx] = (a2 >= 0 ? a2 : 0) - (qe2);
						y2_shmem[cshmem_idx] = (b2 >= 0 ? b2 : 0) - (qe2);
						// p_local[idx] = d;
					}
				}

				for (int i = 0; i < NUM_SEGMENTS; i++) {
					int id = t + tid + i * blockDim.x;
					if (!(flag & KSW_EZ_APPROX_MAX) && id >= st0 && id < en0_rounded) H[id] = H_local[i];
					if (id <= en) p[row_offset + id] = p_local[i];
					/* Prefetch with bounds checking to prevent illegal memory access */
					int next_id = id + SEGMENT_SIZE;
					int next_q_idx = q_start + next_id;
					t_char_local[i] = (next_id < (tlen_ << 4)) ? target[next_id] : 0;
					if (!(flag & KSW_EZ_APPROX_MAX)) H_local[i] = (next_id < (tlen_ << 4)) ? H[next_id] : 0;
					q_char_local[i] = (next_q_idx >= 0 && next_q_idx < (qlen_ << 4)) ? query[next_q_idx] : 0;
				}

				if (over_shmem) {
					int no_elems = t + SEGMENT_SIZE <= en ? SEGMENT_SIZE : en - t + 1;
					memcpy_to_global(u, v, x, y, x2, y2, u_shmem + curr_off, v_shmem + curr_off, x_shmem + curr_off, y_shmem + curr_off, x2_shmem + curr_off, y2_shmem + curr_off, t, no_elems);

					// int tmp_off = curr_off; curr_off = next_off; next_off = tmp_off;
					curr_off = curr_off + next_off; // swap(curr_off, next_off);
					next_off = curr_off - next_off;
					curr_off = curr_off - next_off;
				}
			}
		}

		__syncthreads();

		// Debug output removed - was causing CIGAR mismatch
		// if(blockIdx.x == 2) printf("%d ", max_H_local);
		// 	////__syncthreads();
		// if(blockIdx.x == 2 && threadIdx.x == 0) printf("\n");

		if (!(flag & KSW_EZ_APPROX_MAX)) { /* find the exact max with a 32-bit score array */

			int32_t max_t = en0;
			/* compute H[], max_H and max_t */
			if (r > 0) {

				int8_t u_en0 = over_shmem ? u[en0] : u_shmem[en0];
				int8_t v_en0 = over_shmem ? v[en0] : v_shmem[en0];

				max_H = en0 > 0 ? max_H + u_en0 : max_H + v_en0; /* special casing the last element */
				H[en0] = max_H;

				// Intra-wavefront reduction via shuffle
				for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
					int tmp_H = __shfl_down(max_H_local, offset);
					int tmp_t = __shfl_down(max_t_local, offset);
					if (tmp_H > max_H_local || (tmp_H == max_H_local && tmp_t < max_t_local))
						max_H_local = tmp_H, max_t_local = tmp_t;
				}

				// Cross-wavefront reduction via shared memory
				if (lane_id == 0) {
					reduce_H[warp_id] = max_H_local;
					reduce_t[warp_id] = max_t_local;
				}
				__syncthreads();

				// Final reduction across warps (all threads read same result)
				{
					int nwarps = blockDim.x / WARP_SIZE;
					int32_t best_H = reduce_H[0];
					int32_t best_t = reduce_t[0];
					for (int w = 1; w < nwarps; w++) {
						if (reduce_H[w] > best_H || (reduce_H[w] == best_H && reduce_t[w] < best_t)) {
							best_H = reduce_H[w];
							best_t = reduce_t[w];
						}
					}
					if (best_H > max_H) max_H = best_H, max_t = best_t;
				}

				/*for (t = en1; t < en0; ++t) {  // for the rest of values that haven't been computed with SSE 
					H[t] += (int32_t)v[t];
					if (H[t] > max_H)
						max_H = H[t], max_t = t;
				}*/

			} else
				H[0] = v_shmem[0] - (qe), max_H = H[0], max_t = 0; /* special casing r==0 */

			/* update ez from a single thread to avoid shared-memory races */
			if (tid == 0) {
				if (en0 == tlen - 1 && H[en0] > ez_local.mte)
					ez_local.mte = H[en0], ez_local.mte_q = r - en0;
				if (r - st0 == qlen - 1 && H[st0] > ez_local.mqe)
					ez_local.mqe = H[st0], ez_local.mqe_t = st0;
				should_break = ksw_apply_zdrop_gpu(&ez_local, max_H, r, max_t, zdrop, e2);
				if (r == qlen + tlen - 2 && en0 == tlen - 1)
					ez_local.score = H[tlen - 1];
			}
			__syncthreads();
			if (should_break) break; // if score drops too fast, interrupt alignment


		} else { /* find approximate max; Z-drop might be inaccurate, too. */

			if (r > 0) {

				int8_t* u_ptr = over_shmem ? u : u_shmem;
				int8_t* v_ptr = over_shmem ? v : v_shmem;

				if (last_H0_t >= st0 && last_H0_t <= en0 && last_H0_t + 1 >= st0 && last_H0_t + 1 <= en0) {
					int32_t d0 = v_ptr[last_H0_t];
					int32_t d1 = u_ptr[last_H0_t + 1];
					if (d0 > d1)
						H0 += d0;
					else
						H0 += d1, ++last_H0_t;
				} else if (last_H0_t >= st0 && last_H0_t <= en0) {
					H0 += v_ptr[last_H0_t];
				} else {
					++last_H0_t, H0 += u_ptr[last_H0_t];
				}
			} else
				H0 = v_shmem[0] - (qe), last_H0_t = 0;

			if (tid == 0) {
				if (flag & KSW_EZ_APPROX_DROP)
					should_break = ksw_apply_zdrop_gpu(&ez_local, H0, r, last_H0_t, zdrop, e2);
				if (r == qlen + tlen - 2 && en0 == tlen - 1)
					ez_local.score = H0;
			}
			__syncthreads();
			if (should_break) break;
		}

		last_st = st, last_en = en;

		if (en == SHMEM_SIZE - 1) {
			memcpy_to_global(u, v, x, y, x2, y2, u_shmem, v_shmem, x_shmem, y_shmem, x2_shmem, y2_shmem, 0, SHMEM_SIZE);
		}

		//for (t = st0; t <= en0; ++t) printf("(%d,%d)\t(%d,%d,%d,%d)\t%d\n", r, t, ((int8_t*)u)[t], ((int8_t*)v)[t], ((int8_t*)x)[t], ((int8_t*)y)[t], H[t]); // for debugging
	}


	__syncthreads();
	if (tid == 0) {
		// if (qlen == 105 && tlen == 112) {
		// 	printf("[KERNEL] bid=%d qlen=%d tlen=%d score=%d zdrop=%d max=%d max_t=%d max_q=%d mqe=%d mqe_t=%d mte=%d mte_q=%d\n",
		// 		bid, qlen, tlen, ez_local.score, ez_local.zdropped, (int)ez_local.max,
		// 		ez_local.max_t, ez_local.max_q, ez_local.mqe, ez_local.mqe_t, ez_local.mte, ez_local.mte_q);
		// }
		ez_d[bid] = ez_local;
	}
}
