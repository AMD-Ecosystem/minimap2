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

#define NUM_SEGMENTS 4 // num segments
#define NUM_ELEMENTS 64 // elements processed simultaneously
#define SEGMENT_SIZE 256 // half-segment number of elements to prefetch
#define SHMEM_SIZE 512 // size of shared memory buffer

#define HALF_SEGMENT_SIZE 64 // half-segment number of elements to prefetch

__device__ size_t get_clock(void) __attribute__((noinline));
__device__ size_t get_clock(void)
{
	return clock64();
}

__device__ __forceinline__ short2 compute_max_asm(short2 a, short2 b)
{
	short2 res;
	asm volatile(
	    "v_pk_max_i16 %0, %1, %2\n"
	    : "=v"(res)
	    : "v"(a), "v"(b));
	return res;
}

__device__ __forceinline__ short2 compute_min_asm(short2 a, short2 b)
{
	short2 res;
	asm volatile(
	    "v_pk_min_i16 %0, %1, %2\n"
	    : "=v"(res)
	    : "v"(a), "v"(b));
	return res;
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

	for (int i = 0; i < HALF_SEGMENT_SIZE; i += BLOCK_DIM)
		if (tid + i < HALF_SEGMENT_SIZE) {
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

	int iters = no_elements / 4;

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

__launch_bounds__(64, 2)
    __global__ __attribute__((amdgpu_waves_per_eu(5, 8))) void ksw_extd2_gpu_packed( // void *km,
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
	const int8_t q, const int8_t e, const int8_t q2, const int8_t e2, const int w_d, const int zdrop, const int end_bonus, const int flag,
	ksw_extz_t* __restrict__ const ez_d,
	const uint32_t* __restrict__ const tlen_ps_d,
	const uint32_t* __restrict__ const qlen_ps_d,
	const uint64_t* __restrict__ const p_ps_d,
	const uint32_t* __restrict__ const off_ps_d,
	const int8_t min_sc)
{

	const int32_t bid = blockIdx.x;
	const int32_t tid = threadIdx.x;

	const bool generic_sc = (flag & KSW_EZ_GENERIC_SC);

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
	memcpy(mat, mat_d, 25);

	__shared__ int8_t u_shmem[SHMEM_SIZE];
	__shared__ int8_t v_shmem[SHMEM_SIZE];
	__shared__ int8_t x_shmem[SHMEM_SIZE];
	__shared__ int8_t y_shmem[SHMEM_SIZE];
	__shared__ int8_t x2_shmem[SHMEM_SIZE];
	__shared__ int8_t y2_shmem[SHMEM_SIZE];

	__shared__ ksw_extz_t ez_local;
	if (tid == 0) {
		ez_local.max_q = ez_local.max_t = ez_local.mqe_t = ez_local.mte_q = -1;
		ez_local.max = 0, ez_local.score = ez_local.mqe = ez_local.mte = KSW_NEG_INF;
		ez_local.n_cigar = 0, ez_local.zdropped = 0, ez_local.reach_end = 0;
	}
	__syncthreads();

	if (qlen <= 0 || tlen <= 0) return; // end due to invalid alignment

	const int8_t qe = q + e;
	const int8_t qe2 = q2 + e2;
	const int8_t sc_mch = mat[0];
	const int8_t sc_mis = mat[1];
	const int8_t sc_N = mat[24] == 0 ? -e2 : mat[24];

	short2 sc_mch_v = make_short2(sc_mch, sc_mch);
	short2 sc_mis_v = make_short2(sc_mis, sc_mis);

	short2 zero = make_short2(0, 0);
	const short2 one = make_short2(1, 1);
	const short2 two = make_short2(2, 2);
	const short2 three = make_short2(3, 3);
	const short2 four = make_short2(4, 4);

	const short2 qe_v = make_short2(qe, qe);
	const short2 qe2_v = make_short2(qe2, qe2);
	const short2 sc_N_v = make_short2(sc_N, sc_N);
	const short2 q_v = make_short2(q, q);
	const short2 q2_v = make_short2(q2, q2);
	const short2 m1_v = make_short2(m1, m1);


	short2 clamp_08_v = make_short2(0x08, 0x08);
	short2 clamp_10_v = make_short2(0x10, 0x10);
	short2 clamp_20_v = make_short2(0x20, 0x20);
	short2 clamp_40_v = make_short2(0x40, 0x40);


	if (-min_sc > 2 * (qe)) return; /* otherwise, we won't see any mismatches */

	int w = w_d;
	if (w < 0) w = tlen > qlen ? tlen : qlen; // set bandwidth in case it is negative (non-banded alignment)
	// const int tlen_ = (tlen + 15)>>4; // / 16;										// round to multiple of 16
	// const int qlen_ = (qlen + 15)>>4; //  / 16;                                     // round to multiple of 16
	int n_col_ = qlen < tlen ? qlen : tlen;
	n_col_ = (((n_col_ < w + 1 ? n_col_ : w + 1) + 15) >> 4) + 1; // round to multiple of 16

	int long_thres = e != e2 ? (q2 - q) / (e - e2) - 1 : 0; // if double affine, set the threshold for long gaps(?) to ((q2 - q) / (e - e2) - 1), otherwise 0
	if ((qe2) + long_thres * e2 > (qe) + long_thres * e)
		++long_thres;

	for (int i = 0; i < SHMEM_SIZE; i += BLOCK_DIM) {
		if (tid + i < SHMEM_SIZE) {
			u_shmem[tid + i] = -qe;
			v_shmem[tid + i] = -qe;
			x_shmem[tid + i] = -qe;
			y_shmem[tid + i] = -qe;
			x2_shmem[tid + i] = -(qe2);
			y2_shmem[tid + i] = -(qe2);
		}
	}

	int32_t last_st = -1;
	int32_t last_en = -1;
	int32_t s_local = 0;

	short2 s_local_v = make_short2(s_local, s_local);

	int32_t H0 = 0, last_H0_t = 0;

	// alignment main loop
	for (int r = 0; r < qlen + tlen - 1; ++r) {

		int32_t st = 0;
		int32_t en = tlen - 1;
		int32_t st0;
		int32_t en0;

		/* find the boundaries */
		const int rql1 = r - qlen + 1;
		const int rw1 = (r - w + 1) >> 1;
		const int rw2 = (r + w) >> 1;
		st = st > (rql1 > rw1 ? rql1 : rw1) ? st : (rql1 > rw1 ? rql1 : rw1);
		en = en < (r < rw2 ? r : rw2) ? en : (r < rw2 ? r : rw2);

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

		const int32_t q_start = qlen - 1 - r;
		int8_t q_char_local[NUM_SEGMENTS] = {4};
		int8_t t_char_local[NUM_SEGMENTS] = {4};
		int32_t H_local[NUM_SEGMENTS] = {KSW_NEG_INF};
		int8_t p_local[NUM_SEGMENTS] = {0};

		for (int i = 0; i < NUM_SEGMENTS; i += 2) {
			int id = st + tid * 2 + i * blockDim.x;
			if (id >= st0) {
				*(int16_t*)(q_char_local + i) = *(int16_t*)(query + q_start + id);
				*(int16_t*)(t_char_local + i) = *(int16_t*)(target + id);
				*(int64_t*)(H_local + i) = *(int64_t*)(H + id);
			} else if (id + 1 >= st0) {
				t_char_local[i + 1] = target[id + 1];
				q_char_local[i + 1] = query[q_start + id + 1];
				H_local[i + 1] = H[id + 1];
			}
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

		if (over_shmem)
			memcpy_to_shared(u, v, x, y, x2, y2, u_shmem, v_shmem, x_shmem, y_shmem, x2_shmem, y2_shmem, st);

		int32_t max_H = en0 > 0 ? H[en0 - 1] : H[en0]; /* special casing the last element */
		int32_t max_H_local = KSW_NEG_INF;
		int32_t max_t_local = -1;

		const int en0_rounded = ((en0 - st0) / 16 + 1) * 16 + st0;

		int8_t* curr_u = u_shmem;
		int8_t* curr_v = v_shmem;
		int8_t* curr_x = x_shmem;
		int8_t* curr_y = y_shmem;
		int8_t* curr_x2 = x2_shmem;
		int8_t* curr_y2 = y2_shmem;

		int8_t* next_u = u_shmem + SEGMENT_SIZE;
		int8_t* next_v = v_shmem + SEGMENT_SIZE;
		int8_t* next_x = x_shmem + SEGMENT_SIZE;
		int8_t* next_y = y_shmem + SEGMENT_SIZE;
		int8_t* next_x2 = x2_shmem + SEGMENT_SIZE;
		int8_t* next_y2 = y2_shmem + SEGMENT_SIZE;


		__syncthreads();


		//  		if (flag&KSW_EZ_SCORE_ONLY) { // score only
		//
		//  			for (int t = st; t <= en; t += SEGMENT_SIZE) {
		//
		//  				int shift = !over_shmem ? t : 0;
		//
		//  				// __syncthreads();
		//
		//  				if(over_shmem) { // Start issue data from global memory for the following portion, number of elements depends on the antidiagonal size
		//  					// int no_elems2 = SEGMENT_SIZE; // t + 2 * SEGMENT_SIZE <= en ? SEGMENT_SIZE : en - (t + SEGMENT_SIZE) + 1;
		//  					memcpy_to_shared(u, v, x, y, x2, y2, next_u, next_v, next_x, next_y, next_x2, next_y2, t + SEGMENT_SIZE);
		//  				}
		//
		//  				for (int i = 0; i < SEGMENT_SIZE; i += BLOCK_DIM){
		//
		//  					int shmem_idx = tid + i + shift;
		//  					int global_idx = tid + i + t;
		//
		//  					int idx = i / BLOCK_DIM;
		//
		//  					int8_t	q_char = q_char_local[idx];
		//  					int8_t	t_char = t_char_local[idx];
		//  					int32_t	H_val = H_local[idx];
		//
		//  					t_char_local[idx] = target[global_idx + SEGMENT_SIZE];
		//  					q_char_local[idx] = query[q_start + global_idx + SEGMENT_SIZE];
		//
		//  					int8_t z = s_local;
		//
		//  					int8_t ut = curr_u[shmem_idx]; 			//  u_shmem[shmem_idx];
		//  					int8_t v_val = curr_v[shmem_idx]; 		//  v_shmem[shmem_idx];
		//  					int8_t x_val = curr_x[shmem_idx]; 		//  x_shmem[shmem_idx];
		//  					int8_t y_val = curr_y[shmem_idx]; 		//  y_shmem[shmem_idx];
		//  					int8_t x2_val = curr_x2[shmem_idx]; 	//  x2_shmem[shmem_idx];
		//  					int8_t y2_val = curr_y2[shmem_idx]; 	//  y2_shmem[shmem_idx];
		//
		//  					if (global_idx >= st0 && global_idx < en0_rounded){
		//  						if (!(flag & KSW_EZ_GENERIC_SC)) {
		//  							z = t_char == q_char ? sc_mch : sc_mis;
		//  							z = t_char == m1 || q_char == m1 ? sc_N : z;
		//  							if(t == st) s_local = z;
		//  						} else {
		//  							z = mat[t_char * m + q_char]; // ((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
		//  							if(t == st) s_local = z;
		//  						}
		//  					}
		//
		//  					int8_t xt1 = __shfl_up(x_val, 1);
		//  					int8_t vt1 = __shfl_up(v_val , 1);
		//  					int8_t x2t1 = __shfl_up(x2_val, 1);
		//
		//  					if(tid == 0){
		//  						xt1 = x1;
		//  						x2t1 = x21;
		//  						vt1 = v1;
		//  					}
		//
		//  					x1 = __shfl(x_val, blockDim.x - 1);
		//  					x21 = __shfl(x2_val, blockDim.x - 1);
		//  					v1 = __shfl(v_val, blockDim.x - 1);
		//
		//  					int8_t a = xt1 + vt1;
		//  					int8_t b = y_val + ut;
		//
		//  					int8_t a2 = x2t1 + vt1;
		//  					int8_t b2 = y2_val + ut;
		//
		//  					z = z > a ? z : a;
		//  					z = z > b ? z : b;
		//  					z = z > a2 ? z : a2;
		//  					z = z > b2 ? z : b2;
		//
		//  					z = z < sc_mch ? z : sc_mch;
		//
		//  					v_val = z - ut;
		//  					H_val += v_val;
		//
		//  					int32_t tmp = H_val > max_H_local ? 1 : 0;
		//  					max_H_local = tmp ? H_val : max_H_local;
		//  					max_t_local = tmp ? global_idx : max_t_local;
		//  					H_local[idx] = H_val;
		//
		//  					a = a - (z - q);
		//  					b = b - (z - q);
		//  					a2 = a2 - (z - q2);
		//  					b2 = b2 - (z - q2);
		//
		//  					int8_t u_val = z - vt1;
		//  					x_val = (a > 0 ? a : 0) - (qe);
		//  					y_val = (b > 0 ? b : 0) - (qe);
		//  					x2_val = (a2 > 0 ? a2 : 0) - (qe2);
		//  					y2_val = (b2 > 0 ? b2 : 0) - (qe2);
		//
		//  					if (global_idx <= en){
		//  						curr_u[shmem_idx] = u_val;
		//  						curr_v[shmem_idx] = v_val;
		//  						curr_x[shmem_idx] = x_val;
		//  						curr_y[shmem_idx] = y_val;
		//  						curr_x2[shmem_idx] = x2_val;
		//  						curr_y2[shmem_idx] = y2_val;
		//  					}
		//  				}
		//
		//  				for(int i = 0; i < NUM_SEGMENTS; i++){
		//  					int id = t + tid + i*blockDim.x;
		//  					if(id >= st0){
		//  						H[id] = H_local[i];
		//  						// t_char_local[i] = target[id + SEGMENT_SIZE];
		//  						// q_char_local[i] = query[q_start + id + SEGMENT_SIZE];
		//  					}
		//  					H_local[i] = H[id + SEGMENT_SIZE];
		//  				}
		//
		//  				if(over_shmem) {
		//  					int no_elems = t + SEGMENT_SIZE <= en ? SEGMENT_SIZE : en - t + 1;
		//  					__syncthreads();
		//  					memcpy_to_global(u, v, x, y, x2, y2, curr_u, curr_v, curr_x, curr_y, curr_x2, curr_y2, t, no_elems);
		//
		//  					swap_pointers(curr_u, next_u);
		//  					swap_pointers(curr_v, next_v);
		//  					swap_pointers(curr_x, next_x);
		//  					swap_pointers(curr_y, next_y);
		//  					swap_pointers(curr_x2, next_x2);
		//  					swap_pointers(curr_y2, next_y2);
		//  				}
		//
		//  			}
		//
		//  		} else if (!(flag&KSW_EZ_RIGHT)) { /* gap left-alignment */

		const int row_offset = ((r * n_col_) >> 4) - st;
		const uint8_t* query_ptr = query + q_start;

		for (int t = st; t <= en; t += SEGMENT_SIZE) {

			int shift = !over_shmem ? t : 0;

			if (over_shmem) { // Start issue data from global memory for the following portion, number of elements depends on the antidiagonal size
				memcpy_to_shared(u, v, x, y, x2, y2, next_u, next_v, next_x, next_y, next_x2, next_y2, t + SEGMENT_SIZE);
			}

			int idx = 0;

			for (int i = 0; i < SEGMENT_SIZE; i += BLOCK_DIM * 2) {

				int32_t shmem_idx = tid * 2 + i + shift;
				int32_t global_idx = tid * 2 + i + t;

				const bool in_range_x = global_idx >= st0 && global_idx < en0_rounded;
				const bool in_range_y = (global_idx + 1) >= st0 && (global_idx + 1) < en0_rounded;
				const short2 in_range = make_short2(in_range_x, in_range_y);
				const short2 not_in_range = make_short2(!in_range_x, !in_range_y);

				short2 ut, v_val, x_val, y_val, x2_val, y2_val;

				ut.x = curr_u[shmem_idx];
				v_val.x = curr_v[shmem_idx];
				x_val.x = curr_x[shmem_idx];
				y_val.x = curr_y[shmem_idx];
				x2_val.x = curr_x2[shmem_idx];
				y2_val.x = curr_y2[shmem_idx];

				ut.y = curr_u[shmem_idx + 1];
				v_val.y = curr_v[shmem_idx + 1];
				x_val.y = curr_x[shmem_idx + 1];
				y_val.y = curr_y[shmem_idx + 1];
				x2_val.y = curr_x2[shmem_idx + 1];
				y2_val.y = curr_y2[shmem_idx + 1];

				char2 q_char, t_char;

				q_char.x = q_char_local[idx];
				q_char.y = q_char_local[idx + 1];

				t_char.x = t_char_local[idx];
				t_char.y = t_char_local[idx + 1];

				// preload next segment
				*(int16_t*)(t_char_local + idx) = *(int16_t*)(target + global_idx + SEGMENT_SIZE);
				*(int16_t*)(q_char_local + idx) = *(int16_t*)(query_ptr + global_idx + SEGMENT_SIZE);
				//t_char_local[idx] = target[global_idx + SEGMENT_SIZE];
				//t_char_local[idx + 1] = target[global_idx + SEGMENT_SIZE + 1];
				//q_char_local[idx] = query[q_start + global_idx + SEGMENT_SIZE];
				//q_char_local[idx + 1] = query[q_start + global_idx + SEGMENT_SIZE + 1];
				// -------

				short2 z = s_local_v;

				short2 z1;
				z1.x = (t_char.x == q_char.x) ? sc_mch_v.x : sc_mis_v.x;
				z1.y = (t_char.y == q_char.y) ? sc_mch_v.y : sc_mis_v.y;

				short2 ambiguous;

				ambiguous.x = (t_char.x == m1) || (q_char.x == m1);
				ambiguous.y = (t_char.y == m1) || (q_char.y == m1);

				z1.x = ambiguous.x ? sc_N_v.x : z1.x;
				z1.y = ambiguous.y ? sc_N_v.y : z1.y;

				short2 z2;
				z2.x = mat[t_char.x * m + q_char.x];
				z2.y = mat[t_char.y * m + q_char.y];


				if (!generic_sc) {
					z.x = (in_range_x) ? z1.x : z.x;
					z.y = (in_range_y) ? z1.y : z.y;
				} else {
					z.x = (in_range_x) ? z2.x : z.x;
					z.y = (in_range_y) ? z2.y : z.y;
				}

				if (t == st && in_range_x) s_local_v.x = z.x;
				if (t == st && in_range_y) s_local_v.y = z.y;

				short2 xt1, vt1, x2t1;

				// shuffle neighboring data
				xt1.x = __shfl_up(x_val.y, 1);
				xt1.y = x_val.x;
				vt1.x = __shfl_up(v_val.y, 1);
				vt1.y = v_val.x;
				x2t1.x = __shfl_up(x2_val.y, 1);
				x2t1.y = x2_val.x;

				xt1.x = (tid == 0) ? x1 : xt1.x;
				x2t1.x = (tid == 0) ? x21 : x2t1.x;
				vt1.x = (tid == 0) ? v1 : vt1.x;

				x1 = __shfl(x_val.y, blockDim.x - 1);
				x21 = __shfl(x2_val.y, blockDim.x - 1);
				v1 = __shfl(v_val.y, blockDim.x - 1);

				short2 a = xt1 + vt1;
				short2 b = y_val + ut;
				short2 a2 = x2t1 + vt1;
				short2 b2 = y2_val + ut;

				short2 d;

				d.x = a.x > z.x ? one.x : zero.x;
				d.y = a.y > z.y ? one.y : zero.y;

				asm volatile("v_pk_max_i16 %0, %1, %2\n" : "=v"(z) : "v"(a), "v"(z));
				// z = compute_max_asm(z, a);
				// z.x = z.x > a.x ? z.x : a.x;
				// z.y = z.y > a.y ? z.y : a.y;

				d.x = b.x > z.x ? two.x : d.x;
				d.y = b.y > z.y ? two.y : d.y;

				asm volatile("v_pk_max_i16 %0, %1, %2\n" : "=v"(z) : "v"(b), "v"(z));
				// z = compute_max_asm(z, b);
				// z.x = z.x > b.x ? z.x : b.x;
				// z.y = z.y > b.y ? z.y : b.y;

				d.x = a2.x > z.x ? three.x : d.x;
				d.y = a2.y > z.y ? three.y : d.y;

				asm volatile("v_pk_max_i16 %0, %1, %2\n" : "=v"(z) : "v"(a2), "v"(z));
				// z = compute_max_asm(z, a2);
				// z.x = z.x > a2.x ? z.x : a2.x;
				// z.y = z.y > a2.y ? z.y : a2.y;

				d.x = b2.x > z.x ? four.x : d.x;
				d.y = b2.y > z.y ? four.y : d.y;

				asm volatile("v_pk_max_i16 %0, %1, %2\n" : "=v"(z) : "v"(b2), "v"(z));
				// z = compute_max_asm(z, b2);
				// z.x = z.x > b2.x ? z.x : b2.x;
				// z.y = z.y > b2.y ? z.y : b2.y;

				asm volatile("v_pk_min_i16 %0, %1, %2\n" : "=v"(z) : "v"(sc_mch_v), "v"(z));
				// z.x = z.x < sc_mch_v.x ? z.x : sc_mch_v.x;
				// z.y = z.y < sc_mch_v.y ? z.y : sc_mch_v.y;

				a = a - z + q_v;
				b = b - z + q_v;
				a2 = a2 - z + q2_v;
				b2 = b2 - z + q2_v;

				d.x = a.x > zero.x ? clamp_08_v.x : zero.x;
				d.y = a.y > zero.y ? clamp_08_v.y : zero.y;

				d.x = b.x > zero.x ? clamp_10_v.x : zero.x;
				d.y = b.y > zero.y ? clamp_10_v.y : zero.y;

				d.x = a2.x > zero.x ? clamp_20_v.x : zero.x;
				d.y = a2.y > zero.y ? clamp_20_v.y : zero.y;

				d.x = b2.x > zero.x ? clamp_40_v.x : zero.x;
				d.y = b2.y > zero.y ? clamp_40_v.y : zero.y;

				asm volatile("v_pk_max_i16 %0, %1, %2\n" : "=v"(x_val) : "v"(a), "v"(zero));
				// x_val = compute_max_asm(a, zero);
				// x_val.x = a.x > zero.x ? a.x : zero.x;
				// x_val.y = a.y > zero.y ? a.y : zero.y;

				asm volatile("v_pk_max_i16 %0, %1, %2\n" : "=v"(y_val) : "v"(b), "v"(zero));
				// y_val = compute_max_asm(b, zero);
				// y_val.x = b.x > zero.x ? b.x : zero.x;
				// y_val.y = b.y > zero.y ? b.y : zero.y;

				asm volatile("v_pk_max_i16 %0, %1, %2\n" : "=v"(x2_val) : "v"(a2), "v"(zero));
				// x2_val = compute_max_asm(a2, zero);
				// x2_val.x = a2.x > zero.x ? a2.x : zero.x;
				// x2_val.y = a2.y > zero.y ? a2.y : zero.y;

				asm volatile("v_pk_max_i16 %0, %1, %2\n" : "=v"(y2_val) : "v"(b2), "v"(zero));
				// y2_val = compute_max_asm(b2, zero);
				// y2_val.x = b2.x > zero.x ? b2.x : zero.x;
				// y2_val.y = b2.y > zero.y ? b2.y : zero.y;

				v_val = z - ut;
				ut = z - vt1;
				x_val -= qe_v;
				y_val -= qe_v;
				x2_val -= qe2_v;
				y2_val -= qe2_v;

				H_local[idx] = H_local[idx] + v_val.x;
				H_local[idx + 1] = H_local[idx + 1] + v_val.y;

				int32_t tmp = H_local[idx] > max_H_local ? 1 : 0;
				max_H_local = tmp ? H_local[idx] : max_H_local;
				max_t_local = tmp ? global_idx : max_t_local;

				tmp = H_local[idx + 1] > max_H_local ? 1 : 0;
				max_H_local = tmp ? H_local[idx + 1] : max_H_local;
				max_t_local = tmp ? (global_idx + 1) : max_t_local;

				p_local[idx] = (global_idx <= en) ? d.x : 0;
				p_local[idx + 1] = (global_idx + 1 <= en) ? d.y : 0;

				idx += 2;

				if (global_idx <= en) {
					curr_u[shmem_idx] = ut.x;
					curr_v[shmem_idx] = v_val.x;
					curr_x[shmem_idx] = x_val.x;
					curr_y[shmem_idx] = y_val.x;
					curr_x2[shmem_idx] = x2_val.x;
					curr_y2[shmem_idx] = y2_val.x;
				}

				if (global_idx + 1 <= en) {
					curr_u[shmem_idx + 1] = ut.y;
					curr_v[shmem_idx + 1] = v_val.y;
					curr_x[shmem_idx + 1] = x_val.y;
					curr_y[shmem_idx + 1] = y_val.y;
					curr_x2[shmem_idx + 1] = x2_val.y;
					curr_y2[shmem_idx + 1] = y2_val.y;
				}
			}

			// for(int i = 0; i < NUM_SEGMENTS; i++){
			// 	int id = t + tid + i*blockDim.x;
			// 	if(id >= st0){
			// 		H[id] = H_local[i];
			// 	}
			// 	p[row_offset + id] = p_local[i];
			// 	H_local[i] = H[id + SEGMENT_SIZE];
			// }

			__syncthreads();

			for (int i = 0; i < NUM_SEGMENTS; i += 2) {
				int id = t + tid * 2 + i * blockDim.x;
				if (id >= st0) {
					*(int64_t*)(H + id) = *(int64_t*)(H_local + i);
				} else if (id + 1 >= st0) {
					H[id + 1] = H_local[i + 1];
				}

				*(int16_t*)(p + row_offset + id) = *(int16_t*)(p_local + i);
				// p[row_offset + id] = p_local[i];
				// p[row_offset + id + 1] = p_local[i + 1];

				*(int64_t*)(H_local + i) = *(int64_t*)(H + id + SEGMENT_SIZE);
			}

			if (over_shmem) {
				int no_elems = t + SEGMENT_SIZE <= en ? SEGMENT_SIZE : en - t + 1;

				memcpy_to_global(u, v, x, y, x2, y2, curr_u, curr_v, curr_x, curr_y, curr_x2, curr_y2, t, no_elems);

				swap_pointers(curr_u, next_u);
				swap_pointers(curr_v, next_v);
				swap_pointers(curr_x, next_x);
				swap_pointers(curr_y, next_y);
				swap_pointers(curr_x2, next_x2);
				swap_pointers(curr_y2, next_y2);
			}
		}


		//  		} else { /* gap right-alignment */
		//
		//  			int row_offset = ((r * n_col_)>>4) - st;
		//
		//  			for (int t = st; t <= en; t += SEGMENT_SIZE) {
		//
		//  				int shift = !over_shmem ? t : 0;
		//
		//  				if(over_shmem) { // Start issue data from global memory for the following portion, number of elements depends on the antidiagonal size
		//  					// int no_elems2 = SEGMENT_SIZE; //  t + 2 * SEGMENT_SIZE <= en ? SEGMENT_SIZE : en - (t + SEGMENT_SIZE) + 1;
		//  					memcpy_to_shared(u, v, x, y, x2, y2, next_u, next_v, next_x, next_y, next_x2, next_y2, t + SEGMENT_SIZE);
		//  				}
		//
		//  				for (int i = 0; i < SEGMENT_SIZE; i += BLOCK_DIM){
		//
		//  					int shmem_idx = tid + shift + i;
		//  					int global_idx = tid + t + i;
		//
		//  					int idx = i / BLOCK_DIM;
		//
		//  					int8_t	q_char = q_char_local[idx];
		//  					int8_t	t_char = t_char_local[idx];
		//  					int32_t	H_val = H_local[idx];
		//
		//  					t_char_local[idx] = target[global_idx + SEGMENT_SIZE];
		//  					q_char_local[idx] = query[q_start + global_idx + SEGMENT_SIZE];
		//
		//  					int8_t z = s_local;
		//
		//  					int8_t x_val = curr_x[shmem_idx];
		//  					int8_t x2_val = curr_x2[shmem_idx];
		//  					int8_t v_val = curr_v[shmem_idx];
		//  					int8_t y_val = curr_y[shmem_idx];
		//  					int8_t y2_val = curr_y2[shmem_idx];
		//  					int8_t ut = curr_u[shmem_idx];
		//
		//  					if (global_idx >= st0 && global_idx < en0_rounded){
		//  						if (!(flag & KSW_EZ_GENERIC_SC)) {
		//  							z = t_char == q_char ? sc_mch : sc_mis;
		//  							z = t_char == m1 || q_char == m1 ? sc_N : z;
		//  							if(t == st) s_local = z;
		//  						} else {
		//  							z = mat[t_char * m + q_char]; // ((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
		//  							if(t == st) s_local = z;
		//  						}
		//  					}
		//
		//  					int8_t xt1 = __shfl_up(x_val, 1);
		//  					int8_t vt1 = __shfl_up(v_val , 1);
		//  					int8_t x2t1 = __shfl_up(x2_val, 1);
		//
		//  					if(tid == 0){
		//  						xt1 = x1;
		//  						x2t1 = x21;
		//  						vt1 = v1;
		//  					}
		//
		//  					x1 = __shfl(x_val, blockDim.x - 1);
		//  					x21 = __shfl(x2_val, blockDim.x - 1);
		//  					v1 = __shfl(v_val, blockDim.x - 1);
		//
		//  					int8_t a = xt1 + vt1;
		//  					int8_t b = y_val + ut;
		//
		//  					int8_t a2 = x2t1 + vt1;
		//  					int8_t b2 = y2_val + ut;
		//
		//  					int8_t d;
		//
		//  					d = z > a ? 0 : 1;
		//  					z = z > a ? z : a;
		//  					d = z > b ? d : 2;
		//  					z = z > b ? z : b;
		//  					d = z > a2 ? d : 3;
		//  					z = z > a2 ? z : a2;
		//  					d = z > b2 ? d : 4;
		//  					z = z > b2 ? z : b2;
		//  					z = z < sc_mch ? z : sc_mch;
		//
		//  					a = a - (z - q);
		//  					b = b - (z - q);
		//  					a2 = a2 - (z - q2);
		//  					b2 = b2 - (z - q2);
		//
		//  					d = a > 0 ? 0x08 : 0;
		//  					d = b > 0 ? 0x10 : 0;
		//  					d = a2 > 0 ? 0x20 : 0;
		//  					d = b2 > 0 ? 0x40 : 0;
		//
		//  					H_val += z - ut;
		//  					H_local[idx] = H_val;
		//  					int32_t tmp = H_val > max_H_local ? 1: 0;
		//  					max_H_local = tmp ? H_val : max_H_local;
		//  					max_t_local = tmp ? global_idx : max_t_local;
		//
		//  					p_local[idx] = (global_idx <= en) ? d : 0;
		//
		//  					if (global_idx <= en){
		//  						curr_u[shmem_idx] = z - vt1;
		//  						curr_v[shmem_idx] = z - ut;
		//  						curr_x[shmem_idx] = (a > 0 ? a : 0)  - (qe);
		//  						curr_y[shmem_idx] = (b > 0 ? b : 0) - (qe);
		//  						curr_x2[shmem_idx] = (a2 > 0 ? a2 : 0) - (qe2);
		//  						curr_y2[shmem_idx] = (b2 > 0 ? b2 : 0) - (qe2);
		//  						// p_local[idx] = d;
		//  					}
		//  				}
		//
		//  				for(int i = 0; i < NUM_SEGMENTS; i++){
		//  					int id = t + tid + i*blockDim.x;
		//  					if(id >= st0){
		//  						H[id] = H_local[i];
		//  						// t_char_local[i] = target[id + SEGMENT_SIZE];
		//  						// q_char_local[i] = query[q_start + id + SEGMENT_SIZE];
		//  					}
		//  					H_local[i] = H[id + SEGMENT_SIZE];
		//  					p[row_offset + id] = p_local[i];
		//  				}
		//
		//  				if(over_shmem) {
		//  					int no_elems = t + SEGMENT_SIZE <= en ? SEGMENT_SIZE : en - t + 1;
		//  					__syncthreads();
		//  					memcpy_to_global(u, v, x, y, x2, y2, curr_u, curr_v, curr_x, curr_y, curr_x2, curr_y2, t, no_elems);
		//
		//  					swap_pointers(curr_u, next_u);
		//  					swap_pointers(curr_v, next_v);
		//  					swap_pointers(curr_x, next_x);
		//  					swap_pointers(curr_y, next_y);
		//  					swap_pointers(curr_x2, next_x2);
		//  					swap_pointers(curr_y2, next_y2);
		//  				}
		//
		//  			}
		//
		//  		}
		//
		// 		__syncthreads();

		if (!(flag & KSW_EZ_SCORE_ONLY)) off[r] = st, off_end[r] = en;

		if (!(flag & KSW_EZ_APPROX_MAX)) { /* find the exact max with a 32-bit score array */

			int32_t max_t = en0;
			/* compute H[], max_H and max_t */
			if (r > 0) {

				int8_t u_en0 = over_shmem ? u[en0] : u_shmem[en0];
				int8_t v_en0 = over_shmem ? v[en0] : v_shmem[en0];

				max_H = en0 > 0 ? max_H + u_en0 : max_H + v_en0; /* special casing the last element */
				H[en0] = max_H;

				for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
					int tmp_H = __shfl_down(max_H_local, offset);
					int tmp_t = __shfl_down(max_t_local, offset);
					if (tmp_H > max_H_local) max_H_local = tmp_H, max_t_local = tmp_t;
					__syncthreads();
				}

				int tmp_H = __shfl(max_H_local, 0);
				int tmp_t = __shfl(max_t_local, 0);

				if (tmp_H > max_H) max_H = tmp_H, max_t = tmp_t;

				/*for (t = en1; t < en0; ++t) {  // for the rest of values that haven't been computed with SSE 
					H[t] += (int32_t)v[t];
					if (H[t] > max_H)
						max_H = H[t], max_t = t;
				}*/

			} else
				H[0] = v_shmem[0] - (qe), max_H = H[0], max_t = 0; /* special casing r==0 */

			/* update ez */
			if (en0 == tlen - 1 && H[en0] > ez_local.mte)
				ez_local.mte = H[en0], ez_local.mte_q = r - en0;
			if (r - st0 == qlen - 1 && H[st0] > ez_local.mqe)
				ez_local.mqe = H[st0], ez_local.mqe_t = st0;
			if (ksw_apply_zdrop_gpu(&ez_local, max_H, r, max_t, zdrop, e2)) break; // if score drops too fast, interrupt alignment
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez_local.score = H[tlen - 1];


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

			if ((flag & KSW_EZ_APPROX_DROP) && ksw_apply_zdrop_gpu(&ez_local, H0, r, last_H0_t, zdrop, e2)) break;
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez_local.score = H0;
		}

		last_st = st, last_en = en;

		if (en == SHMEM_SIZE - 1) {
			memcpy_to_global(u, v, x, y, x2, y2, u_shmem, v_shmem, x_shmem, y_shmem, x2_shmem, y2_shmem, 0, SHMEM_SIZE);
		}

		//for (t = st0; t <= en0; ++t) printf("(%d,%d)\t(%d,%d,%d,%d)\t%d\n", r, t, ((int8_t*)u)[t], ((int8_t*)v)[t], ((int8_t*)x)[t], ((int8_t*)y)[t], H[t]); // for debugging
	}


	memcpy(&ez_d[bid], &ez_local, sizeof(ksw_extz_t));

	// long long int end_kernel = get_clock();
	// long long int duration = end_kernel - start_kernel;
	// if(threadIdx.x == 0) {printf("%d, %d, %d, %ld\n", blockIdx.x, tlen, qlen, duration);};

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


// __launch_bounds__(64,2)
//__global__ __attribute__((amdgpu_waves_per_eu(5,8))) void ksw_extd2_gpu_packed( // void *km,
//					int8_t* __restrict__ const u_d,
//					int8_t* __restrict__ const v_d,
//					int8_t* __restrict__ const x_d,
//					int8_t* __restrict__ const y_d,
//					int8_t* __restrict__ const x2_d,
//					int8_t* __restrict__ const y2_d,
//					// int8_t* __restrict__ const s_d,
//					uint8_t* __restrict__ const p_d,
//					int32_t* __restrict__ const H_d,
//					int* __restrict__ const off_d,
//					int* __restrict__ const off_end_d,
//					const int* __restrict__ const qlen_d,
//					const uint8_t* __restrict__ const query_d,
//					const int* __restrict__ const tlen_d,
//					const uint8_t* __restrict__ const target_d,
//					// const int8_t m,
//					const int8_t* const mat_d,
//					const int8_t q, const int8_t e, const int8_t q2, const int8_t e2, const int w_d, const int zdrop, const int end_bonus, const int flag,
//					ksw_extz_t* __restrict__ const ez_d,
//					const uint32_t* __restrict__ const tlen_ps_d,
//					const uint32_t* __restrict__ const qlen_ps_d,
//					const uint32_t* __restrict__ const p_ps_d,
//					const uint32_t* __restrict__ const off_ps_d,
//					const int8_t min_sc)
//{
//
//	const int32_t bid = blockIdx.x;
//	const int32_t tid = threadIdx.x;
//
//	const bool generic_sc = (flag & KSW_EZ_GENERIC_SC);
//
//	const uint32_t offset_t_ps = tlen_ps_d[bid];
//	const uint32_t offset_q_ps = qlen_ps_d[bid];
//	const uint32_t offset_p_ps = p_ps_d[bid];
//	const uint32_t offset_o_ps = !(flag&KSW_EZ_SCORE_ONLY) ? off_ps_d[bid] : 0;
//
//	const int qlen = qlen_d[bid];
//	const int tlen = tlen_d[bid];
//
//	const uint8_t* const target = target_d + offset_t_ps;
//
//	int8_t* const u = u_d + offset_t_ps;
//	int8_t* const v = v_d + offset_t_ps;
//	int8_t* const x = x_d + offset_t_ps;
//	int8_t* const y = y_d + offset_t_ps;
//	int8_t* const x2 = x2_d + offset_t_ps;
//	int8_t* const y2 = y2_d + offset_t_ps;
//	int32_t* const H = H_d + offset_t_ps;
//
//	const uint8_t* const query = query_d + offset_q_ps;
//	uint8_t* const p = p_d + offset_p_ps;
//
//	int* const off = !(flag&KSW_EZ_SCORE_ONLY) ? off_d + offset_o_ps : 0;
//	int* const off_end = !(flag&KSW_EZ_SCORE_ONLY) ? off_end_d + offset_o_ps : 0;
//
//	__shared__ int8_t mat[25];
//	memcpy(mat, mat_d, 25);
//
//	__shared__ int8_t u_shmem[SHMEM_SIZE];
//	__shared__ int8_t v_shmem[SHMEM_SIZE];
//	__shared__ int8_t x_shmem[SHMEM_SIZE];
//	__shared__ int8_t y_shmem[SHMEM_SIZE];
//	__shared__ int8_t x2_shmem[SHMEM_SIZE];
//	__shared__ int8_t y2_shmem[SHMEM_SIZE];
//
//	__shared__ ksw_extz_t ez_local;
//	if(tid == 0){
//		ez_local.max_q = ez_local.max_t = ez_local.mqe_t = ez_local.mte_q = -1;
//		ez_local.max = 0, ez_local.score = ez_local.mqe = ez_local.mte = KSW_NEG_INF;
//		ez_local.n_cigar = 0, ez_local.zdropped = 0, ez_local.reach_end = 0;
//	}
//	__syncthreads();
//
//	if (qlen <= 0 || tlen <= 0) return; // end due to invalid alignment
//
//	const int8_t qe = q + e;
//	const int8_t qe2 = q2 + e2;
//	const int8_t sc_mch = mat[0];
//	const int8_t sc_mis = mat[1];
//	const int8_t sc_N = mat[24] == 0? -e2 : mat[24];
//
//	if (-min_sc > 2 * (qe)) return; /* otherwise, we won't see any mismatches */
//
//	int w = w_d;
//	if (w < 0) w = tlen > qlen? tlen : qlen;										// set bandwidth in case it is negative (non-banded alignment)
//	// const int tlen_ = (tlen + 15)>>4; // / 16;										// round to multiple of 16
//	// const int qlen_ = (qlen + 15)>>4; //  / 16;                                     // round to multiple of 16
//	int n_col_ = qlen < tlen? qlen : tlen;
//    n_col_ = (((n_col_ < w + 1? n_col_ : w + 1) + 15)>>4) + 1;              		// round to multiple of 16
//
//	int long_thres = e != e2 ? (q2 - q) / (e - e2) - 1 : 0; // if double affine, set the threshold for long gaps(?) to ((q2 - q) / (e - e2) - 1), otherwise 0
//	if ((qe2) + long_thres * e2 > (qe) + long_thres * e)
//		++long_thres;
//
//	for(int i = 0; i < SHMEM_SIZE; i += BLOCK_DIM){
//		if(tid + i < SHMEM_SIZE) {
//			u_shmem[tid + i] = -qe;
//			v_shmem[tid + i] = -qe;
//			x_shmem[tid + i] = -qe;
//			y_shmem[tid + i] = -qe;
//			x2_shmem[tid + i] = -(qe2);
//			y2_shmem[tid + i] = -(qe2);
//		}
//	}
//
//	// for(int i = 0; i < tlen_<<4; i += blockDim.x){
//	// 	if(tid + i < tlen_<<4) {
//	// 		u[tid + i] = -qe;
//	// 		v[tid + i] = -qe;
//	// 		x[tid + i] = -qe;
//	// 		y[tid + i] = -qe;
//	// 		x2[tid + i] = -(qe2);
//	// 		y2[tid + i] = -(qe2);
//	// 		if (!(flag&KSW_EZ_APPROX_MAX))  H[tid + i] = KSW_NEG_INF;
//	// 	}
//	// }
//
//	int32_t last_st = -1;
//	int32_t last_en = -1;
//	int32_t s_local = 0;
//	int32_t H0 = 0, last_H0_t = 0;
//
//	// alignment main loop
//	for (int r = 0; r < qlen + tlen - 1; ++r) {
//
//		int32_t st = 0;
//		int32_t en = tlen - 1;
//		int32_t st0;
//		int32_t en0;
//
//		/* find the boundaries */
//		// if (st < r - qlen + 1) st = r - qlen + 1;
//		// if (en > r) en = r;
//		// if (st < (r-w+1)>>1) st = (r-w+1)>>1; 	/* take the ceil */
//		// if (en > (r+w)>>1) en = (r+w)>>1; 		/* take the floor */
//
//		const int rql1 = r - qlen + 1;
//		const int rw1 = (r - w + 1) >> 1;
//		const int rw2 = (r + w) >> 1;
//		st = st > (rql1 > rw1 ? rql1 : rw1) ? st : (rql1 > rw1 ? rql1 : rw1);
//		en = en < (r < rw2 ? r : rw2) ? en : (r < rw2 ? r : rw2);
//
//		if (st > en) {
//			ez_local.zdropped = 1;
//			break;
//		}
//
//
//		st0 = st, en0 = en;
//		st = (st >> 4) << 4, en = (((en + 16) >> 4) << 4) - 1;
//
//		int8_t x1 = -qe, x21 = -qe2;
//		int8_t v1 = -qe;
//
//		bool over_shmem = en >= SHMEM_SIZE;
//
//		__syncthreads();
//
//		const int32_t q_start = qlen - 1 - r;
//		int8_t q_char_local[NUM_SEGMENTS] = {4};
//		int8_t t_char_local[NUM_SEGMENTS] = {4};
//		int32_t H_local[NUM_SEGMENTS] = {KSW_NEG_INF};
//		int8_t p_local[NUM_SEGMENTS] = {0};
//
//		for(int i = 0; i < NUM_SEGMENTS; i++){
//			int id = st + tid + i*blockDim.x;
//			if(id >= st0){
//				t_char_local[i] = target[id];
//				q_char_local[i] = query[q_start + id];
//				H_local[i] = H[id];
//			}
//		}
//
//		/* set boundary conditions */
//
//		if (st > 0 && st - 1 >= last_st && st - 1 <= last_en) {
//			x1 = !over_shmem ? x_shmem[st - 1] : x[st - 1];
//			x21 = !over_shmem ? x2_shmem[st - 1] : x2[st - 1];
//			v1 = !over_shmem ? v_shmem[st - 1] : v[st - 1];	/* (r-1,s-1) calculated in the last round */
//		}
//
//		int8_t tmp_val = r == 0? -qe : r < long_thres? -e : r == long_thres? (long_thres * (e - e2) + q - qe2) : -e2;
//
//		if (st == 0)
//			v1 = tmp_val; // r == 0? -qe : r < long_thres? -e : r == long_thres? (long_thres * (e - e2) + q - qe2) : -e2;
//
//		if (en >= r) {
//			if (!over_shmem){
//				y_shmem[r] = -qe;
//				y2_shmem[r] = -qe2; 	// ((int8_t*)y)[r] = -q - e, ((int8_t*)y2)[r] = -q2 - e2;
//				u_shmem[r] = tmp_val; 	// r == 0? -(qe) : r < long_thres? -e : r == long_thres? (long_thres * (e - e2) + q - qe2) : -e2;
//			} else {
//				y[r] = -qe;
//				y2[r] = -qe2; 			// ((int8_t*)y)[r] = -q - e, ((int8_t*)y2)[r] = -q2 - e2;
//				u[r] = tmp_val; 		// r == 0? -(qe) : r < long_thres? -e : r == long_thres? (long_thres * (e - e2) + q - qe2) : -e2;
//			}
//		}
//
//		if(over_shmem)
//			memcpy_to_shared(u, v, x, y, x2, y2, u_shmem, v_shmem, x_shmem, y_shmem, x2_shmem, y2_shmem, st);
//
//		int32_t max_H = en0 > 0? H[en0-1] : H[en0]; /* special casing the last element */
//		int32_t max_H_local = KSW_NEG_INF;
//		int32_t max_t_local = -1;
//
//		const int en0_rounded = ((en0 - st0)/16 + 1)*16 + st0;
//
//		int8_t* curr_u = u_shmem;
//		int8_t* curr_v = v_shmem;
//		int8_t* curr_x = x_shmem;
//		int8_t* curr_y = y_shmem;
//		int8_t* curr_x2 = x2_shmem;
//		int8_t* curr_y2 = y2_shmem;
//
//		int8_t* next_u = u_shmem + SEGMENT_SIZE;
//		int8_t* next_v = v_shmem + SEGMENT_SIZE;
//		int8_t* next_x = x_shmem + SEGMENT_SIZE;
//		int8_t* next_y = y_shmem + SEGMENT_SIZE;
//		int8_t* next_x2 = x2_shmem + SEGMENT_SIZE;
//		int8_t* next_y2 = y2_shmem + SEGMENT_SIZE;
//
//		__syncthreads();
//
////  		if (flag&KSW_EZ_SCORE_ONLY) { // score only
////
////  			for (int t = st; t <= en; t += SEGMENT_SIZE) {
////
////  				int shift = !over_shmem ? t : 0;
////
////  				// __syncthreads();
////
////  				if(over_shmem) { // Start issue data from global memory for the following portion, number of elements depends on the antidiagonal size
////  					// int no_elems2 = SEGMENT_SIZE; // t + 2 * SEGMENT_SIZE <= en ? SEGMENT_SIZE : en - (t + SEGMENT_SIZE) + 1;
////  					memcpy_to_shared(u, v, x, y, x2, y2, next_u, next_v, next_x, next_y, next_x2, next_y2, t + SEGMENT_SIZE);
////  				}
////
////  				for (int i = 0; i < SEGMENT_SIZE; i += BLOCK_DIM){
////
////  					int shmem_idx = tid + i + shift;
////  					int global_idx = tid + i + t;
////
////  					int idx = i / BLOCK_DIM;
////
////  					int8_t	q_char = q_char_local[idx];
////  					int8_t	t_char = t_char_local[idx];
////  					int32_t	H_val = H_local[idx];
////
////  					t_char_local[idx] = target[global_idx + SEGMENT_SIZE];
////  					q_char_local[idx] = query[q_start + global_idx + SEGMENT_SIZE];
////
////  					int8_t z = s_local;
////
////  					int8_t ut = curr_u[shmem_idx]; 			//  u_shmem[shmem_idx];
////  					int8_t v_val = curr_v[shmem_idx]; 		//  v_shmem[shmem_idx];
////  					int8_t x_val = curr_x[shmem_idx]; 		//  x_shmem[shmem_idx];
////  					int8_t y_val = curr_y[shmem_idx]; 		//  y_shmem[shmem_idx];
////  					int8_t x2_val = curr_x2[shmem_idx]; 	//  x2_shmem[shmem_idx];
////  					int8_t y2_val = curr_y2[shmem_idx]; 	//  y2_shmem[shmem_idx];
////
////  					if (global_idx >= st0 && global_idx < en0_rounded){
////  						if (!(flag & KSW_EZ_GENERIC_SC)) {
////  							z = t_char == q_char ? sc_mch : sc_mis;
////  							z = t_char == m1 || q_char == m1 ? sc_N : z;
////  							if(t == st) s_local = z;
////  						} else {
////  							z = mat[t_char * m + q_char]; // ((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
////  							if(t == st) s_local = z;
////  						}
////  					}
////
////  					int8_t xt1 = __shfl_up(x_val, 1);
////  					int8_t vt1 = __shfl_up(v_val , 1);
////  					int8_t x2t1 = __shfl_up(x2_val, 1);
////
////  					if(tid == 0){
////  						xt1 = x1;
////  						x2t1 = x21;
////  						vt1 = v1;
////  					}
////
////  					x1 = __shfl(x_val, blockDim.x - 1);
////  					x21 = __shfl(x2_val, blockDim.x - 1);
////  					v1 = __shfl(v_val, blockDim.x - 1);
////
////  					int8_t a = xt1 + vt1;
////  					int8_t b = y_val + ut;
////
////  					int8_t a2 = x2t1 + vt1;
////  					int8_t b2 = y2_val + ut;
////
////  					z = z > a ? z : a;
////  					z = z > b ? z : b;
////  					z = z > a2 ? z : a2;
////  					z = z > b2 ? z : b2;
////
////  					z = z < sc_mch ? z : sc_mch;
////
////  					v_val = z - ut;
////  					H_val += v_val;
////
////  					int32_t tmp = H_val > max_H_local ? 1 : 0;
////  					max_H_local = tmp ? H_val : max_H_local;
////  					max_t_local = tmp ? global_idx : max_t_local;
////  					H_local[idx] = H_val;
////
////  					a = a - (z - q);
////  					b = b - (z - q);
////  					a2 = a2 - (z - q2);
////  					b2 = b2 - (z - q2);
////
////  					int8_t u_val = z - vt1;
////  					x_val = (a > 0 ? a : 0) - (qe);
////  					y_val = (b > 0 ? b : 0) - (qe);
////  					x2_val = (a2 > 0 ? a2 : 0) - (qe2);
////  					y2_val = (b2 > 0 ? b2 : 0) - (qe2);
////
////  					if (global_idx <= en){
////  						curr_u[shmem_idx] = u_val;
////  						curr_v[shmem_idx] = v_val;
////  						curr_x[shmem_idx] = x_val;
////  						curr_y[shmem_idx] = y_val;
////  						curr_x2[shmem_idx] = x2_val;
////  						curr_y2[shmem_idx] = y2_val;
////  					}
////  				}
////
////  				for(int i = 0; i < NUM_SEGMENTS; i++){
////  					int id = t + tid + i*blockDim.x;
////  					if(id >= st0){
////  						H[id] = H_local[i];
////  						// t_char_local[i] = target[id + SEGMENT_SIZE];
////  						// q_char_local[i] = query[q_start + id + SEGMENT_SIZE];
////  					}
////  					H_local[i] = H[id + SEGMENT_SIZE];
////  				}
////
////  				if(over_shmem) {
////  					int no_elems = t + SEGMENT_SIZE <= en ? SEGMENT_SIZE : en - t + 1;
////  					__syncthreads();
////  					memcpy_to_global(u, v, x, y, x2, y2, curr_u, curr_v, curr_x, curr_y, curr_x2, curr_y2, t, no_elems);
////
////  					swap_pointers(curr_u, next_u);
////  					swap_pointers(curr_v, next_v);
////  					swap_pointers(curr_x, next_x);
////  					swap_pointers(curr_y, next_y);
////  					swap_pointers(curr_x2, next_x2);
////  					swap_pointers(curr_y2, next_y2);
////  				}
////
////  			}
////
////  		} else if (!(flag&KSW_EZ_RIGHT)) { /* gap left-alignment */
//
//			const int row_offset = ((r * n_col_)>>4) - st;
//
//			for (int t = st; t <= en; t += SEGMENT_SIZE) {
//
//				int shift = !over_shmem ? t : 0;
//
//				if(over_shmem && t + SEGMENT_SIZE <= en) { // Start issue data from global memory for the following portion, number of elements depends on the antidiagonal size
//					memcpy_to_shared(u, v, x, y, x2, y2, next_u, next_v, next_x, next_y, next_x2, next_y2, t + SEGMENT_SIZE);
//				}
//
//				for (int i = 0; i < SEGMENT_SIZE; i += BLOCK_DIM){
//
//					int32_t shmem_idx = tid + i + shift;
//					int32_t global_idx = tid + i + t;
//
//					int idx = i / BLOCK_DIM;
//
//					int8_t ut = curr_u[shmem_idx];
//					int8_t v_val = curr_v[shmem_idx];
//					int8_t x_val = curr_x[shmem_idx];
//					int8_t y_val = curr_y[shmem_idx];
//					int8_t x2_val = curr_x2[shmem_idx];
//					int8_t y2_val = curr_y2[shmem_idx];
//
//					int8_t	q_char = q_char_local[idx];
//					int8_t	t_char = t_char_local[idx];
//					int32_t	H_val = H_local[idx];
//
//					t_char_local[idx] = target[global_idx + SEGMENT_SIZE];
//					q_char_local[idx] = query[q_start + global_idx + SEGMENT_SIZE];
//
//					int8_t z = s_local;
//
//					// if (global_idx >= st0 && global_idx < en0_rounded){
//					// 	if (!generic_sc) {
//					// 		z = t_char == q_char ? sc_mch : sc_mis;
//					// 		z = t_char == m1 || q_char == m1 ? sc_N : z;
//					// 		if(t == st) s_local = z;
//					// 	} else {
//					// 		z = mat[t_char * m + q_char]; // ((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
//					// 		if(t == st) s_local = z;
//					// 	}
//					// }
//
//					const bool in_range = global_idx >= st0 && global_idx < en0_rounded;
//
//					int8_t z1 = t_char == q_char ? sc_mch : sc_mis;
//					const bool ambiguous = (t_char == m1) | (q_char == m1);
//					z1 = ambiguous ? sc_N : z1;
//
//					const int8_t z2 = mat[t_char * m + q_char];
//
//					z = (in_range && !generic_sc) ? z1 : z;
//					z = (in_range && generic_sc) ? z2 : z;
//
//					s_local = (t == st) ? z : s_local;
//
//					int8_t xt1 = __shfl_up(x_val, 1);
//					int8_t vt1 = __shfl_up(v_val , 1);
//					int8_t x2t1 = __shfl_up(x2_val, 1);
//
//					xt1 = (tid == 0) ? x1 : xt1;
//					x2t1 = (tid == 0) ? x21 : x2t1;
//					vt1 = (tid == 0) ? v1 : vt1;
//
//					x1 = __shfl(x_val, BLOCK_DIM - 1);
//					x21 = __shfl(x2_val, BLOCK_DIM - 1);
//					v1 = __shfl(v_val, BLOCK_DIM - 1);
//
//					int8_t a = xt1 + vt1;
//					int8_t b = y_val + ut;
//
//					int8_t a2 = x2t1 + vt1;
//					int8_t b2 = y2_val + ut;
//
//					int8_t d = a > z ? 1 : 0;
//
//					z = z > a ? z : a;
//					d = b > z ? 2 : d;
//
//					z = z > b ? z : b;
//					d = a2 > z ? 3 : d;
//
//					z = z > a2 ? z : a2;
//					d = b2 > z ? 4 : d;
//
//					z = z > b2 ? z : b2;
//					z = z < sc_mch ? z : sc_mch;
//
//					a = a - z + q;
//					b = b - z + q;
//					a2 = a2 - z + q2;
//					b2 = b2 - z + q2;
//
//					v_val = z - ut;
//
//					H_val += v_val;
//
//					int32_t tmp = H_val > max_H_local ? 1 : 0;
//					max_H_local = tmp ? H_val : max_H_local;
//					max_t_local = tmp ? global_idx : max_t_local;
//					H_local[idx] = H_val;
//
//					d = a > 0 ? 0x08 : 0;
//					d = b > 0 ? 0x10 : 0;
//					d = a2 > 0 ? 0x20 : 0;
//					d = b2 > 0 ? 0x40 : 0;
//
//					ut = z - vt1;
//					x_val = (a > 0 ? a : 0) - (qe);
//					y_val = (b > 0 ? b : 0) - (qe);
//					x2_val = (a2 > 0 ? a2 : 0) - (qe2);
//					y2_val = (b2 > 0 ? b2 : 0) - (qe2);
//
//					p_local[idx] = (global_idx <= en) ? d : 0;
//
//					if (global_idx <= en){
//						curr_u[shmem_idx] = ut;
//						curr_v[shmem_idx] = v_val;
//						curr_x[shmem_idx] = x_val;
//						curr_y[shmem_idx] = y_val;
//						curr_x2[shmem_idx] = x2_val;
//						curr_y2[shmem_idx] = y2_val;
//					}
//				}
//
//				for(int i = 0; i < NUM_SEGMENTS; i++){
//					int id = t + tid + i*blockDim.x;
//					if(id >= st0){
//						H[id] = H_local[i];
//					}
//					p[row_offset + id] = p_local[i];
//					H_local[i] = H[id + SEGMENT_SIZE];
//				}
//
//				if(over_shmem) {
//					int no_elems = t + SEGMENT_SIZE <= en ? SEGMENT_SIZE : en - t + 1;
//					__syncthreads();
//					memcpy_to_global(u, v, x, y, x2, y2, curr_u, curr_v, curr_x, curr_y, curr_x2, curr_y2, t, no_elems);
//
//					swap_pointers(curr_u, next_u);
//					swap_pointers(curr_v, next_v);
//					swap_pointers(curr_x, next_x);
//					swap_pointers(curr_y, next_y);
//					swap_pointers(curr_x2, next_x2);
//					swap_pointers(curr_y2, next_y2);
//				}
//
//			}
//
////  		} else { /* gap right-alignment */
////
////  			int row_offset = ((r * n_col_)>>4) - st;
////
////  			for (int t = st; t <= en; t += SEGMENT_SIZE) {
////
////  				int shift = !over_shmem ? t : 0;
////
////  				if(over_shmem) { // Start issue data from global memory for the following portion, number of elements depends on the antidiagonal size
////  					// int no_elems2 = SEGMENT_SIZE; //  t + 2 * SEGMENT_SIZE <= en ? SEGMENT_SIZE : en - (t + SEGMENT_SIZE) + 1;
////  					memcpy_to_shared(u, v, x, y, x2, y2, next_u, next_v, next_x, next_y, next_x2, next_y2, t + SEGMENT_SIZE);
////  				}
////
////  				for (int i = 0; i < SEGMENT_SIZE; i += BLOCK_DIM){
////
////  					int shmem_idx = tid + shift + i;
////  					int global_idx = tid + t + i;
////
////  					int idx = i / BLOCK_DIM;
////
////  					int8_t	q_char = q_char_local[idx];
////  					int8_t	t_char = t_char_local[idx];
////  					int32_t	H_val = H_local[idx];
////
////  					t_char_local[idx] = target[global_idx + SEGMENT_SIZE];
////  					q_char_local[idx] = query[q_start + global_idx + SEGMENT_SIZE];
////
////  					int8_t z = s_local;
////
////  					int8_t x_val = curr_x[shmem_idx];
////  					int8_t x2_val = curr_x2[shmem_idx];
////  					int8_t v_val = curr_v[shmem_idx];
////  					int8_t y_val = curr_y[shmem_idx];
////  					int8_t y2_val = curr_y2[shmem_idx];
////  					int8_t ut = curr_u[shmem_idx];
////
////  					if (global_idx >= st0 && global_idx < en0_rounded){
////  						if (!(flag & KSW_EZ_GENERIC_SC)) {
////  							z = t_char == q_char ? sc_mch : sc_mis;
////  							z = t_char == m1 || q_char == m1 ? sc_N : z;
////  							if(t == st) s_local = z;
////  						} else {
////  							z = mat[t_char * m + q_char]; // ((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
////  							if(t == st) s_local = z;
////  						}
////  					}
////
////  					int8_t xt1 = __shfl_up(x_val, 1);
////  					int8_t vt1 = __shfl_up(v_val , 1);
////  					int8_t x2t1 = __shfl_up(x2_val, 1);
////
////  					if(tid == 0){
////  						xt1 = x1;
////  						x2t1 = x21;
////  						vt1 = v1;
////  					}
////
////  					x1 = __shfl(x_val, blockDim.x - 1);
////  					x21 = __shfl(x2_val, blockDim.x - 1);
////  					v1 = __shfl(v_val, blockDim.x - 1);
////
////  					int8_t a = xt1 + vt1;
////  					int8_t b = y_val + ut;
////
////  					int8_t a2 = x2t1 + vt1;
////  					int8_t b2 = y2_val + ut;
////
////  					int8_t d;
////
////  					d = z > a ? 0 : 1;
////  					z = z > a ? z : a;
////  					d = z > b ? d : 2;
////  					z = z > b ? z : b;
////  					d = z > a2 ? d : 3;
////  					z = z > a2 ? z : a2;
////  					d = z > b2 ? d : 4;
////  					z = z > b2 ? z : b2;
////  					z = z < sc_mch ? z : sc_mch;
////
////  					a = a - (z - q);
////  					b = b - (z - q);
////  					a2 = a2 - (z - q2);
////  					b2 = b2 - (z - q2);
////
////  					d = a > 0 ? 0x08 : 0;
////  					d = b > 0 ? 0x10 : 0;
////  					d = a2 > 0 ? 0x20 : 0;
////  					d = b2 > 0 ? 0x40 : 0;
////
////  					H_val += z - ut;
////  					H_local[idx] = H_val;
////  					int32_t tmp = H_val > max_H_local ? 1: 0;
////  					max_H_local = tmp ? H_val : max_H_local;
////  					max_t_local = tmp ? global_idx : max_t_local;
////
////  					p_local[idx] = (global_idx <= en) ? d : 0;
////
////  					if (global_idx <= en){
////  						curr_u[shmem_idx] = z - vt1;
////  						curr_v[shmem_idx] = z - ut;
////  						curr_x[shmem_idx] = (a > 0 ? a : 0)  - (qe);
////  						curr_y[shmem_idx] = (b > 0 ? b : 0) - (qe);
////  						curr_x2[shmem_idx] = (a2 > 0 ? a2 : 0) - (qe2);
////  						curr_y2[shmem_idx] = (b2 > 0 ? b2 : 0) - (qe2);
////  						// p_local[idx] = d;
////  					}
////  				}
////
////  				for(int i = 0; i < NUM_SEGMENTS; i++){
////  					int id = t + tid + i*blockDim.x;
////  					if(id >= st0){
////  						H[id] = H_local[i];
////  						// t_char_local[i] = target[id + SEGMENT_SIZE];
////  						// q_char_local[i] = query[q_start + id + SEGMENT_SIZE];
////  					}
////  					H_local[i] = H[id + SEGMENT_SIZE];
////  					p[row_offset + id] = p_local[i];
////  				}
////
////  				if(over_shmem) {
////  					int no_elems = t + SEGMENT_SIZE <= en ? SEGMENT_SIZE : en - t + 1;
////  					__syncthreads();
////  					memcpy_to_global(u, v, x, y, x2, y2, curr_u, curr_v, curr_x, curr_y, curr_x2, curr_y2, t, no_elems);
////
////  					swap_pointers(curr_u, next_u);
////  					swap_pointers(curr_v, next_v);
////  					swap_pointers(curr_x, next_x);
////  					swap_pointers(curr_y, next_y);
////  					swap_pointers(curr_x2, next_x2);
////  					swap_pointers(curr_y2, next_y2);
////  				}
////
////  			}
////
////  		}
////
//// 		__syncthreads();
//
//		if(!(flag&KSW_EZ_SCORE_ONLY)) off[r] = st, off_end[r] = en;
//
//		if (!(flag&KSW_EZ_APPROX_MAX)) { /* find the exact max with a 32-bit score array */
//
//			int32_t max_t = en0;
//			/* compute H[], max_H and max_t */
//			if (r > 0) {
//
//				int8_t u_en0 = over_shmem ? u[en0] : u_shmem[en0];
//				int8_t v_en0 = over_shmem ? v[en0] : v_shmem[en0];
//
//				max_H = en0 > 0? max_H + u_en0 : max_H + v_en0; /* special casing the last element */
//				H[en0] = max_H;
//
//				for(int offset = blockDim.x/2; offset > 0; offset /= 2){
//					int tmp_H = __shfl_down(max_H_local, offset);
//					int tmp_t = __shfl_down(max_t_local, offset);
//					if (tmp_H > max_H_local) max_H_local = tmp_H, max_t_local = tmp_t;
//					__syncthreads();
//				}
//
//				int tmp_H = __shfl(max_H_local, 0);
//				int tmp_t = __shfl(max_t_local, 0);
//
//				if(tmp_H > max_H) max_H = tmp_H, max_t = tmp_t;
//
//				/*for (t = en1; t < en0; ++t) {  // for the rest of values that haven't been computed with SSE
//					H[t] += (int32_t)v[t];
//					if (H[t] > max_H)
//						max_H = H[t], max_t = t;
//				}*/
//
//			} else H[0] = v_shmem[0] - (qe), max_H = H[0], max_t = 0; /* special casing r==0 */
//
//			/* update ez */
//			if (en0 == tlen - 1 && H[en0] > ez_local.mte)
//				ez_local.mte = H[en0], ez_local.mte_q = r - en0;
//			if (r - st0 == qlen - 1 && H[st0] > ez_local.mqe)
//				ez_local.mqe = H[st0], ez_local.mqe_t = st0;
//			if (ksw_apply_zdrop_gpu(&ez_local, max_H, r, max_t, zdrop, e2)) break; // if score drops too fast, interrupt alignment
//			if (r == qlen + tlen - 2 && en0 == tlen - 1)
//				ez_local.score = H[tlen - 1];
//
//
//		} else { /* find approximate max; Z-drop might be inaccurate, too. */
//
//			if (r > 0) {
//
//				int8_t* u_ptr = over_shmem ? u : u_shmem;
//				int8_t* v_ptr = over_shmem ? v : v_shmem;
//
//				if (last_H0_t >= st0 && last_H0_t <= en0 && last_H0_t + 1 >= st0 && last_H0_t + 1 <= en0) {
//					int32_t d0 = v_ptr[last_H0_t];
//					int32_t d1 = u_ptr[last_H0_t + 1];
//					if (d0 > d1) H0 += d0;
//					else H0 += d1, ++last_H0_t;
//				} else if (last_H0_t >= st0 && last_H0_t <= en0) {
//					H0 += v_ptr[last_H0_t];
//				} else {
//					++last_H0_t, H0 += u_ptr[last_H0_t];
//				}
//			} else H0 = v_shmem[0] - (qe), last_H0_t = 0;
//
//			if ((flag & KSW_EZ_APPROX_DROP) && ksw_apply_zdrop_gpu(&ez_local, H0, r, last_H0_t, zdrop, e2)) break;
//			if (r == qlen + tlen - 2 && en0 == tlen - 1)
//				ez_local.score = H0;
//
//		}
//
//		last_st = st, last_en = en;
//
//		if(en == SHMEM_SIZE - 1){
//			memcpy_to_global(u, v, x, y, x2, y2, u_shmem, v_shmem, x_shmem, y_shmem, x2_shmem, y2_shmem, 0, SHMEM_SIZE);
//		}
//
//		//for (t = st0; t <= en0; ++t) printf("(%d,%d)\t(%d,%d,%d,%d)\t%d\n", r, t, ((int8_t*)u)[t], ((int8_t*)v)[t], ((int8_t*)x)[t], ((int8_t*)y)[t], H[t]); // for debugging
//	}
//
//
//	memcpy(&ez_d[bid], &ez_local, sizeof(ksw_extz_t));
//
//	// long long int end_kernel = get_clock();
//	// long long int duration = end_kernel - start_kernel;
//	// if(threadIdx.x == 0) {printf("%d, %d, %d, %ld\n", blockIdx.x, tlen, qlen, duration);};
//
//	// kfree(km, mem);
//	// if (!approx_max) kfree(km, H);
//	// if (with_cigar) { // backtrack
//	// 	int rev_cigar = !!(flag & KSW_EZ_REV_CIGAR);
//	// 	if (!ez->zdropped && !(flag&KSW_EZ_EXTZ_ONLY)) {
//	// 		ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*16, tlen-1, qlen-1, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
//	// 	} else if (!ez->zdropped && (flag&KSW_EZ_EXTZ_ONLY) && ez->mqe + end_bonus > (int)ez->max) {
//	// 		ez->reach_end = 1;
//	// 		ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*16, ez->mqe_t, qlen-1, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
//	// 	} else if (ez->max_t >= 0 && ez->max_q >= 0) {
//	// 		ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*16, ez->max_t, ez->max_q, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
//	// 	}
//	//
//	// 	// // @todo: check whether we can remove it
//	// 	// if (flag & KSW_EZ_EQX) {
//	// 	// 	int32_t nc0 = ez->n_cigar;
//	// 	// 	uint32_t *ci0;
//	// 	// 	ci0 = (uint32_t*)kmalloc(km, nc0 * sizeof(uint32_t));
//	// 	// 	memcpy(ci0, ez->cigar, nc0 * sizeof(uint32_t));
//	// 	// 	ksw_cigar2eqx(km, query, target, nc0, ci0, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
//	// 	// 	kfree(km, ci0);
//	// 	// }
//	// 	kfree(km, mem2); kfree(km, off);
//	// }
//}