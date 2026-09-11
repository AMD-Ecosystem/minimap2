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

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "plscore.cuh"
#include "hipify.cuh"
#include "mm_log.h"

/* 

Parallel chaining helper functions with CUDA

*/

// The former __device__ curr_long_segid global has been retired. The work-
// stealing counter is now per-worker (deviceMemPtr::d_long_segid), zeroed
// via cudaMemsetAsync before each long-kernel launch. score_generation_long_map
// adds gridDim.x to its atomic returns to recover the original "start at
// gridDim.x" dispatch range.

/* arithmetic functions begin */

// Matches CPU mg_log2() from misc.h - polynomial approximation using IEEE 754 representation
__device__ static inline float cuda_mg_log2(float x) // NB: this doesn't work when x<2
{
	union {
		float f;
		uint32_t i;
	} z = {x};
	float log_2 = ((z.i >> 23) & 255) - 128;
	z.i &= ~(255 << 23);
	z.i += 127 << 23;
	log_2 += (-0.34484843f * z.f + 2.02466578f) * z.f - 0.67487759f;
	return log_2;
}

inline __device__ int32_t comput_sc(const int32_t ai_x, const int32_t ai_y, const int32_t aj_x, const int32_t aj_y,
    const int32_t sidi, const int32_t sidj,
    const int32_t q_span, // actual q_span from anchor, not hardcoded
    const int32_t max_dist_x, const int32_t max_dist_y,
    const int32_t bw, const float chn_pen_gap,
    const float chn_pen_skip, const int is_cdna, const int n_seg)
{
	// Match CPU logic from lchain.c:comput_sc exactly
	const int32_t dq = ai_y - aj_y;
	if (dq <= 0 || dq > max_dist_x) return INT32_MIN;
	const int32_t dr = ai_x - aj_x;

	// Cache repeated comparisons for better performance
	const bool same_seg = (sidi == sidj); // same segment ID
	const bool diff_seg = !same_seg;
	const bool dr_gt_dq = (dr > dq);
	const bool dr_eq_0 = (dr == 0);

	if (same_seg && (dr_eq_0 || dq > max_dist_y)) return INT32_MIN;
	const int32_t dd = dr_gt_dq ? dr - dq : dq - dr; // abs(dr - dq)
	if (same_seg && dd > bw) return INT32_MIN;
	if (n_seg > 1 && !is_cdna && same_seg && dr > max_dist_y) return INT32_MIN;

	const int32_t dg = dr < dq ? dr : dq;
	int32_t sc = q_span < dg ? q_span : dg;

	if (dd || dg > q_span) {
		float lin_pen, log_pen;
		lin_pen = chn_pen_gap * (float)dd + chn_pen_skip * (float)dg;
		log_pen = dd >= 1 ? cuda_mg_log2(dd + 1) : 0.0f; // Match CPU mg_log2
		if (is_cdna || diff_seg) {
			if (diff_seg && dr_eq_0)
				++sc; // minor bonus for overlapping paired ends
			else if (dr_gt_dq || diff_seg)
				sc -= (int)(lin_pen < log_pen ? lin_pen : log_pen); // deletion or jump
			else
				sc -= (int)(lin_pen + 0.5f * log_pen);
		} else {
			sc -= (int)(lin_pen + 0.5f * log_pen);
		}
	}
	return sc;
}


/* arithmetic functions end */

// Upper bound on the predecessor window (= max_chain_iter). The window-local
// t[] marker buffer lives in shared memory and is sized to this bound; larger
// --max-chain-iter values are clamped to it.
#define PULL_MAX_WIN 8192

// Wave size on gfx942.
#define PULL_WARP 64

// Warp-level inclusive scans (wave-synchronous, no __syncthreads). blockDim is
// always a multiple of PULL_WARP on the pull paths.
// Inclusive max with arg (walk pos k); on a value tie keep the smaller k. An
// arg < 0 marks "no element / invalid".
__device__ __forceinline__ void warp_incl_max(int32_t& v, int32_t& k)
{
	const int lane = threadIdx.x & (PULL_WARP - 1);
#pragma unroll
	for (int d = 1; d < PULL_WARP; d <<= 1) {
		const int32_t nv = __shfl_up(v, d);
		const int32_t nk = __shfl_up(k, d);
		if (lane >= d && (nv > v || (nv == v && nk >= 0 && (k < 0 || nk < k)))) {
			v = nv;
			k = nk;
		}
	}
}

// Fused inclusive scan over delta producing both the prefix sum P and the
// prefix-min of the prefix sum M (the (sum, min-prefix) monoid: combining left
// (sL,mL) before right (sR,mR) gives (sL+sR, min(mL, sL+mR))). Block totals are
// left in wss[(blockDim>>6)-1] (sum) and wsm[...] (min-prefix).
__device__ __forceinline__ void warp_incl_summin(int32_t& s, int32_t& m)
{
	const int lane = threadIdx.x & (PULL_WARP - 1);
#pragma unroll
	for (int d = 1; d < PULL_WARP; d <<= 1) {
		const int32_t ns = __shfl_up(s, d);
		const int32_t nm = __shfl_up(m, d);
		if (lane >= d) {
			const int32_t cand = ns + m;
			m = nm < cand ? nm : cand;
			s = ns + s;
		}
	}
}
__device__ __forceinline__ void block_incl_summin(int32_t& s, int32_t& m, int32_t* wss, int32_t* wsm)
{
	const int lane = threadIdx.x & (PULL_WARP - 1), wid = threadIdx.x >> 6;
	warp_incl_summin(s, m);
	if (lane == PULL_WARP - 1) {
		wss[wid] = s;
		wsm[wid] = m;
	}
	__syncthreads();
	if (wid == 0) {
		const int nw = blockDim.x >> 6;
		int32_t ts = (lane < nw) ? wss[lane] : 0;
		int32_t tm = (lane < nw) ? wsm[lane] : INT32_MAX;
		warp_incl_summin(ts, tm);
		if (lane < nw) {
			wss[lane] = ts;
			wsm[lane] = tm;
		}
	}
	__syncthreads();
	if (wid > 0) {
		const int32_t ps = wss[wid - 1], pm = wsm[wid - 1];
		const int32_t cand = ps + m;
		m = pm < cand ? pm : cand;
		s = ps + s;
	}
}
// Block min reduce, broadcast to all lanes via ws[0].
__device__ __forceinline__ int32_t block_reduce_min(int32_t v, int32_t* ws)
{
	const int lane = threadIdx.x & (PULL_WARP - 1), wid = threadIdx.x >> 6;
#pragma unroll
	for (int d = PULL_WARP >> 1; d >= 1; d >>= 1) {
		const int32_t n = __shfl_xor(v, d);
		if (n < v) v = n;
	}
	if (lane == 0) ws[wid] = v;
	__syncthreads();
	if (wid == 0) {
		const int nw = blockDim.x >> 6;
		int32_t t = (lane < nw) ? ws[lane] : INT32_MAX;
#pragma unroll
		for (int d = PULL_WARP >> 1; d >= 1; d >>= 1) {
			const int32_t n = __shfl_xor(t, d);
			if (n < t) t = n;
		}
		if (lane == 0) ws[0] = t;
	}
	__syncthreads();
	const int32_t r = ws[0];
	__syncthreads();
	return r;
}
// Block EXCLUSIVE max-with-arg: pmv/pmk = max over walk positions strictly
// before this thread (tie -> smaller k); totv/totk = inclusive block max. No
// full shared array needed (unlike the inclusive variant). No final barrier:
// the caller separates this from the next ws_* user with its own __syncthreads.
__device__ __forceinline__ void block_excl_max(int32_t v0, int32_t k0, int32_t& pmv, int32_t& pmk,
    int32_t& totv, int32_t& totk, int32_t* wsv, int32_t* wsi)
{
	const int lane = threadIdx.x & (PULL_WARP - 1), wid = threadIdx.x >> 6;
	int32_t iv = v0, ik = k0;
	warp_incl_max(iv, ik); // inclusive within warp
	int32_t ev = __shfl_up(iv, 1), ek = __shfl_up(ik, 1); // exclusive within warp
	if (lane == 0) {
		ev = INT32_MIN;
		ek = -1;
	}
	if (lane == PULL_WARP - 1) {
		wsv[wid] = iv;
		wsi[wid] = ik;
	}
	__syncthreads();
	if (wid == 0) {
		const int nw = blockDim.x >> 6;
		int32_t tv = (lane < nw) ? wsv[lane] : INT32_MIN, tk = (lane < nw) ? wsi[lane] : -1;
		warp_incl_max(tv, tk);
		if (lane < nw) {
			wsv[lane] = tv;
			wsi[lane] = tk;
		}
	}
	__syncthreads();
	const int32_t prv = (wid > 0) ? wsv[wid - 1] : INT32_MIN, prk = (wid > 0) ? wsi[wid - 1] : -1;
	pmv = prv;
	pmk = prk;
	if (ev > pmv || (ev == pmv && ek >= 0 && (pmk < 0 || ek < pmk))) {
		pmv = ev;
		pmk = ek;
	}
	const int nw = blockDim.x >> 6;
	totv = wsv[nw - 1];
	totk = wsi[nw - 1];
}
// Block reduce: max value with LARGEST-index tie-break (matches CPU max_ii
// recompute), broadcast via wsv[0]/wsi[0].
__device__ __forceinline__ void block_reduce_argmax(int32_t v, int32_t idx, int32_t& ov, int32_t& oi,
    int32_t* wsv, int32_t* wsi)
{
	const int lane = threadIdx.x & (PULL_WARP - 1), wid = threadIdx.x >> 6;
#pragma unroll
	for (int d = PULL_WARP >> 1; d >= 1; d >>= 1) {
		const int32_t nv = __shfl_xor(v, d), ni = __shfl_xor(idx, d);
		if (nv > v || (nv == v && ni > idx)) {
			v = nv;
			idx = ni;
		}
	}
	if (lane == 0) {
		wsv[wid] = v;
		wsi[wid] = idx;
	}
	__syncthreads();
	if (wid == 0) {
		const int nw = blockDim.x >> 6;
		int32_t tv = (lane < nw) ? wsv[lane] : INT32_MIN, ti = (lane < nw) ? wsi[lane] : -1;
#pragma unroll
		for (int d = PULL_WARP >> 1; d >= 1; d >>= 1) {
			const int32_t nv = __shfl_xor(tv, d), ni = __shfl_xor(ti, d);
			if (nv > tv || (nv == tv && ni > ti)) {
				tv = nv;
				ti = ni;
			}
		}
		if (lane == 0) {
			wsv[0] = tv;
			wsi[0] = ti;
		}
	}
	__syncthreads();
	ov = wsv[0];
	oi = wsi[0];
	__syncthreads();
}

// Per-warp marker window for the warp-per-segment kernel. Caps the predecessor
// window; segments whose window could exceed this are handled by the block
// kernel instead. Must equal PULL_MAX_WIN so warp-routed and block-routed
// segments clamp --max-chain-iter identically (otherwise chain composition
// would depend on segment size for max_iter in (old cap, PULL_MAX_WIN]).
#define WARP_MARK_WIN PULL_MAX_WIN

// Segments larger than this (anchor count) stay on the block-per-segment kernel
// (one such giant would stall a single warp); smaller long segments run on the
// high-occupancy warp-per-segment kernel.
#define GIANT_CUT 16384

// Warp-per-segment pull DP. One 64-lane warp owns one segment and runs the
// CPU-faithful backward scan entirely wave-synchronously (warp shuffles, no
// __syncthreads), so many warps in a block process independent segments
// concurrently -> high occupancy. markbuf is this warp's private window buffer.
// Sequential per-target state (st, carries, max_ii) is kept redundantly in every
// lane's registers; the prefix scans are warp shuffles whose results all lanes
// see. f[i]/p[i] are written by all lanes (identical value).
inline __device__ void compute_sc_seg_pull_warp(const int32_t* anchors_x, const int32_t* anchors_y,
    const int8_t* sid, const uint8_t* qspan, const int32_t* xrev, uint8_t* markbuf,
    const size_t start_idx, const size_t end_idx, int32_t* f, uint16_t* p, Misc blk_misc)
{
	const int lane = threadIdx.x & (PULL_WARP - 1);
	const int32_t max_skip = blk_misc.max_skip;
	int64_t max_iter = blk_misc.max_iter;
	if (max_iter > WARP_MARK_WIN) max_iter = WARP_MARK_WIN;
	const int32_t max_dist_x = blk_misc.max_dist_x;

	for (size_t i = start_idx + lane; i < end_idx; i += PULL_WARP) {
		f[i] = qspan[i];
		p[i] = 0;
	}

	int64_t st = (int64_t)start_idx;
	int64_t max_ii = -1;
	for (size_t ii = start_idx; ii < end_idx; ii++) {
		const int64_t i = (int64_t)ii;
		const int32_t ax_i = anchors_x[i];
		const int32_t ay_i = anchors_y[i];
		const int32_t sid_i = sid[i];
		while (st < i && (xrev[i] != xrev[st] || ax_i - anchors_x[st] > max_dist_x)) ++st;
		if (i - st > max_iter) st = i - max_iter;
		const int64_t win = i - st;

		for (int64_t r = lane; r < win; r += PULL_WARP) markbuf[r] = 0;

		int32_t cmax = qspan[i], cmaxk = -1, cQ = 0, cminQ = 0;
		int32_t r_max_f = qspan[i], r_max_j = -1;
		int64_t r_end_j = st - 1;
		bool broke = false;
		for (int64_t base = i - 1, chunkStartK = 0; base >= st; base -= PULL_WARP, chunkStartK += PULL_WARP) {
			const int64_t j = base - lane;
			const bool inwin = (j >= st);
			int32_t sc = INT32_MIN, fj = 0;
			if (inwin) {
				sc = comput_sc(ax_i, ay_i, anchors_x[j], anchors_y[j], sid_i, sid[j], qspan[j],
				    max_dist_x, blk_misc.max_dist_y, blk_misc.bw, blk_misc.chn_pen_gap,
				    blk_misc.chn_pen_skip, blk_misc.is_cdna, blk_misc.n_seg);
				fj = f[j];
			}
			const bool valid = inwin && (sc != INT32_MIN);
			const int32_t s_val = valid ? sc + fj : INT32_MIN;
			const int32_t kpos = (int32_t)(chunkStartK + lane);

			if (valid) {
				const uint16_t prel = p[j];
				if (prel != 0) {
					const int64_t jp = j - (int64_t)prel;
					if (jp >= st) markbuf[jp - st] = 1;
				}
			}

			// exclusive prefix-max of s_val within the warp (tie -> smaller k).
			int32_t iv = s_val, ik = valid ? kpos : -1;
			warp_incl_max(iv, ik);
			int32_t ev = __shfl_up(iv, 1), ek = __shfl_up(ik, 1);
			if (lane == 0) {
				ev = INT32_MIN;
				ek = -1;
			}
			int32_t pm_v = cmax, pm_k = cmaxk;
			if (ev > pm_v) {
				pm_v = ev;
				pm_k = ek;
			}
			const int32_t chunk_max_v = __shfl(iv, PULL_WARP - 1);
			const int32_t chunk_max_k = __shfl(ik, PULL_WARP - 1);

			const bool improving = valid && (s_val > pm_v);
			const int32_t marked = inwin ? markbuf[j - st] : 0;
			const int32_t delta = improving ? -1 : ((valid && marked) ? 1 : 0);

			int32_t P = delta, M = delta;
			warp_incl_summin(P, M);
			const int32_t Q = cQ + P;
			const int32_t mp = cQ + M;
			const int32_t runminQ = cminQ < mp ? cminQ : mp;
			const int32_t c_k = Q - runminQ;
			const int32_t chunk_sum = cQ + __shfl(P, PULL_WARP - 1);
			const int32_t totMp = cQ + __shfl(M, PULL_WARP - 1);
			const int32_t chunk_minQ = cminQ < totMp ? cminQ : totMp;

			int32_t cand = (delta == 1 && c_k > max_skip) ? kpos : INT32_MAX;
#pragma unroll
			for (int d = PULL_WARP >> 1; d >= 1; d >>= 1) {
				const int32_t n = __shfl_xor(cand, d);
				if (n < cand) cand = n;
			}
			if (cand != INT32_MAX) {
				// break at walk pos `cand`; max_f/max_j come from the pm of that lane.
				int32_t bf = (kpos == cand) ? pm_v : INT32_MIN;
				int32_t bj = (kpos == cand) ? ((pm_k >= 0) ? (int32_t)(i - 1 - pm_k) : -1) : INT32_MIN;
#pragma unroll
				for (int d = PULL_WARP >> 1; d >= 1; d >>= 1) {
					const int32_t nf = __shfl_xor(bf, d), nj = __shfl_xor(bj, d);
					if (nf > bf) {
						bf = nf;
						bj = nj;
					}
				}
				r_max_f = bf;
				r_max_j = bj;
				r_end_j = i - 1 - cand;
				broke = true;
				break;
			}
			if (chunk_max_v > cmax) {
				cmax = chunk_max_v;
				cmaxk = chunk_max_k;
			}
			cQ = chunk_sum;
			cminQ = chunk_minQ;
		}
		if (!broke) {
			r_max_f = cmax;
			r_max_j = (cmaxk >= 0) ? (int32_t)(i - 1 - cmaxk) : -1;
		}

		// Lazy max_ii maintenance + far-link recovery (matches CPU).
		if (max_ii < 0 || xrev[i] != xrev[max_ii] || ax_i - anchors_x[max_ii] > max_dist_x) {
			int32_t lv = INT32_MIN, li = -1;
			for (int64_t jj = st + lane; jj < i; jj += PULL_WARP) {
				if (f[jj] > lv || (f[jj] == lv && (int32_t)jj > li)) {
					lv = f[jj];
					li = (int32_t)jj;
				}
			}
#pragma unroll
			for (int d = PULL_WARP >> 1; d >= 1; d >>= 1) {
				const int32_t nv = __shfl_xor(lv, d), ni = __shfl_xor(li, d);
				if (nv > lv || (nv == lv && ni > li)) {
					lv = nv;
					li = ni;
				}
			}
			max_ii = li;
		}
		if (max_ii >= 0 && max_ii < r_end_j) {
			const int32_t tmp = comput_sc(ax_i, ay_i, anchors_x[max_ii], anchors_y[max_ii], sid_i, sid[max_ii], qspan[max_ii],
			    max_dist_x, blk_misc.max_dist_y, blk_misc.bw, blk_misc.chn_pen_gap,
			    blk_misc.chn_pen_skip, blk_misc.is_cdna, blk_misc.n_seg);
			if (tmp != INT32_MIN && r_max_f < tmp + f[max_ii]) {
				r_max_f = tmp + f[max_ii];
				r_max_j = (int32_t)max_ii;
			}
		}
		// All lanes hold identical r_max_f/r_max_j; redundant write guarantees the
		// store is visible to this warp's later f[] reads.
		f[i] = r_max_f;
		p[i] = r_max_j >= 0 ? (uint16_t)(i - (int64_t)r_max_j) : 0;
		if (max_ii < 0 || (xrev[i] == xrev[max_ii] && ax_i - anchors_x[max_ii] <= max_dist_x && f[max_ii] < r_max_f))
			max_ii = i;
	}
}

// Pull-model chaining DP (AIOSS-3664). Replicates the CPU mg_lchain_dp backward
// predecessor scan with the max_skip early-stop, the t[] "already on a chain"
// marker, and the max_ii far-link recovery, so GPU output matches the CPU at a
// finite --max-chain-skip (the forward-push kernels only reproduce the infinity
// case). One target anchor is processed at a time by the whole block: threads
// evaluate comput_sc over a chunk of predecessors in parallel, scatter the t[]
// markers into a shared window buffer, and lane 0 applies the order-dependent
// skip/marker logic reading only shared memory. The forward `range` buffer is
// unused on this path (the window is derived from anchor positions here).
inline __device__ void compute_sc_seg_pull(const int32_t* anchors_x, const int32_t* anchors_y,
    const int8_t* sid, const uint8_t* qspan, const int32_t* xrev, int32_t* t_scratch,
    const size_t start_idx, const size_t end_idx,
    int32_t* f, uint16_t* p, Misc blk_misc)
{
	const int tid = threadIdx.x;
	const int32_t max_skip = blk_misc.max_skip;
	int64_t max_iter = blk_misc.max_iter;
	if (max_iter > PULL_MAX_WIN) max_iter = PULL_MAX_WIN; // bound by shared markbuf
	const int32_t max_dist_x = blk_misc.max_dist_x;

	// Per-segment init of f[] and p[].
	for (size_t i = start_idx + tid; i < end_idx; i += blockDim.x) {
		f[i] = qspan[i];
		p[i] = 0;
	}
	__syncthreads();

	__shared__ int32_t ws_a[32], ws_b[32], ws_c[32]; // warp-scan scratch (<= blockDim/64 warps)
	__shared__ uint8_t markbuf[PULL_MAX_WIN]; // window-local t[] markers (LDS, not global)
	__shared__ int64_t sh_st;
	__shared__ int64_t sh_max_ii;
	__shared__ int32_t r_max_f, r_max_j, r_broke, r_recompute;
	__shared__ int64_t r_end_j;
	// Carries threaded across chunks of the backward scan (walk order).
	__shared__ int32_t sh_cmax, sh_cmaxk; // running max of (f[j]+sc) and its walk pos
	__shared__ int32_t sh_cQ, sh_cminQ; // delta prefix sum and its running min (>= floor 0)

	if (tid == 0) {
		sh_st = (int64_t)start_idx;
		sh_max_ii = -1;
	}
	__syncthreads();

	for (size_t ii = start_idx; ii < end_idx; ii++) {
		const int64_t i = (int64_t)ii;
		const int32_t ax_i = anchors_x[i];
		const int32_t ay_i = anchors_y[i];
		const int32_t sid_i = sid[i];

		// Advance the window lower bound st (monotonic) and cap by max_iter.
		// The rid/strand prefix check (xrev) is required because a GPU segment
		// can span multiple references (the range kernel only cuts at some
		// rid boundaries); without it cross-rid anchors with coincidentally
		// close low-32 positions would be admitted as predecessors.
		if (tid == 0) {
			int64_t st = sh_st;
			while (st < i && (xrev[i] != xrev[st] || ax_i - anchors_x[st] > max_dist_x)) ++st;
			if (i - st > max_iter) st = i - max_iter;
			sh_st = st;
			r_max_f = qspan[i];
			r_max_j = -1;
			r_broke = 0;
			r_end_j = st - 1; // value of j when the scan completes the whole window
			sh_cmax = qspan[i]; // running max seeded with the anchor's own score
			sh_cmaxk = -1; // walk pos of the running max (-1 = no predecessor)
			sh_cQ = 0;
			sh_cminQ = 0;
		}
		__syncthreads();
		const int64_t st = sh_st;
		const int64_t win = i - st;

		// Clear the window-local marker buffer (only [0, win)).
		for (int64_t r = tid; r < win; r += blockDim.x) markbuf[r] = 0;
		__syncthreads();

		// Backward predecessor scan, j = i-1 .. st, in chunks of blockDim (walk
		// order = increasing tid). The CPU n_skip/break recurrence is replaced
		// by parallel scans: prefix-max of (f[j]+sc) gives the "improving" flag;
		// the clamped skip counter n_skip(k) = Q_k - min(0, min_{m<=k} Q_m) is a
		// prefix-sum (Q) plus a prefix-min; the break is the first marked,
		// non-improving position whose counter exceeds max_skip. Carries thread
		// the running max / Q / min(Q) across chunks; ~94% of targets break in
		// the first chunk so later chunks rarely run.
		for (int64_t base = i - 1, chunkStartK = 0; base >= st;
		     base -= blockDim.x, chunkStartK += blockDim.x) {
			const int64_t j = base - tid;
			const bool inwin = (j >= st);
			int32_t sc = INT32_MIN, fj = 0;
			if (inwin) {
				sc = comput_sc(ax_i, ay_i, anchors_x[j], anchors_y[j], sid_i, sid[j], qspan[j],
				    max_dist_x, blk_misc.max_dist_y, blk_misc.bw, blk_misc.chn_pen_gap,
				    blk_misc.chn_pen_skip, blk_misc.is_cdna, blk_misc.n_seg);
				fj = f[j];
			}
			const bool valid = inwin && (sc != INT32_MIN);
			const int32_t s_val = valid ? sc + fj : INT32_MIN;
			const int32_t kpos = (int32_t)(chunkStartK + tid); // walk position

			// Scatter markers: CPU sets t[p[j]]=i for every VALID-sc link. Only
			// p[j] >= st can be re-read in this window; marks at visited
			// positions are never spurious (a smaller-index j never points up).
			if (valid) {
				const uint16_t prel = p[j];
				if (prel != 0) {
					const int64_t jp = j - (int64_t)prel;
					if (jp >= st) markbuf[jp - st] = 1;
				}
			}

			// (1) exclusive prefix-max of s_val over the chunk (tie -> smaller k).
			int32_t pm_cv, pm_ck, chunk_max_v, chunk_max_k;
			block_excl_max(s_val, valid ? kpos : -1, pm_cv, pm_ck, chunk_max_v, chunk_max_k, ws_a, ws_b);
			// pm = max strictly before this position, incl. carry. On a value tie
			// keep the carry: it is earlier in walk order (smaller k), and CPU
			// only updates the argmax on a strict improvement.
			int32_t pm_v = sh_cmax, pm_k = sh_cmaxk;
			if (pm_cv > pm_v) {
				pm_v = pm_cv;
				pm_k = pm_ck;
			}
			__syncthreads();

			const bool improving = valid && (s_val > pm_v);
			const int32_t marked = inwin ? markbuf[j - st] : 0;
			const int32_t delta = improving ? -1 : ((valid && marked) ? 1 : 0);

			// (2+3) fused prefix-sum (Q) and prefix-min-of-prefix-sum in one scan.
			int32_t P = delta, M = delta;
			block_incl_summin(P, M, ws_a, ws_b);
			const int32_t nw = blockDim.x >> 6;
			const int32_t Q = sh_cQ + P;
			const int32_t mp = sh_cQ + M; // min prefix of Q over [start..k]
			const int32_t runminQ = sh_cminQ < mp ? sh_cminQ : mp;
			const int32_t chunk_sum = sh_cQ + ws_a[nw - 1];
			const int32_t totMp = sh_cQ + ws_b[nw - 1];
			const int32_t chunk_minQ = sh_cminQ < totMp ? sh_cminQ : totMp;
			const int32_t c_k = Q - runminQ; // clamped n_skip at this position

			// (4) break = first (smallest-k) marked non-improving pos over budget.
			const bool cand = (delta == 1) && (c_k > max_skip);
			const int32_t bk = block_reduce_min(cand ? kpos : INT32_MAX, ws_c);
			if (bk != INT32_MAX) {
				if (kpos == bk) {
					r_max_f = pm_v;
					r_max_j = (pm_k >= 0) ? (int32_t)(i - 1 - pm_k) : -1;
					r_end_j = i - 1 - bk;
					r_broke = 1;
				}
				__syncthreads();
				break;
			}
			// No break: advance carries to the next (farther) chunk. On a value
			// tie keep the existing carry (earlier walk order = smaller k).
			if (tid == 0) {
				if (chunk_max_v > sh_cmax) {
					sh_cmax = chunk_max_v;
					sh_cmaxk = chunk_max_k;
				}
				sh_cQ = chunk_sum;
				sh_cminQ = chunk_minQ;
			}
			__syncthreads();
		}
		// No break anywhere: max_f is the full-window running max.
		if (tid == 0 && !r_broke) {
			r_max_f = sh_cmax;
			r_max_j = (sh_cmaxk >= 0) ? (int32_t)(i - 1 - sh_cmaxk) : -1;
		}
		__syncthreads();

		// Far-link recovery using CPU-faithful lazy max_ii maintenance: max_ii
		// is carried across the i-loop and recomputed ONLY when it leaves the
		// max_dist_x window (matches mg_lchain_dp). Recomputing it every i would
		// always find the true windowed argmax, which is MORE optimal than CPU
		// and therefore diverges on near-tie reads.
		if (tid == 0)
			r_recompute = (sh_max_ii < 0 || xrev[i] != xrev[sh_max_ii] || ax_i - anchors_x[sh_max_ii] > max_dist_x)
			    ? 1
			    : 0;
		__syncthreads();
		if (r_recompute) {
			// argmax f over [st, i); CPU scans descending with strict '<', i.e.
			// keeps the LARGEST index among ties.
			int32_t local_v = INT32_MIN, local_i = -1;
			for (int64_t j = st + tid; j < i; j += blockDim.x) {
				if (f[j] > local_v || (f[j] == local_v && (int32_t)j > local_i)) {
					local_v = f[j];
					local_i = (int32_t)j;
				}
			}
			int32_t mv, mi;
			block_reduce_argmax(local_v, local_i, mv, mi, ws_a, ws_b);
			if (tid == 0) sh_max_ii = mi;
			__syncthreads();
		}
		if (tid == 0) {
			const int64_t max_ii = sh_max_ii;
			if (max_ii >= 0 && max_ii < r_end_j) {
				const int32_t tmp = comput_sc(ax_i, ay_i, anchors_x[max_ii], anchors_y[max_ii], sid_i, sid[max_ii], qspan[max_ii],
				    max_dist_x, blk_misc.max_dist_y, blk_misc.bw, blk_misc.chn_pen_gap,
				    blk_misc.chn_pen_skip, blk_misc.is_cdna, blk_misc.n_seg);
				if (tmp != INT32_MIN && r_max_f < tmp + f[max_ii]) {
					r_max_f = tmp + f[max_ii];
					r_max_j = max_ii;
				}
			}
			f[i] = r_max_f;
			p[i] = r_max_j >= 0 ? (uint16_t)(i - (int64_t)r_max_j) : 0;
			// End-of-iteration max_ii update (matches CPU).
			if (sh_max_ii < 0 || (xrev[i] == xrev[sh_max_ii] && ax_i - anchors_x[sh_max_ii] <= max_dist_x && f[sh_max_ii] < f[i]))
				sh_max_ii = i;
		}
		__syncthreads();
	}
}

inline __device__ void compute_sc_seg_one_wf(const int32_t* anchors_x, const int32_t* anchors_y, const int8_t* sid, const uint8_t* qspan, const int32_t* xrev, const int32_t* range,
    const size_t start_idx, const size_t end_idx,
    int32_t* f, uint16_t* p, Misc blk_misc)
{
	// Finite max_chain_skip: use the CPU-faithful pull-model DP (the forward
	// push below only reproduces max_chain_skip=infinity). range[] is unused
	// here and serves as the t[] scratch. max_skip==0 is a finite (most
	// aggressive) early-stop, so route it to the pull path too.
	if (blk_misc.max_skip >= 0 && blk_misc.max_skip < INT32_MAX) {
		compute_sc_seg_pull(anchors_x, anchors_y, sid, qspan, xrev, (int32_t*)range, start_idx, end_idx, f, p, blk_misc);
		return;
	}
	int tid = threadIdx.x;
	// init f and p using actual q_span values from anchors
	for (size_t i = start_idx + tid; i < end_idx; i += blockDim.x) {
		f[i] = qspan[i]; // Use actual q_span instead of MM_QSPAN
		p[i] = 0;
	}
#ifndef USEHIP
	__syncwarp(); // NOTE: single warp, no need to sync
#endif // USEHIP
	for (size_t i = start_idx; i < end_idx; i++) {
		int32_t range_i = range[i];

		const int32_t q_span_i = qspan[i];
		const int32_t anchors_x_i = anchors_x[i];
		const int32_t anchors_y_i = anchors_y[i];
		const int32_t sid_i = sid[i];
		const int32_t f_i = f[i];

		for (int32_t j = tid; j < range_i; j += blockDim.x) {
			// Compute i+j+1 once per iteration
			const size_t idx_ij1 = i + j + 1;

			int32_t sc = comput_sc(
			    anchors_x[idx_ij1],
			    anchors_y[idx_ij1],
			    anchors_x_i,
			    anchors_y_i,
			    sid[idx_ij1],
			    sid_i,
			    q_span_i,
			    blk_misc.max_dist_x, blk_misc.max_dist_y, blk_misc.bw, blk_misc.chn_pen_gap,
			    blk_misc.chn_pen_skip, blk_misc.is_cdna, blk_misc.n_seg);
			if (sc == INT32_MIN) continue;
			sc += f_i;
			int32_t init_score = qspan[idx_ij1];
			if (sc >= f[idx_ij1] && sc != init_score) {
				f[idx_ij1] = sc;
				p[idx_ij1] = j + 1;
			}
		}
#ifndef USEHIP
		__syncwarp(); // NOTE: single warp, no need to sync
#endif // USEHIP
	}
}


inline __device__ void compute_sc_seg_multi_wf(const int32_t* anchors_x, const int32_t* anchors_y, const int8_t* sid, const uint8_t* qspan, const int32_t* xrev, const int32_t* range,
    const size_t start_idx, const size_t end_idx,
    int32_t* f, uint16_t* p, Misc blk_misc)
{
	// Finite max_chain_skip: use the CPU-faithful pull-model DP (see
	// compute_sc_seg_one_wf). range[] is unused here and serves as t[] scratch.
	// max_skip==0 is a finite early-stop, so route it to the pull path too.
	if (blk_misc.max_skip >= 0 && blk_misc.max_skip < INT32_MAX) {
		compute_sc_seg_pull(anchors_x, anchors_y, sid, qspan, xrev, (int32_t*)range, start_idx, end_idx, f, p, blk_misc);
		return;
	}
	int tid = threadIdx.x;
	// init f and p using actual q_span values from anchors
	for (size_t i = start_idx + tid; i < end_idx; i += blockDim.x) {
		f[i] = qspan[i]; // Use actual q_span instead of MM_QSPAN
		p[i] = 0;
	}
	__syncthreads();
	for (size_t i = start_idx; i < end_idx; i++) {
		int32_t range_i = range[i];

		// Hoist invariant loads outside inner loop
		const int32_t q_span_i = qspan[i];
		const int32_t anchors_x_i = anchors_x[i];
		const int32_t anchors_y_i = anchors_y[i];
		const int32_t sid_i = sid[i];
		const int32_t f_i = f[i];

		for (int32_t j = tid; j < range_i; j += blockDim.x) {
			// Compute i+j+1 once per iteration
			const size_t idx_ij1 = i + j + 1;

			int32_t sc = comput_sc(
			    anchors_x[idx_ij1],
			    anchors_y[idx_ij1],
			    anchors_x_i,
			    anchors_y_i,
			    sid[idx_ij1],
			    sid_i,
			    q_span_i,
			    blk_misc.max_dist_x, blk_misc.max_dist_y, blk_misc.bw, blk_misc.chn_pen_gap,
			    blk_misc.chn_pen_skip, blk_misc.is_cdna, blk_misc.n_seg);
			if (sc == INT32_MIN) continue;
			sc += f_i;
			int32_t init_score = qspan[idx_ij1];
			if (sc >= f[idx_ij1] && sc != init_score) {
				f[idx_ij1] = sc;
				p[idx_ij1] = j + 1;
			}
		}
		__syncthreads();
	}
}

#define NUM_ANCHORS_PREFETCH 1024

// Forward-push DP for max_chain_skip = infinity (no early-stop). Each source i
// (processed sequentially by the block) pushes its score to every successor in
// range[i] in parallel. Standalone (no pull-path shared memory), so the kernel
// that calls it gets minimal LDS and high occupancy. Bit-exact with the CPU
// infinity case.
inline __device__ void compute_sc_seg_push(const int32_t* anchors_x, const int32_t* anchors_y,
    const int8_t* sid, const uint8_t* qspan, const int32_t* range,
    const size_t start_idx, const size_t end_idx, int32_t* f, uint16_t* p, Misc blk_misc)
{
	const int tid = threadIdx.x;
	for (size_t i = start_idx + tid; i < end_idx; i += blockDim.x) {
		f[i] = qspan[i];
		p[i] = 0;
	}
	__syncthreads();
	for (size_t i = start_idx; i < end_idx; i++) {
		const int32_t range_i = range[i];
		const int32_t q_span_i = qspan[i];
		const int32_t anchors_x_i = anchors_x[i];
		const int32_t anchors_y_i = anchors_y[i];
		const int32_t sid_i = sid[i];
		const int32_t f_i = f[i];
		for (int32_t j = tid; j < range_i; j += blockDim.x) {
			const size_t idx_ij1 = i + j + 1;
			int32_t sc = comput_sc(anchors_x[idx_ij1], anchors_y[idx_ij1], anchors_x_i, anchors_y_i,
			    sid[idx_ij1], sid_i, q_span_i, blk_misc.max_dist_x, blk_misc.max_dist_y, blk_misc.bw,
			    blk_misc.chn_pen_gap, blk_misc.chn_pen_skip, blk_misc.is_cdna, blk_misc.n_seg);
			if (sc == INT32_MIN) continue;
			sc += f_i;
			const int32_t init_score = qspan[idx_ij1];
			if (sc >= f[idx_ij1] && sc != init_score) {
				f[idx_ij1] = sc;
				p[idx_ij1] = j + 1;
			}
		}
		__syncthreads();
	}
}

// inline __device__ void compute_sc_seg_shared(const int64_t* anchors_x, const int64_t* anchors_y, int32_t* range,
//                     size_t start_idx, size_t end_idx,
//                     int32_t* f, uint16_t* p
// ){
//     Misc blk_misc = misc;
//     int tid = threadIdx.x;
//     int bid = blockIdx.x;
//     // init f and p
//     for (size_t i=start_idx+tid; i < end_idx; i += blockDim.x) {
//         f[i] = anchors_y[i] >> 32 & 0xff;
//         p[i] = 0;
//     }
//     __syncthreads();
//     // assert(range[end_idx-1] == 0);
//     __shared__ int64_t anchors_x_shared[NUM_ANCHORS_PREFETCH];

//     __shared__ int64_t anchors_y_shared[NUM_ANCHORS_PREFETCH];
//     size_t prefetch_end_idx = 0;
//     unsigned int prefetch_smem_offset = 0;
//     for (size_t i = start_idx; i < end_idx; i++) {
//         int32_t range_i = range[i];
//         // if (range_i + i >= end_idx)
//         //     printf("range_i %d i %lu start_idx %lu, end_idx %lu\n", range_i, i, start_idx, end_idx);
//         // assert(range_i + i < end_idx);
//         for (int32_t j = tid; j < range_i; j += blockDim.x) {
//             int32_t sc = comput_sc(
//                                 anchors_x[i+j+1],
//                                 anchors_y[i+j+1],
//                                 anchors_x[i],
//                                 anchors_y[i],
//                                 blk_misc.max_dist_x, blk_misc.max_dist_y, blk_misc.bw, blk_misc.chn_pen_gap,
//                                 blk_misc.chn_pen_skip, blk_misc.is_cdna, blk_misc.n_seg);
//             if (sc == INT32_MIN) continue;
//             sc += f[i];
//             if (sc >= f[i+j+1] && sc != (anchors_y[i+j+1]>>32 & 0xff)) {
//                 f[i+j+1] = sc;
//                 p[i+j+1] = j+1;

//             }
//         }
//         __syncthreads();
//     }
// }

// inline __device__ void compute_sc_long_seg_one_wf(const int64_t* anchors_x, const int64_t* anchors_y, int32_t* range,
//                     size_t start_idx, size_t end_idx,
//                     int32_t* f, uint16_t* p
// ){
//     Misc blk_misc = misc;
//     int tid = threadIdx.x;
//     // int bid = blockIdx.x;
//     // NOTE: smallest alignd offset that is greater than start_idx
//     //      anchor_offset = tid;
//     //      while (anchor_offset <= start_idx) anchor_offset += blockDim.x;
//     int anchor_offset = tid + (start_idx - tid + blockDim.x) / blockDim.x * blockDim.x;
//     // init f and p
//     for (size_t i=anchor_offset; i < end_idx; i += blockDim.x) {
//         f[i] = anchors_y[i] >> 32 & 0xff;
//         p[i] = 0;
//     }
//     // int64_t local_anchors[10];
//     int64_t anchor_x = anchors_x[anchor_offset];
//     int64_t anchor_y = anchors_y[anchor_offset];
//     __syncthreads();
//     // assert(range[end_idx-1] == 0);
//     for (size_t i=start_idx; i < end_idx; i++) {
//         int32_t range_i = range[i];
//         // if (range_i + i >= end_idx)
//         //     printf("range_i %d i %lu start_idx %lu, end_idx %lu\n", range_i, i, start_idx, end_idx);
//         // assert(range_i + i < end_idx);
//         // for (int32_t j = tid; j < range_i; j += blockDim.x) {
//         for (unsigned j = anchor_offset; j < i+range_i+1; j += blockDim.x) {
//             anchor_x = anchors_x[j];
//             anchor_y = anchors_y[j];
//             int32_t sc = comput_sc(
//                                 anchor_x,
//                                 anchor_y,
//                                 anchors_x[i],
//                                 anchors_y[i],
//                                 blk_misc.max_dist_x, blk_misc.max_dist_y, blk_misc.bw, blk_misc.chn_pen_gap,
//                                 blk_misc.chn_pen_skip, blk_misc.is_cdna, blk_misc.n_seg);
//             if (sc == INT32_MIN) continue;
//             sc += f[i];
//             if (sc >= f[j] && sc != (anchors_y[j]>>32 & 0xff)) {
//                 f[j] = sc;
//                 p[j] = j+1;
//             }
//         }
//         anchor_offset += (anchor_offset <= i+1) * blockDim.x; // update anchor offset
//         __syncthreads();
//     }

// }


/* kernels begin */


template <size_t short_block_size>
__launch_bounds__(short_block_size)
    __global__ void score_generation_short(
	/* Input: Anchor & Range Inputs */
	int32_t* anchors_x, int32_t* anchors_y, int8_t* sid, uint8_t* qspan, int32_t* xrev, int32_t* range,
	/* Input: Segmentations */
	size_t* seg_start_arr,
	/* Output: Score and Previous Anchor */
	int32_t* f, uint16_t* p,
	/* Sizes*/
	size_t total_n, size_t seg_count,
	/* Output: Long segs */
	int32_t* a_x_long, int32_t* a_y_long, int8_t* sid_long, uint8_t* qspan_long, int32_t* xrev_long, int32_t* range_long, /* aggregated memory space for long seg */
	size_t* total_n_long, size_t buffer_size_long, seg_t* long_seg, seg_t* long_seg_og, unsigned int* long_seg_count, seg_t* mid_seg, unsigned int* mid_seg_count,
	Misc misc, int long_seg_cutoff, int mid_seg_cutoff)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	size_t long_seg_start_idx;
#ifndef USEHIP
	__shared__ size_t long_seg_start_idx_shared;
#endif

	for (int segid = bid; segid < seg_count; segid += gridDim.x) {
		size_t start_idx = seg_start_arr[segid];
		if (start_idx == SIZE_MAX) continue; // start at a failed cut: continue to next iteration
		size_t end_idx = SIZE_MAX;
		int end_segid = segid + 1;
		while (true) {
			if (end_segid >= seg_count) {
				end_idx = total_n;
				break;
			}
			if (seg_start_arr[end_segid] != SIZE_MAX) {
				end_idx = seg_start_arr[end_segid];
				break;
			}
			++end_segid;
		}
		if (end_segid > segid + long_seg_cutoff) {
			if (tid == 0) {
				/* Allocate space in long seg buffer */
				long_seg_start_idx = atomicAdd((unsigned long long int*)total_n_long, (unsigned long long int)end_idx - start_idx);
				if (long_seg_start_idx + (end_idx - start_idx) >= buffer_size_long) { // long segement buffer is full
					/* rollback total_n_long */
#ifdef USEHIP
					atomicSub((unsigned long long int*)total_n_long, (unsigned long long int)end_idx - start_idx);
#else // CUDA. CUDA does not support atomicSub for unsigned long long int.
					atomicAdd((unsigned long long int*)total_n_long, (unsigned long long int)(start_idx - end_idx));

#endif // USEHIP
					long_seg_start_idx = SIZE_MAX;
					// fallback to mid kernel
					int mid_seg_idx = atomicAdd((unsigned long long int*)mid_seg_count, 1);
					mid_seg[mid_seg_idx].start_idx = start_idx;
					mid_seg[mid_seg_idx].end_idx = end_idx;
				} else {
					int long_seg_idx = atomicAdd((unsigned long long int*)long_seg_count, 1);
					long_seg[long_seg_idx].start_idx = long_seg_start_idx;
					long_seg[long_seg_idx].end_idx = long_seg_start_idx + (end_idx - start_idx);
					long_seg_og[long_seg_idx].start_idx = start_idx;
					long_seg_og[long_seg_idx].end_idx = end_idx;
//DEBUG: used for debug plchain_cal_long_seg_range_dis LONG_SEG_RANGE_DIS
#ifdef DEBUG_VERBOSE
					long_seg_og[long_seg_idx].start_segid = segid;
					long_seg_og[long_seg_idx].end_segid = end_segid;
#endif // DEBUG_VERBOSE
				}
			}
			// broadcast long_seg_start_idx to all scalar registers
#ifdef USEHIP
			long_seg_start_idx = __builtin_amdgcn_readfirstlane(long_seg_start_idx);
#else
			if (tid == 0) long_seg_start_idx_shared = long_seg_start_idx;
			__syncwarp();
			long_seg_start_idx = long_seg_start_idx_shared;
			__syncwarp();
#endif
			if (long_seg_start_idx == SIZE_MAX)
				continue; // failed to allocate long_seg buffer
			for (uint64_t idx = tid; idx < end_idx - start_idx; idx += blockDim.x) {
				a_x_long[long_seg_start_idx + idx] = anchors_x[start_idx + idx];
				a_y_long[long_seg_start_idx + idx] = anchors_y[start_idx + idx];
				sid_long[long_seg_start_idx + idx] = sid[start_idx + idx];
				qspan_long[long_seg_start_idx + idx] = qspan[start_idx + idx];
				xrev_long[long_seg_start_idx + idx] = xrev[start_idx + idx];
				range_long[long_seg_start_idx + idx] = range[start_idx + idx];
			}
			continue;
		} else if (end_segid > segid + mid_seg_cutoff) {
			if (tid == 0) {
				int mid_seg_idx = atomicAdd(mid_seg_count, 1);
				mid_seg[mid_seg_idx].start_idx = start_idx;
				mid_seg[mid_seg_idx].end_idx = end_idx;
			}
			continue;
		}
		compute_sc_seg_one_wf(anchors_x, anchors_y, sid, qspan, xrev, range, start_idx, end_idx, f, p, misc);
	}
}


template <size_t mid_block_size>
__launch_bounds__(mid_block_size)
    __global__ void score_generation_mid(int32_t* anchors_x, int32_t* anchors_y, int8_t* sid, uint8_t* qspan, int32_t* xrev, int32_t* range,
	seg_t* long_seg, unsigned int* long_seg_count,
	int32_t* f, uint16_t* p, Misc misc)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	for (int segid = bid; segid < *long_seg_count; segid += gridDim.x) {
		seg_t seg = long_seg[segid];
		compute_sc_seg_multi_wf(anchors_x, anchors_y, sid, qspan, xrev, range, seg.start_idx, seg.end_idx, f, p, misc);
	}
}

template <size_t long_block_size>
__launch_bounds__(long_block_size)
    __global__ void score_generation_long(int32_t* anchors_x, int32_t* anchors_y, int8_t* sid, uint8_t* qspan, int32_t* xrev, int32_t* range,
	seg_t* long_seg, unsigned int* long_seg_count,
	int32_t* f, uint16_t* p, Misc misc)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	for (int segid = bid; segid < *long_seg_count; segid += gridDim.x) {
		seg_t seg = long_seg[segid];
		compute_sc_seg_multi_wf(anchors_x, anchors_y, sid, qspan, xrev, range, seg.start_idx, seg.end_idx, f, p, misc);
	}
}

// FIXME: merge together
template <size_t long_block_size>
__launch_bounds__(long_block_size)
    __global__ void score_generation_long_map(int32_t* anchors_x, int32_t* anchors_y, int8_t* sid, uint8_t* qspan, int32_t* xrev, int32_t* range,
	seg_t* long_seg, unsigned int* long_seg_count,
	int32_t* f, uint16_t* p, unsigned int* map,
	unsigned int* d_long_segid, Misc misc, size_t min_seg_n)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	unsigned int seg_count = 0;

	// #ifdef DEBUG_CHECK
	// auto start = clock64();
	// #endif

	// d_long_segid is initialised to 0 by the host (cudaMemsetAsync) before
	// launch. Each block first processes its own bid; subsequent work is
	// grabbed via atomicAdd, with gridDim.x added back so segIDs hand out
	// gridDim.x, gridDim.x+1, ... matching the original __device__-global
	// semantics without the cross-run race.
	__shared__ unsigned int segid;
	if (tid == 0) {
		segid = bid;
	}

	__syncthreads();
	while (segid < *long_seg_count) {
		seg_t seg = long_seg[map[segid]]; // sorted
		// seg_t seg = long_seg[segid]; // unsorted
		if (seg.end_idx - seg.start_idx > min_seg_n)
			compute_sc_seg_multi_wf(anchors_x, anchors_y, sid, qspan, xrev, range, seg.start_idx, seg.end_idx, f, p, misc);
		seg_count++;
		if (tid == 0) segid = atomicAdd(d_long_segid, 1) + gridDim.x;
		__syncthreads();
	}
}

// Forward-push long kernel (max_chain_skip = infinity). Dedicated so it carries
// no pull-path shared memory and can target a higher occupancy than the
// branching long_map kernel. __launch_bounds__ second arg caps registers to fit
// 2 blocks/CU -> roughly doubles occupancy on the VALU-bound push DP.
//
// NOTE (output parity): this push DP is the fast default for --gpu-chain and is
// bit-exact with the CPU on the vast majority of reads, but it is NOT guaranteed
// bit-exact at max_chain_skip = infinity. On a tiny fraction of reads (~0.01% on
// map-ont; high-anchor repeat reads where the max_chain_iter window cap binds),
// it can pick a different chain among equal-scoring predecessors at the window
// boundary, yielding a slightly different cm:i / chain composition (same locus,
// identical dv:f). The backward-scan pull kernels (compute_sc_seg_pull*, selected
// by any finite --max-chain-skip) replicate the CPU window/tie-break exactly and
// ARE bit-exact; use a finite --max-chain-skip when byte-identical output is
// required. See USAGE.md "GPU Known Limitations".
template <size_t block_size>
__launch_bounds__(block_size, 2)
    __global__ void score_generation_long_push(int32_t* anchors_x, int32_t* anchors_y, int8_t* sid, uint8_t* qspan, int32_t* xrev, int32_t* range,
	seg_t* long_seg, unsigned int* long_seg_count,
	int32_t* f, uint16_t* p, unsigned int* map,
	unsigned int* d_long_segid, Misc misc)
{
	const int tid = threadIdx.x;
	__shared__ unsigned int segid;
	if (tid == 0) segid = blockIdx.x;
	__syncthreads();
	while (segid < *long_seg_count) {
		seg_t seg = long_seg[map[segid]];
		compute_sc_seg_push(anchors_x, anchors_y, sid, qspan, range, seg.start_idx, seg.end_idx, f, p, misc);
		if (tid == 0) segid = atomicAdd(d_long_segid, 1) + gridDim.x;
		__syncthreads();
	}
}

// Warp-per-segment long kernel: each 64-lane warp owns one segment (size <=
// GIANT_CUT) and runs the wave-synchronous pull DP, so block_size/64 segments
// are processed concurrently per block. Larger segments are skipped here and
// handled by score_generation_long_map.
template <size_t block_size>
__launch_bounds__(block_size)
    __global__ void score_generation_long_warp(int32_t* anchors_x, int32_t* anchors_y, int8_t* sid, uint8_t* qspan, int32_t* xrev, int32_t* range,
	seg_t* long_seg, unsigned int* long_seg_count,
	int32_t* f, uint16_t* p, unsigned int* map,
	unsigned int* d_long_segid, Misc misc)
{
	const int lane = threadIdx.x & (PULL_WARP - 1);
	const int warpId = threadIdx.x >> 6;
	const int nwarp = block_size >> 6;
	__shared__ uint8_t markbuf[(block_size >> 6) * WARP_MARK_WIN];
	uint8_t* mymark = markbuf + warpId * WARP_MARK_WIN;
	unsigned int segid = blockIdx.x * nwarp + warpId;
	while (segid < *long_seg_count) {
		seg_t seg = long_seg[map[segid]];
		if (seg.end_idx - seg.start_idx <= GIANT_CUT)
			compute_sc_seg_pull_warp(anchors_x, anchors_y, sid, qspan, xrev, mymark, seg.start_idx, seg.end_idx, f, p, misc);
		unsigned int nxt = 0;
		if (lane == 0) nxt = atomicAdd(d_long_segid, 1) + gridDim.x * nwarp;
		segid = __shfl(nxt, 0);
	}
}

__global__ void score_generation_naive(int32_t* anchors_x, int32_t* anchors_y, int8_t* sid, uint8_t* qspan, int32_t* xrev, int32_t* range,
    size_t* seg_start_arr,
    int32_t* f, uint16_t* p, size_t total_n, size_t seg_count, Misc misc)
{

	// NOTE: each block deal with one batch
	// the number of threads in a block is fixed, so we need to calculate iter
	// n = end_idx_arr - start_idx_arr
	// iter = (range[i] - 1) / num_threads + 1

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	for (int segid = bid; segid < seg_count; segid += gridDim.x) {
		/* calculate the segement for current block */
		size_t start_idx = seg_start_arr[segid];
		if (start_idx == SIZE_MAX) continue; // start at a failed cut: continue to next iteration
		size_t end_idx = SIZE_MAX;
		int end_segid = segid + 1;
		while (true) {
			if (end_segid >= seg_count) {
				end_idx = total_n;
				break;
			}
			if (seg_start_arr[end_segid] != SIZE_MAX) {
				end_idx = seg_start_arr[end_segid];
				break;
			}
			++end_segid;
		}
		// assert(end_idx <= total_n);
		compute_sc_seg_one_wf(anchors_x, anchors_y, sid, qspan, xrev, range, start_idx, end_idx, f, p, misc);
	}
}

/* kernels end */

/* host functions begin */
// score_kernel_config global has been retired; configs now live on each
// gpu_chain_ctx_t (ctx->score_config).

void plscore_async_short_mid_forward_dp(const gpu_chain_ctx_t* ctx, deviceMemPtr* dev_mem, cudaStream_t* stream, Misc misc)
{
	size_t total_n = dev_mem->total_n;
	size_t cut_num = dev_mem->num_cut;

	// Skip kernel launch if no work to do
	if (total_n == 0 || cut_num == 0) {
		return;
	}

	size_t buffer_size_long = dev_mem->buffer_size_long;
	dim3 shortDimGrid(ctx->score_config.short_griddim, 1, 1);
	dim3 midDimGrid(ctx->score_config.mid_griddim, 1, 1);
	dim3 shortDimBlock(ctx->score_config.short_blockdim, 1, 1);

#ifdef DEBUG_VERBOSE
	mm_log_debug("plscore_async_short_mid_forward_dp: total_n={}, cut_num={}, short_griddim={}, short_blockdim={}",
	    total_n, cut_num, ctx->score_config.short_griddim, ctx->score_config.short_blockdim);
#endif

	// Run kernel;
	cudaMemsetAsync(dev_mem->d_mid_seg_count, 0, sizeof(unsigned int),
	    *stream);

	if (ctx->score_config.short_blockdim == 32) {
		score_generation_short<32><<<shortDimGrid, dim3(32, 1, 1), 0, *stream>>>(
		    dev_mem->d_ax, dev_mem->d_ay, dev_mem->d_sid, dev_mem->d_qspan, dev_mem->d_xrev, dev_mem->d_range,
		    dev_mem->d_cut, dev_mem->d_f, dev_mem->d_p, total_n, cut_num,
		    dev_mem->d_ax_long, dev_mem->d_ay_long, dev_mem->d_sid_long, dev_mem->d_qspan_long, dev_mem->d_xrev_long, dev_mem->d_range_long,
		    dev_mem->d_total_n_long, buffer_size_long,
		    dev_mem->d_long_seg, dev_mem->d_long_seg_og, dev_mem->d_long_seg_count,
		    dev_mem->d_mid_seg, dev_mem->d_mid_seg_count,
		    misc, ctx->score_config.long_seg_cutoff, ctx->score_config.mid_seg_cutoff);
	} else if (ctx->score_config.short_blockdim == 64) {
		score_generation_short<64><<<shortDimGrid, dim3(64, 1, 1), 0, *stream>>>(
		    dev_mem->d_ax, dev_mem->d_ay, dev_mem->d_sid, dev_mem->d_qspan, dev_mem->d_xrev, dev_mem->d_range,
		    dev_mem->d_cut, dev_mem->d_f, dev_mem->d_p, total_n, cut_num,
		    dev_mem->d_ax_long, dev_mem->d_ay_long, dev_mem->d_sid_long, dev_mem->d_qspan_long, dev_mem->d_xrev_long, dev_mem->d_range_long,
		    dev_mem->d_total_n_long, buffer_size_long,
		    dev_mem->d_long_seg, dev_mem->d_long_seg_og, dev_mem->d_long_seg_count,
		    dev_mem->d_mid_seg, dev_mem->d_mid_seg_count,
		    misc, ctx->score_config.long_seg_cutoff, ctx->score_config.mid_seg_cutoff);
	} else {
		mm_log_error("Unsupported warpsize: {}. mm2-gb only supports device "
			     "with a warpsize of 32 / 64. ",
		    ctx->score_config.short_blockdim);
		exit(1);
	}
	cudaCheck();


	if (ctx->score_config.mid_blockdim == 128) {
		score_generation_mid<128><<<midDimGrid, dim3(128, 1, 1), 0, *stream>>>(
		    dev_mem->d_ax, dev_mem->d_ay, dev_mem->d_sid, dev_mem->d_qspan, dev_mem->d_xrev, dev_mem->d_range, dev_mem->d_mid_seg,
		    dev_mem->d_mid_seg_count, dev_mem->d_f, dev_mem->d_p, misc);
	} else if (ctx->score_config.mid_blockdim == 256) {
		score_generation_mid<256><<<midDimGrid, dim3(256, 1, 1), 0, *stream>>>(
		    dev_mem->d_ax, dev_mem->d_ay, dev_mem->d_sid, dev_mem->d_qspan, dev_mem->d_xrev, dev_mem->d_range, dev_mem->d_mid_seg,
		    dev_mem->d_mid_seg_count, dev_mem->d_f, dev_mem->d_p, misc);
	} else if (ctx->score_config.mid_blockdim == 512) {
		score_generation_mid<512><<<midDimGrid, dim3(512, 1, 1), 0, *stream>>>(
		    dev_mem->d_ax, dev_mem->d_ay, dev_mem->d_sid, dev_mem->d_qspan, dev_mem->d_xrev, dev_mem->d_range, dev_mem->d_mid_seg,
		    dev_mem->d_mid_seg_count, dev_mem->d_f, dev_mem->d_p, misc);
	} else if (ctx->score_config.mid_blockdim == 1024) {
		score_generation_mid<1024><<<midDimGrid, dim3(1024, 1, 1), 0, *stream>>>(
		    dev_mem->d_ax, dev_mem->d_ay, dev_mem->d_sid, dev_mem->d_qspan, dev_mem->d_xrev, dev_mem->d_range, dev_mem->d_mid_seg,
		    dev_mem->d_mid_seg_count, dev_mem->d_f, dev_mem->d_p, misc);
	} else {
		mm_log_error("Unsupported mid_blockdim: {}. mm2-gb only supports a "
			     "blockdim of 128/256/512/1024 for mid kernel \n\n"
			     "Please adjust score_kernel:mid_blockdim in gpu config file. ",
		    ctx->score_config.mid_blockdim);
		exit(1);
	}
	cudaCheck();

#ifdef DEBUG_PRINT
	// fprintf(stderr, "[Info] %s (%s:%d) short mid score kernel launched\n", __func__, __FILE__, __LINE__);
#endif

	cudaCheck();
}

void plscore_async_long_forward_dp(const gpu_chain_ctx_t* ctx, deviceMemPtr* dev_mem, cudaStream_t* stream, Misc misc)
{
	size_t total_n = dev_mem->total_n;
	size_t cut_num = dev_mem->num_cut;
	size_t buffer_size_long = dev_mem->buffer_size_long;
	dim3 longDimGrid(ctx->score_config.long_griddim, 1, 1);

	// Reset the per-worker work-stealing counter to 0 before each launch.
	// The kernel adds gridDim.x to atomic returns, so dispatched segIDs are
	// gridDim.x, gridDim.x+1, ... -- the same range as the old __device__
	// curr_long_segid, but isolated per worker so concurrent ctxs don't race.
	cudaMemsetAsync(dev_mem->d_long_segid, 0, sizeof(unsigned), *stream);

#ifdef DEBUG_VERBOSE
	mm_log_debug("{} ({}:{}) Long Grid Dim = {}", __func__, __FILE__, __LINE__, longDimGrid.x);
#endif // DEBUG_VERBOSE


	if (ctx->score_config.long_blockdim == 1024) {
		// max_skip==0 is a finite (most aggressive) early-stop, not infinity, so
		// it belongs on the CPU-faithful pull path, not the forward-push kernel.
		const bool pull = (misc.max_skip >= 0 && misc.max_skip < INT32_MAX);
		if (pull) {
			// High-occupancy warp-per-segment kernel for all but the largest
			// segments, then the block kernel for the few giant ones. The two
			// launches are serialized on the stream and share d_long_segid (reset
			// between), so they never race.
			score_generation_long_warp<256><<<longDimGrid, dim3(256, 1, 1), 0, *stream>>>(
			    dev_mem->d_ax_long, dev_mem->d_ay_long, dev_mem->d_sid_long, dev_mem->d_qspan_long, dev_mem->d_xrev_long, dev_mem->d_range_long, dev_mem->d_long_seg,
			    dev_mem->d_long_seg_count, dev_mem->d_f_long, dev_mem->d_p_long, dev_mem->d_map, dev_mem->d_long_segid, misc);
			cudaMemsetAsync(dev_mem->d_long_segid, 0, sizeof(unsigned), *stream);
			score_generation_long_map<1024><<<longDimGrid, dim3(1024, 1, 1), 0, *stream>>>(
			    dev_mem->d_ax_long, dev_mem->d_ay_long, dev_mem->d_sid_long, dev_mem->d_qspan_long, dev_mem->d_xrev_long, dev_mem->d_range_long, dev_mem->d_long_seg,
			    dev_mem->d_long_seg_count, dev_mem->d_f_long, dev_mem->d_p_long, dev_mem->d_map, dev_mem->d_long_segid, misc, GIANT_CUT);
		} else {
			score_generation_long_push<1024><<<longDimGrid, dim3(1024, 1, 1), 0, *stream>>>(
			    dev_mem->d_ax_long, dev_mem->d_ay_long, dev_mem->d_sid_long, dev_mem->d_qspan_long, dev_mem->d_xrev_long, dev_mem->d_range_long, dev_mem->d_long_seg,
			    dev_mem->d_long_seg_count, dev_mem->d_f_long, dev_mem->d_p_long, dev_mem->d_map, dev_mem->d_long_segid, misc);
		}
	} else {
		mm_log_error("Unsupported MaxThreadsPerBlock: {}. mm2-gb only supports a blockdim of 1024 for long kernel ",
		    ctx->score_config.long_blockdim);
		exit(1);
	}

	cudaCheck();

#ifdef DEBUG_PRINT
	// fprintf(stderr, "[Info] %s (%s:%d) long score generation launched\n", __func__, __FILE__, __LINE__);
#endif

	cudaCheck();
	(void)total_n;
	(void)cut_num;
	(void)buffer_size_long;
}

void plscore_async_naive_forward_dp(const gpu_chain_ctx_t* ctx,
    deviceMemPtr* dev_mem,
    cudaStream_t* stream, Misc misc)
{
	size_t total_n = dev_mem->total_n;
	size_t cut_num = dev_mem->num_cut;
	dim3 DimBlock(ctx->score_config.long_blockdim, 1, 1);
	dim3 longDimGrid(ctx->score_config.long_griddim, 1, 1);
	dim3 shortDimGrid(ctx->score_config.short_griddim, 1, 1);

	// Run kernel
	// printf("Grid Dim, %d\n", DimGrid.x);
	score_generation_naive<<<shortDimGrid, DimBlock, 0, *stream>>>(
	    dev_mem->d_ax, dev_mem->d_ay, dev_mem->d_sid, dev_mem->d_qspan, dev_mem->d_xrev, dev_mem->d_range, dev_mem->d_cut,
	    dev_mem->d_f, dev_mem->d_p, total_n, cut_num, misc);
	cudaCheck();
#ifdef DEBUG_VERBOSE
	mm_log_debug("[M::{}] score generation kernel launch success", __func__);
#endif

	cudaCheck();
}
