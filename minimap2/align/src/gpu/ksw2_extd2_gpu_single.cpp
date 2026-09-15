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
#include "mm_log.h"
#include <assert.h>
#include "ksw2.h"

#include <hip/hip_runtime.h>

#include <emmintrin.h>

#define CHECK(call) \
	do { \
		hipError_t err = call; \
		if (err != hipSuccess) { \
			mm_log_error("GPU Error: {}", hipGetErrorString(err)); \
			exit(1); \
		} \
	} while (0)

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

__global__ void ksw_extd2_gpu_single( // void *km,
    uint8_t* mem, uint8_t* mem2, int32_t* H, int* off,
    int qlen, const uint8_t* query, int tlen, const uint8_t* target, int8_t m, const int8_t* mat,
    int8_t q, int8_t e, int8_t q2, int8_t e2, int w, int zdrop, int end_bonus, int flag, ksw_extz_t* ez)
// ~KSW_CPU_DISPATCH
{
#define __dp_code_block1 \
	for (int i = 0; i < VECT_SIZE; i++) z[i] = s[t * VECT_SIZE + i]; /* z = _mm_load_si128(&s[t]); */ \
	for (int i = 0; i < VECT_SIZE; i++) xt1[i] = x[t * VECT_SIZE + i]; /* xt1 <- x[r-1][t..t+15] */ \
	for (int i = 0; i < VECT_SIZE; i++) tmp[i] = 0; \
	tmp[0] = xt1[VECT_SIZE - 1]; /* tmp = _mm_srli_si128(xt1, 15);    tmp <- x[r-1][t+15], logical right shift of 15 bytes*/ \
	for (int i = 1; i < VECT_SIZE; i++) tmp1[i] = xt1[i - 1]; \
	tmp1[0] = 0; /* xt1 = _mm_or_si128(_mm_slli_si128(xt1, 1), x1_); 		 xt1 <- x[r-1][t-1..t+14], logical left shift of 15 bytes */ \
	for (int i = 0; i < VECT_SIZE; i++) xt1[i] = tmp1[i] | x1_[i]; \
	for (int i = 0; i < VECT_SIZE; i++) x1_[i] = tmp[i]; /* x1_ = tmp; */ \
	for (int i = 0; i < VECT_SIZE; i++) vt1[i] = v[t * VECT_SIZE + i]; /* vt1 = _mm_load_si128(&v[t]);         vt1 <- v[r-1][t..t+15] */ \
	for (int i = 0; i < VECT_SIZE; i++) tmp[i] = 0; \
	tmp[0] = vt1[VECT_SIZE - 1]; /* tmp = _mm_srli_si128(vt1, 15);   tmp <- v[r-1][t+15] */ \
	for (int i = 1; i < VECT_SIZE; i++) tmp1[i] = vt1[i - 1]; \
	tmp1[0] = 0; /* vt1 = _mm_or_si128(_mm_slli_si128(vt1, 1), v1_);   vt1 <- v[r-1][t-1..t+14] */ \
	for (int i = 0; i < VECT_SIZE; i++) vt1[i] = tmp1[i] | v1_[i]; \
	for (int i = 0; i < VECT_SIZE; i++) v1_[i] = tmp[i]; /* v1_ = tmp; */ \
	for (int i = 0; i < VECT_SIZE; i++) a[i] = xt1[i] + vt1[i]; /* a = _mm_add_epi8(xt1, vt1);           a <- x[r-1][t-1..t+14] + v[r-1][t-1..t+14] */ \
	for (int i = 0; i < VECT_SIZE; i++) ut[i] = u[t * VECT_SIZE + i]; /* ut <- u[t..t+15] */ \
	for (int i = 0; i < VECT_SIZE; i++) b[i] = y[t * VECT_SIZE + i] + ut[i]; /**((__m128i*)b) = _mm_add_epi8(_mm_load_si128(&((__m128i*)y)[t]), *ptr_ut); 	--> b = _mm_add_epi8(_mm_load_si128(&y[t]), ut);      b <- y[r-1][t..t+15] + u[r-1][t..t+15] */ \
	for (int i = 0; i < VECT_SIZE; i++) x2t1[i] = x2[t * VECT_SIZE + i]; /* x2t1= _mm_load_si128(&x2[t]); */ \
	for (int i = 0; i < VECT_SIZE; i++) tmp[i] = 0; \
	tmp[0] = x2t1[VECT_SIZE - 1]; /* tmp = _mm_srli_si128(x2t1, 15); */ \
	for (int i = 1; i < VECT_SIZE; i++) tmp1[i] = x2t1[i - 1]; \
	tmp1[0] = 0; /* x2t1= _mm_or_si128(_mm_slli_si128(x2t1, 1), x21_); */ \
	for (int i = 0; i < VECT_SIZE; i++) x2t1[i] = tmp1[i] | x21_[i]; \
	for (int i = 0; i < VECT_SIZE; i++) x21_[i] = tmp[i]; /* x21_= tmp; */ \
	for (int i = 0; i < VECT_SIZE; i++) a2[i] = x2t1[i] + vt1[i]; /* a2= _mm_add_epi8(x2t1, vt1); */ \
	for (int i = 0; i < VECT_SIZE; i++) b2[i] = y2[t * VECT_SIZE + i] + ut[i]; /* b2= _mm_add_epi8(_mm_load_si128(&y2[t]), ut); */

#define __dp_code_block2 \
	for (int i = 0; i < VECT_SIZE; i++) u[t * VECT_SIZE + i] = z[i] - vt1[i]; /* _mm_store_si128(&u[t], _mm_sub_epi8(z, vt1));   u[r][t..t+15] <- z - v[r-1][t-1..t+14] */ \
	for (int i = 0; i < VECT_SIZE; i++) v[t * VECT_SIZE + i] = z[i] - ut[i]; /* _mm_store_si128(&v[t], _mm_sub_epi8(z, ut));    v[r][t..t+15] <- z - u[r-1][t..t+15] */ \
	for (int i = 0; i < VECT_SIZE; i++) tmp[i] = z[i] - q_[i]; /* tmp = _mm_sub_epi8(z, *ptr_q); */ \
	for (int i = 0; i < VECT_SIZE; i++) a[i] = a[i] - tmp[i]; /* a = _mm_sub_epi8(a, tmp); */ \
	for (int i = 0; i < VECT_SIZE; i++) b[i] = b[i] - tmp[i]; /* b = _mm_sub_epi8(b, tmp); */ \
	for (int i = 0; i < VECT_SIZE; i++) tmp[i] = z[i] - q2_[i]; /* tmp = _mm_sub_epi8(z, *ptr_q2); */ \
	for (int i = 0; i < VECT_SIZE; i++) a2[i] = a2[i] - tmp[i]; /* a2= _mm_sub_epi8(a2, tmp); */ \
	for (int i = 0; i < VECT_SIZE; i++) b2[i] = b2[i] - tmp[i]; /* b2= _mm_sub_epi8(b2, tmp); */

	int r, t, qe = q + e, n_col_, /* *off = 0, */ *off_end = 0, tlen_, qlen_, last_st, last_en, wl, wr, max_sc, min_sc, long_thres, long_diff;
	int with_cigar = !(flag & KSW_EZ_SCORE_ONLY), approx_max = !!(flag & KSW_EZ_APPROX_MAX);
	int32_t H0 = 0, last_H0_t = 0;
	uint8_t *qr, *sf;

	int8_t q_[VECT_SIZE], q2_[VECT_SIZE], qe_[VECT_SIZE], qe2_[VECT_SIZE], zero_[VECT_SIZE], sc_mch_[VECT_SIZE], sc_mis_[VECT_SIZE], m1_[VECT_SIZE], sc_N_[VECT_SIZE]; // 128-bit intrinsic data type, packed values are in integer format
	int8_t *u, *v, *x, *y, *x2, *y2, *s, *p = 0; // 128-bit intrinsic data type, packed values are in integer format

	ez->max_q = ez->max_t = ez->mqe_t = ez->mte_q = -1;
	ez->max = 0, ez->score = ez->mqe = ez->mte = KSW_NEG_INF;
	ez->n_cigar = 0, ez->zdropped = 0, ez->reach_end = 0;
	if (m <= 1 || qlen <= 0 || tlen <= 0) return; // end due to invalid alignment

	if (q2 + e2 < q + e) t = q, q = q2, q2 = t, t = e, e = e2, e2 = t; // make sure q+e no larger than q2+e2

	for (int i = 0; i < VECT_SIZE; i++) zero_[i] = 0; // zero_   = _mm_set1_epi8(0); // replicates the 8-bit input value in all 16 positions
	for (int i = 0; i < VECT_SIZE; i++) q_[i] = q; // q_      = _mm_set1_epi8(q);
	for (int i = 0; i < VECT_SIZE; i++) q2_[i] = q2; // q2_     = _mm_set1_epi8(q2);
	for (int i = 0; i < VECT_SIZE; i++) qe_[i] = q + e; // qe_     = _mm_set1_epi8(q + e);
	for (int i = 0; i < VECT_SIZE; i++) qe2_[i] = q2 + e2; // qe2_    = _mm_set1_epi8(q2 + e2);
	for (int i = 0; i < VECT_SIZE; i++) sc_mch_[i] = mat[0]; // sc_mch_ = _mm_set1_epi8(mat[0]);
	for (int i = 0; i < VECT_SIZE; i++) sc_mis_[i] = mat[1]; // sc_mis_ = _mm_set1_epi8(mat[1]);
	for (int i = 0; i < VECT_SIZE; i++) sc_N_[i] = mat[m * m - 1] == 0 ? -e2 : mat[m * m - 1]; // sc_N_   = mat[m*m-1] == 0? _mm_set1_epi8(-e2) : _mm_set1_epi8(mat[m*m-1]);
	for (int i = 0; i < VECT_SIZE; i++) m1_[i] = m - 1; // m1_     = _mm_set1_epi8(m - 1); /* wildcard */

	if (w < 0) w = tlen > qlen ? tlen : qlen; // set bandwidth in case it is negative (non-banded alignment)
	wl = wr = w;
	tlen_ = (tlen + (VECT_SIZE - 1)) / VECT_SIZE; // tlen_ = (tlen + 15) / 16; // round to multiple of 16
	n_col_ = qlen < tlen ? qlen : tlen;
	n_col_ = ((n_col_ < w + 1 ? n_col_ : w + 1) + (VECT_SIZE - 1)) / VECT_SIZE + 1; // n_col_ = ((n_col_ < w + 1? n_col_ : w + 1) + 15) / 16 + 1; // round to multiple of 16
	qlen_ = (qlen + (VECT_SIZE - 1)) / VECT_SIZE; // qlen_ = (qlen + 15) / 16; // round to multiple of 16
	for (t = 1, max_sc = mat[0], min_sc = mat[1]; t < m * m; ++t) { // set max_sc to maximum score in mat, and min_sc to minimum score in mat
		max_sc = max_sc > mat[t] ? max_sc : mat[t];
		min_sc = min_sc < mat[t] ? min_sc : mat[t];
	}
	if (-min_sc > 2 * (q + e)) return; /* otherwise, we won't see any mismatches */

	long_thres = e != e2 ? (q2 - q) / (e - e2) - 1 : 0; // if double affine, set the threshold for long gaps(?) to ((q2 - q) / (e - e2) - 1), otherwise 0
	if (q2 + e2 + long_thres * e2 > q + e + long_thres * e)
		++long_thres;
	long_diff = long_thres * (e - e2) - (q2 - q) - e2;

	u = (int8_t*)mem; // (int8_t*)(((size_t)mem + 15) >> 4 << 4); // u = (__m128i*)(((size_t)mem + 15) >> 4 << 4); /* 16-byte aligned */
	v = (u + tlen_ * VECT_SIZE), x = (v + tlen_ * VECT_SIZE), y = (x + tlen_ * VECT_SIZE), x2 = y + tlen_ * VECT_SIZE, y2 = x2 + tlen_ * VECT_SIZE; // v = (u + tlen_*16), x = (v + tlen_*16), y = (x + tlen_*16), x2 = y + tlen_*16, y2 = x2 + tlen_*16; // v = u + tlen_, x = v + tlen_, y = x + tlen_, x2 = (y + tlen_), y2 = (x2 + tlen_); // set these values to compute the following ones
	s = (y2 + tlen_ * VECT_SIZE), sf = (uint8_t*)(s + tlen_ * VECT_SIZE), qr = sf + tlen_ * VECT_SIZE; // s = (y2 + tlen_*16), sf = (uint8_t*)(s + tlen_*16), qr = sf + tlen_ * 16;		// s = y2 + tlen_, sf = (uint8_t*)(s + tlen_), qr = sf + tlen_ * 16;
	memset(u, -q - e, tlen_ * VECT_SIZE); // 16); // set all 16 bytes of variable to a specific value (overwrites)
	memset(v, -q - e, tlen_ * VECT_SIZE); // 16);
	memset(x, -q - e, tlen_ * VECT_SIZE); // 16);
	memset(y, -q - e, tlen_ * VECT_SIZE); // 16);
	memset(x2, -q2 - e2, tlen_ * VECT_SIZE); // 16);
	memset(y2, -q2 - e2, tlen_ * VECT_SIZE); // 16);

	if (!approx_max) { // if !approximated alignment
		// check line 97  --> // H = (int32_t*)kmalloc(km, tlen_ * 16 * 4); // malloc of (tlen_ * 16 * 4) bytes
		for (t = 0; t < tlen_ * VECT_SIZE; ++t) H[t] = KSW_NEG_INF; // initialize to -inf
	}

	if (with_cigar) { // if CIGAR is required
		p = (int8_t*)mem2; // (((size_t)mem2 + 15) >> 4 << 4);  // p = (__m128i*)(((size_t)mem2 + 15) >> 4 << 4);
		// off = (int*)kmalloc(km, (qlen + tlen - 1) * sizeof(int) * 2);
		off_end = off + qlen + tlen - 1;
	}

	for (t = 0; t < qlen; ++t) qr[t] = query[qlen - 1 - t]; // reverse query to read aligned memory
	memcpy(sf, target, tlen);

	// alignment main loop
	for (r = 0, last_st = last_en = -1; r < qlen + tlen - 1; ++r) {
		int st = 0, en = tlen - 1, st0, en0, st_, en_;
		int8_t x1, x21, v1;
		uint8_t* qrr = qr + (qlen - 1 - r);
		int8_t *u8 = u, *v8 = v, *x8 = x; // *u8 = (int8_t*)u, *v8 = (int8_t*)v, *x8 = (int8_t*)x;
		int8_t* x28 = (int8_t*)x2; // *x28 = (int8_t*)x2;
		// __m128i x1_, x21_, v1_;
		uint8_t x1_[VECT_SIZE], x21_[VECT_SIZE], v1_[VECT_SIZE];

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
		// st = st / VECT_SIZE * VECT_SIZE, en = (en + VECT_SIZE) / VECT_SIZE * VECT_SIZE - 1;

		/* set boundary conditions */
		if (st > 0) {
			if (st - 1 >= last_st && st - 1 <= last_en) {
				x1 = x8[st - 1], x21 = x28[st - 1], v1 = v8[st - 1]; /* (r-1,s-1) calculated in the last round */
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
			u8[r] = r == 0 ? -q - e : r < long_thres ? -e
			    : r == long_thres			 ? long_diff
								 : -e2;
		}

		/* loop fission: set scores first */
		if (!(flag & KSW_EZ_GENERIC_SC)) {
			for (t = st0; t <= en0; t += VECT_SIZE) {

				uint8_t sq[VECT_SIZE], st[VECT_SIZE], mask[VECT_SIZE], tmp[VECT_SIZE];

				uint8_t tmp1[VECT_SIZE], tmp2[VECT_SIZE];

				for (int i = 0; i < VECT_SIZE; i++) sq[i] = sf[t + i]; // sq = _mm_loadu_si128((__m128i*)&sf[t]); // load 128 bits from memory (does not require memory to be aligned)
				for (int i = 0; i < VECT_SIZE; i++) st[i] = qrr[t + i]; // st = _mm_loadu_si128((__m128i*)&qrr[t]);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[VECT_SIZE - 1 - i] = sq[VECT_SIZE - 1 - i] == m1_[VECT_SIZE - 1 - i] ? 0xFF : 0;
				for (int i = 0; i < VECT_SIZE; i++) tmp2[VECT_SIZE - 1 - i] = st[VECT_SIZE - 1 - i] == m1_[VECT_SIZE - 1 - i] ? 0xFF : 0;
				for (int i = 0; i < VECT_SIZE; i++) mask[i] = tmp1[i] | tmp2[i];


				for (int i = 0; i < VECT_SIZE; i++) tmp[VECT_SIZE - 1 - i] = sq[VECT_SIZE - 1 - i] == st[VECT_SIZE - 1 - i] ? 0xFF : 0;

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & sc_mis_[i];
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = tmp[i] & sc_mch_[i];
				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~mask[i]) & tmp[i];
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = mask[i] & sc_N_[i];
				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) s[t + i] = tmp[i]; // _mm_storeu_si128((__m128i*)((int8_t*)s + t), *((__m128i*)tmp)); // store 128-bit register into memory without requiring addresses to be 16-bit aligned
			}
		} else {
			for (t = st0; t <= en0; ++t)
				s[t] = mat[sf[t] * m + qrr[t]]; // ((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
		}

		/* core loop */
		for (int i = 0; i < VECT_SIZE; i++) x1_[i] = 0;
		x1_[0] = (uint8_t)x1; // x1_  = _mm_cvtsi32_si128((uint8_t)x1); // convert x1 into a 128 bit, placing x1 in the lowest 8 bits of the vector, rest of vector is 0
		for (int i = 0; i < VECT_SIZE; i++) x21_[i] = 0;
		x21_[0] = (uint8_t)x21; // x21_ = _mm_cvtsi32_si128((uint8_t)x21);
		for (int i = 0; i < VECT_SIZE; i++) v1_[i] = 0;
		v1_[0] = (uint8_t)v1; // v1_  = _mm_cvtsi32_si128((uint8_t)v1);

		// st_ = st / 16, en_ = en / 16;
		st_ = st / VECT_SIZE, en_ = en / VECT_SIZE;

		// printf("dlen %d\t\t%d %d %d %d %d %d\n", en - st, st, en, st_, en_, st0, en0);

		assert(en_ - st_ + 1 <= n_col_);

		if (!with_cigar) { // score only

			for (t = st_; t <= en_; ++t) {

				int8_t a2[VECT_SIZE];
				int8_t a[VECT_SIZE];
				int8_t b2[VECT_SIZE];
				int8_t vt1[VECT_SIZE];
				int8_t x2t1[VECT_SIZE];
				int8_t xt1[VECT_SIZE];
				int8_t tmp[VECT_SIZE], tmp2[VECT_SIZE], ut[VECT_SIZE];
				int8_t b[VECT_SIZE];
				int8_t z[VECT_SIZE];
				int8_t tmp1[VECT_SIZE];

				__dp_code_block1;

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = a[i] > z[i] ? 0xFF : 0; // signed > on 8 bits				// tmp = _mm_cmpgt_epi8(a,  z); // signed > on 8 bits

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & a[i]; // z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = (~tmp[i]) & z[i];
				for (int i = 0; i < VECT_SIZE; i++) z[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = b[i] > z[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(b,  z);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & b[i]; // z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, b));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = (~tmp[i]) & z[i];
				for (int i = 0; i < VECT_SIZE; i++) z[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = a2[i] > z[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(a2, z);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & a2[i]; // z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a2));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = (~tmp[i]) & z[i];
				for (int i = 0; i < VECT_SIZE; i++) z[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = b2[i] > z[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(b2, z);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & b2[i]; // z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, b2));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = (~tmp[i]) & z[i];
				for (int i = 0; i < VECT_SIZE; i++) z[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = sc_mch_[i] < z[i] ? 0xFF : 0; // tmp = _mm_cmplt_epi8(*ptr_sc_mch, z); // signed < on 8 bits

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & sc_mch_[i]; // z = _mm_or_si128(_mm_and_si128(tmp, *ptr_sc_mch), _mm_andnot_si128(tmp, z));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = (~tmp[i]) & z[i];
				for (int i = 0; i < VECT_SIZE; i++) z[i] = tmp1[i] | tmp2[i];

				__dp_code_block2;

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = a[i] > zero_[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(a, zero_);

				for (int i = 0; i < VECT_SIZE; i++) x[t * VECT_SIZE + i] = (tmp[i] & a[i]) - qe_[i]; // _mm_store_si128(&x[t],  _mm_sub_epi8(_mm_and_si128(tmp, a), qe_));

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = b[i] > zero_[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(b, zero_);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & b[i]; // _mm_store_si128(&y[t],  _mm_sub_epi8(_mm_and_si128(tmp, b), qe_));
				for (int i = 0; i < VECT_SIZE; i++) y[t * VECT_SIZE + i] = tmp1[i] - qe_[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = a2[i] > zero_[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(a2, zero_);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & a2[i]; // _mm_store_si128(&x2[t], _mm_sub_epi8(_mm_and_si128(tmp, a2), qe2_));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = tmp1[i] - qe2_[i];
				for (int i = 0; i < VECT_SIZE; i++) x2[t * VECT_SIZE + i] = tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = b2[i] > zero_[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(b2, zero_);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & b2[i]; // _mm_store_si128(&y2[t], _mm_sub_epi8(_mm_and_si128(tmp, b2), *ptr_qe2));
				for (int i = 0; i < VECT_SIZE; i++) y2[t * VECT_SIZE + i] = tmp1[i] - qe2_[i];
			}

		} else if (!(flag & KSW_EZ_RIGHT)) { /* gap left-alignment */
			// int8_t *pr = p + ((size_t)r * n_col_ - st_)*16; // __m128i *pr = p + (size_t)r * n_col_ - st_;
			int8_t* pr = p + ((size_t)r * n_col_ - st_) * VECT_SIZE;
			off[r] = st, off_end[r] = en;
			for (t = st_; t <= en_; ++t) {

				int8_t x2t1[VECT_SIZE];
				int8_t xt1[VECT_SIZE];
				int8_t vt1[VECT_SIZE];
				int8_t tmp[VECT_SIZE], ut[VECT_SIZE];
				int8_t b2[VECT_SIZE];
				int8_t b[VECT_SIZE];
				int8_t a2[VECT_SIZE];
				int8_t a[VECT_SIZE];
				int8_t z[VECT_SIZE];
				int8_t d[VECT_SIZE];
				int8_t tmp1[VECT_SIZE], tmp2[VECT_SIZE];

				__dp_code_block1;

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = a[i] > z[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(a,  z);

				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp[i] & 1; // d = _mm_and_si128(tmp, _mm_set1_epi8(1));

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & a[i]; // z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = (~tmp[i]) & z[i];
				for (int i = 0; i < VECT_SIZE; i++) z[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = b[i] > z[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(b,  z);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & 2; // d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, _mm_set1_epi8(2)));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = (~tmp[i]) & d[i];
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & b[i]; // z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, b));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = (~tmp[i]) & z[i];
				for (int i = 0; i < VECT_SIZE; i++) z[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = a2[i] > z[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(a2, z);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & 3; // d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, _mm_set1_epi8(3)));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = (~tmp[i]) & d[i];
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & a2[i]; // z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a2));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = (~tmp[i]) & z[i];
				for (int i = 0; i < VECT_SIZE; i++) z[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = b2[i] > z[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(b2, z);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & 4; // d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, _mm_set1_epi8(4)));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = (~tmp[i]) & d[i];
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & b2[i]; // z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, b2));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = (~tmp[i]) & z[i];
				for (int i = 0; i < VECT_SIZE; i++) z[i] = tmp1[i] | tmp2[i];


				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = sc_mch_[i] < z[i] ? 0xFF : 0; // tmp = _mm_cmplt_epi8(*ptr_sc_mch, z);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & sc_mch_[i]; // z = _mm_or_si128(_mm_and_si128(tmp, *ptr_sc_mch), _mm_andnot_si128(tmp, z));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = (~tmp[i]) & z[i];
				for (int i = 0; i < VECT_SIZE; i++) z[i] = tmp1[i] | tmp2[i];

				__dp_code_block2;

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = a[i] > zero_[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(a, zero_);

				for (int i = 0; i < VECT_SIZE; i++) x[t * VECT_SIZE + i] = (tmp[i] & a[i]) - qe_[i]; // _mm_store_si128(&x[t],  _mm_sub_epi8(_mm_and_si128(tmp, a), qe_));

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & 0x08; // d = a > 0? 1<<3 : 0 	// d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x08))); // d = a > 0? 1<<3 : 0
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | d[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = b[i] > zero_[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(b, zero_);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & b[i]; // _mm_store_si128(&y[t],  _mm_sub_epi8(_mm_and_si128(tmp, b), qe_));
				for (int i = 0; i < VECT_SIZE; i++) y[t * VECT_SIZE + i] = tmp1[i] - qe_[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & 0x10; // d = b > 0? 1<<4 : 0 	// d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x10))); // d = b > 0? 1<<4 : 0
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | d[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = a2[i] > zero_[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(a2, zero_);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & a2[i]; // _mm_store_si128(&x2[t], _mm_sub_epi8(_mm_and_si128(tmp, a2), qe2_));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = tmp1[i] - qe2_[i];
				for (int i = 0; i < VECT_SIZE; i++) x2[t * VECT_SIZE + i] = tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & 0x20; // d = a > 0? 1<<5 : 0  // d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x20)));
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | d[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = b2[i] > zero_[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(b2, zero_);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & b2[i]; // _mm_store_si128(&y2[t], _mm_sub_epi8(_mm_and_si128(tmp, b2), *ptr_qe2));
				for (int i = 0; i < VECT_SIZE; i++) y2[t * VECT_SIZE + i] = tmp1[i] - qe2_[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & 0x40; // d = b > 0? 1<<6 : 0 	// d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x40)));
				for (int i = 0; i < VECT_SIZE; i++) d[i] = d[i] | tmp1[i];

				for (int i = 0; i < VECT_SIZE; i++) pr[t * VECT_SIZE + i] = d[i]; // _mm_store_si128(&pr[t], d);
			}
		} else { /* gap right-alignment */
			// int8_t *pr = p + ((size_t)r * n_col_ - st_)*16; // __m128i *pr = p + (size_t)r * n_col_ - st_;
			int8_t* pr = p + ((size_t)r * n_col_ - st_) * VECT_SIZE;
			off[r] = st, off_end[r] = en;

			for (t = st_; t <= en_; ++t) {

				int8_t tmp[VECT_SIZE], ut[VECT_SIZE];
				int8_t vt1[VECT_SIZE];
				int8_t x2t1[VECT_SIZE];
				int8_t xt1[VECT_SIZE];
				int8_t b2[VECT_SIZE];
				int8_t b[VECT_SIZE];
				int8_t a2[VECT_SIZE];
				int8_t a[VECT_SIZE];
				int8_t z[VECT_SIZE];
				int8_t d[VECT_SIZE];
				int8_t tmp1[VECT_SIZE];
				int8_t tmp2[VECT_SIZE];

				__dp_code_block1;

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = z[i] > a[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(z, a);
				for (int i = 0; i < VECT_SIZE; i++) d[i] = (~tmp[i]) & 1; // d = _mm_andnot_si128(tmp, _mm_set1_epi8(1));

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & a[i]; // z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, a));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = tmp[i] & z[i];
				for (int i = 0; i < VECT_SIZE; i++) z[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = z[i] > b[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(z, b);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & 2; // d = _mm_or_si128(_mm_and_si128(tmp, d), _mm_andnot_si128(tmp, _mm_set1_epi8(2)));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = tmp[i] & d[i];
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & b[i]; // z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, b));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = tmp[i] & z[i];
				for (int i = 0; i < VECT_SIZE; i++) z[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = z[i] > a2[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(z, a2);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & 3; // d = _mm_or_si128(_mm_and_si128(tmp, d), _mm_andnot_si128(tmp, _mm_set1_epi8(3)));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = tmp[i] & d[i];
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & a2[i]; // z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, a2));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = tmp[i] & z[i];
				for (int i = 0; i < VECT_SIZE; i++) z[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = z[i] > b2[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(z, b2);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & 4; // d = _mm_or_si128(_mm_and_si128(tmp, d), _mm_andnot_si128(tmp, _mm_set1_epi8(4)));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = tmp[i] & d[i];
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & b2[i]; // z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, b2));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = tmp[i] & z[i];
				for (int i = 0; i < VECT_SIZE; i++) z[i] = tmp1[i] | tmp2[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = z[i] > sc_mch_[i] ? 0xFF : 0; // tmp = _mm_cmplt_epi8(sc_mch, z);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & z[i]; // z = _mm_or_si128(_mm_and_si128(tmp, *ptr_sc_mch), _mm_andnot_si128(tmp, z));
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = tmp[i] & sc_mch_[i];
				for (int i = 0; i < VECT_SIZE; i++) z[i] = tmp1[i] | tmp2[i];

				__dp_code_block2;

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = zero_[i] > a[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(zero_, a);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & a[i]; // _mm_store_si128(&x[t],  _mm_sub_epi8(_mm_andnot_si128(tmp, a),  *ptr_qe));
				for (int i = 0; i < VECT_SIZE; i++) x[t * VECT_SIZE + i] = tmp1[i] - qe_[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & 0x08; // d = a > 0? 1<<3 : 0 	// d = _mm_or_si128(d, _mm_andnot_si128(tmp, _mm_set1_epi8(0x08)));
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | d[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = zero_[i] > b[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(*ptr_zero, b);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & b[i]; // _mm_store_si128(&((__m128i*)y)[t],  _mm_sub_epi8(_mm_andnot_si128(tmp, b),  *ptr_qe));
				for (int i = 0; i < VECT_SIZE; i++) y[t * VECT_SIZE + i] = tmp1[i] - qe_[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & 0x10; // d = b > 0? 1<<4 : 0		// d = _mm_or_si128(d, _mm_andnot_si128(tmp, _mm_set1_epi8(0x10)));
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | d[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = zero_[i] > a2[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(*ptr_zero, a2);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & a2[i]; // _mm_store_si128(&x2[t], _mm_sub_epi8(_mm_andnot_si128(tmp, a2), qe2_));
				for (int i = 0; i < VECT_SIZE; i++) x2[t * VECT_SIZE + i] = tmp1[i] - qe2_[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & 0x20; // d = a > 0? 1<<5 : 0		// d = _mm_or_si128(d, _mm_andnot_si128(tmp, _mm_set1_epi8(0x20)));
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | d[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = zero_[i] > b2[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(zero_, b2);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & b2[i]; // _mm_store_si128(&((__m128i*)y2)[t], _mm_sub_epi8(_mm_andnot_si128(tmp, b2), qe2_));
				for (int i = 0; i < VECT_SIZE; i++) y2[t * VECT_SIZE + i] = tmp1[i] - qe2_[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & 0x40; // d = b > 0? 1<<6 : 0		// d = _mm_or_si128(d, _mm_andnot_si128(tmp, _mm_set1_epi8(0x40)));
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | d[i];

				for (int i = 0; i < VECT_SIZE; i++) pr[t * VECT_SIZE + i] = d[i]; // _mm_store_si128(&pr[t], d);
			}
		}

		if (!approx_max) { /* find the exact max with a 32-bit score array */
			int32_t max_H, max_t;
			/* compute H[], max_H and max_t */
			if (r > 0) {
				int32_t HH[4], tt[4], en1 = st0 + (en0 - st0) / 4 * 4, i;
				int32_t max_H_[4], max_t_[4];
				max_H = H[en0] = en0 > 0 ? H[en0 - 1] + u8[en0] : H[en0] + v8[en0]; /* special casing the last element */
				max_t = en0;
				for (int i = 0; i < 4; i++) max_H_[i] = max_H; // *((__m128i*)max_H_) = _mm_set1_epi32(max_H); // initializes the register by broadcasting the 32-bit value to all 4 elements
				for (int i = 0; i < 4; i++) max_t_[i] = max_t; // *((__m128i*)max_t_) = _mm_set1_epi32(max_t);
				for (t = st0; t < en1; t += 4) { /* this implements: H[t]+=v8[t]-qe; if(H[t]>max_H) max_H=H[t],max_t=t; */
					int32_t H1[4], tmp[4], t_[4];
					for (int i = 0; i < 4; i++) H1[i] = H[t + i]; // H1 = _mm_loadu_si128((__m128i*)&H[t]);
					t_[0] = v8[t]; // *((__m128i*)t_) = _mm_setr_epi32(v8[t], v8[t+1], v8[t+2], v8[t+3]);
					t_[1] = v8[t + 1];
					t_[2] = v8[t + 2];
					t_[3] = v8[t + 3];
					for (int i = 0; i < 4; i++) H1[i] = H1[i] + t_[i]; // *((__m128i*)H1) = _mm_add_epi32(*((__m128i*)H1), *((__m128i*)t_));
					for (int i = 0; i < 4; i++) H[t + i] = H1[i]; // _mm_storeu_si128((__m128i*)&H[t], *((__m128i*)H1));
					for (int i = 0; i < 4; i++) t_[i] = t; // *((__m128i*)t_) = _mm_set1_epi32(t);
					for (int i = 0; i < 4; i++) tmp[i] = H1[i] > max_H_[i] ? 0xFFFFFFFF : 0; // *((__m128i*)tmp) = _mm_cmpgt_epi32(*((__m128i*)H1), *((__m128i*)max_H_));

					for (int i = 0; i < 4; i++) max_H_[i] = (tmp[i] & H1[i]) | ((~tmp[i]) & max_H_[i]); // *((__m128i*)max_H_) = _mm_or_si128(_mm_and_si128(*((__m128i*)tmp), *((__m128i*)H1)), _mm_andnot_si128(*((__m128i*)tmp), *((__m128i*)max_H_)));
					for (int i = 0; i < 4; i++) max_t_[i] = (tmp[i] & t_[i]) | ((~tmp[i]) & max_t_[i]); // *((__m128i*)max_t_) = _mm_or_si128(_mm_and_si128(*((__m128i*)tmp), *((__m128i*)t_)), _mm_andnot_si128(*((__m128i*)tmp), *((__m128i*)max_t_)));
				}
				for (int i = 0; i < 4; i++) HH[i] = max_H_[i]; // _mm_storeu_si128((__m128i*)HH, *((__m128i*)max_H_));
				for (int i = 0; i < 4; i++) tt[i] = max_t_[i]; // _mm_storeu_si128((__m128i*)tt, *((__m128i*)max_t_));
				for (i = 0; i < 4; ++i)
					if (max_H < HH[i]) max_H = HH[i], max_t = tt[i] + i;
#ifdef DEBUG_GPU_KERNEL
				if (threadIdx.x == 0) printf("%d %d\n", max_H, max_t);
#endif
				for (; t < en0; ++t) { /* for the rest of values that haven't been computed with SSE */
					H[t] += (int32_t)v8[t];
					if (H[t] > max_H)
						max_H = H[t], max_t = t;
				}
			} else
				H[0] = v8[0] - qe, max_H = H[0], max_t = 0; /* special casing r==0 */
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
					int32_t d0 = v8[last_H0_t];
					int32_t d1 = u8[last_H0_t + 1];
					if (d0 > d1)
						H0 += d0;
					else
						H0 += d1, ++last_H0_t;
				} else if (last_H0_t >= st0 && last_H0_t <= en0) {
					H0 += v8[last_H0_t];
				} else {
					++last_H0_t, H0 += u8[last_H0_t];
				}
			} else
				H0 = v8[0] - qe, last_H0_t = 0;
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
