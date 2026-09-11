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

__global__ void ksw_extz2_gpu_single( //void *km,
    uint8_t* mem, uint8_t* mem2, int32_t* H, int* off,
    int qlen, const uint8_t* query, int tlen, const uint8_t* target, int8_t m, const int8_t* mat, int8_t q, int8_t e, int w, int zdrop, int end_bonus, int flag, ksw_extz_t* ez)
// ~KSW_CPU_DISPATCH
{
#define __dp_code_block1 \
	for (int i = 0; i < VECT_SIZE; i++) z[i] = s[t * VECT_SIZE + i] + qe2_[i]; /* z = _mm_add_epi8(_mm_load_si128(&s[t]), qe2_);  */ \
	for (int i = 0; i < VECT_SIZE; i++) xt1[i] = x[t * VECT_SIZE + i]; /* xt1 = _mm_load_si128(&x[t]);                    */ \
	for (int i = 0; i < VECT_SIZE; i++) tmp[i] = 0; \
	tmp[0] = xt1[VECT_SIZE - 1]; /* tmp = _mm_srli_si128(xt1, 15);                  */ \
	for (int i = 1; i < VECT_SIZE; i++) tmp1[i] = xt1[i - 1]; \
	tmp1[0] = 0; /* xt1 = _mm_or_si128(_mm_slli_si128(xt1, 1), x1_);*/ \
	for (int i = 0; i < VECT_SIZE; i++) xt1[i] = tmp1[i] | x1_[i]; \
	for (int i = 0; i < VECT_SIZE; i++) x1_[i] = tmp[i]; /* x1_ = tmp;                                      */ \
	for (int i = 0; i < VECT_SIZE; i++) vt1[i] = v[t * VECT_SIZE + i]; /* vt1 = _mm_load_si128(&v[t]);                    */ \
	for (int i = 0; i < VECT_SIZE; i++) tmp[i] = 0; \
	tmp[0] = vt1[VECT_SIZE - 1]; /* tmp = _mm_srli_si128(vt1, 15);                  */ \
	for (int i = 1; i < VECT_SIZE; i++) tmp1[i] = vt1[i - 1]; \
	tmp1[0] = 0; /* vt1 = _mm_or_si128(_mm_slli_si128(vt1, 1), v1_);*/ \
	for (int i = 0; i < VECT_SIZE; i++) vt1[i] = tmp1[i] | v1_[i]; \
	for (int i = 0; i < VECT_SIZE; i++) v1_[i] = tmp[i]; /* v1_ = tmp;                                      */ \
	for (int i = 0; i < VECT_SIZE; i++) a[i] = xt1[i] + vt1[i]; /* a = _mm_add_epi8(xt1, vt1);                     */ \
	for (int i = 0; i < VECT_SIZE; i++) ut[i] = u[t * VECT_SIZE + i]; /* ut = _mm_load_si128(&u[t]);                     */ \
	for (int i = 0; i < VECT_SIZE; i++) b[i] = y[t * VECT_SIZE + i] + ut[i]; /* b = _mm_add_epi8(_mm_load_si128(&y[t]), ut);    */

#define __dp_code_block2 \
	for (int i = 0; i < VECT_SIZE; i++) z[i] = z[i] > b[i] ? z[i] : b[i]; /* z = _mm_max_epu8(z, b);   z = max(z, b); this works because both are non-negative */ \
	for (int i = 0; i < VECT_SIZE; i++) z[i] = z[i] < max_sc_[i] ? z[i] : max_sc_[i]; /* z = _mm_min_epu8(z, max_sc_); 	              */ \
	for (int i = 0; i < VECT_SIZE; i++) u[t * VECT_SIZE + i] = z[i] - vt1[i]; /* _mm_store_si128(&u[t], _mm_sub_epi8(z, vt1));  */ /* u[r][t..t+15] <- z - v[r-1][t-1..t+14] */ \
	for (int i = 0; i < VECT_SIZE; i++) v[t * VECT_SIZE + i] = z[i] - ut[i]; /* _mm_store_si128(&v[t], _mm_sub_epi8(z, ut));   */ /* v[r][t..t+15] <- z - u[r-1][t..t+15] */ \
	for (int i = 0; i < VECT_SIZE; i++) z[i] = z[i] - q_[i]; /* z = _mm_sub_epi8(z, q_);                       */ \
	for (int i = 0; i < VECT_SIZE; i++) a[i] = a[i] - z[i]; /* a = _mm_sub_epi8(a, z);                        */ \
	for (int i = 0; i < VECT_SIZE; i++) b[i] = b[i] - z[i]; /* b = _mm_sub_epi8(b, z);                        */

	int r, t, qe = q + e, n_col_, /* *off = 0, */ *off_end = 0, tlen_, qlen_, last_st, last_en, wl, wr, max_sc, min_sc;
	int with_cigar = !(flag & KSW_EZ_SCORE_ONLY), approx_max = !!(flag & KSW_EZ_APPROX_MAX);
	// int32_t *H = 0;
	int32_t H0 = 0, last_H0_t = 0;
	uint8_t *qr, *sf; //, *mem, *mem2 = 0;
	// __m128i q_,        qe2_,            zero_,            flag1_,            flag2_,            flag8_,            flag16_,            sc_mch_,            sc_mis_,            sc_N_,            m1_,            max_sc_;
	int8_t q_[VECT_SIZE], qe2_[VECT_SIZE], zero_[VECT_SIZE], flag1_[VECT_SIZE], flag2_[VECT_SIZE], flag8_[VECT_SIZE], flag16_[VECT_SIZE], sc_mch_[VECT_SIZE], sc_mis_[VECT_SIZE], sc_N_[VECT_SIZE], m1_[VECT_SIZE], max_sc_[VECT_SIZE]; // 128-bit intrinsic data type, packed values are in integer format
	// __m128i *u, *v, *x, *y, *s, *p = 0;
	int8_t *u, *v, *x, *y, *s, *p = 0; // 128-bit intrinsic data type, packed values are in integer format

	// ksw_reset_extz(ez);
	if (m <= 0 || qlen <= 0 || tlen <= 0) return;

	for (int i = 0; i < VECT_SIZE; i++) zero_[i] = 0; // zero_   = _mm_set1_epi8(0);
	for (int i = 0; i < VECT_SIZE; i++) q_[i] = q; // q_      = _mm_set1_epi8(q);
	for (int i = 0; i < VECT_SIZE; i++) qe2_[i] = (q + e) * 2; // qe2_    = _mm_set1_epi8((q + e) * 2);
	for (int i = 0; i < VECT_SIZE; i++) flag1_[i] = 1; // flag1_  = _mm_set1_epi8(1);
	for (int i = 0; i < VECT_SIZE; i++) flag1_[i] = 2; // flag2_  = _mm_set1_epi8(2);
	for (int i = 0; i < VECT_SIZE; i++) flag1_[i] = 0x08; // flag8_  = _mm_set1_epi8(0x08);
	for (int i = 0; i < VECT_SIZE; i++) flag1_[i] = 0x10; // flag16_ = _mm_set1_epi8(0x10);
	for (int i = 0; i < VECT_SIZE; i++) sc_mch_[i] = mat[0]; // sc_mch_ = _mm_set1_epi8(mat[0]);
	for (int i = 0; i < VECT_SIZE; i++) sc_mis_[i] = mat[1]; // sc_mis_ = _mm_set1_epi8(mat[1]);
	for (int i = 0; i < VECT_SIZE; i++) sc_N_[i] = mat[m * m - 1] == 0 ? -e : mat[m * m - 1]; // sc_N_   = mat[m*m-1] == 0? _mm_set1_epi8(-e) : _mm_set1_epi8(mat[m*m-1]);
	for (int i = 0; i < VECT_SIZE; i++) m1_[i] = m - 1; // m1_     = _mm_set1_epi8(m - 1); // wildcard
	for (int i = 0; i < VECT_SIZE; i++) max_sc_[i] = mat[0] + (q + e) * 2; // max_sc_ = _mm_set1_epi8(mat[0] + (q + e) * 2);

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
	if (-min_sc > 2 * (q + e)) return; // otherwise, we won't see any mismatches

	// mem = (uint8_t*)kcalloc(km, tlen_ * 6 + qlen_ + 1, 16); // calloc an array of (tlen_ * 8 + qlen_ + 1) elements, each of 16 bytes
	u = (int8_t*)(((size_t)mem + 15) >> 4 << 4); // u = (__m128i*)(((size_t)mem + 15) >> 4 << 4); /* 16-byte aligned */

	// v = u + tlen_,   x = v + tlen_,      y = x + tlen_,      s = y + tlen_,      sf = (uint8_t*)(s + tlen_),    qr = sf + tlen_ * 16;
	v = (u + tlen_ * 16), x = (v + tlen_ * 16), y = (x + tlen_ * 16), s = (y + tlen_ * 16), sf = (uint8_t*)(s + tlen_ * 16), qr = sf + tlen_ * 16;

	if (!approx_max) {
		// H = (int32_t*)kmalloc(km, tlen_ * 16 * 4);
		for (t = 0; t < tlen_ * 16; ++t) H[t] = KSW_NEG_INF;
	}

	if (with_cigar) {
		// mem2 = (uint8_t*)kmalloc(km, ((size_t)(qlen + tlen - 1) * n_col_ + 1) * 16);
		// p = (__m128i*)(((size_t)mem2 + 15) >> 4 << 4);
		p = (int8_t*)(((size_t)mem2 + 15) >> 4 << 4);
		// off = (int*)kmalloc(km, (qlen + tlen - 1) * sizeof(int) * 2);
		off_end = off + qlen + tlen - 1;
	}

	for (t = 0; t < qlen; ++t) qr[t] = query[qlen - 1 - t]; // reverse query to read aligned memory
	memcpy(sf, target, tlen);

	for (r = 0, last_st = last_en = -1; r < qlen + tlen - 1; ++r) {
		int st = 0, en = tlen - 1, st0, en0, st_, en_;
		int8_t x1, v1;
		uint8_t* qrr = qr + (qlen - 1 - r);
		int8_t *u8 = u, *v8 = v;
		//__m128i x1_, v1_;
		uint8_t x1_[VECT_SIZE], v1_[VECT_SIZE];

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
			if (st - 1 >= last_st && st - 1 <= last_en)
				x1 = x[st - 1], v1 = v8[st - 1]; // (r-1,s-1) calculated in the last round
			else
				x1 = v1 = 0; // not calculated; set to zeros
		} else
			x1 = 0, v1 = r ? q : 0;
		if (en >= r) y[r] = 0, u8[r] = r ? q : 0;
		// loop fission: set scores first
		if (!(flag & KSW_EZ_GENERIC_SC)) {
			for (t = st0; t <= en0; t += 16) {
				// __m128i sq, st, tmp, mask;
				uint8_t sq[VECT_SIZE], st[VECT_SIZE], mask[VECT_SIZE], tmp[VECT_SIZE];

				uint8_t tmp1[VECT_SIZE], tmp2[VECT_SIZE];
				for (int i = 0; i < VECT_SIZE; i++) sq[i] = sf[t + i]; // sq = _mm_loadu_si128((__m128i*)&sf[t]); // load 128 bits from memory (does not require memory to be aligned)
				for (int i = 0; i < VECT_SIZE; i++) st[i] = qrr[t + i]; // st = _mm_loadu_si128((__m128i*)&qrr[t]);

				// mask = _mm_or_si128(_mm_cmpeq_epi8(sq, m1_), _mm_cmpeq_epi8(st, m1_));
				for (int i = 0; i < VECT_SIZE; i++) tmp1[VECT_SIZE - 1 - i] = sq[VECT_SIZE - 1 - i] == m1_[VECT_SIZE - 1 - i] ? 0xFF : 0;
				for (int i = 0; i < VECT_SIZE; i++) tmp2[VECT_SIZE - 1 - i] = st[VECT_SIZE - 1 - i] == m1_[VECT_SIZE - 1 - i] ? 0xFF : 0;
				for (int i = 0; i < VECT_SIZE; i++) mask[i] = tmp1[i] | tmp2[i];
				// tmp = _mm_cmpeq_epi8(sq, st);
				for (int i = 0; i < VECT_SIZE; i++) tmp[VECT_SIZE - 1 - i] = sq[VECT_SIZE - 1 - i] == st[VECT_SIZE - 1 - i] ? 0xFF : 0;

				// tmp = _mm_or_si128(_mm_andnot_si128(tmp,  sc_mis_), _mm_and_si128(tmp,  sc_mch_));
				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & sc_mis_[i];
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = tmp[i] & sc_mch_[i];
				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = tmp1[i] | tmp2[i];

				// tmp = _mm_or_si128(_mm_andnot_si128(mask, tmp),     _mm_and_si128(mask, sc_N_));
				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~mask[i]) & tmp[i];
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = mask[i] & sc_N_[i];
				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = tmp1[i] | tmp2[i];
				for (int i = 0; i < VECT_SIZE; i++) s[t + i] = tmp[i]; // _mm_storeu_si128((__m128i*)((uint8_t*)s + t), tmp);
			}
		} else {
			for (t = st0; t <= en0; ++t)
				s[t] = mat[sf[t] * m + qrr[t]]; // ((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
		}
		/* core loop */
		for (int i = 0; i < VECT_SIZE; i++) x1_[i] = 0;
		x1_[0] = (uint8_t)x1; // x1_  = _mm_cvtsi32_si128(x1);
		for (int i = 0; i < VECT_SIZE; i++) v1_[i] = 0;
		v1_[0] = (uint8_t)v1; // v1_  = _mm_cvtsi32_si128(v1);
		st_ = st / 16, en_ = en / 16;
		assert(en_ - st_ + 1 <= n_col_);
		if (!with_cigar) { // score only
			for (t = st_; t <= en_; ++t) {
				// __m128i z, a, b, xt1, vt1, ut, tmp;
				int8_t z[VECT_SIZE];
				int8_t a[VECT_SIZE];
				int8_t b[VECT_SIZE];
				int8_t xt1[VECT_SIZE];
				int8_t vt1[VECT_SIZE];

				int8_t tmp[VECT_SIZE];
				int8_t ut[VECT_SIZE];
				int8_t tmp1[VECT_SIZE];

				__dp_code_block1;

				for (int i = 0; i < VECT_SIZE; i++) z[i] = z[i] & (z[i] > zero_[i] ? 0xFF : 0); // z = _mm_and_si128(z, _mm_cmpgt_epi8(z, zero_));  // z = z > 0? z : 0;

				for (int i = 0; i < VECT_SIZE; i++) z[i] = z[i] > a[i] ? z[i] : a[i]; // z = _mm_max_epu8(z, a);                          // z = max(z, a); this works because both are non-negative


				__dp_code_block2;

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = a[i] > zero_[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(a, zero_);
				for (int i = 0; i < VECT_SIZE; i++) x[t * VECT_SIZE + i] = (tmp[i] & a[i]); // _mm_store_si128(&x[t], _mm_and_si128(a, tmp));
				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = b[i] > zero_[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(b, zero_);
				for (int i = 0; i < VECT_SIZE; i++) y[t * VECT_SIZE + i] = b[i] & tmp[i]; // _mm_store_si128(&y[t], _mm_and_si128(b, tmp));
			}
		} else if (!(flag & KSW_EZ_RIGHT)) { /* gap left-alignment */
			int8_t* pr = p + ((size_t)r * n_col_ - st_) * 16; // __m128i *pr = p + (size_t)r * n_col_ - st_;
			off[r] = st, off_end[r] = en;
			for (t = st_; t <= en_; ++t) {
				// __m128i d, z, a, b, xt1, vt1, ut, tmp;
				int8_t d[VECT_SIZE];
				int8_t z[VECT_SIZE];
				int8_t a[VECT_SIZE];
				int8_t b[VECT_SIZE];
				int8_t xt1[VECT_SIZE];
				int8_t vt1[VECT_SIZE];
				int8_t ut[VECT_SIZE];
				int8_t tmp[VECT_SIZE], tmp1[VECT_SIZE], tmp2[VECT_SIZE];

				__dp_code_block1;

				for (int i = 0; i < VECT_SIZE; i++) d[i] = (a[i] > z[i] ? 0xFF : 0) & flag1_[i]; // d = _mm_and_si128(_mm_cmpgt_epi8(a, z), flag1_); // d = a > z? 1 : 0
				for (int i = 0; i < VECT_SIZE; i++) z[i] = (z[i] > zero_[i] ? 0xFF : 0) & z[i]; // z = _mm_and_si128(z, _mm_cmpgt_epi8(z, zero_));  // z = z > 0? z : 0;
				for (int i = 0; i < VECT_SIZE; i++) z[i] = a[i] > z[i] ? a[i] : z[i]; // z = _mm_max_epu8(z, a);                          // z = max(z, a); this works because both are non-negative
				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = b[i] > z[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(b, z);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & flag2_[i]; // d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, flag2_)); // d = b > z? 2 : d; emulating blendv
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = (~tmp[i]) & d[i];
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | tmp2[i];


				__dp_code_block2;

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = a[i] > zero_[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(a, zero_);

				for (int i = 0; i < VECT_SIZE; i++) x[t * VECT_SIZE + i] = (tmp[i] & a[i]); // _mm_store_si128(&x[t], _mm_and_si128(tmp, a));

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & flag8_[i]; // d = _mm_or_si128(d, _mm_and_si128(tmp, flag8_));  // d = a > 0? 0x08 : 0
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | d[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = b[i] > zero_[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(b, zero_);

				for (int i = 0; i < VECT_SIZE; i++) y[t * VECT_SIZE + i] = tmp[i] & b[i]; // _mm_store_si128(&y[t], _mm_and_si128(tmp, b));

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = tmp[i] & flag16_[i]; // d = _mm_or_si128(d, _mm_and_si128(tmp, flag16_)); // d = b > 0? 0x10 : 0
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | d[i];

				for (int i = 0; i < VECT_SIZE; i++) pr[t * VECT_SIZE + i] = d[i]; // _mm_store_si128(&pr[t], d);
			}
		} else { /* gap right-alignment */
			int8_t* pr = p + ((size_t)r * n_col_ - st_) * 16; // __m128i *pr = p + (size_t)r * n_col_ - st_;
			off[r] = st, off_end[r] = en;
			for (t = st_; t <= en_; ++t) {
				// __m128i d, z, a, b, xt1, vt1, ut, tmp;
				int8_t ut[VECT_SIZE];
				int8_t vt1[VECT_SIZE];
				int8_t xt1[VECT_SIZE];
				int8_t b[VECT_SIZE];
				int8_t d[VECT_SIZE];
				int8_t z[VECT_SIZE];
				int8_t a[VECT_SIZE];
				int8_t tmp[VECT_SIZE];
				int8_t tmp1[VECT_SIZE];
				int8_t tmp2[VECT_SIZE];

				__dp_code_block1;
				for (int i = 0; i < VECT_SIZE; i++) d[i] = (~(z[i] > a[i] ? 0xFF : 0)) & flag1_[i]; // d = _mm_andnot_si128(_mm_cmpgt_epi8(z, a), flag1_); // d = z > a? 0 : 1

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = z[i] > zero_[i] ? 0xFF : 0; // z = _mm_and_si128(z, _mm_cmpgt_epi8(z, zero_));  // z = z > 0? z : 0;
				for (int i = 0; i < VECT_SIZE; i++) z[i] = tmp1[i] & z[i];

				for (int i = 0; i < VECT_SIZE; i++) z[i] = z[i] > a[i] ? z[i] : a[i]; // z = _mm_max_epu8(z, a);                          // z = max(z, a); this works because both are non-negative

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = z[i] > b[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(z, b);

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & flag2_[i]; // d = _mm_or_si128(_mm_andnot_si128(tmp, flag2_), _mm_and_si128(tmp, d)); // d = z > b? d : 2; emulating blendv
				for (int i = 0; i < VECT_SIZE; i++) tmp2[i] = tmp[i] & d[i];
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | tmp2[i];

				__dp_code_block2;
				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = zero_[i] > a[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(zero_, a);

				for (int i = 0; i < VECT_SIZE; i++) x[t * VECT_SIZE + i] = (~tmp[i]) & a[i]; // _mm_store_si128(&x[t], _mm_andnot_si128(tmp, a));

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & flag8_[i]; //  d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag8_));  // d = 0 > a? 0 : 0x08
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | d[i];

				for (int i = 0; i < VECT_SIZE; i++) tmp[i] = zero_[i] > b[i] ? 0xFF : 0; // tmp = _mm_cmpgt_epi8(zero_, b);

				for (int i = 0; i < VECT_SIZE; i++) y[t * VECT_SIZE + i] = (~tmp[i]) & b[i]; // _mm_store_si128(&y[t], _mm_andnot_si128(tmp, b));

				for (int i = 0; i < VECT_SIZE; i++) tmp1[i] = (~tmp[i]) & flag16_[i]; // d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag16_)); // d = 0 > b? 0 : 0x10
				for (int i = 0; i < VECT_SIZE; i++) d[i] = tmp1[i] | d[i];

				for (int i = 0; i < VECT_SIZE; i++) pr[t * VECT_SIZE + i] = d[i]; // _mm_store_si128(&pr[t], d);
			}
		}
		if (!approx_max) { /* find the exact max with a 32-bit score array */
			int32_t max_H, max_t;
			/* compute H[], max_H and max_t */
			if (r > 0) {
				int32_t HH[4], tt[4], en1 = st0 + (en0 - st0) / 4 * 4, i;
				int32_t max_H_[4], max_t_[4], qe_[4]; // __m128i max_H_, max_t_, qe_;
				max_H = H[en0] = en0 > 0 ? H[en0 - 1] + u8[en0] - qe : H[en0] + v8[en0] - qe; /* special casing the last element */
				max_t = en0;
				for (int i = 0; i < 4; i++) max_H_[i] = max_H; // max_H_ = _mm_set1_epi32(max_H);
				for (int i = 0; i < 4; i++) max_t_[i] = max_t; // max_t_ = _mm_set1_epi32(max_t);
				for (int i = 0; i < 4; i++) qe_[i] = q + e; // qe_    = _mm_set1_epi32(q + e);
				for (t = st0; t < en1; t += 4) { /* this implements: H[t]+=v8[t]-qe; if(H[t]>max_H) max_H=H[t],max_t=t; */
					int32_t H1[4], tmp[4], t_[4]; // __m128i H1, tmp, t_;
					for (int i = 0; i < 4; i++) H1[i] = H[t + i]; // H1 = _mm_loadu_si128((__m128i*)&H[t]);
					t_[0] = v8[t]; // t_ = _mm_setr_epi32(v8[t], v8[t+1], v8[t+2], v8[t+3]);
					t_[1] = v8[t + 1];
					t_[2] = v8[t + 2];
					t_[3] = v8[t + 3];
					for (int i = 0; i < 4; i++) H1[i] = H1[i] + t_[i]; // H1 = _mm_add_epi32(H1, t_);
					for (int i = 0; i < 4; i++) H1[i] = H1[i] - qe_[i]; // H1 = _mm_sub_epi32(H1, qe_);
					for (int i = 0; i < 4; i++) H[t + i] = H1[i]; // _mm_storeu_si128((__m128i*)&H[t], H1);
					for (int i = 0; i < 4; i++) t_[i] = t; // t_ = _mm_set1_epi32(t);
					for (int i = 0; i < 4; i++) tmp[i] = H1[i] > max_H_[i] ? 0xFFFFFFFF : 0; // tmp = _mm_cmpgt_epi32(H1, max_H_);

					for (int i = 0; i < 4; i++) max_H_[i] = (tmp[i] & H1[i]) | ((~tmp[i]) & max_H_[i]); // max_H_ = _mm_or_si128(_mm_and_si128(tmp, H1), _mm_andnot_si128(tmp, max_H_));
					for (int i = 0; i < 4; i++) max_t_[i] = (tmp[i] & t_[i]) | ((~tmp[i]) & max_t_[i]); // max_t_ = _mm_or_si128(_mm_and_si128(tmp, t_), _mm_andnot_si128(tmp, max_t_));
				}
				for (int i = 0; i < 4; i++) HH[i] = max_H_[i]; // _mm_storeu_si128((__m128i*)HH, max_H_);
				for (int i = 0; i < 4; i++) tt[i] = max_t_[i]; // _mm_storeu_si128((__m128i*)tt, max_t_);
				for (i = 0; i < 4; ++i)
					if (max_H < HH[i]) max_H = HH[i], max_t = tt[i] + i;
				for (; t < en0; ++t) { /* for the rest of values that haven't been computed with SSE */
					H[t] += (int32_t)v8[t] - qe;
					if (H[t] > max_H)
						max_H = H[t], max_t = t;
				}
			} else
				H[0] = v8[0] - qe - qe, max_H = H[0], max_t = 0; // special casing r==0
			// update ez
			if (en0 == tlen - 1 && H[en0] > ez->mte)
				ez->mte = H[en0], ez->mte_q = r - en0;
			if (r - st0 == qlen - 1 && H[st0] > ez->mqe)
				ez->mqe = H[st0], ez->mqe_t = st0;
			if (ksw_apply_zdrop_gpu(ez, 1, max_H, r, max_t, zdrop, e)) break;
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez->score = H[tlen - 1];
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
				if ((flag & KSW_EZ_APPROX_DROP) && ksw_apply_zdrop_gpu(ez, 1, H0, r, last_H0_t, zdrop, e)) break;
			} else
				H0 = v8[0] - qe - qe, last_H0_t = 0;
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez->score = H0;
		}
		last_st = st, last_en = en;
		//for (t = st0; t <= en0; ++t) printf("(%d,%d)\t(%d,%d,%d,%d)\t%d\n", r, t, ((int8_t*)u)[t], ((int8_t*)v)[t], ((int8_t*)x)[t], ((int8_t*)y)[t], H[t]); // for debugging
	}

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
