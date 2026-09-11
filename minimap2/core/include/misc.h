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

#ifndef MM2_MISC_H
#define MM2_MISC_H

#include <stdio.h>
#include <stdint.h>
#include "minimap.h"

#ifdef __cplusplus
extern "C" {
#endif

#define mm_seq4_set(s, i, c) ((s)[(i) >> 3] |= (uint32_t)(c) << (((i) & 7) << 2))
#define mm_seq4_get(s, i) ((s)[(i) >> 3] >> (((i) & 7) << 2) & 0xf)

#define MALLOC(type, len) ((type*)malloc((len) * sizeof(type)))
#define CALLOC(type, len) ((type*)calloc((len), sizeof(type)))
#define REALLOC(type, ptr, cnt) ((type*)realloc((ptr), (cnt) * sizeof(type)))

#ifndef kroundup32
#define kroundup32(x) (--(x), (x) |= (x) >> 1, (x) |= (x) >> 2, (x) |= (x) >> 4, (x) |= (x) >> 8, (x) |= (x) >> 16, ++(x))
#endif

/* Application-level global defaults; see minimap.h for full comment.
 * Pipeline code should prefer core_ctx->verbose / core_ctx->dbg_flag. */
extern int mm_verbose;
extern int mm_dbg_flag;
extern double mm_realtime0;

/* Debug flags */
#define MM_DBG_NO_KALLOC 0x1
#define MM_DBG_PRINT_QNAME 0x2
#define MM_DBG_PRINT_SEED 0x4
#define MM_DBG_PRINT_ALN_SEQ 0x8
#define MM_DBG_PRINT_CHAIN 0x10
#define MM_DBG_SEED_FREQ 0x20

#define KSW_SPSC_OFFSET 64

/* Utility functions */
double cputime(void);
double realtime(void);
long peakrss(void);

/* Radix sort functions */
void radix_sort_128x(mm128_t* beg, mm128_t* end);
void radix_sort_64(uint64_t* beg, uint64_t* end);

/* ksmall function */
uint32_t ks_ksmall_uint32_t(size_t n, uint32_t arr[], size_t kk);

/* Error handling functions */
void mm_err_puts(const char* str);
void mm_err_fwrite(const void* p, size_t size, size_t nitems, FILE* fp);
void mm_err_fread(void* p, size_t size, size_t nitems, FILE* fp);

/* Inline utility functions */
static inline float mg_log2(float x) // NB: this doesn't work when x<2
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

#ifdef __cplusplus
}
#endif

#endif /* MM2_MISC_H */
