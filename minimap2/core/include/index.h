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

#ifndef MM2_INDEX_H
#define MM2_INDEX_H

#include <stdio.h>
#include <stdint.h>
#include "minimap.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	int32_t off, off2, cnt;
	int16_t strand;
	uint16_t flag;
} mm_idx_jjump1_t;

/* Index query functions */
const uint64_t* mm_idx_get(const mm_idx_t* mi, uint64_t minier, int* n);
int32_t mm_idx_cal_max_occ(const mm_idx_t* mi, float f);

/* Sequence retrieval - mm_idx_getseq2 adds is_rev parameter for reverse complement */
int mm_idx_getseq2(const mm_idx_t* mi, int is_rev, uint32_t rid, uint32_t st, uint32_t en, uint8_t* seq);
int mm_idx_jjump_read(mm_idx_t* mi, const char* fn, int flag, int min_sc);
const mm_idx_jjump1_t* mm_idx_jump_get(const mm_idx_t* db, int32_t cid, int32_t st, int32_t en, int32_t* n);

#ifdef __cplusplus
}
#endif

#endif /* MM2_INDEX_H */
