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

#ifndef MM2_SPLITIDX_H
#define MM2_SPLITIDX_H

#include <stdio.h>
#include <stdint.h>
#include "minimap.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Split index functions - defined in splitidx.c */
FILE* mm_split_init(const char* prefix, const mm_idx_t* mi);
mm_idx_t* mm_split_merge_prep(const char* prefix, int n_splits, FILE** fp, uint32_t* n_seq_part);
int mm_split_merge(int n_segs, const char** fn, const mm_mapopt_t* opt, int n_split_idx);
void mm_split_rm_tmp(const char* prefix, int n_splits);

#ifdef __cplusplus
}
#endif

#endif /* MM2_SPLITIDX_H */
