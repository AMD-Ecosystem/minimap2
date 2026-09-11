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

#ifndef MM2_FORMAT_H
#define MM2_FORMAT_H

#include <stdint.h>
#include "minimap.h"
#include "bseq.h"
#include "kseq.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Output/format functions - defined in format.c */
int mm_write_sam_hdr(const mm_idx_t* mi, const char* rg, const char* ver, int argc, char* argv[]);
void mm_write_paf(kstring_t* s, const mm_idx_t* mi, const mm_bseq1_t* t, const mm_reg1_t* r, void* km, int64_t opt_flag);
void mm_write_paf3(kstring_t* s, const mm_idx_t* mi, const mm_bseq1_t* t, const mm_reg1_t* r, void* km, int64_t opt_flag, int rep_len);
void mm_write_paf4(kstring_t* s, const mm_idx_t* mi, const mm_bseq1_t* t, const mm_reg1_t* r, void* km, int64_t opt_flag, int rep_len, int n_seg, int seg_idx);
void mm_write_sam(kstring_t* s, const mm_idx_t* mi, const mm_bseq1_t* t, const mm_reg1_t* r, int n_regs, const mm_reg1_t* regs);
void mm_write_sam2(kstring_t* s, const mm_idx_t* mi, const mm_bseq1_t* t, int seg_idx, int reg_idx, int n_seg, const int* n_regs, const mm_reg1_t* const* regs, void* km, int64_t opt_flag);
void mm_write_sam3(kstring_t* s, const mm_idx_t* mi, const mm_bseq1_t* t, int seg_idx, int reg_idx, int n_seg, const int* n_regss, const mm_reg1_t* const* regss, void* km, int64_t opt_flag, int rep_len);
void mm_write_junc(kstring_t* s, const mm_idx_t* mi, const mm_bseq1_t* t, const mm_reg1_t* r);

#ifdef __cplusplus
}
#endif

#endif /* MM2_FORMAT_H */
