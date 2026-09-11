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

/**
 * @file align_internal.h
 * @brief Alignment primitives in align.c that the GPU translation unit calls.
 *
 * Mirror of gpu/align_gpu.h: this header carries the align.c → align_gpu.cpp
 * direction; gpu/align_gpu.h carries the align_gpu.cpp → align.c direction.
 * Neither is part of the cross-module API — for that, see align_priv.h.
 *
 * Implementation: align/src/align.c
 */

#ifndef ALIGN_INTERNAL_H
#define ALIGN_INTERNAL_H

#include "minimap.h" /* mm_core_ctx_t fwd typedef, mm_idx_t, mm_mapopt_t, mm_reg1_t, mm128_t */
#include "ksw2.h" /* ksw_extz_t */
#include "mm2_align.h" /* mm_gpu_align_ctx_t fwd typedef */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Scoring / CIGAR / sequence helpers (defined in align.c) ---- */

void ksw_gen_simple_mat(int m, int8_t* mat, int8_t a, int8_t b, int8_t sc_ambi);

void ksw_gen_ts_mat(int m, int8_t* mat, int8_t a, int8_t b, int8_t transition, int8_t sc_ambi);

void mm_seq_rev(uint32_t len, uint8_t* seq);

int mm_test_zdrop(void* km, const mm_mapopt_t* opt, const uint8_t* qseq, const uint8_t* tseq,
    uint32_t n_cigar, uint32_t* cigar, const int8_t* mat);

void mm_update_extra(mm_reg1_t* r, const uint8_t* qseq, const uint8_t* tseq, const int8_t* mat,
    int8_t q, int8_t e, int is_eqx, int log_gap);

void mm_append_cigar(mm_reg1_t* r, uint32_t n_cigar, const uint32_t* cigar);

void mm_align_pair(void* km, const mm_mapopt_t* opt, int qlen, const uint8_t* qseq, int tlen,
    const uint8_t* tseq, const uint8_t* junc, const int8_t* mat, int w, int end_bonus, int zdrop,
    int flag, ksw_extz_t* ez, const mm_core_ctx_t* ctx, mm_gpu_align_ctx_t* gpu_align_ctx);

void mm_adjust_minier(const mm_idx_t* mi, uint8_t* const qseq0[2], mm128_t* a, int32_t* r,
    int32_t* q);

/* ---- Setup helpers (defined in align.c) ---- */

void mm_filter_bad_seeds(void* km, int as1, int cnt1, mm128_t* a, int min_gap, int diff_thres,
    int max_ext_len, int max_ext_cnt);

void mm_filter_bad_seeds_alt(void* km, int as1, int cnt1, mm128_t* a, int min_gap, int max_ext);

void mm_fix_bad_ends(const mm_reg1_t* r, const mm128_t* a, int bw, int min_match, int32_t* as,
    int32_t* cnt);

void mm_max_stretch(const mm_reg1_t* r, const mm128_t* a, int32_t* as, int32_t* cnt);

void mm_fix_bad_ends_splice(void* km, const mm_mapopt_t* opt, const mm_idx_t* mi,
    const mm_reg1_t* r, const int8_t mat[25], int qlen, uint8_t* qseq0[2], const mm128_t* a,
    int* as1, int* cnt1);

/* ---- Orchestration helpers (defined in align.c) ---- */

void mm_align1(void* km, const mm_mapopt_t* opt, const mm_idx_t* mi, int qlen, uint8_t* qseq0[2],
    mm_reg1_t* r, mm_reg1_t* r2, int n_a, mm128_t* a, ksw_extz_t* ez, int splice_flag,
    const mm_core_ctx_t* ctx, mm_gpu_align_ctx_t* gpu_align_ctx, int tid);

int mm_align1_inv(void* km, const mm_mapopt_t* opt, const mm_idx_t* mi, int qlen,
    uint8_t* qseq0[2], const mm_reg1_t* r1, const mm_reg1_t* r2, mm_reg1_t* r_inv, ksw_extz_t* ez,
    const mm_core_ctx_t* ctx);

mm_reg1_t* mm_insert_reg(const mm_reg1_t* r, int i, int* n_regs, mm_reg1_t* regs);

#ifdef __cplusplus
}
#endif

#endif /* ALIGN_INTERNAL_H */
