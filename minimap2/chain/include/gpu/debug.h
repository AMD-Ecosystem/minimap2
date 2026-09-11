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

#ifndef __DEBUG_H__
#define __DEBUG_H__
#include "plutils.h"
#include "chain_priv.h"

#ifdef __cplusplus
extern "C" {
#endif

const char debug_folder[] = "debug";

// #define ITER_LIMIT 10000
// #define MAX_READ_NUM 100000
// #define MEM_CPU (96-6) // 96 - 6 GB for possible exceed read
// #define MEM_GPU (16-4) // 16 - 4 GB as memory pool = 16760832(0xffc000) KB
// #define SATURATE_FACTOR (0.7) // NOTE: how much portion of cpu memory shall be allocated, < 1


// /* Input File Names: Set by command line arguments in main.c */
// extern char input_filename[];   // plaintxt chaining inputs & score
// extern char range_infile[];     // plaintxt range
// extern char binary_file[];      // binary chaining inputs & score
// extern char binary_range[];     // binary range

#ifndef DEBUG_CHECK
#define ASSERT(X)
// Chaining Debug Checker: checks chaining score against input.
#elif DEBUG_CHECK
#define ASSERT(X) assert(X)
// Read score from file for comparison
int debug_read_score(const char input_filename[], chain_read_t* in, void* km);
int debug_build_score(chain_read_t* in, void* km);

// Check score
int debug_check_score(const int64_t* p, const int32_t* f, const int64_t* p_gold,
    const int32_t* f_gold, int64_t n, char* qname);
void debug_check_range(const int32_t* range, size_t n);
int debug_check_cut(const size_t* cut, const int32_t* range, size_t max_cut,
    size_t n, size_t offset);

#ifdef DEBUG_CHECK_FORCE
mm128_t* mg_lchain_dp(int max_dist_x, int max_dist_y, int bw, int max_skip,
    int max_iter, int min_cnt, int min_sc, float chn_pen_gap,
    float chn_pen_skip, int is_cdna, int n_seg,
    int64_t n, // NOTE: n is number of anchors
    mm128_t* a, // NOTE: a is ptr to anchors.
    int* n_u_, uint64_t** _u, void* km, chain_read_t* input,
    int32_t* f_, int64_t* p_);
#endif // DEBUG_CHECK_FORCE


// Analyze Distribution

void debug_cal_long_seg_range_dis(size_t total_n, size_t num_cut, int32_t* range);
void debug_cal_mid_range_dis(size_t total_n, size_t num_cut, int32_t* range);
void debug_cal_range_dis(size_t total_n, size_t num_cut, int32_t* range);
void debug_cal_sc_pair_density(size_t total_n, size_t num_cut, size_t* cut, int32_t* range);
#endif // DEBUG_CHECK

#ifdef DEBUG_VERBOSE
typedef struct {
	mm_seq_meta_t* refs;
	int n_refs;
	Misc misc;
} input_meta_t;

static inline void free_meta_struct(input_meta_t* meta, void* km)
{
	if (meta->refs) kfree(km, meta->refs);
}
// Print Input Files
void debug_output_anchors(const char debug_folder[], chain_read_t* in);
void debug_output_score(const char debug_folder[], chain_read_t* in);
void debug_output_meta(const char debug_folder[], input_meta_t* meta);

void debug_print_successor_range(int32_t* range, int64_t n);
int debug_print_cut(const size_t* cut, size_t max_cut, size_t n, size_t offset, char* qname);
void debug_print_score(const int64_t* p, const int32_t* score, int64_t n);
void debug_print_score_rel_p(const uint16_t* p, const int32_t* score, int64_t n);
void debug_print_chain(mm128_t* a, uint64_t* u, int32_t n_u, char* qname);
void debug_print_backtrack_params(char* qname, int64_t n, int32_t* f, int64_t* p, int32_t min_cnt, int64_t min_score, float max_drop);
void debug_print_regs(mm_reg1_t* regs, int n_u, char* qname);
void debug_print_segs(seg_t* segs, chain_read_t* reads, int num_segs, int num_reads);
void debug_check_anchors(seg_t* segs, int num_segs, int32_t* ax_aggregated, int32_t* ax);
#endif // DEBUG_VERBOSE

#ifdef __cplusplus
}
#endif

#endif // __DEBUG_H__
