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

#ifndef MM_DUMP_H
#define MM_DUMP_H

#include <stdint.h>
#include "minimap.h"
#include "core_ctx.h"

#ifdef __cplusplus
extern "C" {
#endif

struct chain_read_s;
typedef struct chain_read_s chain_read_t;

int mm_dump_init(mm_core_ctx_t* ctx);
void mm_dump_cleanup(mm_core_ctx_t* ctx);

int mm_dump_seeds(const mm_core_ctx_t* ctx, const char* qname,
    const chain_read_t* read, const mm_idx_t* mi);

int mm_dump_chains(const mm_core_ctx_t* ctx, const char* qname,
    const chain_read_t* read, const mm_idx_t* mi);

int mm_dump_alignments(const mm_core_ctx_t* ctx, const char* qname,
    int n_segs, int qlen_sum, const int* n_regs, mm_reg1_t** regs,
    const mm_idx_t* mi);

int mm_dump_write_timer_json(const char* dir, const mm_timer_stats_t* ts,
    double wall_time, double cpu_time, double peak_rss_gb, int n_threads);

#ifdef __cplusplus
}
#endif

#endif // MM_DUMP_H
