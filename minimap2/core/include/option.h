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
 * @file option.h
 * @brief Minimap2 options initialization declarations
 * 
 * Contains function declarations for option initialization.
 * Implementations are in core/src/options.c
 */

#ifndef MM2_OPTION_H
#define MM2_OPTION_H

#include "minimap.h"

#define MM_JUNC_ANNO 0x1
#define MM_JUNC_MISC 0x2

/* GPU alignment batching defaults */
#define MM_GPU_FLUSH_THRESHOLD_DEFAULT 8192
#define MM_GPU_BATCH_MAX_ALIGNMENTS_DEFAULT 16384
#define MM_GPU_BATCH_DEFAULT_MEM (4LL * 1024 * 1024 * 1024)
#define MM_GPU_BATCH_MAX_MEM_CAP (4LL * 1024 * 1024 * 1024) /* hard cap per-thread batch mem */

/* GPU VRAM budget formula constants (used by both chain and align).
 * per_thread = free_vram * UTIL_FRAC / n_threads, clamped to BUDGET_FLOOR. */
#define MM_GPU_VRAM_UTIL_FRAC 0.50 /* fraction of free VRAM allocatable for alignment */
#define MM_GPU_VRAM_UTIL_FRAC_XNACK 0.45 /* reduced when XNACK is enabled (pinned = HBM) */
#define MM_GPU_VRAM_BUDGET_FLOOR (1LL * 1024 * 1024 * 1024) /* 1 GB minimum per thread */
#define MM_GPU_VRAM_WARN_FRAC 0.80 /* warn if floor-clamped aggregate exceeds this */

#ifdef __cplusplus
extern "C" {
#endif

/* Option initialization functions */
void mm_idxopt_init(mm_idxopt_t* opt);
void mm_mapopt_init(mm_mapopt_t* opt);

/* Preset and validation functions */
int mm_set_opt(const char* preset, mm_idxopt_t* io, mm_mapopt_t* mo);
int mm_check_opt(const mm_idxopt_t* io, const mm_mapopt_t* mo);
void mm_mapopt_max_intron_len(mm_mapopt_t* opt, int max_intron_len);

#ifdef __cplusplus
}
#endif

#endif /* MM2_OPTION_H */
