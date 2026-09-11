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

#ifndef MM_TIMER_H
#define MM_TIMER_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Per-stage timer: tracks sum, per-call min/max, and call count
typedef struct {
	double sum; // accumulated time across all calls
	double call_min; // minimum time for a single call
	double call_max; // maximum time for a single call
	uint64_t call_count; // number of calls
} mm_stage_timer_t;

// Per-thread timing accumulators for seed/chain/align stages
typedef struct {
	mm_stage_timer_t seed;
	mm_stage_timer_t chain;
	mm_stage_timer_t align;
} mm_tbuf_timers_t;

// Consolidated run-level timing statistics for seed/chain/align stages.
//
// For each stage:
//   *_sum: accumulated wall time across all threads and all batches
//   *_min: minimum per-thread time observed within any batch over the run
//   *_max: maximum per-thread time observed within any batch over the run
//   *_avg: average per-thread time per batch over the run
//   *_call_min/max/count: per-call (per-read) statistics across the entire run
typedef struct {
	double seed_min, seed_max, seed_avg, seed_sum;
	double chain_min, chain_max, chain_avg, chain_sum;
	double align_min, align_max, align_avg, align_sum;
	uint64_t batch_count;
	uint64_t query_count;
	// Per-call (per-read) statistics
	double seed_call_min, seed_call_max, seed_call_avg;
	double chain_call_min, chain_call_max, chain_call_avg;
	double align_call_min, align_call_max, align_call_avg;
	uint64_t seed_call_count, chain_call_count, align_call_count;
	uint64_t pool_reset_count;
} mm_timer_stats_t;

// Helper to record a single call's duration into a stage timer
static inline void mm_stage_timer_record(mm_stage_timer_t* st, double dt)
{
	if (!st) return;
	st->sum += dt;
	if (st->call_count == 0 || dt < st->call_min) st->call_min = dt;
	if (st->call_count == 0 || dt > st->call_max) st->call_max = dt;
	st->call_count++;
}

// Helper to merge per-call stats from src into dst (call_min/max/count only, not sum)
static inline void mm_stage_timer_merge_calls(mm_stage_timer_t* dst, const mm_stage_timer_t* src)
{
	if (!src || src->call_count == 0) return;
	if (dst->call_count == 0 || src->call_min < dst->call_min) dst->call_min = src->call_min;
	if (dst->call_count == 0 || src->call_max > dst->call_max) dst->call_max = src->call_max;
	dst->call_count += src->call_count;
}

static inline void mm_timer_stats_init(mm_timer_stats_t* ts)
{
	if (ts) memset(ts, 0, sizeof(mm_timer_stats_t));
}

#ifdef __cplusplus
}
#endif

#endif // MM_TIMER_H
