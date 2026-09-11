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
 * @file mm2_chain.h
 * @brief Public API for the mm2_chain library
 *
 * This header provides the public interface for the mm2_chain library,
 * which implements the chaining stage of the minimap2 mapping pipeline.
 * Chaining links individual seed anchors into colinear chains that
 * represent potential alignments.
 *
 * Main entry point: mm_map_chain()
 *
 * Usage (with full minimap2 pipeline):
 * @code
 *   // After seeding stage has populated read_ structure
 *   mm_map_chain(mi, opt, read_, &t_chain, km);
 *   // Process read_->n_u chains in read_->u[], anchors in read_->a[]
 * @endcode
 *
 * Thread safety: mm_map_chain() is thread-safe when each thread uses its own
 * memory pool (km).
 *
 * Library dependencies: mm2_seed (for mm128_t, chain_read_t), mm2_core (for kalloc)
 */

#ifndef MM2_CHAIN_H
#define MM2_CHAIN_H

#include <stdint.h>
#include "mm2_seed.h" /* Provides mm128_t, chain_read_t, and other seed types */
#include "minimap.h" /* Provides mm_idx_t, mm_mapopt_t */

#ifdef __cplusplus
extern "C" {
#endif

/* API export macro */
#ifndef MM_CHAIN_API
#define MM_CHAIN_API
#endif

/**
 * @brief Main chaining stage entry point - link seed hits into chains
 *
 * Second stage of the mapping pipeline. Links individual seed anchors
 * (from mm_map_seed) into colinear chains using dynamic programming
 * or RMQ algorithms.
 *
 * Input/Output through chain_read_t structure:
 * - Input: read_->a (anchors), read_->n (anchor count)
 * - Output: read_->u (chain info), read_->n_u (chain count)
 *
 * @param mi minimap2 index (used for k-mer size and re-seeding)
 * @param opt mapping options (controls chaining parameters)
 * @param read_ query read structure (containing seed anchors from Stage 1)
 * @param t_chain pointer to chain timer (can be NULL to skip timing)
 * @param km memory pool for allocation (NULL uses malloc)
 * @param ctx core context for dump dispatch (can be NULL to skip dumping)
 *
 * Algorithm selection:
 * - MM_F_RMQ flag in opt: Use RMQ-based chaining (faster for ultra-long reads)
 * - Otherwise: Use DP-based chaining (default, suitable for most cases)
 *
 * Special handling:
 * - Long-join re-chaining: When bw_long > bw and sequence appears fragmented,
 *   automatically re-chains with wider bandwidth
 * - Short read re-chaining: When chains incomplete, re-seeds with higher max_occ
 *
 * Thread safety: Thread-safe when each thread uses its own memory pool.
 *
 * @note The anchors array (read_->a) is modified in-place and reordered.
 */
MM_CHAIN_API void mm_map_chain(const mm_idx_t* mi, const mm_mapopt_t* opt,
    chain_read_t* read_, mm_stage_timer_t* t_chain, void* km, const mm_core_ctx_t* ctx);

/* ============================================================================
 * Utility macros for chain result processing
 * ============================================================================ */

/** Extract chain score from u[i] - score is in upper 32 bits */
#define MM_CHAIN_SCORE(u) ((int32_t)((u) >> 32))

/** Extract anchor count from u[i] - count is in lower 32 bits */
#define MM_CHAIN_COUNT(u) ((int32_t)(u))

#ifdef __cplusplus
}
#endif

#endif /* MM2_CHAIN_H */
