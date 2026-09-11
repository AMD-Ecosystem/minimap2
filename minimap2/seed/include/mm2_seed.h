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
 * @file mm2_seed.h
 * @brief Public API for the mm2_seed library
 *
 * This header provides the public interface for the mm2_seed library,
 * which implements the seeding stage of the minimap2 mapping pipeline.
 * Seeding finds k-mer minimizer matches between query and reference sequences.
 *
 * Main entry point: mm_map_seed()
 *
 * Usage (with full minimap2 pipeline):
 * @code
 *   // Initialize read_ structure with query sequences
 *   mm_map_seed(mi, opt, read_, &t_seed, km);
 *   // Process read_->n anchors in read_->a[]
 * @endcode
 *
 * Thread safety: mm_map_seed() is thread-safe when each thread uses its own
 * memory pool (km).
 *
 * Library dependencies: mm2_core (for kalloc, kseq)
 */

#ifndef MM2_SEED_H
#define MM2_SEED_H

#include "minimap.h"
#include "mm_timer.h"
#include "chain_read.h"

#ifdef __cplusplus
extern "C" {
#endif

/* API export macro */
#ifndef MM_SEED_API
#define MM_SEED_API
#endif

/* ============================================================================
 * Seed-related constants
 * ============================================================================ */

#define MM_SEED_LONG_JOIN (1ULL << 40)
#define MM_SEED_IGNORE (1ULL << 41)
#define MM_SEED_TANDEM (1ULL << 42)
#define MM_SEED_SELF (1ULL << 43)

#define MM_SEED_SEG_SHIFT 48
#define MM_SEED_SEG_MASK (0xffULL << (MM_SEED_SEG_SHIFT))

/* ============================================================================
 * Types
 * ============================================================================ */

/**
 * @brief Seed structure containing minimizer hit information
 */
typedef struct {
	uint32_t n; /**< Number of hits for this minimizer */
	uint32_t q_pos; /**< Query position */
	uint32_t q_span : 31; /**< Query span (k-mer length) */
	uint32_t flt : 1; /**< Filter flag */
	uint32_t seg_id : 31; /**< Segment ID */
	uint32_t is_tandem : 1; /**< Tandem repeat flag */
	const uint64_t* cr; /**< Pointer to reference hits */
} mm_seed_t;

/* ============================================================================
 * Public API
 * ============================================================================ */

/**
 * @brief Main seeding stage entry point - find k-mer matches
 *
 * First stage of the mapping pipeline. Collects k-mer minimizers from query,
 * filters by seed quality/occupancy, and looks up in the index to find
 * candidate seed hits.
 *
 * Input through chain_read_t structure:
 * - read_->n_seg: Number of query segments
 * - read_->qlens: Array of query lengths
 * - read_->qseqs: Array of query sequences
 * - read_->seq.name: Query name
 *
 * Output through chain_read_t structure:
 * - read_->a: Array of seed anchors
 * - read_->n: Number of anchors
 * - read_->rep_len: Repetitive length
 * - read_->n_mini_pos: Number of minimizer positions
 * - read_->mini_pos: Minimizer positions array
 * - read_->seq.qlen_sum: Sum of query lengths
 *
 * @param mi      minimap2 index
 * @param opt     mapping options (controls k, w, occurrence thresholds)
 * @param read_   query read structure (containing query sequences)
 * @param t_seed  optional pointer to seed timer (can be NULL)
 * @param km      memory pool for allocation (NULL uses malloc)
 *
 * Thread safety: Thread-safe when each thread uses its own memory pool.
 */
MM_SEED_API void mm_map_seed(const mm_idx_t* mi, const mm_mapopt_t* opt,
    chain_read_t* read_, mm_stage_timer_t* t_seed, void* km,
    const mm_core_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif /* MM2_SEED_H */
