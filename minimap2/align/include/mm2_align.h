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
 * @file mm2_align.h
 * @brief Public API for the mm2_align library
 *
 * This header provides the public interface for the mm2_align library,
 * which implements the alignment stage of the minimap2 mapping pipeline.
 * Alignment performs base-level sequence alignment on chains using KSW2
 * and generates CIGAR strings.
 *
 * Main entry point: mm_map_align()
 *
 * Usage (with full minimap2 pipeline):
 * @code
 *   // After chaining stage has populated read_ structure
 *   mm_reg1_t *regs = NULL;
 *   int n_regs = 0;
 *   mm_map_align(mi, opt, read_, &regs, &n_regs, &t_align, km);
 *   // Process n_regs alignments in regs[]
 * @endcode
 *
 * Thread safety: mm_map_align() is thread-safe when each thread uses its own
 * memory pool (km).
 *
 * Library dependencies: mm2_chain (for chain_read_t), mm2_core (for kalloc)
 */

#ifndef MM2_ALIGN_H
#define MM2_ALIGN_H

#include <stdint.h>
#include "minimap.h" /* Provides mm_idx_t, mm_mapopt_t, mm_reg1_t */
#include "mm_timer.h" /* Provides mm_stage_timer_t */
#include "chain_read.h" /* Provides chain_read_t */

#ifdef __cplusplus
extern "C" {
#endif

/* API export macro */
#ifndef MM_ALIGN_API
#define MM_ALIGN_API
#endif

/* ============================================================================
 * GPU align context — wraps the opaque GPU worker pool + per-thread batches.
 *
 * Lifecycle: mm_align_ctx_create / mm_align_ctx_destroy, called from
 * mm_mapctx_create / mm_mapctx_destroy in map.c.
 * ============================================================================ */
struct mm_gpu_align_ctx_s;
typedef struct mm_gpu_align_ctx_s mm_gpu_align_ctx_t;

struct gpu_align_accum_s;
typedef struct gpu_align_accum_s gpu_align_accum_t;

typedef struct mm_align_ctx_s {
	mm_gpu_align_ctx_t* gpu_align;
} mm_align_ctx_t;

MM_ALIGN_API void mm_align_ctx_destroy(mm_align_ctx_t* ctx);

MM_ALIGN_API gpu_align_accum_t* mm_gpu_align_ctx_acquire_accum(
    mm_gpu_align_ctx_t* ctx, const mm_mapopt_t* opt, const mm_idx_t* mi);

MM_ALIGN_API void mm_gpu_align_ctx_release_accum(mm_gpu_align_ctx_t* ctx,
    gpu_align_accum_t* accum);

MM_ALIGN_API void mm_gpu_align_ctx_flush_all(mm_gpu_align_ctx_t* ctx,
    mm_stage_timer_t* t_align, const mm_core_ctx_t* core_ctx);

MM_ALIGN_API int mm_gpu_align_ctx_pool_size(const mm_gpu_align_ctx_t* ctx);
MM_ALIGN_API int mm_gpu_align_ctx_pool_device_id(const mm_gpu_align_ctx_t* ctx, int slot);

/**
 * @brief Main alignment stage entry point - align chains and generate mappings
 *
 * Third and final stage of the mapping pipeline. Converts chains into
 * alignment regions with base-level alignment using KSW2.
 *
 * Input through chain_read_t structure:
 * - read_->a (anchors), read_->u (chain info), read_->n_u (chain count)
 * - read_->seq (query sequences), read_->qlen (query lengths)
 *
 * Output:
 * - regs: Array of mapping regions with alignment info (CIGAR, scores)
 * - n_regs: Number of mappings per segment
 *
 * Processing steps:
 * 1. Converts chains into mapping regions with reference coordinates
 * 2. Filters regions and estimates error rates
 * 3. Performs sequence alignment if CIGAR strings requested
 * 4. Computes mapping quality (MAPQ) scores
 * 5. Handles paired-end pairing if PE mode
 *
 * @param mi      minimap2 index (for reference sequence access)
 * @param opt     mapping options (scoring, flags, thresholds)
 * @param read_   query read structure (containing chains from Stage 2)
 * @param regs    output array of mapping regions (one per segment)
 * @param n_regs  output number of mappings per segment
 * @param t_align optional pointer to alignment timer (can be NULL)
 * @param km      memory pool for allocation (NULL uses malloc)
 *
 * Thread safety: Thread-safe when each thread uses its own memory pool.
 *
 * @note The output regs array is allocated by this function and must be
 *       freed by the caller using mm_reg_free().
 */
MM_ALIGN_API void mm_map_align(const mm_idx_t* mi, const mm_mapopt_t* opt,
    chain_read_t* read_, mm_reg1_t** regs, int* n_regs,
    mm_stage_timer_t* t_align, void* km,
    const mm_core_ctx_t* ctx,
    mm_align_ctx_t* align_ctx, int tid);

/* ============================================================================
 * Cross-batch GPU alignment accumulator
 * ============================================================================ */

/** Set to 1 to run GAP_FILL on GPU (batched), 0 to run on CPU (sequential) */
#define GPU_GAP_FILL_ENABLED 1

/** Opaque accumulator type for cross-batch GPU alignment
 * (forward-declared above alongside mm_gpu_align_ctx_t) */

MM_ALIGN_API gpu_align_accum_t* mm_gpu_accum_init(const mm_mapopt_t* opt,
    const mm_idx_t* mi, void* mem_pool,
    int64_t batch_max_mem, int batch_max_align, int flush_threshold,
    int device_id);

MM_ALIGN_API void mm_gpu_accum_add_read(gpu_align_accum_t* accum,
    chain_read_t* rd,
    mm_reg1_t** out_regs, int* out_n_regs,
    mm_stage_timer_t* t_align, void* batch_km, const mm_core_ctx_t* core_ctx);

MM_ALIGN_API int mm_gpu_accum_should_flush(const gpu_align_accum_t* accum);

MM_ALIGN_API void mm_gpu_accum_flush(gpu_align_accum_t* accum, mm_stage_timer_t* t_align, const mm_core_ctx_t* core_ctx);

MM_ALIGN_API void mm_gpu_accum_destroy(gpu_align_accum_t* accum);

/* ============================================================================
 * Alignment result utilities
 * ============================================================================ */

/**
 * @brief Calculate event identity from a mapping region
 *
 * Computes the alignment identity based on CIGAR operations.
 * Event identity counts each indel as a single event regardless of length.
 *
 * @param r  Mapping region with populated p field
 * @return   Event identity (0.0-1.0), or NaN if no alignment
 */
MM_ALIGN_API double mm_event_identity(const mm_reg1_t* r);

#ifdef __cplusplus
}
#include "gpu/gpu_device_list_fwd.h"
mm_align_ctx_t* mm_align_ctx_create(const mm_mapopt_t* opt, int n_threads,
    gpu_device_list_t device_list = {});
#endif

#endif /* MM2_ALIGN_H */
