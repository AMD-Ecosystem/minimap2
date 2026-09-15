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
 * @file chain_priv.h
 * @brief Private types and functions for the chaining stage
 *
 * This header contains internal structures and functions used by the
 * chaining stage of the mapping pipeline (anchor chaining using
 * dynamic programming or RMQ-based algorithms).
 */

#ifndef CHAIN_PRIV_H
#define CHAIN_PRIV_H

#include "minimap.h"
#include "seed_priv.h"
#include "chain_read.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Chain anchors using dynamic programming
 * @param max_dist_x  Maximum distance in reference
 * @param max_dist_y  Maximum distance in query
 * @param bw          Bandwidth
 * @param max_skip    Maximum number of anchors to skip
 * @param max_iter    Maximum iterations
 * @param min_cnt     Minimum anchor count per chain
 * @param min_sc      Minimum chain score
 * @param gap_scale   Gap penalty scaling factor
 * @param is_cdna     cDNA mode flag
 * @param n_segs      Number of segments
 * @param n           Number of anchors
 * @param a           Anchor array (sorted by reference position)
 * @param n_u_        Output: number of chains
 * @param _u          Output: chain scores and counts
 * @param km          Memory pool
 * @return            Reordered anchor array
 */
mm128_t* mm_chain_dp(int max_dist_x, int max_dist_y, int bw, int max_skip, int max_iter,
    int min_cnt, int min_sc, float gap_scale, int is_cdna, int n_segs,
    int64_t n, mm128_t* a, int* n_u_, uint64_t** _u, void* km);

/**
 * @brief Chain anchors using linear gap penalty DP
 * @param max_dist_x    Maximum distance in reference
 * @param max_dist_y    Maximum distance in query
 * @param bw            Bandwidth
 * @param max_skip      Maximum number of anchors to skip
 * @param max_iter      Maximum iterations
 * @param min_cnt       Minimum anchor count per chain
 * @param min_sc        Minimum chain score
 * @param chn_pen_gap   Chain gap penalty
 * @param chn_pen_skip  Chain skip penalty
 * @param is_cdna       cDNA mode flag
 * @param n_segs        Number of segments
 * @param n             Number of anchors
 * @param a             Anchor array
 * @param n_u_          Output: number of chains
 * @param _u            Output: chain scores and counts
 * @param km            Memory pool
 * @return              Reordered anchor array
 */
mm128_t* mg_lchain_dp(int max_dist_x, int max_dist_y, int bw, int max_skip, int max_iter,
    int min_cnt, int min_sc, float chn_pen_gap, float chn_pen_skip,
    int is_cdna, int n_segs, int64_t n, mm128_t* a, int* n_u_,
    uint64_t** _u, void* km);

/**
 * @brief Chain anchors using RMQ (Range Maximum Query)
 * @param max_dist        Maximum distance
 * @param max_dist_inner  Maximum inner distance
 * @param bw              Bandwidth
 * @param max_chn_skip    Maximum chain skip
 * @param cap_rmq_size    RMQ size cap
 * @param min_cnt         Minimum anchor count per chain
 * @param min_sc          Minimum chain score
 * @param chn_pen_gap     Chain gap penalty
 * @param chn_pen_skip    Chain skip penalty
 * @param n               Number of anchors
 * @param a               Anchor array
 * @param n_u_            Output: number of chains
 * @param _u              Output: chain scores and counts
 * @param km              Memory pool
 * @return                Reordered anchor array
 */
mm128_t* mg_lchain_rmq(int max_dist, int max_dist_inner, int bw, int max_chn_skip,
    int cap_rmq_size, int min_cnt, int min_sc, float chn_pen_gap,
    float chn_pen_skip, int64_t n, mm128_t* a, int* n_u_,
    uint64_t** _u, void* km);

/**
 * @brief Main chaining stage entry point - link seed hits into chains
 * @param mi minimap2 index (used for k-mer size and re-seeding)
 * @param opt mapping options
 * @param read_ query read structure (containing seed anchors from Stage 1)
 * @param t_chain pointer to chain timer (can be NULL to skip timing)
 * @param km memory pool for allocation
 *
 * Second stage of the mapping pipeline:
 * 1. Calculates chaining gap parameters based on query/reference spans
 * 2. Calls chaining algorithm (mg_lchain_dp or mg_lchain_rmq)
 * 3. Handles long-range chaining if bw_long mode is enabled
 * 4. Re-chains with higher max_occ threshold if chains incomplete (short reads)
 *
 * Links individual seed anchors into chains that represent colinear matches.
 * Output stored in read_->u (chain info) and chains point back into read_->a.
 */
void mm_map_chain(const mm_idx_t* mi, const mm_mapopt_t* opt,
    chain_read_t* read_, mm_stage_timer_t* t_chain, void* km, const mm_core_ctx_t* ctx);

/* Opaque GPU chain context handle.  Full definition lives in plmem.cuh;
 * forward-declared here so mm_chain_ctx_t can hold a pointer without
 * pulling in HIP headers. */
struct mm_gpu_chain_ctx_s;
typedef struct mm_gpu_chain_ctx_s mm_gpu_chain_ctx_t;

/* Chain library context — wraps the GPU chain context (when available) and
 * provides a single handle for any chain-library-wide state.  Currently
 * the only member is gpu_chain; future CPU-side chain tuning knobs or
 * caches can be added here without touching the public API in minimap.h.
 *
 * Lifecycle is managed by mm_chain_ctx_create / mm_chain_ctx_destroy.
 * mm_mapctx_create calls these automatically; callers that bypass mapctx
 * can use the chain ctx directly. */
typedef struct mm_chain_ctx_s {
	mm_gpu_chain_ctx_t* gpu_chain;
} mm_chain_ctx_t;

void mm_chain_ctx_destroy(mm_chain_ctx_t* ctx);

#ifdef __cplusplus
}
#include "gpu/gpu_device_list_fwd.h"
mm_chain_ctx_t* mm_chain_ctx_create(const mm_mapopt_t* opt, int n_threads,
    gpu_device_list_t device_list = {});
#endif

#endif /* CHAIN_PRIV_H */
