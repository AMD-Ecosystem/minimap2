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
 * @file seed_priv.h
 * @brief Private types and functions for the seeding stage
 *
 * This header contains internal functions used by the seeding stage
 * of the mapping pipeline. These functions perform minimizer sketching,
 * seed collection, filtering, and sorting.
 *
 * Implementation: seed/src/seed.c, seed/src/sketch.c
 */

#ifndef SEED_PRIV_H
#define SEED_PRIV_H

#include "mm2_seed.h" /* Public API (mm_map_seed) */

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Minimizer Sketching
 * ============================================================================ */

/**
 * @brief Generate minimizer sketch from a sequence
 * @param km      Memory pool
 * @param str     Input sequence
 * @param len     Sequence length
 * @param w       Window size
 * @param k       K-mer size
 * @param rid     Reference ID
 * @param is_hpc  Use homopolymer compression
 * @param p       Output minimizer vector
 */
void mm_sketch(void* km, const char* str, int len, int w, int k, uint32_t rid, int is_hpc, mm128_v* p);

/**
 * @brief Collect all k-mer minimizers from query sequence(s)
 * @param km          Memory pool
 * @param mi          Minimap2 index (provides k, w, hpc flag)
 * @param n_segs      Number of query segments
 * @param qlens       Array of query lengths
 * @param seqs        Array of query sequences
 * @param sdust_thres SDUST threshold (0 to disable)
 * @param mv          Output vector of minimizers
 */
void mm_collect_minimizers(void* km, const mm_idx_t* mi, int n_segs, const int* qlens,
    const char** seqs, int sdust_thres, mm128_v* mv);

/**
 * @brief Filter minimizers overlapping low-complexity regions (DUST)
 * @param km          Memory pool
 * @param n           Number of minimizers
 * @param a           Array of minimizers to filter
 * @param l_seq       Sequence length
 * @param seq         Sequence string
 * @param sdust_thres SDUST threshold (<=0 disables)
 * @return            Number of minimizers after filtering
 */
int mm_dust_minier(void* km, int n, mm128_t* a, int l_seq, const char* seq, int sdust_thres);

/* ============================================================================
 * Seed Collection and Filtering
 * ============================================================================ */

/**
 * @brief Filter high-occurrence query minimizers in-place
 * @param km          Memory pool
 * @param mv          Vector of minimizers to filter and compact
 * @param q_occ_max   Absolute cap on occurrences before filtering
 * @param q_occ_frac  Fractional cap relative to total minimizers
 */
void mm_seed_mz_flt(void* km, mm128_v* mv, int32_t q_occ_max, float q_occ_frac);

/**
 * @brief Collect all seed matches from index
 * @param km          Memory pool
 * @param mi          Minimap2 index
 * @param mv          Input minimizer vector
 * @param n_m_        Output: number of seeds
 * @return            Array of mm_seed_t entries with index lookup results
 */
mm_seed_t* mm_seed_collect_all(void* km, const mm_idx_t* mi, const mm128_v* mv, int32_t* n_m_);

/**
 * @brief Select seeds based on occurrence thresholds
 * @param n           Number of seeds
 * @param a           Array of seeds
 * @param len         Query length
 * @param max_occ     Maximum occurrence threshold
 * @param max_max_occ Hard maximum occurrence limit
 * @param dist        Distance threshold for selection
 */
void mm_seed_select(int32_t n, mm_seed_t* a, int len, int max_occ, int max_max_occ, int dist);

/**
 * @brief Collect and filter seed matches
 * @param km            Memory pool
 * @param _n_m          Output: number of seeds
 * @param qlen          Query length
 * @param max_occ       Maximum occurrence threshold
 * @param max_max_occ   Hard maximum occurrence limit
 * @param dist          Distance threshold
 * @param mi            Minimap2 index
 * @param mv            Input minimizer vector
 * @param n_a           Output: total number of anchors
 * @param rep_len       Output: repetitive length
 * @param n_mini_pos    Output: number of minimizer positions
 * @param mini_pos      Output: minimizer positions array
 * @return              Array of filtered seeds
 */
mm_seed_t* mm_collect_matches(void* km, int* _n_m, int qlen, int max_occ, int max_max_occ, int dist,
    const mm_idx_t* mi, const mm128_v* mv, int64_t* n_a, int* rep_len,
    int* n_mini_pos, uint64_t** mini_pos);

/* ============================================================================
 * Seed Hit Collection (Sorting Strategies)
 * ============================================================================ */

/**
 * @brief Collect seed hits using heap-based sorting
 * @param km          Memory pool
 * @param opt         Mapping options (flag, max_max_occ, occ_dist used)
 * @param max_occ     Maximum seed occurrence threshold
 * @param mi          Minimap2 index
 * @param qname       Query name
 * @param mv          Minimizers vector
 * @param qlen        Query length
 * @param n_a         Output: number of anchors
 * @param rep_len     Output: repetitive length
 * @param n_mini_pos  Output: number of minimizer positions
 * @param mini_pos    Output: minimizer positions array
 * @return            Array of seed anchors
 */
mm128_t* collect_seed_hits_heap(void* km, const mm_mapopt_t* opt, int max_occ,
    const mm_idx_t* mi, const char* qname, const mm128_v* mv,
    int qlen, int64_t* n_a, int* rep_len,
    int* n_mini_pos, uint64_t** mini_pos);

/**
 * @brief Collect seed hits using radix sort
 * @param km          Memory pool
 * @param opt         Mapping options (flag, max_max_occ, occ_dist used)
 * @param max_occ     Maximum seed occurrence threshold
 * @param mi          Minimap2 index
 * @param qname       Query name
 * @param mv          Minimizers vector
 * @param qlen        Query length
 * @param n_a         Output: number of anchors
 * @param rep_len     Output: repetitive length
 * @param n_mini_pos  Output: number of minimizer positions
 * @param mini_pos    Output: minimizer positions array
 * @return            Array of sorted seed anchors
 */
mm128_t* collect_seed_hits(void* km, const mm_mapopt_t* opt, int max_occ,
    const mm_idx_t* mi, const char* qname, const mm128_v* mv,
    int qlen, int64_t* n_a, int* rep_len,
    int* n_mini_pos, uint64_t** mini_pos);

#ifdef __cplusplus
}
#endif

#endif /* SEED_PRIV_H */
