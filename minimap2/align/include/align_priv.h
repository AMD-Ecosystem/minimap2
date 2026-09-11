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
 * @file align_priv.h
 * @brief Private types and functions for the alignment stage
 *
 * This header contains internal functions used by the alignment stage
 * of the mapping pipeline. These functions perform sequence alignment
 * using dynamic programming (KSW2), compute alignment statistics, and
 * estimate error rates.
 *
 * Implementation: align/src/align.c, align/src/esterr.c
 *
 * Note: Hit processing functions (mm_gen_regs, mm_set_parent, mm_select_sub,
 * mm_set_mapq, etc.) are in hit_priv.h as they operate on the transition
 * between chaining and alignment.
 */

#ifndef ALIGN_PRIV_H
#define ALIGN_PRIV_H

#include "mm2_align.h" /* Public API (mm_map_align, mm_event_identity) */
#include "hit_priv.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Perform skeleton alignment on chained anchors
 *
 * Main alignment function that takes regions from chaining and performs
 * base-level alignment using KSW2. Handles both forward and reverse
 * complement alignments, splice-aware alignment for RNA-seq, and
 * generates CIGAR strings.
 *
 * @param km       Memory pool
 * @param opt      Mapping options (scoring, flags, thresholds)
 * @param mi       Minimap2 index (for reference sequence access)
 * @param qlen     Query length
 * @param qstr     Query sequence string
 * @param n_regs_  Input/output: number of regions
 * @param regs     Input regions from chaining (modified with alignment info)
 * @param a        Anchor array
 * @return         Updated regions with alignment information (p field populated)
 */
mm_reg1_t* mm_align_skeleton(void* km, const mm_mapopt_t* opt, const mm_idx_t* mi,
    int qlen, const char* qstr, int* n_regs_, mm_reg1_t* regs,
    mm128_t* a, const mm_core_ctx_t* ctx,
    mm_gpu_align_ctx_t* gpu_align_ctx, int tid);

/**
 * chain_post: Post-process chains - set parent relationships and select suboptimal chains
 * @param opt mapping options
 * @param max_chain_gap_ref maximum gap on reference sequence
 * @param mi minimap2 index
 * @param km memory pool for allocation
 * @param qlen query length
 * @param n_segs number of query segments
 * @param qlens array of segment lengths
 * @param n_regs number of chains/regions
 * @param regs array of mapping regions
 * @param a array of anchors
 *
 * Performs post-chaining filtering: identifies primary/secondary chains using
 * mm_set_parent() and optionally selects subset of chains with mm_select_sub().
 * This reduces output complexity and identifies conflicting mappings.
 */
void chain_post(const mm_mapopt_t* opt, int max_chain_gap_ref, const mm_idx_t* mi, void* km, int qlen, int n_segs, const int* qlens, int* n_regs, mm_reg1_t* regs, mm128_t* a);

void mm_enlarge_cigar(mm_reg1_t* r, uint32_t n_cigar);

/**
 * @brief Split a spliced alignment at an annotated junction (jump)
 *
 * When the index carries splice-junction hints (mi->J), extends/splits the
 * alignment of a region across an annotated junction.
 *
 * @param km        Memory pool
 * @param mi        Minimap2 index (must carry junction hints)
 * @param opt       Mapping options
 * @param qlen      Query length
 * @param qseq      Query sequence string
 * @param r         Region to split (modified in place)
 * @param ts_strand Transcript strand
 */
void mm_jump_split(void* km, const mm_idx_t* mi, const mm_mapopt_t* opt, int32_t qlen, const uint8_t* qseq, mm_reg1_t* r, int32_t ts_strand);

/**
 * @brief Update DP max scores based on divergence
 *
 * Recalculates the dp_max field for each region based on estimated
 * sequence divergence. Used to adjust expected scores for divergent
 * sequences.
 *
 * @param qlen    Query length
 * @param n_regs  Number of regions
 * @param regs    Region array
 * @param frac    Divergence fraction
 * @param a       Match score
 * @param b       Mismatch penalty
 */
void mm_update_dp_max(int qlen, int n_regs, mm_reg1_t* regs, float frac, int a, int b);

/**
 * @brief Calculate event identity from a mapping region
 *
 * Computes the alignment identity based on CIGAR operations.
 * Event identity counts each indel as a single event regardless of length.
 *
 * @param r  Mapping region with populated p field
 * @return   Event identity (0.0-1.0), or NaN if no alignment
 */
double mm_event_identity(const mm_reg1_t* r);

/**
 * @brief Estimate error rate from minimizer positions
 *
 * Estimates sequence divergence between query and reference based on
 * the density of matching minimizers. Sets the div field in each region.
 *
 * @param mi        Minimap2 index
 * @param qlen      Query length
 * @param n_regs    Number of regions
 * @param regs      Region array (div field will be updated)
 * @param a         Anchor array
 * @param n         Number of minimizer positions
 * @param mini_pos  Minimizer positions array
 */
void mm_est_err(const mm_idx_t* mi, int qlen, int n_regs, mm_reg1_t* regs,
    const mm128_t* a, int32_t n, const uint64_t* mini_pos,
    const mm_core_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif /* ALIGN_PRIV_H */
