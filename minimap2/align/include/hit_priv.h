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
 * @file hit_priv.h
 * @brief Private types and functions for hit processing (chain-to-alignment transition)
 *
 * This header contains internal functions used between the chaining and
 * alignment stages. These functions convert chains to regions, establish
 * parent-child relationships between overlapping hits, filter and select
 * alignments, and compute mapping quality scores.
 *
 * Implementation: align/src/hit.c, align/src/pe.c
 */

#ifndef HIT_PRIV_H
#define HIT_PRIV_H

#include "minimap.h"
#include "mm2_seed.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MM_PARENT_UNSET (-1)
#define MM_PARENT_TMP_PRI (-2)


typedef struct {
	int n_u, n_a;
	uint64_t* u;
	mm128_t* a;
} mm_seg_t;

/**
 * @brief Generate regions from chain scores
 *
 * Converts chain results (score/count pairs and anchors) into mapping
 * regions (mm_reg1_t). Sorts regions by score and assigns IDs.
 *
 * @param km         Memory pool
 * @param hash       Hash for read identification (used for tie-breaking)
 * @param qlen       Query length
 * @param n_u        Number of chains
 * @param u          Chain scores and counts (high 32 bits: score, low 32 bits: anchor count)
 * @param a          Anchor array
 * @param is_qstrand Query strand flag (if true, coordinates are query-relative)
 * @return           Array of mapping regions (caller must free)
 */
mm_reg1_t* mm_gen_regs(void* km, uint32_t hash, int qlen, int n_u, uint64_t* u,
    mm128_t* a, int is_qstrand);

/**
 * @brief Mark alternate mappings in the index
 *
 * Sets the is_alt flag on regions that map to alternate contigs
 * (as defined in the index).
 *
 * @param mi  Minimap2 index
 * @param n   Number of regions
 * @param r   Region array
 */
void mm_mark_alt(const mm_idx_t* mi, int n, mm_reg1_t* r);

/**
 * @brief Split a region into two at a breakpoint
 *
 * Splits region r at anchor position n, creating a new region r2
 * containing anchors [n, r->cnt). Updates coordinates and scores
 * for both regions.
 *
 * @param r          Primary region (modified in place, keeps anchors [0, n))
 * @param r2         Secondary region (output, receives anchors [n, cnt))
 * @param n          Split position (number of anchors to keep in r)
 * @param qlen       Query length
 * @param a          Anchor array
 * @param is_qstrand Query strand flag
 */
void mm_split_reg(mm_reg1_t* r, mm_reg1_t* r2, int n, int qlen, mm128_t* a, int is_qstrand);

/**
 * @brief Synchronize region IDs and parent pointers
 *
 * Resets region IDs to sequential values and updates parent pointers
 * to maintain consistency after regions have been filtered or reordered.
 *
 * @param km      Memory pool
 * @param n_regs  Number of regions
 * @param regs    Region array
 */
void mm_sync_regs(void* km, int n_regs, mm_reg1_t* regs);

/**
 * @brief Squeeze anchor array to remove gaps
 *
 * Compacts the anchor array by removing unused anchors, updating
 * region anchor start indices (as) accordingly.
 *
 * @param km      Memory pool
 * @param n_regs  Number of regions
 * @param regs    Region array
 * @param a       Anchor array (modified in place)
 * @return        New total anchor count
 */
int mm_squeeze_a(void* km, int n_regs, mm_reg1_t* regs, mm128_t* a);

/**
 * @brief Set SAM primary alignment flag
 *
 * Identifies the primary alignment among all regions and sets
 * the sam_pri flag. Primary is the highest-scoring non-secondary hit.
 *
 * @param n  Number of regions
 * @param r  Region array
 * @return   Index of primary alignment (-1 if none)
 */
int mm_set_sam_pri(int n, mm_reg1_t* r);

/**
 * @brief Set parent relationships for overlapping regions
 *
 * Establishes parent-child relationships between overlapping regions.
 * A region becomes a child (secondary) if it overlaps significantly
 * with a higher-scoring region. Also computes suboptimal scores.
 *
 * @param km               Memory pool
 * @param mask_level       Overlap fraction threshold for masking (0.0-1.0)
 * @param mask_len         Maximum uncovered length to still be masked
 * @param n                Number of regions
 * @param r                Region array
 * @param sub_diff         Score difference threshold for suboptimal counting
 * @param hard_mask_level  If true, skip uncovered length calculation
 * @param alt_diff_frac    Score adjustment fraction for alt contigs
 */
void mm_set_parent(void* km, float mask_level, int mask_len, int n, mm_reg1_t* r,
    int sub_diff, int hard_mask_level, float alt_diff_frac);

/**
 * @brief Select suboptimal alignments
 *
 * Filters regions to keep only primary hits and selected secondary hits.
 * Secondary hits are kept if their score is within pri_ratio of the
 * parent or within min_diff absolute score.
 *
 * @param km            Memory pool
 * @param pri_ratio     Secondary/primary score ratio threshold (0.0-1.0)
 * @param min_diff      Minimum absolute score difference to keep
 * @param best_n        Maximum number of secondary alignments to keep
 * @param check_strand  If true, keep opposite-strand hits above min_strand_sc
 * @param min_strand_sc Minimum score for opposite-strand retention
 * @param n_            Input/output: number of regions (updated on return)
 * @param r             Region array (filtered in place)
 */
void mm_select_sub(void* km, float pri_ratio, int min_diff, int best_n,
    int check_strand, int min_strand_sc, int* n_, mm_reg1_t* r);

/**
 * @brief Select suboptimal alignments for multiple segments (paired-end)
 *
 * Extended version of mm_select_sub for paired-end or multi-segment reads.
 * Uses segment-aware scoring thresholds.
 *
 * @param km           Memory pool
 * @param pri_ratio    Secondary/primary score ratio threshold
 * @param pri1         Primary threshold for first segment
 * @param pri2         Primary threshold for second segment
 * @param max_gap_ref  Maximum gap in reference for pairing
 * @param min_diff     Minimum absolute score difference
 * @param best_n       Maximum number of secondary alignments
 * @param n_segs       Number of segments
 * @param qlens        Array of query lengths per segment
 * @param n_           Input/output: number of regions
 * @param r            Region array
 */
void mm_select_sub_multi(void* km, float pri_ratio, float pri1, float pri2,
    int max_gap_ref, int min_diff, int best_n, int n_segs,
    const int* qlens, int* n_, mm_reg1_t* r);

/**
 * @brief Filter strand-retained mappings
 *
 * Removes regions that were only retained for strand checking
 * (strand_retained flag) but are no longer needed.
 *
 * @param n_regs  Number of regions
 * @param r       Region array
 * @return        Number of retained regions
 */
int mm_filter_strand_retained(int n_regs, mm_reg1_t* r);

/**
 * @brief Filter regions based on mapping options
 *
 * Applies final filtering based on mapping options: removes unmapped
 * regions, applies score thresholds, and enforces output limits.
 *
 * @param opt     Mapping options
 * @param qlen    Query length
 * @param n_regs  Input/output: number of regions
 * @param regs    Region array
 */
void mm_filter_regs(const mm_mapopt_t* opt, int qlen, int* n_regs, mm_reg1_t* regs);

/**
 * @brief Sort hits by score and position
 *
 * Sorts regions by score (descending), then by position. Also handles
 * inversion detection and merging of adjacent inversions.
 *
 * @param km             Memory pool
 * @param n_regs         Input/output: number of regions
 * @param r              Region array
 * @param alt_diff_frac  Score adjustment for alt contigs
 */
void mm_hit_sort(void* km, int* n_regs, mm_reg1_t* r, float alt_diff_frac);

/**
 * @brief Set mapping quality scores
 *
 * Computes mapping quality (MAPQ) for each region based on alignment
 * score, uniqueness ratio, and presence of suboptimal alignments.
 *
 * @param km           Memory pool
 * @param n_regs       Number of regions
 * @param regs         Region array
 * @param min_chain_sc Minimum chain score for MAPQ calculation
 * @param match_sc     Match score from scoring parameters
 * @param rep_len      Repetitive length (reduces uniqueness)
 * @param is_sr        Short read mode flag (uses different MAPQ formula)
 * @param is_splice    Splice mode flag (uses different MAPQ formula)
 */
void mm_set_mapq2(void* km, int n_regs, mm_reg1_t* regs, int min_chain_sc, int match_sc, int rep_len, int is_sr, int is_splice);

/* Segment/pair functions - for paired-end and segmented reads */
mm_seg_t* mm_seg_gen(void* km, uint32_t hash, int n_segs, const int* qlens, int n_regs0, const mm_reg1_t* regs0, int* n_regs, mm_reg1_t** regs, const mm128_t* a);
void mm_seg_free(void* km, int n_segs, mm_seg_t* segs);
void mm_pair(void* km, int max_gap_ref, int dp_bonus, int sub_diff, int match_sc, const int* qlens, int* n_regs, mm_reg1_t** regs);

#ifdef __cplusplus
}
#endif

#endif /* HIT_PRIV_H */
