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
 * @file chain_read.h
 * @brief Chain read structure for the mapping pipeline
 *
 * This header contains the chain_read_t structure that is passed through
 * all three stages of the mapping pipeline (seed -> chain -> align).
 * It is shared between map.c, align.c, and the CLI tools.
 */

#ifndef CHAIN_READ_H
#define CHAIN_READ_H

#include "minimap.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Metadata for a sequence being mapped
 */
typedef struct {
	long i; /**< Read id */
	int seg_id; /**< Segment id */
	char name[200]; /**< Name of the sequence */
	uint32_t len; /**< Length of the sequence */

	/* Index metadata */
	int n_alt; /**< Number of alternate mappings */
	int is_alt; /**< Reference sequences only */

	/* Sequence info */
	int qlen_sum; /**< Sum of query lengths */
} mm_seq_meta_t;

/**
 * @brief Chain read structure - holds data for one read through the mapping pipeline
 *
 * This structure carries all data needed by the seed, chain, and align stages.
 * It is populated progressively: seeding fills anchor data, chaining fills
 * chain scores, and alignment produces final mappings.
 */
typedef struct chain_read_s {
	mm_seq_meta_t seq; /**< Sequence metadata */

	/* Input data */
	const char** qseqs; /**< Sequences for each segment */
	int* qlens; /**< Query length for each segment */
	int n_seg; /**< Number of segments */

	int rep_len; /**< Repetitive length */
	int frag_gap; /**< Fragment gap (set by chaining) */

	/* Seeding outputs */
	uint64_t* mini_pos; /**< Minimizer positions */
	int n_mini_pos; /**< Number of minimizer positions */
	int n_minimizers; /**< Number of minimizers collected (mv.n) */
	mm128_t* minimizers_a; /**< Optional cached minimizers for re-chaining */
	int64_t minimizers_n; /**< Number of cached minimizers */

	/* Seeding output, updated in chaining */
	mm128_t* a; /**< Array of anchors */
	int64_t n; /**< Number of anchors (n_a) */

	/* Chaining outputs */
	uint64_t* u; /**< Chain scores and lengths (upper 32: score, lower 32: length) */
	int n_u; /**< Number of chains (n_regs0) */
} chain_read_t;

#ifdef __cplusplus
}
#endif

#endif /* CHAIN_READ_H */
