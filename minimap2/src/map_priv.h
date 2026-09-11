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
 * @file map_priv.h
 * @brief Private data structures and functions for the mapping pipeline
 *
 * This header contains internal structures and functions used by map.c
 * for the three-stage mapping pipeline (seed -> chain -> align).
 * These definitions are not part of the public API.
 */

#ifndef MAP_PRIV_H
#define MAP_PRIV_H

#include <assert.h>
#include "minimap.h"
#include "mm_timer.h"
#include "bseq.h"
#include "kseq.h"
#include "core_ctx.h"

/* Core headers */
#include "misc.h"
#include "index.h"
#include "format.h"
#include "splitidx.h"
#include "option.h"

/* Stage-specific private headers */
#include "mm2_seed.h"
#include "chain_priv.h"
#include "hit_priv.h"
#include "align_priv.h"
#include "mm2_align.h"

/* Shared types */
#include "chain_read.h"

/* Debug flags */
#define MM_DBG_NO_KALLOC 0x1
#define MM_DBG_PRINT_QNAME 0x2
#define MM_DBG_PRINT_SEED 0x4
#define MM_DBG_PRINT_CHAIN 0x10

#define N_ACCUM 64

// Forward declarations
typedef struct pipeline_s pipeline_t;
typedef struct step_s step_t;

// Pipeline structure - global state for multi-threaded mapping
typedef struct pipeline_s {
	int n_processed; // number of sequences processed
	int n_threads; // number of worker threads
	int n_fp; // number of input files
	int64_t mini_batch_size; // sequences per mini-batch
	const mm_mapopt_t* opt; // mapping options
	mm_bseq_file_t** fp; // input file handles
	const mm_idx_t* mi; // minimap2 index
	kstring_t str; // output buffer

	// split index support
	int n_parts; // number of index parts
	uint32_t* rid_shift; // reference ID offsets per part
	FILE* fp_split; // split output file
	FILE** fp_parts; // part input files

	// Chain context, threaded through from the public mm_map*_ctx
	// entry points. NULL = no chain ctx for this run; GPU helpers
	// check chain_ctx->gpu_chain before dispatching to GPU.
	mm_chain_ctx_t* chain_ctx;

	// Align context, same pattern as chain_ctx.  Owns the GPU worker
	// pool and per-thread multialign batches.  NULL when GPU alignment
	// is not enabled.
	mm_align_ctx_t* align_ctx;

	// Core context threaded from the public mm_map*_ctx entry points.
	// Carries the per-run logger, verbose/dbg_flag overrides, and JSON
	// dump config. NULL when called from the legacy non-_ctx wrappers.
	mm_core_ctx_t* core_ctx;

	// Optional timing statistics from the mapping context.
	// NULL when called from legacy non-_ctx wrappers or when timing is
	// not requested.
	mm_timer_stats_t* timer_stats;
} pipeline_t;


// Batch buffer - holds multiple reads for GPU/CPU processing
typedef struct {
	int batchid; // batch identifier
	void* km; // memory pool for this batch
	int count; // number of reads in the batch
	size_t total_n; // total number of anchors in the batch
	chain_read_t* reads; // array of reads
	uint64_t pool_reset_count; // number of pool resets in this batch
} mm_batch_trbuf_t;

// Thread read buffer - 3-batch pipeline for overlapping seed/chain/align stages
typedef struct {
	mm_batch_trbuf_t acc_batch; // accumulation batch (collecting seeds)
	int is_full; // accumulation batch is full

	mm_batch_trbuf_t launched_batch; // launched batch (chaining in progress)
	int has_launched; // has launched batch

	mm_batch_trbuf_t pending_batch; // pending batch (ready for chaining)
	int is_pending; // has pending batch
	int flushed; // set after flush (i_in=-1) completes to prevent double-flush
} mm_trbuf_t;

// Pipeline step structure - data for one batch of sequences
typedef struct step_s {
	const pipeline_t* p; // parent pipeline
	int n_seq; // number of sequences
	int n_frag; // number of fragments
	mm_bseq1_t* seq; // sequence data
	int* n_reg; // number of regions per sequence
	int* seg_off; // segment offsets
	int* n_seg; // number of segments per fragment
	int* rep_len; // repetitive length per sequence
	int* frag_gap; // fragment gap per sequence
	mm_reg1_t** reg; // mapping regions
	mm_tbuf_t** buf; // thread buffers
	int batch_max_reads; // maximum reads per batch
	size_t batch_max_anchors; // maximum anchors per batch
	mm_trbuf_t** trbuf; // thread read buffers
} step_t;

// mm_map_seed is declared in mm2_seed.h
// mm_map_chain is declared in mm2_chain.h
// mm_map_align is declared in align_priv.h

// Internal ctx-aware version of mm_split_merge (which is not part of
// minimap.h). main.c calls this so the process-scope chain ctx covers the
// post-loop split-merge phase too. The non-_ctx mm_split_merge in splitidx.h
// is rewritten as a thin wrapper that passes NULL.
int mm_split_merge_ctx(int n_segs, const char** fn, const mm_mapopt_t* opt,
    int n_split_idx, mm_mapctx_t* mapctx);

// Helper to record a batch of n_calls into a stage timer when only the
// aggregate batch wall time is available (e.g. GPU-batched stages that lack
// per-call granularity). The total duration is added to sum unconditionally;
// when n_calls > 0, call_count is bumped by n_calls and call_min/max are
// updated using the per-call average within this batch (total / n_calls).
// This makes "Avg = sum/count" meaningful at the report level while clearly
// signalling lack of per-call granularity (min == max within a single batch).
static inline void mm_stage_timer_record_batch(mm_stage_timer_t* st, double total_dt, uint64_t n_calls)
{
	if (!st) return;
	st->sum += total_dt;
	if (n_calls == 0) return;
	const double per_call = total_dt / (double)n_calls;
	if (st->call_count == 0 || per_call < st->call_min) st->call_min = per_call;
	if (st->call_count == 0 || per_call > st->call_max) st->call_max = per_call;
	st->call_count += n_calls;
}

#endif // MAP_PRIV_H