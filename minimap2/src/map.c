/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2018-2022 Heng Li; Dana-Farber Cancer Institute; Broad Institute
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

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include "kthread.h"
#include "kvec.h"
#include "kalloc.h"
#include "sdust.h"
#include "map_priv.h"
#include "mm2_seed.h"
#include "align_priv.h"
#include "khash.h"
#include "mm_log.h"
#include "mm_profiler.h"
#include "mm_dump.h"
#include "core_ctx.h"

#ifdef MM_ENABLE_HIP
#include "gpu/plchain.h"
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// Free chain_read_t pointers except a, u, mini_pos because they are freed in mm_map_align.
static inline void free_read(chain_read_t* in, void* km)
{
	if (in->qseqs) kfree(km, in->qseqs);
	if (in->qlens) kfree(km, in->qlens);
	if (in->minimizers_a) kfree(km, in->minimizers_a);

//DEBUG: for SCORE CHECK after chaining
#if defined(DEBUG_CHECK) && 0
	if (in->f) kfree(km, in->f);
	if (in->p) kfree(km, in->p);
	in->f = 0, in->p = 0;
#endif
	in->qseqs = 0, in->qlens = 0;
	in->minimizers_a = 0, in->minimizers_n = 0;
	in->a = 0, in->u = 0;
}

/**
 * Initialize a thread buffer with memory pool.
 * @return pointer to newly initialized mm_tbuf_t structure
 *
 * This function allocates and initializes a per-thread buffer used for memory management
 * during sequence mapping. Each thread gets its own memory pool (km) to minimize contention.
 * The debug flag can suppress km allocation for testing purposes.
 */
mm_tbuf_t* mm_tbuf_init(void)
{
	mm_tbuf_t* b;
	b = (mm_tbuf_t*)calloc(1, sizeof(mm_tbuf_t));
	if (!(mm_dbg_flag & 1))
		b->km = km_init();
	return b;
}

/**
 * Deallocate thread buffer and clean up memory pool.
 * @param b pointer to mm_tbuf_t to destroy
 *
 * Releases all resources associated with a thread buffer including the memory pool.
 * Safe to call with NULL pointer.
 */
void mm_tbuf_destroy(mm_tbuf_t* b)
{
	if (b == 0)
		return;
	km_destroy(b->km);
	free(b);
}

/**
 * Get memory pool pointer from thread buffer.
 * @param b pointer to mm_tbuf_t
 * @return void* pointer to the memory pool (km) contained in the buffer
 *
 * Simple accessor function to retrieve the memory pool from a thread buffer.
 * This allows other functions to allocate memory from the thread-local pool.
 */
void* mm_tbuf_get_km(mm_tbuf_t* b)
{
	return b->km;
}

/**
 * Initialize a batch of reads for pipelined processing.
 * @param batch_ pointer to mm_batch_trbuf_t to initialize
 * @param batch_max_reads maximum number of reads this batch can hold
 *
 * Prepares a read batch by allocating space for chain_read_t structures and
 * creating a dedicated memory pool. Batches accumulate reads for processing
 * (either CPU or GPU chaining depending on configuration).
 */
static void mm_trbuf_batch_init(mm_batch_trbuf_t* batch_, int batch_max_reads)
{
	batch_->count = 0;
	batch_->total_n = 0;
	batch_->reads = (chain_read_t*)malloc(sizeof(chain_read_t) * batch_max_reads);
	memset(batch_->reads, 0, sizeof(chain_read_t) * batch_max_reads);
	batch_->batchid = -1;
	batch_->km = km_init();
}

/**
 * Reset a batch after processing - free reads and manage memory.
 * @param batch_ pointer to mm_batch_trbuf_t to reset
 * @param batch_max_reads (unused) maximum number of reads in batch
 * @param opt mapping options (used for memory limit settings)
 *
 * Clears all reads from a batch and validates memory pool consistency.
 * May destroy and reinitialize the memory pool if it has grown too large.
 * This is crucial for maintaining reasonable memory usage across many reads.
 */
static void mm_trbuf_batch_reset(mm_batch_trbuf_t* batch_, int batch_max_reads, const mm_mapopt_t* opt, const mm_core_ctx_t* ctx)
{
	// free all the reads in the batch
	for (int i = 0; i < batch_->count; i++) {
		free_read(&batch_->reads[i], batch_->km);
	}

	/* reset memory pool km */
	km_stat_t kmst;
	if (batch_->km) {
		chain_read_t* last_read = batch_->reads + batch_->count;
		km_stat(batch_->km, &kmst);
		if (mm_core_ctx_dbg_flag(ctx) & MM_DBG_PRINT_QNAME)
			mm_log_debug("QM\t%s\t%d\tBid=%d\tcap=%ld,avail=%ld,nCore=%ld,largest=%ld",
			    last_read->seq.name, last_read->seq.qlen_sum, batch_->batchid, kmst.capacity, kmst.available, kmst.n_cores, kmst.largest);
		assert(kmst.n_blocks == kmst.n_cores); // memory leak if not equal
		assert(kmst.capacity == kmst.meta_size + kmst.available); // memory accounting error
		if ((opt->cap_kalloc_largest > 0 && kmst.largest > (size_t)opt->cap_kalloc_largest) || (opt->cap_kalloc > 0 && kmst.capacity > opt->cap_kalloc)) {
			if (mm_core_ctx_dbg_flag(ctx) & MM_DBG_PRINT_QNAME)
				mm_log("[W::%s] reset thread-local memory after read %s", __func__, last_read->seq.name);
			km_destroy(batch_->km);
			batch_->km = km_init();
			batch_->pool_reset_count++;
		}
	}

	batch_->count = 0;
	batch_->total_n = 0;
	batch_->batchid = -1;
}

/**
 * mm_trbuf_batch_destroy: Complete cleanup of a batch - destroy reads and memory pool
 * @param batch_ pointer to mm_batch_trbuf_t to destroy
 * 
 * Performs full cleanup of a read batch including all contained reads, associated data,
 * and the dedicated memory pool. Validates memory consistency before destroying.
 */
static void mm_trbuf_batch_destroy(mm_batch_trbuf_t* batch_, const mm_core_ctx_t* ctx)
{
	// free reads in the batch
	for (int i = 0; i < batch_->count; i++) {
		free_read(&batch_->reads[i], batch_->km);
	}
	batch_->batchid = -1;
	batch_->total_n = 0;
	batch_->count = 0;
	/* clean memory pool */
	km_stat_t kmst;
	if (batch_->km) {
		km_stat(batch_->km, &kmst);
		if (mm_core_ctx_dbg_flag(ctx) & MM_DBG_PRINT_QNAME)
			mm_log_debug("Destroy memory pool cap=%ld,avail=%ld,nCore=%ld,largest=%ld",
			    kmst.capacity, kmst.available, kmst.n_cores, kmst.largest);
		assert(kmst.n_blocks == kmst.n_cores); // memory leak if not equal
		assert(kmst.capacity == kmst.meta_size + kmst.available); // memory accounting error
		km_destroy(batch_->km);
		batch_->km = 0;
	}
	free(batch_->reads);
	batch_->reads = 0;
}

/**
 * mm_trbuf_init: Initialize thread read batch buffer with 3-batch pipelining
 * @param batch_max_reads max reads per batch
 * @param opt mapping options
 * @return pointer to initialized mm_trbuf_t structure
 *
 * Creates a pipelined buffering system with 3 batches:
 * - acc_batch: accumulates reads (CPU processes seeds)
 * - pending_batch: ready for chaining (CPU or GPU)
 * - launched_batch: chaining in progress (batch being processed)
 * This enables overlap of seeding, chaining, and alignment stages.
 */
static mm_trbuf_t* mm_trbuf_init(const int batch_max_reads, const mm_mapopt_t* opt)
{
	mm_trbuf_t* tr;
	tr = (mm_trbuf_t*)calloc(1, sizeof(mm_trbuf_t));
	tr->is_full = 0;
	tr->is_pending = 0;
	tr->has_launched = 0;
	mm_trbuf_batch_init(&tr->acc_batch, batch_max_reads);
	tr->acc_batch.batchid = 0;
	mm_trbuf_batch_init(&tr->pending_batch, batch_max_reads);
	tr->pending_batch.batchid = 1;
	mm_trbuf_batch_init(&tr->launched_batch, batch_max_reads);
	tr->launched_batch.batchid = 2;
	return tr;
}

/**
 * mm_trbuf_destroy: Destroy all 3 batches in thread read buffer
 * @param tr pointer to mm_trbuf_t to destroy
 * 
 * Cleans up the entire pipelined batch system by destroying each of the three
 * batches (accumulation, pending, and launched) and freeing the container.
 */
static void mm_trbuf_destroy(mm_trbuf_t* tr, const mm_core_ctx_t* ctx)
{
	if (tr == 0) return;
	mm_trbuf_batch_destroy(&tr->acc_batch, ctx);
	mm_trbuf_batch_destroy(&tr->pending_batch, ctx);
	mm_trbuf_batch_destroy(&tr->launched_batch, ctx);
	free(tr);
}

static step_t* init_step(pipeline_t* p)
{
	int with_qual = (!!(p->opt->flag & MM_F_OUT_SAM) && !(p->opt->flag & MM_F_NO_QUAL));
	int with_comment = !!(p->opt->flag & MM_F_COPY_COMMENT);
	int frag_mode = (p->n_fp > 1 || !!(p->opt->flag & MM_F_FRAG_MODE));
	step_t* s;
	s = (step_t*)calloc(1, sizeof(step_t));
	if (p->n_fp > 1)
		s->seq = mm_bseq_read_frag2(p->n_fp, p->fp, p->mini_batch_size, with_qual, with_comment, &s->n_seq);
	else
		s->seq = mm_bseq_read3(p->fp[0], p->mini_batch_size, with_qual, with_comment, frag_mode, &s->n_seq);
	if (s->seq) {
		s->p = p;
		for (int i = 0; i < s->n_seq; ++i)
			s->seq[i].rid = p->n_processed++;
		s->buf = (mm_tbuf_t**)calloc(p->n_threads, sizeof(mm_tbuf_t*));
		for (int i = 0; i < p->n_threads; ++i)
			s->buf[i] = mm_tbuf_init();

		s->n_reg = (int*)calloc(5 * s->n_seq, sizeof(int));
		s->seg_off = s->n_reg + s->n_seq; // seg_off, n_seg, rep_len and frag_gap are allocated together with n_reg
		s->n_seg = s->seg_off + s->n_seq;
		s->rep_len = s->n_seg + s->n_seq;
		s->frag_gap = s->rep_len + s->n_seq;
		s->reg = (mm_reg1_t**)calloc(s->n_seq, sizeof(mm_reg1_t*));
		for (int i = 1, j = 0; i <= s->n_seq; ++i)
			if (i == s->n_seq || !frag_mode || !mm_qname_same(s->seq[i - 1].name, s->seq[i].name)) {
				s->n_seg[s->n_frag] = i - j;
				s->seg_off[s->n_frag++] = j;
				j = i;
			}
			// MM_F_GPU_CHAIN flag indicates --gpu-chain CLI flag was provided
#ifdef MM_ENABLE_HIP
		if (s->p->opt->flag & MM_F_GPU_CHAIN && s->p->chain_ctx && s->p->chain_ctx->gpu_chain) {
			// GPU mode: use auto-derived anchor threshold, increase batch size
			s->batch_max_anchors = gpu_chain_ctx_target_batch_anchors(s->p->chain_ctx->gpu_chain);
			// Increase max reads to allow accumulation for short-read datasets
			// Assumes worst-case ~10K anchors/read: 125M / 10K = 12,500 reads
			// Cap to GPU's preallocated memory limit
			int gpu_max_reads = gpu_chain_ctx_max_reads(s->p->chain_ctx->gpu_chain);
			size_t computed_max_reads = (s->batch_max_anchors / 10000) + N_ACCUM;
			s->batch_max_reads = (int)MIN(computed_max_reads, gpu_max_reads);
		} else
#endif
		{
			// CPU mode or GPU not available: unlimited anchors, static read threshold
			s->batch_max_anchors = SIZE_MAX;
			s->batch_max_reads = N_ACCUM;
		}
		s->trbuf = (mm_trbuf_t**)calloc(s->p->n_threads, sizeof(mm_trbuf_t*));
		for (int i = 0; i < s->p->n_threads; ++i)
			s->trbuf[i] = mm_trbuf_init(s->batch_max_reads, s->p->opt);
		return s;
	} else
		free(s);
	return NULL;
}

static void free_step(step_t* s)
{
	const mm_core_ctx_t* ctx = s->p->core_ctx;
	for (int i = 0; i < s->p->n_threads; ++i)
		mm_trbuf_destroy(s->trbuf[i], ctx);
	free(s->trbuf);
	for (int k = 0; k < s->n_frag; ++k) {
		int seg_st = s->seg_off[k], seg_en = s->seg_off[k] + s->n_seg[k];
		for (int i = seg_st; i < seg_en; ++i) {
			for (int j = 0; j < s->n_reg[i]; ++j)
				free(s->reg[i][j].p);
			free(s->reg[i]);
			free(s->seq[i].seq);
			free(s->seq[i].name);
			if (s->seq[i].qual)
				free(s->seq[i].qual);
			if (s->seq[i].comment)
				free(s->seq[i].comment);
		}
	}
	for (int i = 0; i < s->p->n_threads; ++i)
		mm_tbuf_destroy(s->buf[i]);
	free(s->buf);
	free(s->reg);
	free(s->n_reg);
	free(s->seq); // seg_off, n_seg, rep_len and frag_gap were allocated with reg; no memory leak here
	if (mm_core_ctx_verbose(ctx) >= 3)
		mm_log("[M::%s::%.3f*%.2f] mapped %d sequences", __func__, realtime() - mm_realtime0, cputime() / (realtime() - mm_realtime0), s->n_seq);
	free(s);
}

/**
 * align_regs: Perform sequence alignment on regions to generate CIGAR strings
 * @param opt mapping options
 * @param mi minimap2 index
 * @param km memory pool for allocation
 * @param qlen query length
 * @param seq query sequence
 * @param n_regs pointer to number of regions
 * @param regs array of mapping regions
 * @param a array of anchors
 * @return aligned regions with CIGAR strings (if MM_F_CIGAR flag set)
 *
 * Performs fine-grained dynamic programming alignment on chain regions if
 * CIGAR strings are requested. Also applies post-chain filtering and selects
 * primary mappings based on score and quality metrics.
 *
 * NOTE: This is only used by mm_map_frag (public API without context).
 * The pipeline path uses align_regs() inside align.c which receives ctx.
 */
static mm_reg1_t* align_regs(const mm_mapopt_t* opt, const mm_idx_t* mi, void* km, int qlen, const char* seq, int* n_regs, mm_reg1_t* regs, mm128_t* a)
{
	if (!(opt->flag & MM_F_CIGAR))
		return regs;
	regs = mm_align_skeleton(km, opt, mi, qlen, seq, n_regs, regs, a, NULL, NULL, 0); // this calls mm_filter_regs()
	if (!(opt->flag & MM_F_ALL_CHAINS)) { // don't choose primary mapping(s)
		mm_set_parent(km, opt->mask_level, opt->mask_len, *n_regs, regs, opt->a * 2 + opt->b, opt->flag & MM_F_HARD_MLEVEL, opt->alt_drop);
		mm_select_sub(km, opt->pri_ratio, mi->k * 2, opt->best_n, 0, opt->max_gap * 0.8, n_regs, regs);
		mm_set_sam_pri(*n_regs, regs);
	}
	return regs;
}

static void mm_timer_stats_record_batch(mm_timer_stats_t* ts,
    const mm_tbuf_timers_t* stage_sum,
    const mm_tbuf_timers_t* stage_min,
    const mm_tbuf_timers_t* stage_max,
    uint64_t query_count,
    int n_threads);

/**
 * mm_map_frag: Orchestrate all three mapping stages - seed, chain, align
 * @param mi minimap2 index
 * @param n_segs number of query segments
 * @param qlens array of segment lengths
 * @param seqs array of segment sequences
 * @param n_regs output number of mappings per segment
 * @param regs output mapping regions
 * @param b thread buffer for memory management
 * @param opt mapping options
 * @param qname query name
 * 
 * High-level wrapper that coordinates the entire three-stage mapping pipeline:
 * 1. Stage 1 (Seed): Finds candidate matches
 * 2. Stage 2 (Chain): Groups matches into colinear chains
 * 3. Stage 3 (Align): Aligns chains and produces final mappings
 * 
 * This is the main entry point for mapping a sequence with optional paired-end support.
 * Handles memory management and error checking across all stages.
 */
#ifdef MM_ENABLE_HIP
/* Per-read GPU chaining: drive the double-buffered submit/finish pipeline for a
 * single, already-seeded read using GPU worker slot `slot`. gpu_chain_submit
 * launches `read` on the GPU and returns the previously launched batch (NULL on
 * a fresh slot, which we ignore); gpu_chain_finish then synchronizes and drains
 * `read` in place, populating read->u / read->n_u for the alignment stage. */
static void mm_chain_one_gpu(gpu_chain_ctx_t* gctx, const mm_idx_t* mi, const mm_mapopt_t* opt, chain_read_t* read, int slot, void* km)
{
	chain_read_t* arr = read;
	int n = 1;
	gpu_chain_submit(gctx, mi, opt, &arr, &n, slot, km); // launches `read`; returns prev batch (ignored)
	chain_read_t* out = NULL;
	int n_out = 0;
	gpu_chain_finish(gctx, mi, opt, &out, &n_out, slot, km); // drains `read` in place
}
#endif

#ifdef HAVE_GPU_ALIGNMENT
/* Per-read GPU alignment: acquire a shared accumulator, add the single read,
 * flush immediately (batch-of-1), then release. mm_gpu_accum_add_read frees
 * read->a / read->u / read->mini_pos from `km`, matching mm_map_align's
 * ownership, and internally falls back to the CPU aligner for reads the GPU
 * kernel cannot handle (multi-segment, splice, SR, or no-CIGAR). */
static void mm_align_one_gpu(mm_align_ctx_t* actx, const mm_idx_t* mi, const mm_mapopt_t* opt, chain_read_t* read, mm_reg1_t** regs, int* n_regs, mm_stage_timer_t* t_align, void* km, const mm_core_ctx_t* core_ctx)
{
	gpu_align_accum_t* accum = mm_gpu_align_ctx_acquire_accum(actx->gpu_align, opt, mi);
	mm_gpu_accum_add_read(accum, read, regs, n_regs, t_align, km, core_ctx);
	mm_gpu_accum_flush(accum, t_align, core_ctx);
	mm_gpu_align_ctx_release_accum(actx->gpu_align, accum);
}
#endif

/* Single-read path. When `mapctx` is NULL (the legacy mm_map / mm_map_frag
 * entry points) every stage runs on the CPU, byte-for-byte as before. When a
 * mapctx is supplied (mm_map_ctx / mm_map_frag_ctx), chaining and alignment are
 * toggled to the GPU per the opt flags + available sub-contexts, exactly like
 * the file pipeline; `slot` selects the per-thread GPU worker slot. GPU stages
 * apply only to single-segment reads; everything else falls back to CPU. */
static void mm_map_frag_core(const mm_idx_t* mi, int n_segs, const int* qlens, const char** seqs, int* n_regs, mm_reg1_t** regs, mm_tbuf_t* b, const mm_mapopt_t* opt, const char* qname, mm_mapctx_t* mapctx, int slot)
{
	int i, qlen_sum = 0;
	chain_read_t read;
	const mm_core_ctx_t* core_ctx = mapctx ? mapctx->core_ctx : NULL;
	(void)slot;

	// Initialize output (must be done even on early return)
	for (i = 0; i < n_segs; ++i) {
		n_regs[i] = 0;
		regs[i] = NULL;
	}

	// Early validation - return before profiling starts
	if (!mi || !qlens || !seqs || !n_regs || !regs || !b || !opt)
		return;
	if (n_segs <= 0 || n_segs > MM_MAX_SEG)
		return;

	// Calculate total query length for validation
	for (i = 0; i < n_segs; ++i)
		qlen_sum += qlens[i];
	if (qlen_sum == 0)
		return;
	if (opt->max_qlen > 0 && qlen_sum > opt->max_qlen)
		return;

	// All validation passed - start profiling
	MM_PROFILE_FUNCTION();

	// Initialize chain_read_t structure
	memset(&read, 0, sizeof(chain_read_t));
	read.n_seg = n_segs;
	read.qlens = (int*)qlens;
	read.qseqs = seqs;
	read.seq.qlen_sum = qlen_sum;

	if (qname) {
		strncpy(read.seq.name, qname, sizeof(read.seq.name) - 1);
		read.seq.name[sizeof(read.seq.name) - 1] = '\0';
	} else {
		read.seq.name[0] = '\0';
	}

	// Stage 1: Seed - Find k-mer matches
	MM_PROFILE_SEED("mm_map_seed");
	mm_map_seed(mi, opt, &read, &b->timers.seed, b->km, core_ctx);
	MM_PROFILE_RANGE_POP();

	// Stage 2: Chain - Link seeds into chains (if seeds were found)
	if (read.n > 0) {
		MM_PROFILE_CHAIN("mm_map_chain");
#ifdef MM_ENABLE_HIP
		if ((opt->flag & MM_F_GPU_CHAIN) && !(opt->flag & MM_F_RMQ) && n_segs == 1 && opt->max_frag_len <= 0 && mapctx && mapctx->chain_ctx && mapctx->chain_ctx->gpu_chain) {
			mm_chain_one_gpu(mapctx->chain_ctx->gpu_chain, mi, opt, &read, slot, b->km);
		} else
#endif
		{
			mm_map_chain(mi, opt, &read, &b->timers.chain, b->km, core_ctx);
		}
		MM_PROFILE_RANGE_POP();
	}

	// Stage 3: Align - Perform sequence alignment (if chains were found)
	if (read.n_u > 0) {
		MM_PROFILE_ALIGN("mm_map_align");
#ifdef HAVE_GPU_ALIGNMENT
		if ((opt->flag & MM_F_GPU_ALIGN) && n_segs == 1 && mapctx && mapctx->align_ctx && mapctx->align_ctx->gpu_align) {
			mm_align_one_gpu(mapctx->align_ctx, mi, opt, &read, regs, n_regs, &b->timers.align, b->km, core_ctx);
		} else
#endif
		{
			mm_map_align(mi, opt, &read, regs, n_regs, &b->timers.align, b->km, core_ctx, NULL, 0);
		}
		MM_PROFILE_RANGE_POP();
	} else {
		// mm_map_align normally frees a, u, mini_pos. When it doesn't run,
		// we must free any km-allocated fields to avoid pool leaks.
		kfree(b->km, read.a);
		kfree(b->km, read.u);
		kfree(b->km, read.mini_pos);
	}
	kfree(b->km, read.minimizers_a);

	// Memory pool management: check and reset if pool grows too large
	if (b->km) {
		km_stat_t kmst;
		km_stat(b->km, &kmst);
		if (mm_dbg_flag & MM_DBG_PRINT_QNAME)
			mm_log_debug("QM\t%s\t%d\tcap=%ld,nCore=%ld,largest=%ld", qname, read.seq.qlen_sum, kmst.capacity, kmst.n_cores, kmst.largest);
		assert(kmst.n_blocks == kmst.n_cores); // memory leak if not equal
		if ((opt->cap_kalloc_largest > 0 && kmst.largest > (size_t)opt->cap_kalloc_largest) || (opt->cap_kalloc > 0 && kmst.capacity > opt->cap_kalloc)) {
			if (mm_dbg_flag & MM_DBG_PRINT_QNAME)
				mm_log("[W::%s] reset thread-local memory after read %s", __func__, qname);
			km_destroy(b->km);
			b->km = km_init();
			b->pool_reset_count++;
		}
	}

	MM_PROFILE_FUNCTION_END();
}

void mm_map_frag(const mm_idx_t* mi, int n_segs, const int* qlens, const char** seqs, int* n_regs, mm_reg1_t** regs, mm_tbuf_t* b, const mm_mapopt_t* opt, const char* qname)
{
	if ((opt->flag & MM_F_WEAK_PAIRING) && n_segs == 2 && opt->pe_ori >= 0 && (opt->flag & MM_F_CIGAR)) {
		int i;
		for (i = 0; i < n_segs; ++i)
			mm_map_frag_core(mi, 1, &qlens[i], &seqs[i], &n_regs[i], &regs[i], b, opt, qname, NULL, 0);
		mm_pair(b->km, opt->max_gap_ref, opt->pe_bonus, opt->a * 2 + opt->b, opt->a, qlens, n_regs, regs);
	} else {
		mm_map_frag_core(mi, n_segs, qlens, seqs, n_regs, regs, b, opt, qname, NULL, 0);
	}
}

/* Ctx-aware twin of mm_map_frag. Threads a shared mm_mapctx_t (created once by
 * the caller, e.g. mappy's Aligner) and a GPU worker slot through the per-read
 * pipeline so chaining/alignment can run on the GPU when enabled. Passing a
 * NULL mapctx is equivalent to mm_map_frag (pure CPU). */
void mm_map_frag_ctx(const mm_idx_t* mi, int n_segs, const int* qlens, const char** seqs, int* n_regs, mm_reg1_t** regs, mm_tbuf_t* b, const mm_mapopt_t* opt, const char* qname, mm_mapctx_t* mapctx, int slot)
{
	// Bind this thread's logger to the context's logger for the duration of the
	// call so any logging (incl. GPU debug logs) uses an owned, live logger,
	// then restore the caller's previous binding on exit. This avoids leaving a
	// stale thread-local pointer behind and avoids permanently changing the
	// caller's (e.g. main thread's) logger. See AIOSS-4548.
	mm_logger_t* prev_tl = mm_tl_logger();
	if (mapctx && mapctx->core_ctx && mapctx->core_ctx->log)
		mm_set_tl_logger(mapctx->core_ctx->log);
	if ((opt->flag & MM_F_WEAK_PAIRING) && n_segs == 2 && opt->pe_ori >= 0 && (opt->flag & MM_F_CIGAR)) {
		int i;
		for (i = 0; i < n_segs; ++i)
			mm_map_frag_core(mi, 1, &qlens[i], &seqs[i], &n_regs[i], &regs[i], b, opt, qname, mapctx, slot);
		mm_pair(b->km, opt->max_gap_ref, opt->pe_bonus, opt->a * 2 + opt->b, opt->a, qlens, n_regs, regs);
	} else {
		mm_map_frag_core(mi, n_segs, qlens, seqs, n_regs, regs, b, opt, qname, mapctx, slot);
	}
	mm_set_tl_logger(prev_tl);
}

/**
 * mm_map: Convenience wrapper for single-sequence mapping. Forwards to
 * mm_map_frag with n_segs=1.
 */
mm_reg1_t* mm_map(const mm_idx_t* mi, int qlen, const char* seq, int* n_regs, mm_tbuf_t* b, const mm_mapopt_t* opt, const char* qname)
{
	mm_reg1_t* regs;
	mm_map_frag(mi, 1, &qlen, &seq, n_regs, &regs, b, opt, qname);
	return regs;
}

/* Ctx-aware twin of mm_map. Forwards to mm_map_frag_ctx with n_segs=1. */
mm_reg1_t* mm_map_ctx(const mm_idx_t* mi, int qlen, const char* seq, int* n_regs, mm_tbuf_t* b, const mm_mapopt_t* opt, const char* qname, mm_mapctx_t* mapctx, int slot)
{
	mm_reg1_t* regs;
	mm_map_frag_ctx(mi, 1, &qlen, &seq, n_regs, &regs, b, opt, qname, mapctx, slot);
	return regs;
}

static void mm_timer_stats_record_batch(mm_timer_stats_t* ts,
    const mm_tbuf_timers_t* stage_sum,
    const mm_tbuf_timers_t* stage_min,
    const mm_tbuf_timers_t* stage_max,
    uint64_t query_count,
    int n_threads)
{
	if (!ts)
		return;

	const uint64_t prev_batches = ts->batch_count;
	const double denom_threads = (double)(n_threads > 0 ? n_threads : 1);

	ts->seed_sum += stage_sum->seed.sum;
	ts->chain_sum += stage_sum->chain.sum;
	ts->align_sum += stage_sum->align.sum;

	if (prev_batches == 0) {
		ts->seed_min = stage_min->seed.sum;
		ts->seed_max = stage_max->seed.sum;
		ts->chain_min = stage_min->chain.sum;
		ts->chain_max = stage_max->chain.sum;
		ts->align_min = stage_min->align.sum;
		ts->align_max = stage_max->align.sum;
	} else {
		ts->seed_min = MIN(ts->seed_min, stage_min->seed.sum);
		ts->chain_min = MIN(ts->chain_min, stage_min->chain.sum);
		ts->align_min = MIN(ts->align_min, stage_min->align.sum);
		ts->seed_max = MAX(ts->seed_max, stage_max->seed.sum);
		ts->chain_max = MAX(ts->chain_max, stage_max->chain.sum);
		ts->align_max = MAX(ts->align_max, stage_max->align.sum);
	}

	ts->batch_count = prev_batches + 1;
	ts->query_count += query_count;

	const double denom = denom_threads * (double)ts->batch_count;
	ts->seed_avg = denom > 0.0 ? (ts->seed_sum / denom) : 0.0;
	ts->chain_avg = denom > 0.0 ? (ts->chain_sum / denom) : 0.0;
	ts->align_avg = denom > 0.0 ? (ts->align_sum / denom) : 0.0;

	// Per-call stats from stage_sum (which has merged per-call data across threads)
	if (stage_sum->seed.call_count > 0) {
		if (ts->seed_call_count == 0) {
			ts->seed_call_min = stage_sum->seed.call_min;
			ts->seed_call_max = stage_sum->seed.call_max;
		} else {
			ts->seed_call_min = MIN(ts->seed_call_min, stage_sum->seed.call_min);
			ts->seed_call_max = MAX(ts->seed_call_max, stage_sum->seed.call_max);
		}
		ts->seed_call_count += stage_sum->seed.call_count;
		ts->seed_call_avg = ts->seed_sum / (double)ts->seed_call_count;
	}
	if (stage_sum->chain.call_count > 0) {
		if (ts->chain_call_count == 0) {
			ts->chain_call_min = stage_sum->chain.call_min;
			ts->chain_call_max = stage_sum->chain.call_max;
		} else {
			ts->chain_call_min = MIN(ts->chain_call_min, stage_sum->chain.call_min);
			ts->chain_call_max = MAX(ts->chain_call_max, stage_sum->chain.call_max);
		}
		ts->chain_call_count += stage_sum->chain.call_count;
		ts->chain_call_avg = ts->chain_sum / (double)ts->chain_call_count;
	}
	if (stage_sum->align.call_count > 0) {
		if (ts->align_call_count == 0) {
			ts->align_call_min = stage_sum->align.call_min;
			ts->align_call_max = stage_sum->align.call_max;
		} else {
			ts->align_call_min = MIN(ts->align_call_min, stage_sum->align.call_min);
			ts->align_call_max = MAX(ts->align_call_max, stage_sum->align.call_max);
		}
		ts->align_call_count += stage_sum->align.call_count;
		ts->align_call_avg = ts->align_sum / (double)ts->align_call_count;
	}
}

// consolidate timers from worker threads
/**
 * mm_consolidate_timers: Consolidate timing statistics from all worker threads
 * @param s step data with worker thread buffers
 * @param p pipeline data
 *
 * Collects and aggregates timing information from each thread's buffer.
 * Calculates run-level min/max/avg times for seed, chain, and align stages.
 * Useful for performance profiling and load balancing analysis.
 */
static void mm_consolidate_timers(step_t* s, pipeline_t* p)
{
	if (p->n_threads <= 0 || !p->timer_stats)
		return;

	mm_timer_stats_t* ts = p->timer_stats;

	// Aggregate this batch across all worker threads.
	mm_tbuf_timers_t batch_sum;
	mm_tbuf_timers_t batch_min, batch_max;
	memset(&batch_sum, 0, sizeof(batch_sum));
	memset(&batch_min, 0, sizeof(batch_min));
	memset(&batch_max, 0, sizeof(batch_max));

	batch_min.seed.sum = batch_max.seed.sum = s->buf[0]->timers.seed.sum;
	batch_min.chain.sum = batch_max.chain.sum = s->buf[0]->timers.chain.sum;
	batch_min.align.sum = batch_max.align.sum = s->buf[0]->timers.align.sum;

	for (int i = 0; i < p->n_threads; ++i) {
		const double t_s = s->buf[i]->timers.seed.sum;
		const double t_c = s->buf[i]->timers.chain.sum;
		const double t_a = s->buf[i]->timers.align.sum;

		batch_sum.seed.sum += t_s;
		batch_sum.chain.sum += t_c;
		batch_sum.align.sum += t_a;

		if (i > 0) {
			batch_min.seed.sum = MIN(batch_min.seed.sum, t_s);
			batch_min.chain.sum = MIN(batch_min.chain.sum, t_c);
			batch_min.align.sum = MIN(batch_min.align.sum, t_a);
			batch_max.seed.sum = MAX(batch_max.seed.sum, t_s);
			batch_max.chain.sum = MAX(batch_max.chain.sum, t_c);
			batch_max.align.sum = MAX(batch_max.align.sum, t_a);
		}

		// Merge per-call stats across threads
		mm_stage_timer_merge_calls(&batch_sum.seed, &s->buf[i]->timers.seed);
		mm_stage_timer_merge_calls(&batch_sum.chain, &s->buf[i]->timers.chain);
		mm_stage_timer_merge_calls(&batch_sum.align, &s->buf[i]->timers.align);
	}

	mm_timer_stats_record_batch(ts, &batch_sum, &batch_min, &batch_max, (uint64_t)s->n_seq, p->n_threads);

	// Consolidate pool reset counts from per-thread batch buffers
	for (int i = 0; i < p->n_threads; ++i) {
		ts->pool_reset_count += s->trbuf[i]->acc_batch.pool_reset_count;
		ts->pool_reset_count += s->trbuf[i]->pending_batch.pool_reset_count;
		ts->pool_reset_count += s->trbuf[i]->launched_batch.pool_reset_count;
		s->trbuf[i]->acc_batch.pool_reset_count = 0;
		s->trbuf[i]->pending_batch.pool_reset_count = 0;
		s->trbuf[i]->launched_batch.pool_reset_count = 0;
	}

	if (mm_core_ctx_dbg_flag(s->p->core_ctx) & MM_DBG_PRINT_QNAME) {
		const double inv_threads = 1.0 / (double)p->n_threads;
		const double batch_savg = batch_sum.seed.sum * inv_threads;
		const double batch_cavg = batch_sum.chain.sum * inv_threads;
		const double batch_aavg = batch_sum.align.sum * inv_threads;

		mm_log_debug("----------------------------------------------------");
		mm_log_debug("              Min (sec)  Max (sec)  Avg (sec)  ");
		mm_log_debug("----------------------------------------------------");
		mm_log_debug("Seed    = %11.3f %11.3f %11.3f", batch_min.seed.sum, batch_max.seed.sum, batch_savg);
		mm_log_debug("Chain   = %11.3f %11.3f %11.3f", batch_min.chain.sum, batch_max.chain.sum, batch_cavg);
		mm_log_debug("Align   = %11.3f %11.3f %11.3f", batch_min.align.sum, batch_max.align.sum, batch_aavg);
		mm_log_debug("----------------------------------------------------");
		mm_log_debug("Avg (seed + chain + align) per thread (this batch) = %.3f secs", (batch_savg + batch_cavg + batch_aavg));
		mm_log_debug("Total (seed + chain + align) (all batches) for %d thread(s) = %.3f secs", p->n_threads, (ts->seed_sum + ts->chain_sum + ts->align_sum));
	}
}

/**
 * mm_trbuf_is_full: Check if accumulation batch is full and manage pipelining
 * @param tr thread read buffer with 3-batch pipeline
 * @param s step data with batch configuration
 * 
 * Monitors accumulation batch size and moves reads to pending batch when
 * accumulation exceeds the anchor limit. This implements the pipelined batching
 * strategy that allows chaining (CPU or GPU) to process pending batch while new
 * seeds are collected into the accumulation batch.
 */
static inline int mm_seg_independent(const mm_mapopt_t* opt); // defined below

static void mm_trbuf_is_full(mm_trbuf_t* tr, step_t* s)
{
	while (tr->acc_batch.total_n > s->batch_max_anchors) { // if the batch is full
		tr->is_full = 1;
		tr->is_pending = 1;
		// move last read from acc_batch to pending batch (another memory poll)
		chain_read_t* read_ptr_acc_batch = &tr->acc_batch.reads[tr->acc_batch.count - 1];
		chain_read_t* read_ptr_pending_batch = &tr->pending_batch.reads[tr->pending_batch.count];
		/* deep copy, with memory pool transaction*/
		*read_ptr_pending_batch = *read_ptr_acc_batch;
		if (mm_seg_independent(s->p->opt)) {
			read_ptr_pending_batch->qlens = (int*)kmalloc(tr->pending_batch.km, sizeof(int));
			read_ptr_pending_batch->qseqs = (const char**)kmalloc(tr->pending_batch.km, sizeof(const char*));
			read_ptr_pending_batch->qlens[0] = read_ptr_acc_batch->qlens[0];
			read_ptr_pending_batch->qseqs[0] = read_ptr_acc_batch->qseqs[0];
		} else {
			read_ptr_pending_batch->qlens = (int*)kmalloc(tr->pending_batch.km, sizeof(int) * read_ptr_acc_batch->n_seg);
			read_ptr_pending_batch->qseqs = (const char**)kmalloc(tr->pending_batch.km, sizeof(const char*) * read_ptr_acc_batch->n_seg);
			memcpy(read_ptr_pending_batch->qlens, read_ptr_acc_batch->qlens, sizeof(int) * read_ptr_acc_batch->n_seg);
			memcpy(read_ptr_pending_batch->qseqs, read_ptr_acc_batch->qseqs, sizeof(const char*) * read_ptr_acc_batch->n_seg);
		}
		read_ptr_pending_batch->mini_pos = (uint64_t*)kmalloc(tr->pending_batch.km, read_ptr_acc_batch->n_mini_pos * sizeof(uint64_t));
		read_ptr_pending_batch->a = (mm128_t*)kmalloc(tr->pending_batch.km, read_ptr_acc_batch->n * sizeof(mm128_t));
		memcpy(read_ptr_pending_batch->mini_pos, read_ptr_acc_batch->mini_pos, read_ptr_acc_batch->n_mini_pos * sizeof(uint64_t));
		memcpy(read_ptr_pending_batch->a, read_ptr_acc_batch->a, read_ptr_acc_batch->n * sizeof(mm128_t));
		strcpy(read_ptr_pending_batch->seq.name, read_ptr_acc_batch->seq.name);
		tr->pending_batch.count++;
		tr->pending_batch.total_n += read_ptr_acc_batch->n;

		// remove read from acc_batch
		tr->acc_batch.count--;
		tr->acc_batch.total_n -= read_ptr_acc_batch->n;
		kfree(tr->acc_batch.km, read_ptr_acc_batch->mini_pos);
		kfree(tr->acc_batch.km, read_ptr_acc_batch->a);
		kfree(tr->acc_batch.km, read_ptr_acc_batch->qlens);
		kfree(tr->acc_batch.km, read_ptr_acc_batch->qseqs);
	}

	// After the while loop that moves excess reads to pending
	// If we're in GPU mode and acc_batch has sufficient anchors, mark full
	if ((s->p->opt->flag & MM_F_GPU_CHAIN) &&
	    tr->acc_batch.total_n >= s->batch_max_anchors) {
		// Anchor threshold reached - mark batch full if we have any reads to dispatch
		if (tr->acc_batch.count > 0) {
			tr->is_full = 1;
		}
	}
}

// ============================================================================
// Stage Wrapper Functions for Profiling and Modularity
// ============================================================================

/**
 * mm_seg_independent: whether a paired read's segments are mapped *independently*
 * (one chain_read_t per segment) rather than as a single joint fragment.
 *
 * True for --pairing no (MM_F_INDEPEND_SEG) and for weak pairing
 * (MM_F_WEAK_PAIRING, under the same gate the mm_map_frag wrapper uses:
 * pe_ori >= 0 && CIGAR). Weak pairing maps each mate on its own and then
 * re-pairs them with mm_pair in a post-alignment pass (see step 1); --pairing no
 * leaves them unpaired. The legacy mm_map_frag wrapper carries this dispatch for
 * the public API, but the staged worker pipeline must implement it here.
 */
static inline int mm_seg_independent(const mm_mapopt_t* opt)
{
	return (opt->flag & MM_F_INDEPEND_SEG) ||
	    ((opt->flag & MM_F_WEAK_PAIRING) && opt->pe_ori >= 0 && (opt->flag & MM_F_CIGAR));
}

/**
 * prepare_read_for_batch: Prepare a single read/fragment for batch processing
 * @param s step_t structure with sequence data
 * @param i fragment index
 * @param read_ptr pointer to chain_read_t to populate
 * @param km memory pool for allocations
 * @param pe_ori paired-end orientation settings
 * @return number of independent reads created (1 for paired, n_seg for independent)
 *
 * Prepares chain_read_t structures for a fragment. In independent mode (--no-pairing),
 * creates separate structures for each segment. In paired mode, creates a single
 * structure containing both segments.
 */
static int prepare_read_for_batch(step_t* s, long i, chain_read_t* read_ptr, void* km, int pe_ori)
{
	int j, off = s->seg_off[i];
	int n_indep_reads = mm_seg_independent(s->p->opt) ? s->n_seg[i] : 1;

	if (mm_seg_independent(s->p->opt)) {
		// Independent mode: each segment gets its own chain_read_t structure
		for (j = 0; j < s->n_seg[i]; ++j) {
			read_ptr->qlens = (int*)kmalloc(km, sizeof(int));
			read_ptr->qseqs = (const char**)kmalloc(km, sizeof(const char*));

			// Apply reverse complement if needed
			if (s->n_seg[i] == 2 && ((j == 0 && (pe_ori >> 1 & 1)) || (j == 1 && (pe_ori & 1))))
				mm_revcomp_bseq(&s->seq[off + j]);

			read_ptr->qlens[0] = s->seq[off + j].l_seq;
			read_ptr->qseqs[0] = s->seq[off + j].seq;
			read_ptr->n_seg = 1;
			read_ptr->seq.i = i;
			read_ptr->seq.seg_id = j;
			strcpy(read_ptr->seq.name, s->seq[off + j].name);
			read_ptr->seq.n_alt = s->p->mi->n_alt;
			read_ptr->seq.is_alt = 0;
			read_ptr++;
		}
	} else {
		// Paired mode: both segments in single chain_read_t
		read_ptr->qlens = (int*)kmalloc(km, s->n_seg[i] * sizeof(int));
		read_ptr->qseqs = (const char**)kmalloc(km, s->n_seg[i] * sizeof(const char*));
		read_ptr->n_seg = s->n_seg[i];

		for (j = 0; j < s->n_seg[i]; ++j) {
			if (s->n_seg[i] == 2 && ((j == 0 && (pe_ori >> 1 & 1)) || (j == 1 && (pe_ori & 1))))
				mm_revcomp_bseq(&s->seq[off + j]);
			read_ptr->qlens[j] = s->seq[off + j].l_seq;
			read_ptr->qseqs[j] = s->seq[off + j].seq;
		}

		read_ptr->seq.i = i;
		read_ptr->seq.seg_id = 0;
		strcpy(read_ptr->seq.name, s->seq[off].name);
		read_ptr->seq.n_alt = s->p->mi->n_alt;
		read_ptr->seq.is_alt = 0;
	}

	return n_indep_reads;
}

/**
 * process_seeding_batch: Run seeding stage on prepared reads
 * @param s step_t structure
 * @param read_ptr pointer to first read (will be decremented)
 * @param n_reads number of reads to process
 * @param timer pointer to seed timer accumulator
 * @param km memory pool
 * @param batch batch structure to update anchor counts
 *
 * Runs mm_map_seed() on each read and updates batch anchor counts.
 * Note: read_ptr is expected to point one past the last read (will decrement).
 */
static void process_seeding_batch(step_t* s, chain_read_t* read_ptr, int n_reads,
    mm_stage_timer_t* timer, void* km, mm_batch_trbuf_t* batch)
{
	for (int j = 0; j < n_reads; j++) {
		read_ptr--;
		mm_map_seed(s->p->mi, s->p->opt, read_ptr, timer, km, s->p->core_ctx);
		batch->total_n += read_ptr->n;
		assert(read_ptr->n_mini_pos >= 0);
	}
}

/**
 * process_chaining_batch_cpu: Run CPU chaining on a batch of reads
 * @param s step_t structure
 * @param batch batch containing reads to chain
 * @param timer pointer to chain timer accumulator
 *
 * Processes all reads in the batch through CPU chaining (mg_lchain_dp).
 */
static void process_chaining_batch_cpu(step_t* s, mm_batch_trbuf_t* batch, mm_stage_timer_t* timer)
{
	for (int iread = 0; iread < batch->count; iread++) {
		mm_map_chain(s->p->mi, s->p->opt, &batch->reads[iread], timer, batch->km, s->p->core_ctx);
	}
}

#ifdef MM_ENABLE_HIP
/**
 * process_chaining_batch_gpu: Run GPU chaining with CPU fallback
 * @param s step_t structure
 * @param tr thread read buffer with batch pipeline
 * @param timer pointer to chain timer accumulator
 * @param tid thread ID
 * @return 1 if GPU processed batch, 0 if batch was queued
 *
 * Submits batch to GPU for chaining. Handles the 3-batch pipeline:
 * - Submits acc_batch to GPU
 * - Retrieves results for launched_batch
 * - Rotates batches through pipeline
 * Falls back to CPU for reads that don't fit in GPU microbatch.
 */
static int process_chaining_batch_gpu(step_t* s, mm_trbuf_t* tr, mm_stage_timer_t* timer, int tid)
{
	const double t_start = realtime();
	/* Snapshot before the rotation below moves acc_batch elsewhere. Used as
	 * n_calls so sum/count stays meaningful on the first submit (when
	 * launched_batch is still empty). */
	const uint64_t n_submitted = (uint64_t)tr->acc_batch.count;

	mm_batch_trbuf_t kernel_batch = tr->acc_batch;
	gpu_chain_submit(s->p->chain_ctx ? s->p->chain_ctx->gpu_chain : NULL, s->p->mi, s->p->opt, &kernel_batch.reads, &kernel_batch.count, tid, tr->launched_batch.km);

	if (kernel_batch.reads) {
		// GPU returned results - process any CPU fallback reads
		assert(tr->has_launched);
		for (; kernel_batch.count < tr->launched_batch.count; kernel_batch.count++) {
			mm_log_debug("[gpu-chain] read '%s' exceeded GPU microbatch capacity; chaining on CPU",
			    kernel_batch.reads[kernel_batch.count].seq.name);
			mm_map_chain(s->p->mi, s->p->opt, &kernel_batch.reads[kernel_batch.count], NULL, tr->launched_batch.km, s->p->core_ctx);
		}
		assert(kernel_batch.count == tr->launched_batch.count);

		// Dump GPU chain results for comparison with CPU
		for (int i = 0; i < tr->launched_batch.count; i++) {
			mm_dump_chains(s->p->core_ctx, tr->launched_batch.reads[i].seq.name,
			    &tr->launched_batch.reads[i], s->p->mi);
		}

		// Rotate batches: launched->pending, acc->launched, pending->acc
		kernel_batch = tr->launched_batch;
		tr->launched_batch = tr->acc_batch;
		tr->acc_batch = tr->pending_batch;
		tr->pending_batch = kernel_batch;
		tr->is_pending = 1;
	} else {
		// No results yet - just rotate
		assert(!tr->has_launched);
		kernel_batch = tr->launched_batch;
		tr->launched_batch = tr->acc_batch;
		tr->acc_batch = tr->pending_batch;
		tr->pending_batch = kernel_batch;
		tr->is_pending = 0;
	}

	tr->is_full = 0;
	tr->has_launched = 1;

	const int gpu_returned_results = kernel_batch.reads ? 1 : 0;
	/* Skip empty submissions (e.g. end-of-input flush of an already-drained
	 * acc_batch) so sum doesn't grow without bumping count. */
	if (n_submitted > 0)
		mm_stage_timer_record_batch(timer, realtime() - t_start, n_submitted);
	return gpu_returned_results;
}

/**
 * finish_chaining_batch_gpu: Finish remaining GPU chaining work
 * @param s step_t structure
 * @param tr thread read buffer
 * @param timer pointer to chain timer accumulator
 * @param tid thread ID
 *
 * Called at end of input to flush any remaining GPU work.
 * Retrieves final batch from GPU and processes CPU fallback reads.
 */
static void finish_chaining_batch_gpu(step_t* s, mm_trbuf_t* tr, mm_stage_timer_t* timer, int tid)
{
	const double t_start = realtime();
	const uint64_t n_reads = (uint64_t)tr->launched_batch.count;

	mm_batch_trbuf_t kernel_batch;
	gpu_chain_finish(s->p->chain_ctx ? s->p->chain_ctx->gpu_chain : NULL, s->p->mi, s->p->opt, &kernel_batch.reads, &kernel_batch.count, tid, tr->launched_batch.km);

	// CPU fallback for reads that didn't fit in microbatch
	for (; kernel_batch.count < tr->launched_batch.count; kernel_batch.count++) {
		mm_log_debug("[gpu-chain] read '%s' exceeded GPU microbatch capacity; chaining on CPU",
		    kernel_batch.reads[kernel_batch.count].seq.name);
		mm_map_chain(s->p->mi, s->p->opt, &kernel_batch.reads[kernel_batch.count], NULL, kernel_batch.km, s->p->core_ctx);
	}
	assert(kernel_batch.count == tr->launched_batch.count);

	// Dump GPU chain results for comparison with CPU
	for (int i = 0; i < tr->launched_batch.count; i++) {
		mm_dump_chains(s->p->core_ctx, tr->launched_batch.reads[i].seq.name,
		    &tr->launched_batch.reads[i], s->p->mi);
	}

	mm_stage_timer_record_batch(timer, realtime() - t_start, n_reads);

	// Final rotation
	kernel_batch = tr->launched_batch;
	tr->is_full = 0;
	tr->is_pending = 1;
	tr->has_launched = 0;
	tr->launched_batch = tr->acc_batch;
	tr->acc_batch = tr->pending_batch;
	tr->pending_batch = kernel_batch;
}
#endif // MM_ENABLE_HIP

/**
 * process_alignment_batch: Run alignment stage on chained reads
 * @param s step_t structure
 * @param batch batch containing chained reads
 * @param timer pointer to align timer accumulator
 * @param pe_ori paired-end orientation settings
 * @param debug_start_time start time for debug logging (0 to disable)
 *
 * Runs mm_map_align() on each read and performs coordinate conversion
 * for paired-end reads that were reverse-complemented during preparation.
 */
static void process_alignment_batch(step_t* s, mm_batch_trbuf_t* batch, mm_stage_timer_t* timer,
    int pe_ori, double debug_start_time, int tid)
{
	int iread, i, j, off;

	// Copy rep_len & frag_gap to step_t
	for (iread = 0; iread < batch->count; iread++) {
		i = batch->reads[iread].seq.i;
		j = batch->reads[iread].seq.seg_id;
		off = s->seg_off[i] + j;
		for (int k = 0; k < batch->reads[iread].n_seg; k++) {
			s->rep_len[off + k] = batch->reads[iread].rep_len;
			s->frag_gap[off + k] = batch->reads[iread].frag_gap;
		}
	}

	// Run alignment and coordinate conversion
#ifdef HAVE_GPU_ALIGNMENT
	if (s->p->opt->flag & MM_F_GPU_ALIGN) {
		gpu_align_accum_t* gpu_accum = mm_gpu_align_ctx_acquire_accum(
		    s->p->align_ctx->gpu_align, s->p->opt, s->p->mi);

		for (iread = 0; iread < batch->count; iread++) {
			i = batch->reads[iread].seq.i;
			off = s->seg_off[i];
			j = batch->reads[iread].seq.seg_id;
			mm_gpu_accum_add_read(gpu_accum, &batch->reads[iread],
			    &s->reg[off + j], &s->n_reg[off + j],
			    timer, batch->km, s->p->core_ctx);
		}

		if (mm_gpu_accum_should_flush(gpu_accum))
			mm_gpu_accum_flush(gpu_accum, timer, s->p->core_ctx);

		mm_gpu_align_ctx_release_accum(s->p->align_ctx->gpu_align, gpu_accum);
	} else
#endif
	{
		for (iread = 0; iread < batch->count; iread++) {
			i = batch->reads[iread].seq.i;
			off = s->seg_off[i];
			j = batch->reads[iread].seq.seg_id;
			mm_map_align(s->p->mi, s->p->opt, &batch->reads[iread],
			    &s->reg[off + j], &s->n_reg[off + j], timer, batch->km, s->p->core_ctx,
			    s->p->align_ctx, tid);
		}
	}

	// Weak pairing (--pairing weak / splice:sr): mates were mapped independently
	// (mm_seg_independent). Pair each 2-segment fragment now — on the mapping-frame
	// coordinates, BEFORE the strand flip-back below, matching the mm_map_frag
	// wrapper (mm_pair must run before the coordinate conversion below).
	// Both segments of a fragment are always in the same batch (added atomically),
	// so they are aligned by this point.
	if ((s->p->opt->flag & MM_F_WEAK_PAIRING) && pe_ori >= 0 && (s->p->opt->flag & MM_F_CIGAR)) {
		for (iread = 0; iread < batch->count; iread++) {
			if (batch->reads[iread].seq.seg_id != 0)
				continue; // pair once per fragment, from its first segment
			i = batch->reads[iread].seq.i;
			off = s->seg_off[i];
			if (s->n_seg[i] == 2) {
				int qlens2[2];
				qlens2[0] = s->seq[off].l_seq;
				qlens2[1] = s->seq[off + 1].l_seq;
				mm_pair(batch->km, s->p->opt->max_gap_ref, s->p->opt->pe_bonus,
				    s->p->opt->a * 2 + s->p->opt->b, s->p->opt->a, qlens2,
				    &s->n_reg[off], &s->reg[off]);
				// The paired (joint-fragment) mapping path writes the shared
				// rep_len (left by the LAST segment) to *both* segments. Our
				// independent mapping produced per-segment rep_len; match that
				// by giving both segments the last segment's value.
				s->rep_len[off] = s->rep_len[off + 1];
			}
		}
	}

	// Coordinate conversion for reverse-complemented segments
	for (iread = 0; iread < batch->count; iread++) {
		i = batch->reads[iread].seq.i;
		off = s->seg_off[i];
		j = batch->reads[iread].seq.seg_id;

		if (mm_seg_independent(s->p->opt)) {
			if (s->n_seg[i] == 2 && ((j == 0 && (pe_ori >> 1 & 1)) || (j == 1 && (pe_ori & 1)))) {
				mm_revcomp_bseq(&s->seq[off + j]);
				for (int k = 0; k < s->n_reg[off + j]; ++k) {
					mm_reg1_t* r = &s->reg[off + j][k];
					int t = r->qs;
					// Per-segment read: qlens holds only this segment at [0] (seg_id j may be 1).
					r->qs = batch->reads[iread].qlens[0] - r->qe;
					r->qe = batch->reads[iread].qlens[0] - t;
					r->rev = !r->rev;
					// flip trans_strand to match the reverse-complemented read
					if (r->p) {
						if (r->p->trans_strand == 1)
							r->p->trans_strand = 2;
						else if (r->p->trans_strand == 2)
							r->p->trans_strand = 1;
					}
				}
			}
		} else {
			for (j = 0; j < batch->reads[iread].n_seg; ++j) {
				if (s->n_seg[i] == 2 && ((j == 0 && (pe_ori >> 1 & 1)) || (j == 1 && (pe_ori & 1)))) {
					mm_revcomp_bseq(&s->seq[off + j]);
					for (int k = 0; k < s->n_reg[off + j]; ++k) {
						mm_reg1_t* r = &s->reg[off + j][k];
						int t = r->qs;
						r->qs = batch->reads[iread].qlens[j] - r->qe;
						r->qe = batch->reads[iread].qlens[j] - t;
						r->rev = !r->rev;
						if (r->p) {
							if (r->p->trans_strand == 1)
								r->p->trans_strand = 2;
							else if (r->p->trans_strand == 2)
								r->p->trans_strand = 1;
						}
					}
				}
			}
		}

		if ((mm_core_ctx_dbg_flag(s->p->core_ctx) & MM_DBG_PRINT_QNAME) && debug_start_time > 0)
			mm_log_debug("QT\t%s\t%d\t%.6f", s->seq[off].name, 0, realtime() - debug_start_time);
	}
}

// ============================================================================
// Main Worker Function
// ============================================================================

/**
 * worker_for: Worker thread callback for kt_for() parallelization
 * @param _data step_t structure passed to worker
 * @param i_in read index being processed (-1 signals end)
 * @param tid thread ID (used for per-thread buffers)
 * 
 * Main worker function for multi-threaded mapping. Processes one read through
 * all three stages (seed->chain->align). When GPU support is enabled, uses
 * pipelined batching to accumulate reads for GPU-accelerated chaining.
 *
 * Pipeline Stages (via wrapper functions):
 * - prepare_read_for_batch(): Prepare chain_read_t structures
 * - process_seeding_batch(): Find k-mer seeds (always CPU)
 * - process_chaining_batch_cpu/gpu(): Link seeds into chains
 * - process_alignment_batch(): DP alignment and coordinate conversion
 *
 * i_in=-1 signals final flush of pending work
 * 
 * GPU Chaining Limitations:
 * - Only supports DP algorithm; falls back to CPU when MM_F_RMQ flag is set
 * - Reads exceeding GPU microbatch capacity are processed by CPU kernel
 */
static void worker_for(void* _data, long i_in, int tid) // kt_for() callback
{
	step_t* s = (step_t*)_data;
	if (s->p->core_ctx && mm_tl_logger() != s->p->core_ctx->log)
		mm_set_tl_logger(s->p->core_ctx->log);
	long i = i_in;
	int pe_ori = s->p->opt->pe_ori;
	double t = 0.0;
	mm_tbuf_t* b = s->buf[tid];
	mm_trbuf_t* tr = s->trbuf[tid];

	// ========== Phase 1: Read Preparation and Seeding ==========
	if (i != -1) {
		MM_PROFILE_SEED("worker_seed");
		int off = s->seg_off[i];
		assert(s->n_seg[i] <= MM_MAX_SEG);

		if (mm_core_ctx_dbg_flag(s->p->core_ctx) & MM_DBG_PRINT_QNAME) {
			mm_log_debug("QR\t%s\t%d\t%d", s->seq[off].name, tid, s->seq[off].l_seq);
			t = realtime();
		}

		// Determine batch and memory pool
		int n_indep_reads = mm_seg_independent(s->p->opt) ? s->n_seg[i] : 1;
		void* km;
		chain_read_t* read_ptr;
		mm_batch_trbuf_t* target_batch;

		if (n_indep_reads + tr->acc_batch.count <= s->batch_max_reads) {
			target_batch = &tr->acc_batch;
			read_ptr = tr->acc_batch.reads + tr->acc_batch.count;
			tr->acc_batch.count += n_indep_reads;
			km = tr->acc_batch.km;
		} else {
			// Read count threshold reached
			if (s->p->opt->flag & MM_F_GPU_CHAIN) {
				// GPU mode: Keep accumulating in acc_batch beyond N_ACCUM if capacity allows
				// Safety check: ensure we don't overflow allocated capacity
				if (tr->acc_batch.count + n_indep_reads <= s->batch_max_reads) {
					// Still have capacity - continue accumulating
					target_batch = &tr->acc_batch;
					read_ptr = tr->acc_batch.reads + tr->acc_batch.count;
					tr->acc_batch.count += n_indep_reads;
					km = tr->acc_batch.km;
					// Don't set is_full - mm_trbuf_is_full() will decide based on anchors
				} else {
					// Capacity exhausted - mark full and use pending batch
					target_batch = &tr->pending_batch;
					read_ptr = tr->pending_batch.reads + tr->pending_batch.count;
					tr->pending_batch.count += n_indep_reads;
					tr->is_full = 1;
					tr->is_pending = 1;
					km = tr->pending_batch.km;
				}
			} else {
				// CPU mode: use existing behavior (mark full immediately)
				target_batch = &tr->pending_batch;
				read_ptr = tr->pending_batch.reads + tr->pending_batch.count;
				tr->pending_batch.count += n_indep_reads;
				tr->is_full = 1;
				tr->is_pending = 1;
				km = tr->pending_batch.km;
			}
		}

		// Prepare read structures
		prepare_read_for_batch(s, i, read_ptr, km, pe_ori);
		read_ptr += n_indep_reads; // Move past prepared reads

		// Run seeding
		process_seeding_batch(s, read_ptr, n_indep_reads, &b->timers.seed, km, target_batch);

		// Check if batch is full
		mm_trbuf_is_full(tr, s);
		MM_PROFILE_RANGE_POP();
	} else {
		if (tr->flushed) return; // already flushed; skip redundant flush
		tr->is_full = 1; // Signal end of input
	}

	// ========== Phase 2: Chaining ==========
	while (tr->is_full || (i_in == -1 && tr->has_launched)) {
		MM_PROFILE_CHAIN("worker_chain");

		if (tr->is_full) {
#ifdef MM_ENABLE_HIP
			/* Take the GPU branch only when MM_F_GPU_CHAIN is set AND a real
			 * GPU chain ctx is available. mm_gpu_chain_ctx_create returns
			 * NULL on CPU-only builds, when MM_F_GPU_CHAIN is unset, or on
			 * GPU init failure; library users can also pass NULL into the
			 * *_ctx() entry points.
			 * Without this gate, an MM_F_GPU_CHAIN flag with a NULL ctx
			 * routes through process_chaining_batch_gpu / finish_chaining_batch_gpu
			 * which delegate to gpu_chain_submit / gpu_chain_finish; those
			 * are NULL-safe and return (NULL, 0), but the surrounding
			 * CPU-fallback loop then dereferences kernel_batch.reads (NULL)
			 * and segfaults. Falling through to the CPU path here is the
			 * documented behaviour of the *_ctx() API (NULL ctx -> CPU). */
			if ((s->p->opt->flag & MM_F_GPU_CHAIN) && !(s->p->opt->flag & MM_F_RMQ) && s->p->chain_ctx != NULL && s->p->chain_ctx->gpu_chain != NULL) {
				// GPU chaining path: helper records its own batch timing
				// (sum + per-batch call_count/min/max) into b->timers.chain.
				MM_PROFILE_CHAIN("gpu_chain_batch");
				process_chaining_batch_gpu(s, tr, &b->timers.chain, tid);
				MM_PROFILE_RANGE_POP();
			} else {
#endif
				// CPU chaining path: per-read timing is recorded inside mm_map_chain.
				process_chaining_batch_cpu(s, &tr->acc_batch, &b->timers.chain);

				// Rotate batches
				mm_batch_trbuf_t kernel_batch = tr->acc_batch;
				tr->acc_batch = tr->pending_batch;
				tr->pending_batch = kernel_batch;
				tr->has_launched = 0;
				tr->is_full = 0;
				tr->is_pending = 1;
#ifdef MM_ENABLE_HIP
			}
		} else if ((s->p->opt->flag & MM_F_GPU_CHAIN) && !(s->p->opt->flag & MM_F_RMQ) && s->p->chain_ctx != NULL && s->p->chain_ctx->gpu_chain != NULL) {
			// Finish GPU chaining at end of input: helper records its own batch timing.
			assert(i_in == -1 && tr->has_launched);
			MM_PROFILE_CHAIN("gpu_chain_finish");
			finish_chaining_batch_gpu(s, tr, &b->timers.chain, tid);
			MM_PROFILE_RANGE_POP();
#endif
		}

		MM_PROFILE_RANGE_POP();

		// ========== Phase 3: Alignment ==========
		if (tr->is_pending) {
			MM_PROFILE_ALIGN("worker_align");
			mm_batch_trbuf_t* batch = &tr->pending_batch;

			process_alignment_batch(s, batch, &b->timers.align, pe_ori, t, tid);

			// Reset batch for reuse
			mm_trbuf_batch_reset(batch, s->batch_max_reads, s->p->opt, s->p->core_ctx);
			tr->pending_batch = *batch;
			tr->is_pending = 0;
			tr->pending_batch.batchid = tr->acc_batch.batchid + 1;
			MM_PROFILE_RANGE_POP();
		}
	}

#ifdef HAVE_GPU_ALIGNMENT
	if (i_in == -1 && (s->p->opt->flag & MM_F_GPU_ALIGN)) {
		/* flush_all is called once from the pipeline step after kt_for;
		 * nothing to do per-thread here. */
	}
#endif
	if (i_in == -1)
		tr->flushed = 1;
}

/**
 * merge_hits: Merge mapping results from split indices
 * @param s step_t structure containing sequences and output arrays
 *
 * When minimap2 splits the reference into multiple indices for parallel
 * mapping, this function merges the partial results back together:
 * 1. Reads partial results from multiple temporary files
 * 2. Combines regions from each index partition
 * 3. Restores proper reference IDs by applying offsets
 * 4. Applies post-processing (error estimation, MAPQ calculation)
 *
 * This enables efficient split-index mapping and reduction of peak memory.
 */
static void merge_hits(step_t* s)
{
	int f, i, k0, k, max_seg = 0, *n_reg_part, *rep_len_part, *frag_gap_part, *qlens;
	void* km;
	FILE** fp = s->p->fp_parts;
	const mm_mapopt_t* opt = s->p->opt;

	km = km_init();
	for (f = 0; f < s->n_frag; ++f)
		max_seg = max_seg > s->n_seg[f] ? max_seg : s->n_seg[f];
	qlens = CALLOC(int, max_seg + s->p->n_parts * 3);
	n_reg_part = qlens + max_seg;
	rep_len_part = n_reg_part + s->p->n_parts;
	frag_gap_part = rep_len_part + s->p->n_parts;
	for (f = 0, k = k0 = 0; f < s->n_frag; ++f) {
		k0 = k;
		for (i = 0; i < s->n_seg[f]; ++i, ++k) {
			int j, l, t, rep_len = 0;
			qlens[i] = s->seq[k].l_seq;
			for (j = 0, s->n_reg[k] = 0; j < s->p->n_parts; ++j) {
				mm_err_fread(&n_reg_part[j], sizeof(int), 1, fp[j]);
				mm_err_fread(&rep_len_part[j], sizeof(int), 1, fp[j]);
				mm_err_fread(&frag_gap_part[j], sizeof(int), 1, fp[j]);
				s->n_reg[k] += n_reg_part[j];
				if (rep_len < rep_len_part[j])
					rep_len = rep_len_part[j];
			}
			s->reg[k] = CALLOC(mm_reg1_t, s->n_reg[k]);
			for (j = 0, l = 0; j < s->p->n_parts; ++j) {
				for (t = 0; t < n_reg_part[j]; ++t, ++l) {
					mm_reg1_t* r = &s->reg[k][l];
					uint32_t capacity;
					mm_err_fread(r, sizeof(mm_reg1_t), 1, fp[j]);
					r->rid += s->p->rid_shift[j];
					if (opt->flag & MM_F_CIGAR) {
						mm_err_fread(&capacity, 4, 1, fp[j]);
						r->p = (mm_extra_t*)calloc(capacity, 4);
						r->p->capacity = capacity;
						mm_err_fread(r->p, r->p->capacity, 4, fp[j]);
					}
				}
			}
			if (!(opt->flag & MM_F_SR) && s->seq[k].l_seq >= opt->rank_min_len)
				mm_update_dp_max(s->seq[k].l_seq, s->n_reg[k], s->reg[k], opt->rank_frac, opt->a, opt->b);
			for (j = 0; j < s->n_reg[k]; ++j) {
				mm_reg1_t* r = &s->reg[k][j];
				if (r->p)
					r->p->dp_max2 = 0; // reset ->dp_max2 as mm_set_parent() doesn't clear it; necessary with mm_update_dp_max()
				r->subsc = 0; // this may not be necessary
				r->n_sub = 0; // n_sub will be an underestimate as we don't see all the chains now, but it can't be accurate anyway
			}
			mm_hit_sort(km, &s->n_reg[k], s->reg[k], opt->alt_drop);
			mm_set_parent(km, opt->mask_level, opt->mask_len, s->n_reg[k], s->reg[k], opt->a * 2 + opt->b, opt->flag & MM_F_HARD_MLEVEL, opt->alt_drop);
			if (!(opt->flag & MM_F_ALL_CHAINS)) {
				mm_select_sub(km, opt->pri_ratio, s->p->mi->k * 2, opt->best_n, 0, opt->max_gap * 0.8, &s->n_reg[k], s->reg[k]);
				mm_set_sam_pri(s->n_reg[k], s->reg[k]);
			}
			mm_set_mapq2(km, s->n_reg[k], s->reg[k], opt->min_chain_score, opt->a, rep_len, !!(opt->flag & (MM_F_SR | MM_F_SR_RNA)), !!(opt->flag & MM_F_SPLICE));
		}
		if (s->n_seg[f] == 2 && opt->pe_ori >= 0 && (opt->flag & MM_F_CIGAR))
			mm_pair(km, frag_gap_part[0], opt->pe_bonus, opt->a * 2 + opt->b, opt->a, qlens, &s->n_reg[k0], &s->reg[k0]);
	}
	free(qlens);
	km_destroy(km);
}

/**
 * worker_pipeline: Three-stage pipeline worker - coordinates read I/O, mapping, and output
 * @param shared pipeline_t structure with global settings
 * @param step pipeline stage (0=read, 1=map, 2=output)
 * @param in input data from previous stage (or NULL for stage 0)
 * @return output to next stage (NULL for stage 2)
 *
 * Implements kt_pipeline() callback for three-stage processing:
 * - STAGE 0: Reads sequences from input file(s) into batches
 * - STAGE 1: Maps each batch by calling worker_for() on multiple threads
 * - STAGE 2: Outputs results and performs merge_hits() if using split indices
 *
 * Enables I/O parallelization: reading while mapping, outputting while reading.
 */
static void* worker_pipeline(void* shared, int step, void* in)
{
	int i, j, k;
	pipeline_t* p = (pipeline_t*)shared;
	if (p->core_ctx && mm_tl_logger() != p->core_ctx->log)
		mm_set_tl_logger(p->core_ctx->log);
	if (step == 0) { // step 0: read sequences
		return init_step(p);
	} else if (step == 1) { // step 1: map
		step_t* s = (step_t*)in;
		if (p->n_parts > 0)
			merge_hits((step_t*)in);
		else
			kt_for(p->n_threads, worker_for, in, ((step_t*)in)->n_frag);

#ifdef HAVE_GPU_ALIGNMENT
		/* Flush all shared pool accumulators once, after all workers finish. */
		if ((p->opt->flag & MM_F_GPU_ALIGN) && s->p->align_ctx && s->p->align_ctx->gpu_align) {
			step_t* s = (step_t*)in;
			mm_gpu_align_ctx_flush_all(s->p->align_ctx->gpu_align,
			    &s->buf[0]->timers.align, s->p->core_ctx);
		}
#endif

		// Safety net: drain any GPU chain batches still pending per thread.
		// GPU alignment flushing is handled above by
		// mm_gpu_align_ctx_flush_all(), not here. The flushed flag in trbuf
		// ensures this is a no-op if the flush already completed.
		for (int tid = 0; tid < p->n_threads; ++tid)
			worker_for(in, -1, tid);
		return in;
	} else if (step == 2) { // step 2: output
		void* km = 0;
		step_t* s = (step_t*)in;
		const mm_idx_t* mi = p->mi;
		// consolidate timers from threads
		mm_consolidate_timers(s, p);

		if ((p->opt->flag & MM_F_OUT_CS) && !(mm_core_ctx_dbg_flag(p->core_ctx) & MM_DBG_NO_KALLOC))
			km = km_init();
		for (k = 0; k < s->n_frag; ++k) {
			int seg_st = s->seg_off[k], seg_en = s->seg_off[k] + s->n_seg[k];
			for (i = seg_st; i < seg_en; ++i) {
				mm_bseq1_t* t = &s->seq[i];
				if (p->opt->split_prefix && p->n_parts == 0) { // then write to temporary files
					mm_err_fwrite(&s->n_reg[i], sizeof(int), 1, p->fp_split);
					mm_err_fwrite(&s->rep_len[i], sizeof(int), 1, p->fp_split);
					mm_err_fwrite(&s->frag_gap[i], sizeof(int), 1, p->fp_split);
					for (j = 0; j < s->n_reg[i]; ++j) {
						mm_reg1_t* r = &s->reg[i][j];
						mm_err_fwrite(r, sizeof(mm_reg1_t), 1, p->fp_split);
						if (p->opt->flag & MM_F_CIGAR) {
							mm_err_fwrite(&r->p->capacity, 4, 1, p->fp_split);
							mm_err_fwrite(r->p, r->p->capacity, 4, p->fp_split);
						}
					}
				} else if (p->opt->flag & MM_F_OUT_JUNC) { // extra logic for --write-junc
					for (j = 0; j < s->n_reg[i]; ++j) {
						const mm_reg1_t* r = &s->reg[i][j];
						if (r->id != r->parent || r->mapq < 10) continue;
						mm_write_junc(&p->str, mi, t, r);
						if (p->str.l > 0) mm_print_str(p->str.s);
					}
				} else if (s->n_reg[i] > 0) { // the query has at least one hit
					for (j = 0; j < s->n_reg[i]; ++j) {
						const mm_reg1_t* r = &s->reg[i][j];
						assert(!r->sam_pri || r->id == r->parent);
						if ((p->opt->flag & MM_F_NO_PRINT_2ND) && r->id != r->parent)
							continue;
						if (p->opt->flag & MM_F_OUT_SAM)
							mm_write_sam3(&p->str, mi, t, i - seg_st, j, s->n_seg[k], &s->n_reg[seg_st], (const mm_reg1_t* const*)&s->reg[seg_st], km, p->opt->flag, s->rep_len[i]);
						else
							mm_write_paf4(&p->str, mi, t, r, km, p->opt->flag, s->rep_len[i], s->n_seg[k], i - seg_st);
						mm_print_str(p->str.s);
					}
				} else if ((p->opt->flag & MM_F_PAF_NO_HIT) || ((p->opt->flag & MM_F_OUT_SAM) && !(p->opt->flag & MM_F_SAM_HIT_ONLY))) { // output an empty hit, if requested
					if (p->opt->flag & MM_F_OUT_SAM)
						mm_write_sam3(&p->str, mi, t, i - seg_st, -1, s->n_seg[k], &s->n_reg[seg_st], (const mm_reg1_t* const*)&s->reg[seg_st], km, p->opt->flag, s->rep_len[i]);
					else
						mm_write_paf4(&p->str, mi, t, 0, 0, p->opt->flag, s->rep_len[i], s->n_seg[k], i - seg_st);
					mm_print_str(p->str.s);
				}
			}
		}
		km_destroy(km);
		if (mm_core_ctx_verbose(p->core_ctx) >= 3)
			mm_log("[M::%s::%.3f*%.2f] mapped %d sequences", __func__, realtime() - mm_realtime0, cputime() / (realtime() - mm_realtime0), s->n_seq);
		free_step(s);
	}
	return 0;
}

/**
 * open_bseqs: Open sequence input file handles
 * @param n number of files to open
 * @param fn array of filenames
 * @return array of mm_bseq_file_t pointers (opened file handles)
 *
 * Opens multiple sequence input files and returns file handles for batch reading.
 * Handles errors gracefully by closing any successfully opened files if any fail.
 */
static mm_bseq_file_t** open_bseqs(int n, const char** fn)
{
	mm_bseq_file_t** fp;
	int i, j;
	fp = (mm_bseq_file_t**)calloc(n, sizeof(mm_bseq_file_t*));
	for (i = 0; i < n; ++i) {
		if ((fp[i] = mm_bseq_open(fn[i])) == 0) {
			if (mm_verbose >= 1)
				mm_log_error("ERROR: failed to open file '%s': %s", fn[i], strerror(errno));
			for (j = 0; j < i; ++j)
				mm_bseq_close(fp[j]);
			free(fp);
			return 0;
		}
	}
	return fp;
}

/**
 * mm_map_file_frag: Main entry point - map sequence files with multi-threading
 * @param idx minimap2 index
 * @param n_segs number of input file segments (1=single file, 2+=paired-end/fragment files)
 * @param fn array of filenames
 * @param opt mapping options
 * @param n_threads number of worker threads to use
 * @return 0 on success, -1 on error
 *
 * High-level API function that:
 * 1. Opens input files
 * 2. Initializes the three-stage kt_pipeline
 * 3. Orchestrates parallel mapping across multiple threads
 * 4. Handles optional output splitting for large datasets
 *
 * This is typically the main entry point for end-users or applications.
 */
/* Ctx-aware twin. The supplied mapctx carries core_ctx and chain_ctx
 * (which may be NULL) plumbed onto pipeline_t so the worker threads use
 * them for the full duration of this call. Lifecycle is owned by caller. */
int mm_map_file_frag_ctx(const mm_idx_t* idx, int n_segs, const char** fn, const mm_mapopt_t* opt, int n_threads, mm_mapctx_t* mapctx)
{
	int i, pl_threads;
	pipeline_t pl;
	if (n_segs < 1)
		return -1;
	memset(&pl, 0, sizeof(pipeline_t));
	pl.n_fp = n_segs;
	pl.fp = open_bseqs(pl.n_fp, fn);
	if (pl.fp == 0)
		return -1;
	pl.opt = opt, pl.mi = idx;
	pl.n_threads = n_threads > 1 ? n_threads : 1;
	pl.mini_batch_size = opt->mini_batch_size;
	pl.chain_ctx = mapctx ? mapctx->chain_ctx : NULL;
	pl.align_ctx = mapctx ? mapctx->align_ctx : NULL;
	pl.core_ctx = mapctx ? mapctx->core_ctx : NULL;
	pl.timer_stats = mapctx ? mapctx->timer_stats : NULL;
	if (opt->split_prefix)
		pl.fp_split = mm_split_init(opt->split_prefix, idx);
	pl_threads = n_threads == 1 ? 1 : (opt->flag & MM_F_2_IO_THREADS) ? 3
									  : 2;
	kt_pipeline(pl_threads, worker_pipeline, &pl, 3);
	free(pl.str.s);
	if (pl.fp_split)
		fclose(pl.fp_split);
	for (i = 0; i < pl.n_fp; ++i)
		mm_bseq_close(pl.fp[i]);
	free(pl.fp);
	return 0;
}

/* Batched in-memory mapping. Maps n_seqs single-segment reads through the
 * shared mapping context, reusing the same batch machinery (kt_for +
 * worker_for + the GPU chain/align accumulators) that the file pipeline uses,
 * so GPU stages run with real cross-read batching rather than one launch per
 * read. For read i, out_n_regs[i] receives the hit count and out_regs[i] a
 * malloc'd mm_reg1_t array (each element's ->p is malloc'd); the caller owns
 * and frees them exactly like the array returned by mm_map. Single-segment
 * reads only. Returns 0 on success, -1 on error. */
int mm_map_batch_ctx(const mm_idx_t* mi, int n_seqs, const char* const* names, const char* const* seqs, const mm_mapopt_t* opt, int n_threads, mm_mapctx_t* mapctx, int* out_n_regs, mm_reg1_t** out_regs)
{
	int i;
	pipeline_t pl;
	step_t* s;
	if (!mi || !seqs || !opt || !out_n_regs || !out_regs)
		return -1;
	for (i = 0; i < n_seqs; ++i) {
		out_n_regs[i] = 0;
		out_regs[i] = NULL;
	}
	if (n_seqs <= 0)
		return 0;

	// Bind this thread's logger to the context's logger for the duration of the
	// call (kt_for work-stealing may leave the calling thread without running
	// worker_for, so its thread-local logger could otherwise still point at a
	// previously destroyed context's freed logger), then restore the caller's
	// previous binding on exit. See AIOSS-4548.
	mm_logger_t* prev_tl = mm_tl_logger();
	if (mapctx && mapctx->core_ctx && mapctx->core_ctx->log)
		mm_set_tl_logger(mapctx->core_ctx->log);

	memset(&pl, 0, sizeof(pipeline_t));
	pl.opt = opt, pl.mi = mi;
	pl.n_threads = n_threads > 1 ? n_threads : 1;
	pl.mini_batch_size = opt->mini_batch_size;
	pl.chain_ctx = mapctx ? mapctx->chain_ctx : NULL;
	pl.align_ctx = mapctx ? mapctx->align_ctx : NULL;
	pl.core_ctx = mapctx ? mapctx->core_ctx : NULL;
	pl.timer_stats = mapctx ? mapctx->timer_stats : NULL;

	s = (step_t*)calloc(1, sizeof(step_t));
	if (!s) {
		mm_set_tl_logger(prev_tl);
		return -1;
	}
	s->p = &pl;
	s->n_seq = n_seqs;

	s->seq = (mm_bseq1_t*)calloc(n_seqs, sizeof(mm_bseq1_t));
	if (!s->seq)
		goto fail;
	for (i = 0; i < n_seqs; ++i) {
		const char* sq = seqs[i] ? seqs[i] : "";
		int l = (int)strlen(sq);
		const char* nm = (names && names[i]) ? names[i] : "";
		int ln = (int)strlen(nm);
		s->seq[i].l_seq = l;
		s->seq[i].rid = pl.n_processed++;
		s->seq[i].seq = (char*)malloc(l + 1);
		s->seq[i].name = (char*)malloc(ln + 1);
		if (!s->seq[i].seq || !s->seq[i].name)
			goto fail;
		memcpy(s->seq[i].seq, sq, l + 1);
		memcpy(s->seq[i].name, nm, ln + 1);
		s->seq[i].qual = NULL;
		s->seq[i].comment = NULL;
	}

	s->buf = (mm_tbuf_t**)calloc(pl.n_threads, sizeof(mm_tbuf_t*));
	if (!s->buf)
		goto fail;
	for (i = 0; i < pl.n_threads; ++i) {
		s->buf[i] = mm_tbuf_init();
		if (!s->buf[i])
			goto fail;
	}

	s->n_reg = (int*)calloc(5 * (size_t)n_seqs, sizeof(int));
	if (!s->n_reg)
		goto fail;
	s->seg_off = s->n_reg + n_seqs;
	s->n_seg = s->seg_off + n_seqs;
	s->rep_len = s->n_seg + n_seqs;
	s->frag_gap = s->rep_len + n_seqs;
	s->reg = (mm_reg1_t**)calloc(n_seqs, sizeof(mm_reg1_t*));
	if (!s->reg)
		goto fail;

	// One segment per read (no paired/fragment batching in this entry point).
	s->n_frag = n_seqs;
	for (i = 0; i < n_seqs; ++i) {
		s->n_seg[i] = 1;
		s->seg_off[i] = i;
	}

#ifdef MM_ENABLE_HIP
	if ((opt->flag & MM_F_GPU_CHAIN) && pl.chain_ctx && pl.chain_ctx->gpu_chain) {
		s->batch_max_anchors = gpu_chain_ctx_target_batch_anchors(pl.chain_ctx->gpu_chain);
		int gpu_max_reads = gpu_chain_ctx_max_reads(pl.chain_ctx->gpu_chain);
		size_t computed_max_reads = (s->batch_max_anchors / 10000) + N_ACCUM;
		s->batch_max_reads = (int)MIN(computed_max_reads, gpu_max_reads);
	} else
#endif
	{
		s->batch_max_anchors = SIZE_MAX;
		s->batch_max_reads = N_ACCUM;
	}
	s->trbuf = (mm_trbuf_t**)calloc(pl.n_threads, sizeof(mm_trbuf_t*));
	if (!s->trbuf)
		goto fail;
	for (i = 0; i < pl.n_threads; ++i) {
		s->trbuf[i] = mm_trbuf_init(s->batch_max_reads, opt);
		if (!s->trbuf[i])
			goto fail;
	}

	// Run the map stage exactly like worker_pipeline step 1: parallel
	// seed/chain/align, flush the shared GPU align accumulators, then drain
	// any GPU chain batches still pending per thread.
	kt_for(pl.n_threads, worker_for, s, s->n_frag);
#ifdef HAVE_GPU_ALIGNMENT
	if ((opt->flag & MM_F_GPU_ALIGN) && pl.align_ctx && pl.align_ctx->gpu_align)
		mm_gpu_align_ctx_flush_all(pl.align_ctx->gpu_align, &s->buf[0]->timers.align, pl.core_ctx);
#endif
	for (i = 0; i < pl.n_threads; ++i)
		worker_for(s, -1, i);
#ifdef HAVE_GPU_ALIGNMENT
	// Flush any reads aligned during the per-thread chain drain above.
	if ((opt->flag & MM_F_GPU_ALIGN) && pl.align_ctx && pl.align_ctx->gpu_align)
		mm_gpu_align_ctx_flush_all(pl.align_ctx->gpu_align, &s->buf[0]->timers.align, pl.core_ctx);
#endif

	// Transfer per-read region ownership to the caller, then tear the step
	// down (free_step skips the regions we nulled out).
	for (i = 0; i < n_seqs; ++i) {
		out_n_regs[i] = s->n_reg[i];
		out_regs[i] = s->reg[i];
		s->reg[i] = NULL;
		s->n_reg[i] = 0;
	}
	free_step(s);
	mm_set_tl_logger(prev_tl);
	return 0;

fail:
	// NULL-safe teardown of a partially constructed step. free_step() assumes a
	// fully built step (it walks seg_off/n_seg and destroys every buf/trbuf
	// entry), so it cannot be used here.
	if (s) {
		if (s->trbuf) {
			for (i = 0; i < pl.n_threads; ++i)
				if (s->trbuf[i])
					mm_trbuf_destroy(s->trbuf[i], pl.core_ctx);
			free(s->trbuf);
		}
		if (s->buf) {
			for (i = 0; i < pl.n_threads; ++i)
				if (s->buf[i])
					mm_tbuf_destroy(s->buf[i]);
			free(s->buf);
		}
		if (s->seq) {
			for (i = 0; i < n_seqs; ++i) {
				free(s->seq[i].seq);
				free(s->seq[i].name);
			}
			free(s->seq);
		}
		free(s->reg);
		free(s->n_reg); // seg_off/n_seg/rep_len/frag_gap alias into this block
		free(s);
	}
	mm_set_tl_logger(prev_tl);
	return -1;
}

int mm_map_file_frag(const mm_idx_t* idx, int n_segs, const char** fn, const mm_mapopt_t* opt, int n_threads)
{
	mm_logger_t* log = mm_logger_create(0, NULL);
	mm_set_tl_logger(log);
	mm_mapctx_t* ctx = mm_mapctx_create(log, opt, n_threads);
	int ret = mm_map_file_frag_ctx(idx, n_segs, fn, opt, n_threads, ctx);
	mm_mapctx_destroy(ctx);
	mm_logger_destroy(log);
	return ret;
}

/**
 * mm_map_file: Convenience wrapper for mapping a single sequence file
 * @param idx minimap2 index
 * @param fn filename
 * @param opt mapping options
 * @param n_threads number of worker threads
 * @return 0 on success, -1 on error
 *
 * Simplified entry point for single-file mapping.
 * Simply calls mm_map_file_frag() with n_segs=1.
 */
/* Ctx-aware twin. Forwards to mm_map_file_frag_ctx. */
int mm_map_file_ctx(const mm_idx_t* idx, const char* fn, const mm_mapopt_t* opt, int n_threads, mm_mapctx_t* mapctx)
{
	return mm_map_file_frag_ctx(idx, 1, &fn, opt, n_threads, mapctx);
}

int mm_map_file(const mm_idx_t* idx, const char* fn, const mm_mapopt_t* opt, int n_threads)
{
	mm_logger_t* log = mm_logger_create(0, NULL);
	mm_set_tl_logger(log);
	mm_mapctx_t* ctx = mm_mapctx_create(log, opt, n_threads);
	int ret = mm_map_file_ctx(idx, fn, opt, n_threads, ctx);
	mm_mapctx_destroy(ctx);
	mm_logger_destroy(log);
	return ret;
}

/**
 * mm_split_merge: Map sequences using split index files and merge results
 * @param n_segs number of input query sequence files
 * @param fn array of query filenames
 * @param opt mapping options
 * @param n_split_idx number of split index files (parts)
 * @return 0 on success, -1 on error
 *
 * Handles mapping for large reference sequences split into multiple index files.
 * Reconstructs reference ID space by reading from split parts via separate FILE handles.
 *
 * Process:
 * - Open query sequence files and split index files
 * - Compute cumulative rid_shift offsets for stitching reference ID space
 * - Output SAM headers for all reference sequences across all parts
 * - Execute worker_pipeline with 2 threads to read and process queries
 * - Each query is mapped against all index parts simultaneously
 * - Results are merged back to original reference coordinates via rid_shift
 * - Cleanup: close files, destroy index, remove temporary split files
 *
 * Used for reference sequences that exceed single index file capacity.
 */
/* Ctx-aware twin. Internal only -- mm_split_merge is not part of the public
 * ABI in minimap.h, so the _ctx variant is declared in map_priv.h. main.c
 * uses this so the process-scope chain ctx covers the post-loop split-merge
 * phase too. */
int mm_split_merge_ctx(int n_segs, const char** fn, const mm_mapopt_t* opt, int n_split_idx, mm_mapctx_t* mapctx)
{
	if (n_segs < 1 || n_split_idx < 1)
		return -1;

	int i;
	pipeline_t pl;
	mm_idx_t* mi;
	memset(&pl, 0, sizeof(pipeline_t));
	pl.n_fp = n_segs;
	pl.fp = open_bseqs(pl.n_fp, fn);
	if (pl.fp == 0)
		return -1;
	pl.opt = opt;
	pl.mini_batch_size = opt->mini_batch_size;
	pl.chain_ctx = mapctx ? mapctx->chain_ctx : NULL;
	pl.align_ctx = mapctx ? mapctx->align_ctx : NULL;
	pl.core_ctx = mapctx ? mapctx->core_ctx : NULL;
	pl.timer_stats = mapctx ? mapctx->timer_stats : NULL;

	pl.n_parts = n_split_idx;
	pl.fp_parts = CALLOC(FILE*, pl.n_parts);
	pl.rid_shift = CALLOC(uint32_t, pl.n_parts);
	pl.mi = mi = mm_split_merge_prep(opt->split_prefix, n_split_idx, pl.fp_parts, pl.rid_shift);
	if (pl.mi == 0) {
		free(pl.fp_parts);
		free(pl.rid_shift);
		return -1;
	}
	for (i = n_split_idx - 1; i > 0; --i)
		pl.rid_shift[i] = pl.rid_shift[i - 1];
	for (pl.rid_shift[0] = 0, i = 1; i < n_split_idx; ++i)
		pl.rid_shift[i] += pl.rid_shift[i - 1];
	if (opt->flag & MM_F_OUT_SAM)
		for (i = 0; i < (int32_t)pl.mi->n_seq; ++i)
			mm_print("@SQ\tSN:%s\tLN:%d", pl.mi->seq[i].name, pl.mi->seq[i].len);

	kt_pipeline(2, worker_pipeline, &pl, 3);

	free(pl.str.s);
	mm_idx_destroy(mi);
	free(pl.rid_shift);
	for (i = 0; i < n_split_idx; ++i)
		fclose(pl.fp_parts[i]);
	free(pl.fp_parts);
	for (i = 0; i < pl.n_fp; ++i)
		mm_bseq_close(pl.fp[i]);
	free(pl.fp);
	mm_split_rm_tmp(opt->split_prefix, n_split_idx);
	return 0;
}

/* Public, non-ABI-changing wrapper: same signature as before, NULL ctx. The
 * ctx-aware version (mm_split_merge_ctx) is what main.c uses to share the
 * process-scope GPU chain ctx across the split-merge phase. */
int mm_split_merge(int n_segs, const char** fn, const mm_mapopt_t* opt, int n_split_idx)
{
	mm_logger_t* log = mm_logger_create(0, NULL);
	mm_set_tl_logger(log);
	mm_mapctx_t* ctx = mm_mapctx_create(log, opt, 1);
	int ret = mm_split_merge_ctx(n_segs, fn, opt, n_split_idx, ctx);
	mm_mapctx_destroy(ctx);
	mm_logger_destroy(log);
	return ret;
}
