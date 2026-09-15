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

#include <assert.h>
#include <cstdlib>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>


#include "chain_priv.h"
#include "gpu_device_descriptor.h"
#include "plutils.h"
#include "plmem.cuh"
#include "plrange.cuh"
#include "plscore.cuh"
#include "plchain.h"
#include "mm_log.h"
#include "mm_profiler.h"

static inline double gpu_chain_wtime(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#ifdef DEBUG_CHECK
#include "planalyze.cuh"
#include "debug.h"
#endif // DEBUG_CHECK

// utils functions

/**
 * translate relative predecessor index to abs index 
 * Input
 *  rel[]   relative predecessor index
 * Output
 *  p[]     absolute predecessor index (of each read)
 */
void p_rel2idx(const uint16_t* rel, int64_t* p, size_t n)
{
	for (int i = 0; i < n; ++i) {
		if (rel[i] == 0)
			p[i] = -1;
		else
			p[i] = i - rel[i];
	}
}

//////////////////////////////////////////////////////////////////////////
///////////         Backtracking    //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

/**
 * @brief start from end index of the chain, find the location of min score on
 * the chain until anchor has no predecessor OR anchor is in another chain
 *
 * @param max_drop
 * @param z [in] {sc, anchor idx}, sorted by sc
 * @param f [in] score
 * @param p [in] predecessor
 * @param k [in] chain end index
 * @param t [update] 0 for unchained anchor, 1 for chained anchor
 * @return min_i minmute score location in the chain
 */

static int64_t mg_chain_bk_end(int32_t max_drop, const mm128_t* z,
    const int32_t* f, const int64_t* p, int32_t* t,
    int64_t k)
{
	int64_t i = z[k].y, end_i = -1, max_i = i;
	int32_t max_s = 0;
	if (i < 0 || t[i] != 0) return i;
	do {
		int32_t s;
		t[i] = 2;
		end_i = i = p[i];
		s = i < 0 ? z[k].x : (int32_t)z[k].x - f[i];
		if (s > max_s)
			max_s = s, max_i = i;
		else if (max_s - s > max_drop)
			break;
	} while (i >= 0 && t[i] == 0);
	for (i = z[k].y; i >= 0 && i != end_i; i = p[i]) // reset modified t[]
		t[i] = 0;
	return max_i;
}

void plchain_backtracking(hostMemPtr* host_mem, chain_read_t* reads, Misc misc, void* km)
{
	int max_drop = misc.bw;
	if (misc.max_dist_x < misc.bw) misc.max_dist_x = misc.bw;
	if (misc.max_dist_y < misc.bw && !misc.is_cdna) misc.max_dist_y = misc.bw;
	if (misc.is_cdna) max_drop = INT32_MAX;

	size_t n_read = host_mem->size;

	uint16_t* p_hostmem = host_mem->p;
	int32_t* f = host_mem->f;
	for (int i = 0; i < n_read; i++) {
		int64_t* p;
		KMALLOC(km, p, reads[i].n);
		p_rel2idx(p_hostmem, p, reads[i].n);
// print scores
#if defined(DEBUG_VERBOSE) && 0
		debug_print_score(p, f, reads[i].n);
#endif
// Check score w.r.t to input (MAKE SURE INPUT SCORE EXISTS: search for SCORE CHECK)
#if defined(DEBUG_CHECK) && 0
		debug_check_score(p, f, reads[i].p, reads[i].f, reads[i].n);
#endif

		/* Backtracking */
		uint64_t* u;
		int32_t *v, *t;
		KMALLOC(km, v, reads[i].n);
		KCALLOC(km, t, reads[i].n);
		int32_t n_u, n_v;

#ifdef DEBUG_VERBOSE
		debug_print_backtrack_params(reads[i].seq.name, reads[i].n, f, p, misc.min_cnt, misc.min_score, max_drop);
#endif

		u = mg_chain_backtrack(km, reads[i].n, f, p, v, t, misc.min_cnt, misc.min_score, max_drop, &n_u, &n_v);
		reads[i].u = u;
		reads[i].n_u = n_u;
		kfree(km, p);
		// here f is not managed by km memory pool
		kfree(km, t);
		if (n_u == 0) {
			kfree(km, reads[i].a);
			kfree(km, v);
			reads[i].a = 0;

			f += reads[i].n;
			p_hostmem += reads[i].n;
			continue;
		}

		mm128_t* new_a = compact_a(km, n_u, u, n_v, v, reads[i].a);
		reads[i].a = new_a;

#ifdef DEBUG_VERBOSE
		debug_print_chain(reads[i].a, reads[i].u, reads[i].n_u, reads[i].seq.name);
#endif
		f += reads[i].n;
		p_hostmem += reads[i].n;
	}
}


//////////////////////////////////////////////////////////////////////////
///////////         Stream Management    /////////////////////////////////
//////////////////////////////////////////////////////////////////////////


/*
 * Accepts a worker that has already been synced, and finished processing a batch
 * Finish and cleanup the worker, save primary chain results to unpinned CPU memory.  
 * RETURN: number of reads in last batch 
*/
int plchain_post_gpu_helper(gpu_chain_ctx_t* ctx, int wid, Misc misc, void* km)
{
	gpu_worker_t* w = &ctx->workers[wid];
	hipSetDevice(w->device_id);
	int n_reads = 0; // Number of reads in the batch

	seg_t* long_segs = w->long_mem.long_segs_og_idx;
	seg_t* long_segs_buf = w->long_mem.long_segs_buf_idx;
	size_t long_seg_idx = 0;
	for (int uid = 0; uid < ctx->score_config.micro_batch; uid++) {
		if (w->host_mems[uid].size == 0) continue;
		// reorg long to each host mem ptr
		// NOTE: this is the number of long segs till this microbatch
		unsigned int long_segs_num = w->host_mems[uid].long_segs_num[0];
		size_t total_n_long_segs = 0;
		for (; long_seg_idx < long_segs_num; long_seg_idx++) {
			const size_t seg_len = long_segs[long_seg_idx].end_idx -
			    long_segs[long_seg_idx].start_idx;
			const size_t dst = long_segs[long_seg_idx].start_idx;
			const size_t src = long_segs_buf[long_seg_idx].start_idx;
			memcpy(&w->host_mems[uid].f[dst], &w->long_mem.f_long[src],
			    seg_len * sizeof(int32_t));
			memcpy(&w->host_mems[uid].p[dst], &w->long_mem.p_long[src],
			    seg_len * sizeof(uint16_t));
			total_n_long_segs += seg_len;
		}

		// backtrack after p/f is copied
		plchain_backtracking(&w->host_mems[uid], w->reads + n_reads, misc, km);
		// accumulate n_reads
		n_reads += w->host_mems[uid].size;
	}

	return n_reads;
}


/* 
 * 1. synchronize stream and process previous batch. cleanup stream
 * 2. launch kernels (asynchornizely) for the input batch 
 */

void plchain_cal_score_async(chain_read_t** reads_, int* n_read_, Misc misc, gpu_chain_ctx_t* ctx, int wid, void* km)
{
	gpu_worker_t* w = &ctx->workers[wid];
	hipSetDevice(w->device_id);

	chain_read_t* reads = *reads_;
	*reads_ = NULL;
	int n_read = *n_read_;
	*n_read_ = 0;

	double t_sync_prev = 0, t_backtrack = 0;

	/* sync worker and process previous batch */
	if (w->busy) {
		double t0 = gpu_chain_wtime();
		cudaStreamSynchronize(w->cudastream);
		double t1 = gpu_chain_wtime();
		*n_read_ = plchain_post_gpu_helper(ctx, wid, misc, km);
		double t2 = gpu_chain_wtime();
		*reads_ = w->reads;
		w->busy = false;
		t_sync_prev = t1 - t0;
		t_backtrack = t2 - t1;
	}

	size_t total_n = 0;
	for (int i = 0; i < n_read; i++) {
		total_n += reads[i].n;
	}

	// reset long seg counters
	cudaMemsetAsync(w->dev_mem.d_long_seg_count, 0, sizeof(unsigned int), w->cudastream);
	cudaMemsetAsync(w->dev_mem.d_total_n_long, 0, sizeof(size_t), w->cudastream);
	cudaCheck();
	w->long_mem.total_long_segs_num[0] = 0;
	w->long_mem.total_long_segs_n[0] = 0;

	for (int uid = 0; uid < ctx->score_config.micro_batch; uid++) {
		w->host_mems[uid].long_segs_num[0] = 0;
		w->host_mems[uid].index = uid;
		w->host_mems[uid].griddim = 0;
		w->host_mems[uid].size = 0;
		w->host_mems[uid].total_n = 0;
		w->host_mems[uid].cut_num = 0;
	}

	w->reads = reads;
	w->n_read = n_read;
	int read_start = 0;

	double t_reorg_sum = 0, t_queue_sum = 0;

	for (int uid = 0; uid < ctx->score_config.micro_batch; uid++) {
		if (read_start == n_read) continue;
		MM_PROFILE_CHAIN("microbatch");
		// decide the size of micro batch
		size_t batch_n = 0;
		int read_end = 0;
		size_t cut_num = 0;
		int griddim = 0;
		for (read_end = read_start; read_end < n_read; read_end++) {
			if (batch_n + reads[read_end].n > ctx->max_anchors_per_worker) {
				break;
			}
			batch_n += reads[read_end].n;
			int an_p_block = ctx->range_config.anchor_per_block;
			int an_p_cut = ctx->range_config.blockdim;
			int block_num = (reads[read_end].n - 1) / an_p_block + 1;
			griddim += block_num;
			cut_num += (reads[read_end].n - 1) / an_p_cut + 1;
		}

		assert(ctx->max_anchors_per_worker >= batch_n);
		assert(ctx->max_range_grid >= griddim);
		assert(ctx->max_num_cut >= cut_num);

		double t_r0 = gpu_chain_wtime();
		MM_PROFILE_CHAIN("reorg");
		plmem_reorg_input_arr(ctx, reads + read_start, read_end - read_start,
		    &w->host_mems[uid], w->staging_raw);
		double t_r1 = gpu_chain_wtime();
		MM_PROFILE_RANGE_POP();

		plmem_async_h2d_short_memcpy(w, uid);
		plmem_async_soa_extract(w, uid);
		plrange_async_range_selection(ctx, &w->dev_mem, &w->cudastream, misc);
		plscore_async_short_mid_forward_dp(ctx, &w->dev_mem, &w->cudastream, misc);
		plmem_async_d2h_short_memcpy(w, uid);
		double t_r2 = gpu_chain_wtime();

		t_reorg_sum += t_r1 - t_r0;
		t_queue_sum += t_r2 - t_r1;
		read_start = read_end;

		MM_PROFILE_RANGE_POP();
	}

	if (read_start < n_read) {
		mm_log_warn("Unable to fit reads {} - {} into a microbatch. Fall back to cpu chaining", read_start, n_read - 1);
	}

	// step6: sort long segments on device and launch long kernel (all on-stream, no CPU sync)
	double t_long0 = gpu_chain_wtime();
	plmem_async_sort_long_segs(w, w->max_long_segs);
	plscore_async_long_forward_dp(ctx, &w->dev_mem, &w->cudastream, misc);
	plmem_async_d2h_long_memcpy(w);
	double t_long_queue = gpu_chain_wtime();

	w->busy = true;
	cudaCheck();

	mm_log_debug("[gpu-chain] cal_score: sync_prev={:.3f}s backtrack={:.3f}s "
		     "reorg={:.3f}s queue_short={:.3f}s queue_long={:.3f}s | "
		     "n_read={} anchors={}",
	    t_sync_prev, t_backtrack,
	    t_reorg_sum, t_queue_sum, t_long_queue - t_long0,
	    n_read, total_n);
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

gpu_stream_config_t parse_gpu_config(const char* gpu_config_file)
{
	if (gpu_config_file == NULL || gpu_config_file[0] == '\0') {
		gpu_stream_config_t empty;
		memset(&empty, 0, sizeof(empty));
		return empty;
	}
	return plmem_parse_stream_config(gpu_config_file);
}

/* Heap allocator for gpu_chain_ctx_t. Implemented here because C callers
 * (map.c) cannot sizeof the struct (plmem.cuh pulls in hip/hip_runtime.h
 * for cudaDeviceProp / pthread types). zero-init matches the assumption
 * that gpu_chain_free()/_dealloc()/uninit ctx are interchangeable empty states. */
extern "C" gpu_chain_ctx_t* gpu_chain_alloc(void)
{
	return (gpu_chain_ctx_t*)std::calloc(1, sizeof(gpu_chain_ctx_t));
}

extern "C" void gpu_chain_dealloc(gpu_chain_ctx_t* ctx)
{
	std::free(ctx);
}

#ifdef __cplusplus
}
#endif

void gpu_chain_init(gpu_chain_ctx_t* ctx, const gpu_stream_config_t* config,
    int n_threads, int n_gpu_workers, gpu_device_list_t device_list)
{
	assert(ctx != NULL);
	assert(!device_list.empty() && "gpu_chain_init: caller must provide a non-empty device list");
	// No more writes to mm_mapopt_t::gpu_chain_*; the resolved batch sizing is
	// captured directly on the ctx (max_anchors_per_batch / max_reads_per_batch /
	// min_n_anchors) and read via the gpu_chain_ctx_* getters.
	plmem_stream_initialize(ctx, NULL, NULL, NULL, config, n_threads, n_gpu_workers, std::move(device_list));
#ifdef DEBUG_PRINT
	mm_gpu_log_info("gpu initialized for chaining");
	mm_gpu_log_info("Compile time config:");
#ifdef USEHIP
	mm_gpu_log_info("  USE HIP");
#else
	mm_gpu_log_info("  USE CUDA");
#endif // USEHIP
#ifdef MAX_MICRO_BATCH
	mm_gpu_log_info("  MAX MICRO BATCH: {}", MAX_MICRO_BATCH);
#endif // MAX_MICRO_BATCH
#endif // DEBUG_PRINT
}

/**
 * post_chaining_helper: Helper for optional re-chaining with different parameters
 * @param mi minimap2 index
 * @param opt mapping options
 * @param read query read structure
 * @param misc chaining configuration
 * @param km memory pool for allocation
 * 
 * Checks if re-chaining is needed (when chains don't cover all segments or
 * when rescue mode is needed for long reads). If so, re-collects seeds and
 * re-chains with adjusted parameters for better sensitivity.
 * This function is called from plchain.cu after GPU chaining completes.
 */
void post_chaining_helper(const mm_idx_t* mi, const mm_mapopt_t* opt, chain_read_t* read, Misc misc, void* km)
{
	int n_segs = read->n_seg;
	const char* qname = read->seq.name;
	int* rep_len = &read->rep_len;
	int* frag_gap = &read->frag_gap;
	int* qlen_sum = &read->seq.qlen_sum;
	int* n_regs0 = &read->n_u;
	int* n_mini_pos = &read->n_mini_pos;
	uint64_t** mini_pos = &read->mini_pos;
	int64_t* n_a = &read->n;
	uint64_t** u = &read->u;
	mm128_t** a = &read->a;

	int i;
	mm128_v mv = {0, 0, 0};
	int regenerated_mv = 0;

	if (read->minimizers_a && read->minimizers_n > 0) {
		mv.a = read->minimizers_a;
		mv.n = mv.m = (size_t)read->minimizers_n;
	}

	if (opt->bw_long > opt->bw &&
	    (opt->flag & (MM_F_SPLICE | MM_F_SR | MM_F_NO_LJOIN)) == 0 &&
	    n_segs == 1 && *n_regs0 > 1) { // re-chain/long-join for long sequences
		int32_t st = (int32_t)(*a)[0].y, en = (int32_t)(*a)[(int32_t)(*u)[0] - 1].y;
		if (*qlen_sum - (en - st) > opt->rmq_rescue_size || en - st > *qlen_sum * opt->rmq_rescue_ratio) {
			int32_t i;
			for (i = 0, *n_a = 0; i < *n_regs0; ++i) *n_a += (int32_t)(*u)[i];
			kfree(km, *u);
			radix_sort_128x(*a, (*a) + *n_a);
			*a = mg_lchain_rmq(opt->max_gap, opt->rmq_inner_dist, opt->bw_long, opt->max_chain_skip, opt->rmq_size_cap, opt->min_cnt, opt->min_chain_score,
			    misc.chn_pen_gap, misc.chn_pen_skip, *n_a, *a, n_regs0, u, km);
		}
	} else if (opt->max_occ > opt->mid_occ && *rep_len > 0 &&
	    !(opt->flag & MM_F_RMQ)) { // re-chain, mostly for short reads
		int rechain = 0;
		if (*n_regs0 > 0) { // test if the best chain has all the segments
			int n_chained_segs = 1, max = 0, max_i = -1, max_off = -1, off = 0;
			for (i = 0; i < *n_regs0; ++i) { // find the best chain
				if (max < (int)((*u)[i] >> 32)) max = (*u)[i] >> 32, max_i = i, max_off = off;
				off += (uint32_t)(*u)[i];
			}
			for (i = 1; i < (int32_t)(*u)[max_i]; ++i) // count the number of segments in the best chain
				if (((*a)[max_off + i].y & MM_SEED_SEG_MASK) != ((*a)[max_off + i - 1].y & MM_SEED_SEG_MASK))
					++n_chained_segs;
			if (n_chained_segs < n_segs)
				rechain = 1;
		} else
			rechain = 1;
		if (rechain) { // redo chaining with a higher max_occ threshold
			kfree(km, *a);
			kfree(km, *u);
			kfree(km, *mini_pos);
			if (mv.a == NULL) {
				// Fallback for paths that don't cache minimizers from seed stage.
				mm_collect_minimizers(km, mi, n_segs, read->qlens, read->qseqs, opt->sdust_thres, &mv);
				if (opt->q_occ_frac > 0.0f) mm_seed_mz_flt(km, &mv, opt->mid_occ, opt->q_occ_frac);
				regenerated_mv = 1;
			}
			if (opt->flag & MM_F_HEAP_SORT)
				*a = collect_seed_hits_heap(km, opt, opt->max_occ, mi, qname, &mv, *qlen_sum, n_a, rep_len, n_mini_pos, mini_pos);
			else
				*a = collect_seed_hits(km, opt, opt->max_occ, mi, qname, &mv, *qlen_sum, n_a, rep_len, n_mini_pos, mini_pos);
			*a = mg_lchain_dp(misc.max_dist_x, misc.max_dist_y, opt->bw, opt->max_chain_skip, opt->max_chain_iter, opt->min_cnt, opt->min_chain_score,
			    misc.chn_pen_gap, misc.chn_pen_skip, misc.is_cdna, n_segs, *n_a, *a, n_regs0, u, km);
		}
	}
	if (mv.a) {
		kfree(km, mv.a);
		if (!regenerated_mv) {
			read->minimizers_a = NULL;
			read->minimizers_n = 0;
		}
	}
	*frag_gap = misc.max_dist_x;
}

/**
 * worker for launching forward chaining on gpu (streaming)
 * use KMALLOC and kfree for cpu memory management
 * [in/out] in_arr_: ptr to array of reads, updated to a batch launched in previous run 
 *                  (NULL if no finishing batch)
 * [in/out] n_read_: ptr to num of reads in array, updated to a batch launched in previous run
 *                  (NULL if no finishing batch)
*/
void gpu_chain_submit(gpu_chain_ctx_t* ctx, const mm_idx_t* mi, const mm_mapopt_t* opt, chain_read_t** in_arr_, int* n_read_,
    int thread_id, void* km)
{
	if (ctx == NULL || ctx->n_workers == 0) {
		// CPU-only fast path: no GPU chain initialized, nothing to submit.
		if (in_arr_) *in_arr_ = NULL;
		if (n_read_) *n_read_ = 0;
		return;
	}
	// assume only one seg. and qlen_sum desn't matter
	assert(opt->max_frag_len <= 0);
	// GPU chaining uses DP algorithm; RMQ mode requires mg_lchain_rmq which is CPU-only
	assert(!(opt->flag & MM_F_RMQ));
	Misc misc = build_misc(mi, opt, 0, 1);

	// Acquire a GPU worker from the pool BEFORE taking the per-device
	// const-mem lock. Pool acquisition can block waiting for another thread
	// to release a worker; holding the const-mem lock during that wait would
	// needlessly serialize unrelated ctxs on the same device.
	double t0 = gpu_chain_wtime();
	int wid = plmem_pool_acquire(ctx, thread_id);
	double t1 = gpu_chain_wtime();

	plchain_cal_score_async(in_arr_, n_read_, misc, ctx, wid, km);
	double t2 = gpu_chain_wtime();

	if (in_arr_) {
		int n_read = *n_read_;
		chain_read_t* out_arr = *in_arr_;
		for (int i = 0; i < n_read; i++) {
			post_chaining_helper(mi, opt, &out_arr[i], misc, km);
		}
	}
	double t3 = gpu_chain_wtime();

	mm_log_debug("[gpu-chain] submit: pool_wait={:.3f}s cal_score={:.3f}s post_chain={:.3f}s total={:.3f}s",
	    t1 - t0, t2 - t1, t3 - t2, t3 - t0);
}


/**
 * worker for finish all forward chaining kernenls on gpu
 * use KMALLOC and kfree for cpu memory management
 * [out] batches:   array of batches
 * [out] num_reads: array of number of reads in each batch
 */
void gpu_chain_finish(gpu_chain_ctx_t* ctx, const mm_idx_t* mi, const mm_mapopt_t* opt, chain_read_t** reads_,
    int* n_read_, int t, void* km)
{
	if (ctx == NULL || ctx->n_workers == 0) {
		// CPU-only: nothing to flush.
		if (reads_) *reads_ = NULL;
		if (n_read_) *n_read_ = 0;
		return;
	}
	// assume only one seg. and qlen_sum desn't matter
	assert(opt->max_frag_len <= 0);
	// GPU chaining uses DP algorithm; RMQ mode requires mg_lchain_rmq which is CPU-only
	assert(!(opt->flag & MM_F_RMQ));
	Misc misc = build_misc(mi, opt, 0, 1);

	// Look up which worker this thread holds
	int wid = ctx->thread_worker_map[t];
	if (wid < 0 || !ctx->workers[wid].busy) {
		*reads_ = NULL;
		*n_read_ = 0;
		// Release worker if held but not busy
		if (wid >= 0) plmem_pool_release(ctx, t);
		return;
	}

	chain_read_t* reads;
	int n_read = 0;
	hipSetDevice(ctx->workers[wid].device_id);
	cudaStreamSynchronize(ctx->workers[wid].cudastream);
	cudaCheck();

	n_read = plchain_post_gpu_helper(ctx, wid, misc, km);
	reads = ctx->workers[wid].reads;
	ctx->workers[wid].busy = false;

	/* Release the worker now, before the CPU-heavy post_chaining_helper
	 * loop.  'reads' points to the trbuf's launched_batch array which is
	 * thread-local — not owned by the worker — so it remains valid after
	 * release.  The next thread that acquires this worker only touches
	 * host_mems / dev_mem and sets its own w->reads pointer. */
	plmem_pool_release(ctx, t);

	for (int i = 0; i < n_read; i++) {
		post_chaining_helper(mi, opt, &reads[i], misc, km);
	}

	*reads_ = reads;
	*n_read_ = n_read;
}


void gpu_chain_free(gpu_chain_ctx_t* ctx, int n_threads)
{
	(void)n_threads;
	/* NULL-safe: CPU-only runs never call gpu_chain_init(); skip HIP teardown and driver queries. */
	if (ctx == NULL || ctx->n_workers == 0)
		return;
	plmem_stream_cleanup(ctx);
#ifdef DEBUG_PRINT
	{
		size_t gpu_free_mem = 0, gpu_total_mem = 0;
		cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);
		mm_gpu_log_info("GPU free mem: {:.2f} GB, total mem: {:.2f} GB (after cleanup)", (float)gpu_free_mem / OneG, (float)gpu_total_mem / OneG);
	}
#endif
}

#ifdef __cplusplus
extern "C" {
#endif

/* Internal getters used by map.c so the chaining pipeline can size per-thread
 * batches without touching mm_mapopt_t::gpu_chain_*. NULL ctx -> 0 (CPU-only).
 */
extern "C" size_t gpu_chain_ctx_max_anchors(const gpu_chain_ctx_t* ctx)
{
	return ctx ? ctx->max_anchors_per_batch : 0;
}
extern "C" int gpu_chain_ctx_max_reads(const gpu_chain_ctx_t* ctx)
{
	return ctx ? ctx->max_reads_per_batch : 0;
}
extern "C" int gpu_chain_ctx_min_n(const gpu_chain_ctx_t* ctx)
{
	return ctx ? ctx->min_n_anchors : 0;
}
extern "C" size_t gpu_chain_ctx_target_batch_anchors(const gpu_chain_ctx_t* ctx)
{
	// Return auto-derived value, or SIZE_MAX for CPU mode
	return ctx ? ctx->target_batch_anchors : SIZE_MAX;
}
extern "C" int gpu_chain_ctx_n_workers(const gpu_chain_ctx_t* ctx)
{
	return ctx ? ctx->n_workers : 0;
}
extern "C" int gpu_chain_ctx_n_threads(const gpu_chain_ctx_t* ctx)
{
	return ctx ? ctx->n_threads : 0;
}

extern "C" size_t gpu_query_free_vram(void)
{
	size_t free_mem = 0, total_mem = 0;
	if (hipMemGetInfo(&free_mem, &total_mem) != hipSuccess)
		return 0;
	return free_mem;
}

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
