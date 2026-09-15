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

#ifndef _PLMEM_CUH_
#define _PLMEM_CUH_
#include <pthread.h>
#include "hipify.cuh"
#include "plchain.h"
#include "plutils.h"

#ifndef MAX_MICRO_BATCH
#define MAX_MICRO_BATCH 8
#endif // MAX_MICRO_BATCH

#define OneK 1024
#define OneM (OneK * 1024)
#define OneG (OneM * 1024)


typedef struct {
	void* pinned_base; // consolidated pinned allocation (NULL if legacy per-field)
	int index; // read index / batch index
	int griddim; // grid for range selection kernel.
	int size; // number of reads in the batch
	size_t total_n; // number of anchors in the batch
	size_t cut_num; // number of cuts in the batch

	// outputs (array size: number of anchors in the batch)
	int32_t* f; // score
	uint16_t* p; // predecessor

	// array size: number of cuts in the batch / long_seg_cut
	// total long segs number till this batch
	unsigned int* long_segs_num;

	// start index for each block in range selection
	/***** range selection block assiagnment
     * One block only gets assgined one read or part of one read.
     *  start_idx:      idx of the first anchor assigned to each block
     *  read_end_idx:   idx of the last anchor OF THE READ assigned to each
     * block if a read is devided into several blocks, all the blocks take the
     * last anchor index of the read cut_start_idx:  idx of the first cut this
     * block needs to make
     */
	// array size: grid dimension
	size_t* start_idx;
	size_t* read_end_idx;
	size_t* cut_start_idx;
} hostMemPtr;

typedef struct {
	void* pinned_base; // consolidated pinned allocation (NULL if legacy per-field)
	// array size: number of cuts in the batch / long_seg_cut
	seg_t* long_segs_og_idx; // start & end idx of long segs in the original micro batch
	seg_t* long_segs_buf_idx; // start & end idx of long segs in the long seg buffer
	unsigned int* total_long_segs_num; // sum of mini batch long_segs_num
	size_t* total_long_segs_n; // number of anchors in all the long segs
	int32_t* f_long; // score for long segs
	uint16_t* p_long; // predecessor for long segs
} longMemPtr;

typedef struct {
	void* dev_base; // consolidated device allocation (NULL if legacy per-field / mempool)
	int size;
	int griddim;
	size_t total_n;
	size_t num_cut;
	// device memory ptrs
	uint64_t* d_anchors_raw; // raw (x,y) pairs, 2 entries per anchor
	// SoA arrays extracted on device from d_anchors_raw
	int32_t* d_ax;
	int32_t* d_ay;
	int8_t* d_sid; // a[].y >> 48 & 0xff (segment ID)
	uint8_t* d_qspan; // a[].y >> 32 & 0xff (q_span for scoring)
	int32_t* d_xrev; // a[].x >> 32
	int32_t* d_range;
	int32_t* d_f; // score
	uint16_t* d_p; // predecessor

	// range selection index
	size_t* d_start_idx;
	size_t* d_read_end_idx;
	size_t* d_cut_start_idx;

	// cut
	size_t* d_cut; // cut
	unsigned int* d_long_seg_count; // total number of long seg (aggregated accross micro batches)
	seg_t* d_long_seg; // start & end idx of long segs in the long seg buffer (aggregated across micro batches)
	seg_t* d_long_seg_og; // start & end idx of long seg in the micro batch. (aggregated accross micro batches)
	unsigned int* d_mid_seg_count; // private to micro batch
	seg_t* d_mid_seg; // private to micro batch

	// long segement buffer
	unsigned* d_map;
	// Per-worker work-stealing counter for score_generation_long_map.
	// Was a __device__ global (curr_long_segid in plscore.cu) which raced
	// across concurrent runs on the same physical device. Now per-worker so
	// concurrent ctxs do not collide. Initialised to 0 via cudaMemsetAsync
	// on the worker stream before each long-kernel launch; the kernel adds
	// gridDim.x to atomic results to recover the original "start at gridDim.x"
	// dispatch range.
	unsigned* d_long_segid;
	int32_t *d_ax_long, *d_ay_long;
	int8_t* d_sid_long;
	uint8_t* d_qspan_long;
	int32_t* d_range_long;
	int32_t* d_xrev_long; // rid/strand prefix for long segs (pull-model window)
	size_t* d_total_n_long;
	size_t buffer_size_long;
	int32_t* d_f_long; // score, size: buffer_size_long * sizeof(int32_t)
	uint16_t* d_p_long; // predecessor, size: buffer_size_long * sizeof(uint16_t)
} deviceMemPtr;

struct mm_gpu_chain_ctx_s;
typedef struct gpu_worker_t {
	// Back-pointer to the owning ctx so worker-taking helpers (memcpy / analyze)
	// can read ctx->score_config / ctx->range_config without each helper having
	// to take a separate ctx parameter. Set at plmem_stream_initialize time;
	// intra-ctx only (does not violate the "no back-pointer" rule that applies
	// to gpu_chain_ctx_t -> future parent mm_run_ctx_t).
	const struct mm_gpu_chain_ctx_s* parent_ctx;

	chain_read_t* reads;
	size_t n_read;
	hostMemPtr host_mems[MAX_MICRO_BATCH];
	longMemPtr long_mem;
	deviceMemPtr dev_mem;
	cudaStream_t cudastream;
	bool busy = false;
	size_t max_long_segs; // capacity of long-seg device sort buffers

	int device_id; // HIP device id this worker is assigned to (multi-GPU support)
	cudaDeviceProp device_prop;
	hipMemPool_t device_mempool;

	uint64_t* staging_raw;
} gpu_worker_t;

typedef struct mm_gpu_chain_ctx_s {
	int n_workers;
	gpu_worker_t* workers;
	size_t max_anchors_per_worker, max_num_cut, long_seg_buffer_size;
	int max_range_grid;

	// Worker pool: n_threads threads share n_workers GPU workers.
	// Uses an O(1) free-stack instead of O(n_workers × n_threads) scan.
	int n_threads;
	int* thread_worker_map; // [n_threads] tid -> wid, -1 = no worker assigned
	int* free_stack; // [n_workers] stack of available worker indices
	int free_top; // index of next free slot (0 = empty)
	pthread_mutex_t pool_mutex;
	pthread_cond_t pool_cond;

	// Kernel launch configs (formerly file-scope globals score_kernel_config /
	// range_kernel_config in plscore.cu / plrange.cu). Populated once at init
	// from the gpu_stream_config_t and the cached cudaDeviceProp.
	score_kernel_config_t score_config;
	range_kernel_config_t range_config;

	// Per-batch sizing previously written back into mm_mapopt_t::gpu_chain_*.
	// gpu_chain_init no longer mutates mm_mapopt_t; map.c reads via getters.
	size_t max_anchors_per_batch;
	int max_reads_per_batch;
	int min_n_anchors;
	size_t target_batch_anchors; // Auto-derived optimal batch size for accumulation

	// Cached gpu_stream_config_t (parsed once at init / create); avoids
	// re-parsing the JSON config file per index part.
	gpu_stream_config_t stream_config;

	// Multi-GPU device list: vector<shared_ptr<GPU_device>>* cast to void*
	// Passed in at init time; worker devices and device-local pools are
	// derived from this list.
	gpu_device_list_t device_list;
} gpu_chain_ctx_t;

/* memory management methods */
// Hardware-derived default config (no JSON file needed)
gpu_stream_config_t plmem_config_defaults(const cudaDeviceProp* prop);

// configuration parsing (file path -> config struct, sparse overrides)
gpu_stream_config_t plmem_parse_stream_config(const char* gpu_config_file);

// initialization and cleanup
//
// plmem_stream_initialize: max_total_n / max_read / min_n out-params are
// optional (pass NULL if uninterested) and exist solely to let tests inspect
// the post-derivation batch sizing without going through the ctx getters.
// The same values are written to ctx->max_anchors_per_batch / max_reads_per_batch
// / min_n_anchors regardless.
void plmem_stream_initialize(gpu_chain_ctx_t* ctx, size_t* max_total_n, int* max_read, int* min_n, const gpu_stream_config_t* config, int n_threads, int n_gpu_workers, gpu_device_list_t device_list);
void plmem_stream_cleanup(gpu_chain_ctx_t* ctx);

// Worker pool acquire/release
int plmem_pool_acquire(gpu_chain_ctx_t* ctx, int tid);
void plmem_pool_release(gpu_chain_ctx_t* ctx, int tid);

// alloc and free
//
// Helpers that size buffers from score_kernel_config / range_kernel_config
// now read those values via ctx->score_config / ctx->range_config.
void plmem_malloc_host_mem(hostMemPtr* host_mem, size_t anchor_per_batch,
    int range_grid_size);
void plmem_malloc_long_mem(const gpu_chain_ctx_t* ctx, longMemPtr* long_mem, size_t buffer_size_long);
void plmem_free_host_mem(hostMemPtr* host_mem);
void plmem_free_long_mem(longMemPtr* long_mem);
void plmem_malloc_device_mem(const gpu_chain_ctx_t* ctx, hipMemPool_t mempool, deviceMemPtr* dev_mem, size_t anchor_per_batch,
    int range_grid_size, int num_cut, cudaStream_t stream);
void plmem_free_device_mem(deviceMemPtr* dev_mem, cudaStream_t stream);

// data movement
void plmem_reorg_input_arr(const gpu_chain_ctx_t* ctx,
    chain_read_t* reads, int n_read,
    hostMemPtr* host_mem, uint64_t* staging_raw);
void plmem_async_h2d_short_memcpy(gpu_worker_t* worker, size_t uid);
void plmem_async_soa_extract(gpu_worker_t* worker, size_t uid);
void plmem_async_d2h_short_memcpy(gpu_worker_t* worker, size_t uid);
void plmem_async_d2h_long_memcpy(gpu_worker_t* worker);
void plmem_async_sort_long_segs(gpu_worker_t* worker, unsigned int num_long_seg);
#endif // _PLMEM_CUH_