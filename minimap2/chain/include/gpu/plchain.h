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

#ifndef _PLCHAIN_H_
#define _PLCHAIN_H_

#include <stddef.h>
#include "minimap.h"
#include "chain_read.h"
#ifdef __cplusplus
#include "gpu_device_list_fwd.h"
#endif

/* GPU stream configuration - all values parsed from a JSON config file.
 * Can also be constructed directly (e.g. in tests) to avoid file I/O. */
typedef struct {
	// batch config
	int min_n;
	size_t max_total_n;
	int max_read;
	size_t long_seg_buffer_size;
	size_t avg_read_n;

	// range kernel config
	int range_blockdim;
	int range_cut_check_anchors;
	int range_anchor_per_block;

	// score kernel config
	int score_micro_batch;
	int score_mid_blockdim;
	int score_short_griddim;
	int score_long_griddim;
	int score_mid_griddim;
	int score_long_seg_cutoff;
	int score_mid_seg_cutoff;

	// Device memory reserved for other GPU subsystems (e.g. GPU alignment).
	// The auto-detect formula subtracts this from the VRAM budget.
	size_t dev_mem_reserve;
} gpu_stream_config_t;

/* Range Kernel configuration */
typedef struct range_kernel_config_t {
	int blockdim;
	int cut_check_anchors;
	int anchor_per_block;
	int max_dist_x;
	int max_iter;
} range_kernel_config_t;

/* Score Generation Kernel configuration */
typedef struct score_kernel_config_t {
	int micro_batch;
	int short_blockdim;
	int long_blockdim;
	int mid_blockdim;
	int short_griddim;
	int long_griddim;
	int mid_griddim;
	int cut_unit;
	int long_seg_cutoff;
	int mid_seg_cutoff;
} score_kernel_config_t;

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque GPU chaining context handle.
 * Real definition lives in plmem.cuh; non-GPU TUs see only the typedef. */
typedef struct mm_gpu_chain_ctx_s gpu_chain_ctx_t;

void mm_gpu_chain_ctx_destroy(gpu_chain_ctx_t* ctx);

/* Low-level alloc/dealloc for the opaque struct (used inside
 * mm_gpu_chain_ctx_create; exposed for tests that build a ctx manually). */
gpu_chain_ctx_t* gpu_chain_alloc(void);
void gpu_chain_dealloc(gpu_chain_ctx_t* ctx);

/* Configuration: parse_gpu_config parses a JSON file into sparse overrides
 * (zero-valued fields are filled with hardware-derived defaults at init).
 * Pass NULL or "" to get an all-zero config (pure auto-detection). */
gpu_stream_config_t parse_gpu_config(const char* gpu_config_file);

void gpu_chain_free(gpu_chain_ctx_t* ctx, int n_threads);

/* Chaining entry points. */
void gpu_chain_submit(gpu_chain_ctx_t* ctx, const mm_idx_t* mi,
    const mm_mapopt_t* opt, chain_read_t** in_arr_ptr,
    int* n_read_ptr, int thread_id, void* km);
void gpu_chain_finish(gpu_chain_ctx_t* ctx, const mm_idx_t* mi,
    const mm_mapopt_t* opt, chain_read_t** batches,
    int* num_reads, int num_batch, void* km);

/* Getters -- NULL ctx returns 0 for integer getters, SIZE_MAX for size_t getters (CPU-only fast path). */
size_t gpu_chain_ctx_max_anchors(const gpu_chain_ctx_t* ctx);
int gpu_chain_ctx_max_reads(const gpu_chain_ctx_t* ctx);
int gpu_chain_ctx_min_n(const gpu_chain_ctx_t* ctx);
size_t gpu_chain_ctx_target_batch_anchors(const gpu_chain_ctx_t* ctx); // Returns SIZE_MAX on NULL
int gpu_chain_ctx_n_workers(const gpu_chain_ctx_t* ctx);
int gpu_chain_ctx_n_threads(const gpu_chain_ctx_t* ctx);
int gpu_chain_ctx_worker_device_id(const gpu_chain_ctx_t* ctx, int worker_id);

/* Query current free VRAM (bytes). Returns 0 on failure. */
size_t gpu_query_free_vram(void);

#ifdef __cplusplus
}
/* C++ only: take gpu_device_list_t by value. */
void gpu_chain_init(gpu_chain_ctx_t* ctx, const gpu_stream_config_t* config,
	int n_threads, int n_gpu_workers, gpu_device_list_t device_list = {});
gpu_chain_ctx_t* mm_gpu_chain_ctx_create(const mm_mapopt_t* opt, int n_threads,
	gpu_device_list_t device_list = {});
#endif

#endif // _PLCHAIN_H_
