/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * chain_ctx.cpp
 *
 * Public mm_gpu_chain_ctx_t lifecycle. Lives in the chain library because
 * the implementation depends only on chain-internal helpers
 * (gpu_chain_alloc / _init / _free / _dealloc and parse_gpu_config from
 * plchain.h / plchain.cu); it does not touch anything in minimap2/src.
 *
 * map.c keeps the public ABI exposed via minimap.h (the non-_ctx wrappers
 * call into here), but does not own the chain ctx lifecycle. When a future
 * minimap2-wide run context replaces this, the entry points here will be
 * updated to either delegate to the new parent ctx or be retired in favour
 * of it; map.c will not need to change.
 *
 * On CPU-only builds (no MM_ENABLE_HIP) the create/destroy entry points
 * compile to NULL/no-op stubs so external callers (main.c, mm2_chain.cpp,
 * library users) link cleanly without needing to special-case GPU support.
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "minimap.h"
#include "mm_log.h"
#include "option.h"
#include "chain_priv.h"
#include "gpu/plchain.h"
#ifdef MM_ENABLE_HIP
#include "gpu/gpu_device_descriptor.h"
#endif

mm_gpu_chain_ctx_t* mm_gpu_chain_ctx_create(const mm_mapopt_t* opt, int n_threads,
    gpu_device_list_t device_list)
{
	if (opt == NULL) return NULL;
	if (!(opt->flag & MM_F_GPU_CHAIN)) return NULL;
	/* Match mm_map_file_frag_ctx's pipeline-width clamp so the worker pool
	 * (sized to n_threads) cannot end up smaller than the threads that will
	 * call into it. */
	if (n_threads < 1) n_threads = 1;
#ifdef MM_ENABLE_HIP
	if (device_list.empty()) {
		mm_log_error("[GPU] mm_gpu_chain_ctx_create called with no device list");
		return NULL;
	}

	/* opt->gpu_config_file is optional. When NULL or empty, all config
	 * values are auto-derived from GPU hardware properties at init time
	 * (no JSON file needed). When a path is provided, the JSON fields
	 * override the auto-derived defaults. */
	const char* cfg_path = opt->gpu_config_file;
	int has_config_file = (cfg_path != NULL && cfg_path[0] != '\0');

	if (has_config_file) {
		FILE* probe = fopen(cfg_path, "rb");
		if (probe == NULL) {
			mm_log_warn("mm_gpu_chain_ctx_create: cannot open --gpu-cfg "
				    "'%s'; ignoring config file, using auto-detected defaults",
			    cfg_path);
			has_config_file = 0;
		} else {
			fclose(probe);
		}
	}

	/* Heap-allocate via gpu_chain_alloc; the struct (plmem.cuh) pulls in
	 * cudaDeviceProp / pthread types and is intentionally opaque to
	 * non-chain TUs. */
	gpu_chain_ctx_t* ctx = gpu_chain_alloc();
	if (!ctx) return NULL;

	gpu_stream_config_t gpu_config = parse_gpu_config(has_config_file ? cfg_path : NULL);

	/* When GPU alignment is also enabled, reserve its VRAM budget so the
	 * chaining auto-detect does not over-allocate device memory.
	 * gpu_batch_max_mem==0 means alignment will auto-derive; predict its
	 * budget using the same formula (option.h constants). */
	if (opt->flag & MM_F_GPU_ALIGN) {
		size_t align_per_thread = (size_t)opt->gpu_batch_max_mem;
		if (align_per_thread == 0) {
			size_t free_vram = gpu_query_free_vram();
			if (free_vram > 0) {
				int64_t budget = (int64_t)(free_vram * MM_GPU_VRAM_UTIL_FRAC / n_threads);
				if (budget < MM_GPU_VRAM_BUDGET_FLOOR)
					budget = MM_GPU_VRAM_BUDGET_FLOOR;
				int64_t mem_cap = opt->gpu_batch_max_mem_cap > 0
				    ? opt->gpu_batch_max_mem_cap
				    : MM_GPU_BATCH_MAX_MEM_CAP;
				if (budget > mem_cap)
					budget = mem_cap;
				align_per_thread = (size_t)budget;
			} else {
				align_per_thread = (size_t)MM_GPU_VRAM_BUDGET_FLOOR;
			}
		}
		gpu_config.dev_mem_reserve = align_per_thread * n_threads;
	}

	/* Misc is uploaded lazily on the first gpu_chain_submit (it depends on
	 * mi + opt and is rebuilt per-batch); init does not need it any more. */
	gpu_chain_init(ctx, &gpu_config, n_threads, opt->gpu_chain_workers, std::move(device_list));
	return ctx;
#else
	(void)n_threads;
	return NULL;
#endif
}

void mm_gpu_chain_ctx_destroy(mm_gpu_chain_ctx_t* ctx)
{
	if (ctx == NULL) return;
#ifdef MM_ENABLE_HIP
	gpu_chain_free(ctx, /*n_threads=*/0);
	gpu_chain_dealloc(ctx);
#else
	(void)ctx;
#endif
}

#ifndef MM_ENABLE_HIP
/* CPU-only stubs for the chain ctx getters declared in plchain.h.
 *
 * On a CPU-only build the chain library's GPU sources (plchain.cu, where the
 * real getters live) are not compiled, so callers in map.c (which calls
 * gpu_chain_ctx_min_n unconditionally to size per-thread batches) would see
 * an unresolved-symbol link error. The ctx pointer is also always NULL on
 * CPU-only builds (mm_gpu_chain_ctx_create returns NULL), so 0 is the only
 * meaningful value here. */
size_t gpu_chain_ctx_max_anchors(const gpu_chain_ctx_t* ctx)
{
	(void)ctx;
	return 0;
}
int gpu_chain_ctx_max_reads(const gpu_chain_ctx_t* ctx)
{
	(void)ctx;
	return 0;
}
int gpu_chain_ctx_min_n(const gpu_chain_ctx_t* ctx)
{
	(void)ctx;
	return 0;
}
size_t gpu_chain_ctx_target_batch_anchors(const gpu_chain_ctx_t* ctx)
{
	(void)ctx;
	return SIZE_MAX;
}
int gpu_chain_ctx_n_workers(const gpu_chain_ctx_t* ctx)
{
	(void)ctx;
	return 0;
}
int gpu_chain_ctx_n_threads(const gpu_chain_ctx_t* ctx)
{
	(void)ctx;
	return 0;
}
#endif

mm_chain_ctx_t* mm_chain_ctx_create(const mm_mapopt_t* opt, int n_threads,
    gpu_device_list_t device_list)
{
	mm_chain_ctx_t* ctx = (mm_chain_ctx_t*)malloc(sizeof(mm_chain_ctx_t));
	if (!ctx) return NULL;
	ctx->gpu_chain = mm_gpu_chain_ctx_create(opt, n_threads, std::move(device_list));
	return ctx;
}

void mm_chain_ctx_destroy(mm_chain_ctx_t* ctx)
{
	if (!ctx) return;
	mm_gpu_chain_ctx_destroy(ctx->gpu_chain);
	free(ctx);
}
