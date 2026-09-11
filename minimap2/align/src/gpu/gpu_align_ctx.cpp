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
 * gpu_align_ctx.cpp
 *
 * GPU alignment context manager. Owns a shared pool of gpu_align_accum_t
 * objects (cross-read accumulator path) and the device hipMemPool.
 *
 * The accumulator path (gpu_accum.c) handles all GPU alignment work:
 * each thread accumulates reads, then flushes them through its own
 * gpu_batch_t.  The legacy worker-pool and multialign-batch paths
 * have been removed.
 */

#include <hip/hip_runtime.h>

#include <new>
#include <vector>
#include <mutex>
#include <condition_variable>

#include "gpu/gpu_align_ctx.h"
#include "gpu/gpu_device_descriptor.h"
#include "mm2_align.h"
#include "mm_log.h"
#include "option.h"

static inline bool hip_ok_impl(hipError_t err, const char* file, int line, const char* func)
{
	if (err == hipSuccess) return true;
	mm_log_error("[GPU align] HIP error {} ({}) at {}:{}:{}", (int)err,
	    hipGetErrorString(err), file, line, func);
	return false;
}
#define HIP_OK(x) hip_ok_impl((x), __FILE__, __LINE__, __FUNCTION__)

struct mm_gpu_align_ctx_s {
	struct accum_slot_t {
		gpu_align_accum_t* accum;
		std::shared_ptr<GPU_device> device;
		bool available;

		accum_slot_t() : accum(nullptr), device(), available(true) {}
	};

	/* Shared accumulator pool */
	std::vector<accum_slot_t> pool;
	std::mutex pool_mtx;
	std::condition_variable pool_cv;

	int64_t resolved_batch_max_mem;
	int resolved_batch_max_align;
	int resolved_flush_threshold;

	gpu_device_list_t device_list;

	explicit mm_gpu_align_ctx_s(int n_threads, const mm_mapopt_t* opt,
	    gpu_device_list_t selected_devices)
	    : device_list(std::move(selected_devices))
	{
		int device_id = 0;
		if (!device_list.empty())
			device_id = device_list[0]->device_id;
		HIP_OK(hipSetDevice(device_id));
		derive_pool_config(opt, n_threads, device_id);
	}

	~mm_gpu_align_ctx_s()
	{
		mm_log_debug("[GPU align] destroying ctx: pool_size={}", (int)pool.size());
		for (size_t i = 0; i < pool.size(); i++) {
			if (pool[i].accum) {
				mm_log_debug("[GPU align] destroying pool accum[{}]", i);
				mm_gpu_accum_destroy(pool[i].accum);
			}
		}
		mm_log_debug("[GPU align] all pool accums destroyed, calling hipDeviceSynchronize");
		/* Ensure all GPU work is complete before destroying context.
		 * Set to primary device to be explicit about which device is synchronized. */
		if (!device_list.empty())
			HIP_OK(hipSetDevice(device_list[0]->device_id));
		HIP_OK(hipDeviceSynchronize());
		mm_log_debug("[GPU align] ctx destroyed");
	}

      private:
	void derive_pool_config(const mm_mapopt_t* opt, int n_threads, int device_id)
	{
		resolved_batch_max_mem = opt->gpu_batch_max_mem;
		resolved_batch_max_align = opt->gpu_batch_max_align;
		resolved_flush_threshold = opt->gpu_flush_threshold;

		/* Query device properties */
		hipDeviceProp_t prop;
		int n_cus = 48;
		size_t total_vram = 0;
		if (hipGetDeviceProperties(&prop, device_id) == hipSuccess) {
			n_cus = prop.multiProcessorCount;
			total_vram = prop.totalGlobalMem;
		} else {
			mm_log_warn("[GPU align] hipGetDeviceProperties failed; "
				    "using compile-time defaults");
		}

		size_t free_mem = 0, total_mem = 0;
		if (hipMemGetInfo(&free_mem, &total_mem) != hipSuccess)
			free_mem = total_vram;
		size_t usable = (free_mem > 0) ? free_mem : total_vram;
		if (usable == 0)
			usable = MM_GPU_BATCH_DEFAULT_MEM * n_threads;

		int xnack_on = 0;
		(void)hipDeviceGetAttribute(&xnack_on,
		    hipDeviceAttributePageableMemoryAccess, device_id);
		double util_frac = xnack_on ? MM_GPU_VRAM_UTIL_FRAC_XNACK
					    : MM_GPU_VRAM_UTIL_FRAC;
		int64_t total_budget = (int64_t)(usable * util_frac);
		int n_devices = device_list.empty() ? 1 : (int)device_list.size();

		/* --- Accumulators per GPU / total slot count --- */
		int accums_per_gpu = opt->gpu_accum_pool_size;
		if (accums_per_gpu <= 0) {
			int auto_size = (int)(total_budget / MM_GPU_VRAM_BUDGET_FLOOR);
			if (auto_size < 2) auto_size = 2;
			if (auto_size > n_threads) auto_size = n_threads;
			accums_per_gpu = auto_size;
		}
		if (accums_per_gpu > n_threads) accums_per_gpu = n_threads;

		int pool_size = accums_per_gpu * n_devices;

		/* --- Per-accum memory budget --- */
		if (resolved_batch_max_mem <= 0) {
			int64_t mem_cap = opt->gpu_batch_max_mem_cap > 0
			    ? opt->gpu_batch_max_mem_cap
			    : MM_GPU_BATCH_MAX_MEM_CAP;
			int64_t per_accum = total_budget / pool_size;
			if (per_accum > mem_cap)
				per_accum = mem_cap;
			resolved_batch_max_mem = per_accum;
		}

		/* --- Max alignments per batch --- */
		if (resolved_batch_max_align <= 0) {
			int target = n_cus * 4;
			resolved_batch_max_align = (target > 16384) ? target : 16384;
			if (resolved_batch_max_mem < MM_GPU_VRAM_BUDGET_FLOOR) {
				double scale = (double)resolved_batch_max_mem / MM_GPU_VRAM_BUDGET_FLOOR;
				resolved_batch_max_align = (int)(resolved_batch_max_align * scale);
				if (resolved_batch_max_align < 1024)
					resolved_batch_max_align = 1024;
			}
		}

		/* --- Flush threshold --- */
		if (resolved_flush_threshold <= 0) {
			if (resolved_batch_max_mem < MM_GPU_VRAM_BUDGET_FLOOR)
				resolved_flush_threshold = resolved_batch_max_align / 4;
			else
				resolved_flush_threshold = resolved_batch_max_align / 2;
			if (resolved_flush_threshold < 512)
				resolved_flush_threshold = 512;
		}

		/* Initialize pool vectors */
		pool.resize(pool_size);
		if (!device_list.empty()) {
			for (int i = 0; i < pool_size; i++)
				pool[i].device = device_list[i % device_list.size()];
		}

		/* Single consolidated log */
		mm_log_debug("[GPU align] pool_size={}, per_accum={}MB, "
			     "max_align={}, flush={}, aggregate={:.1f}GB "
			     "(per_gpu={}, devices={}, VRAM={:.1f}GB free={:.1f}GB CUs={} threads={}{})",
		    pool_size,
		    (long long)(resolved_batch_max_mem / (1024 * 1024)),
		    resolved_batch_max_align,
		    resolved_flush_threshold,
		    (double)resolved_batch_max_mem * pool_size / (1024.0 * 1024 * 1024),
		    accums_per_gpu,
		    n_devices,
		    total_mem / (1024.0 * 1024 * 1024),
		    free_mem / (1024.0 * 1024 * 1024),
		    n_cus, n_threads,
		    xnack_on ? " XNACK" : "");

		if (resolved_batch_max_mem < MM_GPU_VRAM_BUDGET_FLOOR)
			mm_log_warn("[GPU align] per-accum budget {}MB < floor {}MB; "
				    "batches will be smaller",
			    (long long)(resolved_batch_max_mem / (1024 * 1024)),
			    (long long)(MM_GPU_VRAM_BUDGET_FLOOR / (1024 * 1024)));
	}
};

extern "C" {

int mm_gpu_check_available(void)
{
	int count = 0;
	hipError_t err = hipGetDeviceCount(&count);
	return (err == hipSuccess && count > 0) ? 1 : 0;
}

}

mm_gpu_align_ctx_t* mm_gpu_align_ctx_create(int n_threads,
    const mm_mapopt_t* opt, gpu_device_list_t device_list)
{
	if (n_threads < 1) n_threads = 1;
	mm_gpu_align_ctx_t* ctx = new (std::nothrow) mm_gpu_align_ctx_t(
	    n_threads, opt, std::move(device_list));
	return ctx;
}

extern "C" void mm_gpu_align_ctx_destroy(mm_gpu_align_ctx_t* ctx)
{
	delete ctx;
}

gpu_align_accum_t* mm_gpu_align_ctx_acquire_accum(mm_gpu_align_ctx_t* ctx,
    const mm_mapopt_t* opt, const mm_idx_t* mi)
{
	if (!ctx) return NULL;
	std::unique_lock<std::mutex> lock(ctx->pool_mtx);
	ctx->pool_cv.wait(lock, [&] {
		for (size_t i = 0; i < ctx->pool.size(); i++)
			if (ctx->pool[i].available) return true;
		return false;
	});
	for (size_t i = 0; i < ctx->pool.size(); i++) {
		if (ctx->pool[i].available) {
			ctx->pool[i].available = false;
			if (!ctx->pool[i].accum) {
				auto device = ctx->pool[i].device;
				if (!device) {
					ctx->pool[i].available = true;
					ctx->pool_cv.notify_one();
					return NULL;
				}
				HIP_OK(hipSetDevice(device->device_id));
				ctx->pool[i].accum = mm_gpu_accum_init(opt, mi, device->mempool,
				    ctx->resolved_batch_max_mem,
				    ctx->resolved_batch_max_align,
				    ctx->resolved_flush_threshold,
				    device->device_id);
				if (!ctx->pool[i].accum) {
					/* Lazy init failed (e.g. OOM). Release the
					 * slot so it is not permanently lost, which
					 * would eventually deadlock acquire(). */
					ctx->pool[i].available = true;
					ctx->pool_cv.notify_one();
					return NULL;
				}
			}
			return ctx->pool[i].accum;
		}
	}
	return NULL; /* unreachable */
}

void mm_gpu_align_ctx_release_accum(mm_gpu_align_ctx_t* ctx,
    gpu_align_accum_t* accum)
{
	if (!ctx || !accum) return;
	std::lock_guard<std::mutex> lock(ctx->pool_mtx);
	for (size_t i = 0; i < ctx->pool.size(); i++) {
		if (ctx->pool[i].accum == accum) {
			ctx->pool[i].available = true;
			ctx->pool_cv.notify_one();
			return;
		}
	}
}

void mm_gpu_align_ctx_flush_all(mm_gpu_align_ctx_t* ctx,
    mm_stage_timer_t* t_align, const mm_core_ctx_t* core_ctx)
{
	if (!ctx) return;
	/* Acquire all pool accums, flush any with pending work, release */
	for (size_t i = 0; i < ctx->pool.size(); i++) {
		std::unique_lock<std::mutex> lock(ctx->pool_mtx);
		ctx->pool_cv.wait(lock, [&] { return ctx->pool[i].available; });
		ctx->pool[i].available = false;
		lock.unlock();

		/* Drain unconditionally: any accumulator with pending reads must be
		 * flushed before its originating step_t is freed. Gating this on
		 * mm_gpu_accum_should_flush() would leave sub-threshold reads pending,
		 * whose out_regs point into a step_t that is freed at end of step,
		 * causing a use-after-free on the next flush. mm_gpu_accum_flush() is
		 * a no-op when there is no pending work. */
		if (ctx->pool[i].accum)
			mm_gpu_accum_flush(ctx->pool[i].accum, t_align, core_ctx);

		std::lock_guard<std::mutex> rel(ctx->pool_mtx);
		ctx->pool[i].available = true;
		ctx->pool_cv.notify_one();
	}
}

extern "C" int64_t mm_gpu_align_ctx_get_mem_budget(const mm_gpu_align_ctx_t* ctx)
{
	if (!ctx) return MM_GPU_BATCH_DEFAULT_MEM;
	return ctx->resolved_batch_max_mem;
}

extern "C" int mm_gpu_align_ctx_pool_size(const mm_gpu_align_ctx_t* ctx)
{
	return ctx ? (int)ctx->pool.size() : 0;
}

extern "C" int mm_gpu_align_ctx_pool_device_id(const mm_gpu_align_ctx_t* ctx, int slot)
{
	if (!ctx || slot < 0 || slot >= (int)ctx->pool.size()) return -1;
	return ctx->pool[slot].device ? ctx->pool[slot].device->device_id : -1;
}
