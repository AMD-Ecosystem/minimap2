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

// C++ translation unit for mm_mapctx lifecycle functions.
// Kept separate from map.c so that gpu_device_descriptor.h (C++ only) can be
// included directly without any extern "C" / C-compat wrappers.

#include <cstdlib>
#include <utility>
#include "map_priv.h"
#include "mm_log.h"
#include "core_ctx.h"
#ifdef MM_ENABLE_HIP
#include "gpu/gpu_device_descriptor.h"
#endif

namespace
{
// Scoped thread-local logger binding: sets the calling thread's logger for the
// duration of a context lifecycle/mapping call and restores the previous one on
// exit. This lets per-context logging (e.g. GPU device-list discovery) use an
// owned, live logger without permanently changing the caller's (e.g. the main
// thread's) logger. See AIOSS-4548.
struct ScopedTlLogger {
	mm_logger_t* prev;
	explicit ScopedTlLogger(mm_logger_t* logger) : prev(mm_tl_logger())
	{
		if (logger) mm_set_tl_logger(logger);
	}
	~ScopedTlLogger() { mm_set_tl_logger(prev); }
	ScopedTlLogger(const ScopedTlLogger&) = delete;
	ScopedTlLogger& operator=(const ScopedTlLogger&) = delete;
};
} // namespace

extern "C" {

mm_mapctx_t* mm_mapctx_create(mm_logger_t* logger, const mm_mapopt_t* opt, int n_threads)
{
	ScopedTlLogger tl_guard(logger);
	mm_mapctx_t* ctx = (mm_mapctx_t*)calloc(1, sizeof(mm_mapctx_t));
	if (!ctx) return nullptr;
	ctx->core_ctx = mm_core_ctx_create(logger);

	gpu_device_list_t device_list;
#ifdef MM_ENABLE_HIP
	if (opt->flag & (MM_F_GPU_CHAIN | MM_F_GPU_ALIGN)) {
		device_list = mm_gpu_create_device_list(opt->gpu_device_ids, opt->gpu_num_devices);
		if (device_list.empty()) {
			mm_log_error("[GPU] Failed to create device list");
			mm_core_ctx_destroy(ctx->core_ctx);
			free(ctx);
			return nullptr;
		}
	}
#endif

	ctx->chain_ctx = mm_chain_ctx_create(opt, n_threads, device_list); // copy
	ctx->align_ctx = mm_align_ctx_create(opt, n_threads, std::move(device_list));
	if ((opt->flag & MM_F_GPU_ALIGN) && ctx->align_ctx == nullptr) {
		mm_log_warn("GPU alignment context could not be created; falling back to CPU alignment");
		((mm_mapopt_t*)opt)->flag &= ~MM_F_GPU_ALIGN;
	}
	ctx->timer_stats = (mm_timer_stats_t*)calloc(1, sizeof(mm_timer_stats_t));
	if (ctx->timer_stats) mm_timer_stats_init(ctx->timer_stats);
	return ctx;
}

void mm_mapctx_destroy(mm_mapctx_t* ctx)
{
	if (!ctx) return;
	mm_align_ctx_destroy(ctx->align_ctx);
	mm_chain_ctx_destroy(ctx->chain_ctx);
	mm_core_ctx_destroy(ctx->core_ctx);
	free(ctx->timer_stats);
	free(ctx);
}

void mm_mapctx_reset(mm_mapctx_t* ctx)
{
	if (!ctx) return;
	mm_core_ctx_reset(ctx->core_ctx);
	mm_timer_stats_init(ctx->timer_stats);
}

} // extern "C"
