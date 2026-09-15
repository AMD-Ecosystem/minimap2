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

// C++ translation unit for mm_align_ctx lifecycle.
// Separated from align.c so that gpu_device_descriptor.h (HIP/C++ only) can
// be included directly without extern "C" / C-compat wrappers.

#include <cstdlib>
#include <utility>
#include "mm_log.h"
#include "mm2_align.h"
#include "gpu/align_gpu.h"
#ifdef HAVE_GPU_ALIGNMENT
#include "gpu/gpu_device_descriptor.h"
#endif

mm_align_ctx_t* mm_align_ctx_create(const mm_mapopt_t* opt, int n_threads,
    gpu_device_list_t device_list)
{
#ifdef HAVE_GPU_ALIGNMENT
	if (!opt || !(opt->flag & MM_F_GPU_ALIGN)) return nullptr;
	if (n_threads < 1) n_threads = 1;
	/* Device list is the single point of GPU device configuration, set by
	 * mm_mapctx_create at the pipeline entry point. This is an internal
	 * invariant: mm_align_ctx_create is never called directly by end-user
	 * code, only from map_ctx.cpp. Reject empty list to catch configuration
	 * errors early rather than failing obscurely later during GPU work. */
	if (device_list.empty()) {
		mm_log_error("[GPU] mm_align_ctx_create called with no device list");
		return nullptr;
	}
	mm_align_ctx_t* ctx = (mm_align_ctx_t*)calloc(1, sizeof(mm_align_ctx_t));
	if (!ctx) return nullptr;
	ctx->gpu_align = mm_gpu_align_ctx_create(n_threads, opt, std::move(device_list));
	if (!ctx->gpu_align) {
		free(ctx);
		return nullptr;
	}
	return ctx;
#else
	(void)opt;
	(void)n_threads;
	return nullptr;
#endif
}

extern "C" void mm_align_ctx_destroy(mm_align_ctx_t* ctx)
{
	if (!ctx) return;
#ifdef HAVE_GPU_ALIGNMENT
	mm_gpu_align_ctx_destroy(ctx->gpu_align);
#endif
	free(ctx);
}
