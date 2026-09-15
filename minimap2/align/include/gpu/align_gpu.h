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
 * @file align_gpu.h
 * @brief GPU alignment context lifecycle exports.
 *
 * GPU-only header: include only from translation units that are part of the
 * GPU build, or from CPU-side code wrapped in #ifdef HAVE_GPU_ALIGNMENT.
 */

#ifndef ALIGN_GPU_H
#define ALIGN_GPU_H

#include "mm2_align.h" /* mm_gpu_align_ctx_t fwd typedef */
#ifdef __cplusplus
#include "gpu/gpu_device_list_fwd.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Context lifecycle ---- */
void mm_gpu_align_ctx_destroy(mm_gpu_align_ctx_t* ctx);

/** Return the resolved per-thread VRAM budget (auto-derived or user-set). */
int64_t mm_gpu_align_ctx_get_mem_budget(const mm_gpu_align_ctx_t* ctx);

#ifdef __cplusplus
}
/* C++ only: takes gpu_device_list_t by value. */
mm_gpu_align_ctx_t* mm_gpu_align_ctx_create(int n_threads,
    const mm_mapopt_t* opt, gpu_device_list_t device_list = {});
#endif

#endif /* ALIGN_GPU_H */
