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

#ifndef SEED_GPU_H
#define SEED_GPU_H

#include "minimap.h" /* for mm128_t */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Copy minimizers to array with index tracking (GPU version)
 * Equivalent to: for (i = 0; i < n; ++i) { a[i].x = mv_a[i].x; a[i].y = i; }
 * 
 * @param a      Output array (host memory, pre-allocated)
 * @param mv_a   Input minimizer array (host memory)
 * @param n      Number of elements
 */
void mm_copy_mz_gpu(mm128_t* a, const mm128_t* mv_a, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* SEED_GPU_H */
