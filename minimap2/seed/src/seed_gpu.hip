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

#include <hip/hip_runtime.h>
#include <stdio.h>
#include "seed_gpu.h"

/**
 * GPU kernel: copy minimizer x values and store original indices
 */
__global__ void copy_mz_kernel(mm128_t* da, const mm128_t* d_mv_a, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        da[i].x = d_mv_a[i].x;
        da[i].y = i;
    }
}

/**
 * Host wrapper: copy minimizers with GPU acceleration
 */
void mm_copy_mz_gpu(mm128_t* a, const mm128_t* mv_a, size_t n)
{
    //printf("in seed copy gpu kernel");
    mm128_t *da = nullptr;
    mm128_t *d_mv_a = nullptr;
    hipError_t err = hipSuccess;

    // Allocate device memory
    err = hipMalloc(&da, n * sizeof(mm128_t));
    if (err != hipSuccess) {
        fprintf(stderr, "[mm_copy_mz_gpu] hipMalloc da failed: %s\n", hipGetErrorString(err));
        return;
    }

    err = hipMalloc(&d_mv_a, n * sizeof(mm128_t));
    if (err != hipSuccess) {
        fprintf(stderr, "[mm_copy_mz_gpu] hipMalloc d_mv_a failed: %s\n", hipGetErrorString(err));
        hipFree(da);
        return;
    }

    // Copy input to device
    hipMemcpy(d_mv_a, mv_a, n * sizeof(mm128_t), hipMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 1024;
    int gridSize = (n + blockSize - 1) / blockSize;
    copy_mz_kernel<<<gridSize, blockSize>>>(da, d_mv_a, n);

    // Check for kernel errors
    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[mm_copy_mz_gpu] kernel launch failed: %s\n", hipGetErrorString(err));
    }

    // Copy result back to host
    hipMemcpy(a, da, n * sizeof(mm128_t), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(da);
    hipFree(d_mv_a);
}
