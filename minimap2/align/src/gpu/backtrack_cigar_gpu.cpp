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
 * GPU kernel for reversing CIGAR array
 * This file is compiled with hipcc and provides a C-callable wrapper
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

// GPU kernel to reverse CIGAR array in parallel
__global__ void backtrack_cigar_gpu_kernel(uint32_t* cigar, int n_cigar)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n_cigar / 2) {
		uint32_t tmp = cigar[idx];
		cigar[idx] = cigar[n_cigar - 1 - idx];
		cigar[n_cigar - 1 - idx] = tmp;
	}
}

// C-callable wrapper function
extern "C" void reverse_cigar_gpu(uint32_t* cigar, int n_cigar)
{
	uint32_t* d_cigar = nullptr;

	// Allocate device memory
	hipMalloc(&d_cigar, n_cigar * sizeof(uint32_t));

	// Copy CIGAR to device
	hipMemcpy(d_cigar, cigar, n_cigar * sizeof(uint32_t), hipMemcpyHostToDevice);

	// Launch kernel
	int blockSize = 256;
	int gridSize = (n_cigar + blockSize - 1) / blockSize;
	backtrack_cigar_gpu_kernel<<<gridSize, blockSize>>>(d_cigar, n_cigar);

	// Copy result back to host
	hipMemcpy(cigar, d_cigar, n_cigar * sizeof(uint32_t), hipMemcpyDeviceToHost);

	// Free device memory
	hipFree(d_cigar);
}
