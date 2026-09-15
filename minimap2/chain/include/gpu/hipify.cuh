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

#ifndef __HIPIFY_CUH__
#define __HIPIFY_CUH__

#include "mm_log.h"

#ifdef USEHIP
#include "hip/hip_runtime.h"
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetDevice hipGetDevice
#define cudaMalloc hipMalloc
#define cudaMallocAsync hipMallocAsync
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyToSymbolAsync hipMemcpyToSymbolAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaFree hipFree
#define cudaFreeAsync hipFreeAsync
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamDestroy hipStreamDestroy
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaMemGetInfo hipMemGetInfo
#define hipMemPool_t hipMemPool_t
#define cudaMemPoolCreate hipMemPoolCreate
#define cudaMemPoolDestroy hipMemPoolDestroy
#define cudaMemPoolProps hipMemPoolProps
#define cudaMallocFromPoolAsync hipMallocFromPoolAsync
#define cudaMemPoolAttr hipMemPoolAttr
#define cudaMemPoolSetAttribute hipMemPoolSetAttribute
#define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#define cudaDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool
#define cudaCheck() \
	{ \
		hipError_t err = hipGetLastError(); \
		if (hipSuccess != err) { \
			mm_log_error("GPU error: {}", hipGetErrorString(err)); \
			exit(EXIT_FAILURE); \
		} \
	}
#define cudaWarpSize 64
#else
#define cudaCheck() \
	{ \
		cudaError_t err = cudaGetLastError(); \
		if (cudaSuccess != err) { \
			mm_log_error("GPU error: {}", cudaGetErrorString(err)); \
			exit(EXIT_FAILURE); \
		} \
	}
#include <cuda.h>
// CUDA mempool types (CUDA 11.2+)
#define hipMemPool_t cudaMemPool_t
#define cudaMemPoolCreate cudaMemPoolCreate
#define cudaMemPoolDestroy cudaMemPoolDestroy
#define cudaMemPoolProps cudaMemPoolProps
#define cudaMallocFromPoolAsync cudaMallocFromPoolAsync
#define cudaMemPoolAttr cudaMemPoolAttr
#define cudaMemPoolSetAttribute cudaMemPoolSetAttribute
#define cudaMemPoolAttrReleaseThreshold cudaMemPoolAttrReleaseThreshold
#define cudaDeviceGetDefaultMemPool cudaDeviceGetDefaultMemPool

#endif


#endif // __HIPIFY_CUH__
