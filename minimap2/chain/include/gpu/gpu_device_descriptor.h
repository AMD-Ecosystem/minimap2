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
 * @file gpu_device_descriptor.h
 * @brief Shared GPU device descriptor abstraction for multi-GPU support
 *
 * Defines the GPU_device struct that encapsulates per-device state:
 * device id, HIP mempool, and cached device properties. This header is
 * included by both GPU chain and GPU align subsystems to share the same
 * device objects across contexts.
 *
 * Lifecycle: created once at mapping context init by mm_mapctx_create(),
 * passed to both chain and align contexts, and destroyed when the last
 * reference is released (via shared_ptr).
 */

#ifndef GPU_DEVICE_DESCRIPTOR_H
#define GPU_DEVICE_DESCRIPTOR_H

#include "gpu_device_list_fwd.h"
#include <hip/hip_runtime.h>

struct GPU_device {
	int device_id;
	hipMemPool_t mempool;
	bool mempool_owned;
	hipDeviceProp_t prop;
	char product_name[256];
};

// Create a validated, homogeneous device list.
// device_ids == nullptr / num_devices == 0 means auto-discover from HIP_VISIBLE_DEVICES.
// Returns an empty vector on error (logged internally).
gpu_device_list_t mm_gpu_create_device_list(const int* device_ids, int num_devices);

// Returns the number of visible HIP devices, or 0 on failure.
int mm_gpu_visible_device_count(void);

#endif // GPU_DEVICE_DESCRIPTOR_H
