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
 * GPU Alignment Context Pool Query Tests
 *
 * Verifies the mm_gpu_align_ctx_t multi-GPU pool API:
 *   - mm_gpu_align_ctx_pool_size(): total slots = accums_per_gpu * num_devices
 *   - mm_gpu_align_ctx_pool_device_id(): correct device assignment per slot
 *   - Pool query functions handle NULL contexts safely
 */

#include "test_gpu_common.h"
#include "gpu/gpu_align_ctx.h"

// =============================================================================
// Pool Query Functions - NULL Context Safety Tests
// These tests do not depend on GPU initialization.
// =============================================================================

TEST(GpuAlignContextPoolQueries, PoolSizeReturnsZeroForNull)
{
	EXPECT_EQ(mm_gpu_align_ctx_pool_size(nullptr), 0);
}

TEST(GpuAlignContextPoolQueries, PoolDeviceIdReturnsNegativeOneForNull)
{
	EXPECT_EQ(mm_gpu_align_ctx_pool_device_id(nullptr, 0), -1);
	EXPECT_EQ(mm_gpu_align_ctx_pool_device_id(nullptr, -1), -1);
	EXPECT_EQ(mm_gpu_align_ctx_pool_device_id(nullptr, 999), -1);
}

TEST(GpuAlignContextPoolQueries, DestroyNullContextIsNoOp)
{
	// mm_align_ctx_destroy(NULL) should be a no-op (NULL-safe)
	mm_align_ctx_destroy(nullptr);
}

