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

#ifndef _PLSCORE_CUH_
#define _PLSCORE_CUH_

#include "plmem.cuh"
#include "chain_priv.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MM_QSPAN 15

void plscore_async_naive_forward_dp(const gpu_chain_ctx_t* ctx, deviceMemPtr* dev_mem, cudaStream_t* stream, Misc misc);
void plscore_async_short_mid_forward_dp(const gpu_chain_ctx_t* ctx, deviceMemPtr* dev_mem, cudaStream_t* stream, Misc misc);
void plscore_async_long_forward_dp(const gpu_chain_ctx_t* ctx, deviceMemPtr* dev_mem, cudaStream_t* stream, Misc misc);

// score_kernel_config global has been retired; kernel configs now live on
// each gpu_chain_ctx_t (ctx->score_config). Host-side kernel
// launchers read the config via ctx (passed in or via worker->parent_ctx).

#ifdef __cplusplus
}
#endif

#endif // _PLSCORE_CUH_