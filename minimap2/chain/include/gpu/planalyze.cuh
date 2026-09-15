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

#ifndef __PLANALYZE_H__
#define __PLANALYZE_H__

/* Implement kernel performance analysis that requires extra device
 * synchornization. disabled unless DEBUG_LEVEL is set to analyze.
 * Enable individual verbose prints in planalyze.cu 
 */

#include "hipify.cuh"
#include "plchain.h"
#include "plutils.h"
#include "plmem.cuh"
#include "plscore.cuh"


#ifdef DEBUG_CHECK
void planalyze_short_kernel(const gpu_worker_t* w, int uid, float throughput[]);
void planalyze_long_kernel(const gpu_worker_t* w, float* throughput);
#endif // DEBUG_CHECK

#endif // __PLANALYZE_H__