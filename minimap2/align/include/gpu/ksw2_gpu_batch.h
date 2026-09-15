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
 * @file ksw2_gpu_batch.h
 * @brief GPU batch alignment interface for ksw2_extd2_gpu_multialign
 *
 * This header provides a batching interface that accumulates multiple alignment
 * tasks and executes them together on the GPU for better efficiency.
 */

#ifndef KSW2_GPU_BATCH_H_
#define KSW2_GPU_BATCH_H_

#include <stdint.h>
#include <stddef.h>
#include "ksw2.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle for GPU batch state
 * (Actual struct is defined in ksw2_gpu_batch.cpp using C++ containers)
 */
typedef struct gpu_batch_s gpu_batch_t;

/**
 * Initialize a GPU batch for accumulating alignments
 *
 * @param avail_mem      Available GPU memory in bytes (0 for default)
 * @param max_alignments Maximum alignments per batch before forced flush (0 for default)
 * @param q              Gap open penalty (short gaps)
 * @param e              Gap extension penalty (short gaps)  
 * @param q2             Gap open penalty (long gaps)
 * @param e2             Gap extension penalty (long gaps)
 * @param w              Band width
 * @param zdrop          Z-drop threshold
 * @param a              Match score for scoring matrix generation
 * @param b              Mismatch penalty for scoring matrix generation
 * @param transition     Transition mismatch penalty (A:G, C:T); 0 disables transition scoring
 * @param sc_ambi        Ambiguity score for scoring matrix generation
 * @param stream         HIP stream to use for all GPU operations (caller-owned).
 *                       If NULL, the batch creates its own stream.
 * @param mem_pool       HIP memory pool handle (from hipDeviceGetDefaultMemPool).
 *                       If NULL, pool features are not used.
 * @return               Batch handle, or NULL on error
 */
gpu_batch_t* ksw_extd2_gpu_batch_init(size_t avail_mem, int max_alignments,
    int8_t q, int8_t e, int8_t q2, int8_t e2,
    int w, int zdrop, int8_t a, int8_t b, int8_t transition, int8_t sc_ambi,
    void* stream, void* mem_pool);

/**
 * Add an alignment task to the batch
 *
 * If adding this alignment would exceed memory limits, the batch is 
 * automatically flushed (executed) before adding the new alignment.
 *
 * @param batch      Batch handle from ksw_extd2_gpu_batch_init
 * @param qlen       Query sequence length
 * @param qseq       Query sequence (will be copied; caller can free after call)
 * @param tlen       Target sequence length
 * @param tseq       Target sequence (will be copied; caller can free after call)
 * @param w          Alignment bandwidth for this task (may differ from batch default for long-join gaps)
 * @param end_bonus  End bonus for scoring
 * @param zdrop      Z-drop threshold for this alignment (may differ from batch default,
 *                   e.g. zdrop_inv for inversion-boundary regions). A zdrop change
 *                   triggers an automatic flush before the task is queued.
 * @param flag       Alignment flags (KSW_EZ_*)
 * @param ez         Output structure - results will be written here after flush
 * @return           0 on success, 1 if batch was flushed, -1 on error
 */
int ksw_extd2_gpu_batch_add(gpu_batch_t* batch,
    int qlen, const uint8_t* qseq,
    int tlen, const uint8_t* tseq,
    int w, int end_bonus, int zdrop, int flag,
    ksw_extz_t* ez);

/**
 * Test whether a single alignment can ever fit the batch's VRAM budget.
 *
 * A task whose estimated device memory exceeds the per-batch budget can never
 * be flushed on the GPU (the device alloc would OOM), so the caller should run
 * it on the CPU directly instead of adding it to the batch. Returns 1 if the
 * task fits the budget, 0 if it is too large (or batch is NULL).
 *
 * @param batch      Batch handle
 * @param qlen,tlen  Query / target lengths
 * @param w          Band width (<0 = full)
 * @param flag       KSW_EZ_* flags (affects CIGAR buffer sizing)
 */
int ksw_extd2_gpu_task_fits(gpu_batch_t* batch, int qlen, int tlen, int w, int flag);

/**
 * Flush (execute) all pending alignments in the batch
 *
 * After this call, all ksw_extz_t structures passed to batch_add will
 * be populated with results. The batch is reset and can accept new tasks.
 *
 * @param batch      Batch handle
 * @return           Number of alignments executed, or -1 on error
 */
int ksw_extd2_gpu_batch_flush(gpu_batch_t* batch);

/**
 * Get the number of pending alignments in the batch
 *
 * @param batch      Batch handle
 * @return           Number of alignments waiting to be executed
 */
int ksw_extd2_gpu_batch_pending(gpu_batch_t* batch);

/**
 * Destroy a GPU batch and free all associated resources
 *
 * Any pending alignments that haven't been flushed will be discarded.
 *
 * @param batch      Batch handle
 */
void ksw_extd2_gpu_batch_destroy(gpu_batch_t* batch);

/**
 * Warm up the GPU context (trigger lazy ROCm initialization).
 */
void ksw_gpu_warmup(void);

#ifdef __cplusplus
}
#endif

#endif /* KSW2_GPU_BATCH_H_ */
