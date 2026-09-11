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

#ifndef ALIGN_BATCH_PRIV_H
#define ALIGN_BATCH_PRIV_H

#include "align_priv.h"
#include "ksw2.h"
#include "gpu/ksw2_gpu_batch.h"
#include "mm2_align.h"
#include "minimap.h"
#include "chain_read.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Alignment task types */
#define ALIGN_TASK_LEFT_EXT 0
#define ALIGN_TASK_GAP_FILL 1
#define ALIGN_TASK_RIGHT_EXT 2

typedef struct {
	int task_type;
	int reg_idx;
	int qlen, tlen;
	uint8_t* qseq;
	uint8_t* tseq;
	int w;
	int end_bonus;
	int zdrop;
	int flag;
	ksw_extz_t ez;
	int32_t rs, qs;
	int32_t rs0, qs0;
	int need_rev;
	int gap_idx;
	int32_t re, qe;
	int anchor_idx;
	int bw1;
	int is_last_gap;
} align_task_t;

typedef struct {
	mm_reg1_t* r;
	mm_reg1_t r2;
	int32_t rid, rev;
	int32_t as1, cnt1;
	int32_t rs, re, qs, qe;
	int32_t rs0, re0, qs0, qe0;
	int32_t rs1, qs1, re1, qe1;
	uint8_t* tseq;
	uint8_t* junc;
	int8_t mat[25];
	int bw, bw_long;
	int extra_flag;
	int dropped;
	int completed;
	int gap_task_start;
	int gap_task_count;
} align_state_t;

typedef struct {
	align_task_t* tasks;
	int n_tasks;
	int64_t max_tasks;
	uint8_t* seq_buf;
	size_t seq_buf_size;
	size_t seq_buf_used;
} align_batch_t;

typedef struct {
	chain_read_t* read;
	mm_reg1_t* regs0;
	int n_regs0;
	mm128_t* a;
	int qlen;
	const char* seq;
	uint8_t* qseq0[2];
	int n_a;
	align_state_t* states;
	int n_prepared;
	int use_gpu_batch;
	mm_reg1_t** out_regs;
	int* out_n_regs;
	const char* qname;
	int rep_len;
} read_align_ctx_t;

/* Batch management */
align_batch_t* align_batch_init(void* km, int64_t max_tasks);
void align_batch_reset(align_batch_t* batch);
void align_batch_destroy(void* km, align_batch_t* batch);
int align_batch_execute(const mm_mapopt_t* opt, gpu_batch_t* gpu_batch, align_batch_t* batch);

mm_reg1_t* mm_insert_reg(const mm_reg1_t* r, int i, int* n_regs, mm_reg1_t* regs);

mm_reg1_t* mm_align_batched(void* km, const mm_mapopt_t* opt, const mm_idx_t* mi,
    int qlen, uint8_t* qseq0[2], int* n_regs_,
    mm_reg1_t* regs, mm128_t* a, int n_a,
    gpu_batch_t* gpu_batch, align_batch_t* batch,
    int splice_flag, const mm_core_ctx_t* core_ctx);

#ifdef __cplusplus
}
#endif

#endif /* ALIGN_BATCH_PRIV_H */
