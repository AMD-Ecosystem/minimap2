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

#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <atomic>
#include <hip/hip_runtime.h>
#include "gpu/align_batch_priv.h"
#include "align_internal.h"
#include "misc.h"
#include "index.h"
#include "kalloc.h"
#include "khash.h"
#include "mm_dump.h"
#include "mm_log.h"
#include "mm_timer.h"

#ifdef HAVE_GPU_ALIGNMENT

struct gpu_align_accum_s {
	void* km;
	read_align_ctx_t* ctxs;
	int n_ctxs;
	int cap_ctxs;
	int total_tasks;
	gpu_batch_t* gpu_batch;
	const mm_mapopt_t* opt;
	const mm_idx_t* mi;
	int device_id;
	int is_sr;
	int is_splice;
	int flush_threshold;
};

int mm_align1_prepare_state(void* km, const mm_mapopt_t* opt, const mm_idx_t* mi,
    int qlen, uint8_t* qseq0[2], mm_reg1_t* r,
    int n_a, mm128_t* a, int splice_flag, align_state_t* st);
int mm_batch_add_left_ext(const mm_mapopt_t* opt, const mm_idx_t* mi,
    int qlen, uint8_t* qseq0[2], mm128_t* a,
    align_state_t* st, align_batch_t* batch, int reg_idx);
void mm_process_left_ext_result(align_state_t* st, align_task_t* t);
int mm_batch_collect_gaps(void* km, const mm_mapopt_t* opt, const mm_idx_t* mi,
    int qlen, uint8_t* qseq0[2], mm128_t* a, align_state_t* st,
    align_batch_t* batch, int reg_idx);
void mm_process_gap_fill_results(void* km, const mm_mapopt_t* opt, const mm_idx_t* mi,
    int qlen, uint8_t* qseq0[2], mm128_t* a,
    align_state_t* st, align_batch_t* batch, ksw_extz_t* ez_scratch, const mm_core_ctx_t* core_ctx);
int mm_batch_add_right_ext(const mm_mapopt_t* opt, const mm_idx_t* mi,
    int qlen, uint8_t* qseq0[2],
    align_state_t* st, align_batch_t* batch, int reg_idx);
void mm_process_right_ext_result(int qlen, align_state_t* st, align_task_t* t);
void mm_region_finalize(void* km, const mm_mapopt_t* opt, const mm_idx_t* mi,
    int qlen, uint8_t* qseq0[2], align_state_t* st);
void mm_state_free(void* km, align_state_t* st);
void mm_region_gap_fill(void* km, const mm_mapopt_t* opt, const mm_idx_t* mi,
    int qlen, uint8_t* qseq0[2], mm128_t* a, align_state_t* st,
    ksw_extz_t* ez, int n_a, const mm_core_ctx_t* core_ctx);

gpu_align_accum_t* mm_gpu_accum_init(const mm_mapopt_t* opt, const mm_idx_t* mi, void* mem_pool,
	int64_t batch_max_mem, int batch_max_align, int flush_threshold,
	int device_id)
{
	gpu_align_accum_t* accum = (gpu_align_accum_t*)calloc(1, sizeof(*accum));
	if (!accum) return NULL;

	accum->km = km_init();
	accum->cap_ctxs = 256;
	accum->ctxs = (read_align_ctx_t*)calloc(accum->cap_ctxs, sizeof(read_align_ctx_t));
	accum->opt = opt;
	accum->mi = mi;
	accum->device_id = device_id;
	/* Multi-GPU: set device context for all GPU operations within this
	 * accumulator. The hipSetDevice call adds minimal overhead (register save
	 * on the device, no host-device transfer) and ensures all subsequent
	 * hipMalloc, kernel launches, and HIP calls target the correct GPU. */
	accum->is_sr = !!(opt->flag & MM_F_SR);
	/* Route any splice mode to the CPU path. The GPU aligner does not implement
	 * spliced alignment; gating on the plain MM_F_SPLICE flag (rather than the
	 * both-strand FOR+REV combination) ensures single-strand splice modes
	 * (e.g. --splice -u f) and spsc/jump features also fall back to CPU. */
	accum->is_splice = !!(opt->flag & MM_F_SPLICE);
	accum->flush_threshold = flush_threshold;

	hipSetDevice(device_id);
	accum->gpu_batch = ksw_extd2_gpu_batch_init(
	    batch_max_mem, batch_max_align,
	    opt->q, opt->e, opt->q2, opt->e2,
	    opt->bw, opt->zdrop, opt->a, opt->b, opt->transition, opt->sc_ambi, NULL, mem_pool);
	return accum;
}

void mm_gpu_accum_add_read(gpu_align_accum_t* accum, chain_read_t* rd,
    mm_reg1_t** out_regs, int* out_n_regs,
    mm_stage_timer_t* t_align, void* batch_km, const mm_core_ctx_t* core_ctx)
{
	extern unsigned char seq_nt4_table[256];
	const mm_mapopt_t* opt = accum->opt;
	const mm_idx_t* mi = accum->mi;
	void* akm = accum->km;
	int i;

	/* Route to the CPU aligner unless this is a single-segment, non-splice,
	 * CIGAR read that the GPU kernel can handle. Short-read (MM_F_SR) reads use
	 * the CPU ungapped/SR scoring path the GPU kernel does not implement; they
	 * are normally n_seg==2 (paired) and caught by the n_seg test, but weak/no
	 * pairing makes them n_seg==1, so gate on is_sr explicitly too. */
	if (rd->n_seg != 1 || accum->is_splice || accum->is_sr || !(opt->flag & MM_F_CIGAR)) {
		mm_log_debug("[gpu-align] read '{}' aligned on CPU ({})", rd->seq.name,
		    rd->n_seg != 1	   ? "multi-segment read"
			: accum->is_splice ? "splice mode: no GPU spliced kernel"
			: accum->is_sr	   ? "short-read/SR mode: no GPU SR kernel"
					   : "CIGAR output not requested");
		mm_map_align(mi, opt, rd, out_regs, out_n_regs, t_align, batch_km, core_ctx, NULL, 0);
		return;
	}

	const int* qlens = rd->qlens;
	const char** seqs = rd->qseqs;
	const char* qname = rd->seq.name;
	int qlen_sum = rd->seq.qlen_sum;
	int max_chain_gap_ref = rd->frag_gap;

	*out_n_regs = 0;
	*out_regs = 0;

	uint32_t hash = qname && !(opt->flag & MM_F_NO_HASH_NAME)
	    ? __ac_X31_hash_string(qname)
	    : 0;
	hash ^= __ac_Wang_hash(qlen_sum) + __ac_Wang_hash(opt->seed);
	hash = __ac_Wang_hash(hash);

	mm_reg1_t* regs0 = mm_gen_regs(akm, hash, qlen_sum, rd->n_u, rd->u, rd->a,
	    !!(opt->flag & MM_F_QSTRAND));
	int n_regs0 = rd->n_u;

	if (mi->n_alt) {
		mm_mark_alt(mi, n_regs0, regs0);
		mm_hit_sort(akm, &n_regs0, regs0, opt->alt_drop);
	}

	chain_post(opt, max_chain_gap_ref, mi, akm, qlen_sum, 1, qlens, &n_regs0, regs0, rd->a);
	if (!accum->is_sr && !(opt->flag & MM_F_QSTRAND)) {
		mm_est_err(mi, qlen_sum, n_regs0, regs0, rd->a, rd->n_mini_pos, rd->mini_pos, NULL);
		n_regs0 = mm_filter_strand_retained(n_regs0, regs0);
	}

	kfree(batch_km, rd->u);
	rd->u = 0;
	rd->n_u = 0;
	kfree(batch_km, rd->mini_pos);
	rd->mini_pos = 0;
	rd->n_mini_pos = 0;

	if (n_regs0 < 1) {
		free(regs0);
		kfree(batch_km, rd->a);
		rd->a = 0;
		mm_dump_alignments(core_ctx, qname, 1, qlen_sum, out_n_regs, out_regs, mi);
		return;
	}

	read_align_ctx_t ctx;
	memset(&ctx, 0, sizeof(ctx));
	ctx.read = rd;
	ctx.regs0 = regs0;
	ctx.n_regs0 = n_regs0;
	ctx.qlen = qlens[0];
	ctx.seq = seqs[0];
	ctx.use_gpu_batch = 1;
	ctx.out_regs = out_regs;
	ctx.out_n_regs = out_n_regs;
	ctx.qname = (const char*)kmalloc(akm, strlen(qname) + 1);
	strcpy((char*)ctx.qname, qname);
	ctx.rep_len = rd->rep_len;

	ctx.qseq0[0] = (uint8_t*)kmalloc(akm, ctx.qlen * 2);
	ctx.qseq0[1] = ctx.qseq0[0] + ctx.qlen;
	for (i = 0; i < ctx.qlen; i++) {
		ctx.qseq0[0][i] = seq_nt4_table[(uint8_t)ctx.seq[i]];
		ctx.qseq0[1][ctx.qlen - 1 - i] = ctx.qseq0[0][i] < 4
		    ? 3 - ctx.qseq0[0][i]
		    : 4;
	}

	ctx.a = rd->a;
	ctx.n_a = mm_squeeze_a(akm, ctx.n_regs0, ctx.regs0, ctx.a);

	{
		size_t a_bytes = (size_t)ctx.n_a * sizeof(mm128_t);
		mm128_t* a_copy = (mm128_t*)kmalloc(akm, a_bytes);
		memcpy(a_copy, ctx.a, a_bytes);
		ctx.a = a_copy;
	}
	kfree(batch_km, rd->a);
	rd->a = 0;

	ctx.states = (align_state_t*)kcalloc(akm, ctx.n_regs0, sizeof(align_state_t));
	ctx.n_prepared = 0;
	for (i = 0; i < ctx.n_regs0; i++) {
		if (mm_align1_prepare_state(akm, opt, mi, ctx.qlen, ctx.qseq0,
			&ctx.regs0[i], ctx.n_a, ctx.a,
			opt->flag, &ctx.states[i]) < 0) {
			ctx.states[i].completed = 1;
		} else {
			ctx.n_prepared++;
		}
	}

	if (accum->n_ctxs >= accum->cap_ctxs) {
		accum->cap_ctxs *= 2;
		accum->ctxs = (read_align_ctx_t*)realloc(accum->ctxs,
		    accum->cap_ctxs * sizeof(read_align_ctx_t));
	}

	accum->ctxs[accum->n_ctxs++] = ctx;
	accum->total_tasks += ctx.n_prepared;
}

int mm_gpu_accum_should_flush(const gpu_align_accum_t* accum)
{
	return accum && accum->total_tasks >= accum->flush_threshold;
}

void mm_gpu_accum_flush(gpu_align_accum_t* accum, mm_stage_timer_t* t_align, const mm_core_ctx_t* core_ctx)
{
	if (!accum || accum->n_ctxs == 0) return;
	/* Multi-GPU: set device context to ensure all HIP calls (kernels, hipFree,
	 * hipDeviceSynchronize, etc.) target the correct GPU device. */
	hipSetDevice(accum->device_id);
	mm_log_debug("mm_gpu_accum_flush n_ctxs={}", accum->n_ctxs);

	double t1 = t_align ? realtime() : 0;
	void* km = accum->km;
	const mm_mapopt_t* opt = accum->opt;
	const mm_idx_t* mi = accum->mi;
	int is_sr = accum->is_sr;
	/* mm_set_mapq2() expects the plain --splice flag, matching the CPU path in
	 * mm_map_align(). Re-derive from the flag directly (same as accum->is_splice,
	 * which is set to plain MM_F_SPLICE at line 107). */
	int is_splice = !!(opt->flag & MM_F_SPLICE);
	int r, i, j;
	int n_ctxs = accum->n_ctxs;
	read_align_ctx_t* ctxs = accum->ctxs;
	gpu_batch_t* gpu_batch = accum->gpu_batch;

	int64_t total_regs_est = 0;
	for (r = 0; r < n_ctxs; r++)
		total_regs_est += ctxs[r].n_regs0;
	align_batch_t* batch = align_batch_init(km, total_regs_est * 4 + 256);

	/* Phase 2: LEFT_EXT from all accumulated reads */
	align_batch_reset(batch);
	for (r = 0; r < n_ctxs; r++) {
		read_align_ctx_t* ctx = &ctxs[r];
		if (!ctx->use_gpu_batch) continue;
		for (i = 0; i < ctx->n_regs0; i++) {
			if (ctx->states[i].completed) continue;
			int ret = mm_batch_add_left_ext(opt, mi, ctx->qlen, ctx->qseq0,
			    ctx->a, &ctx->states[i], batch,
			    i | (r << 16));
			if (ret == -2) {
				align_batch_execute(opt, gpu_batch, batch);
				for (j = 0; j < batch->n_tasks; j++) {
					align_task_t* t = &batch->tasks[j];
					int ri = t->reg_idx >> 16, rgi = t->reg_idx & 0xFFFF;
					t->reg_idx = rgi;
					mm_process_left_ext_result(&ctxs[ri].states[rgi], t);
				}
				align_batch_reset(batch);
				mm_batch_add_left_ext(opt, mi, ctx->qlen, ctx->qseq0,
				    ctx->a, &ctx->states[i], batch,
				    i | (r << 16));
			}
		}
	}
	if (batch->n_tasks > 0) {
		align_batch_execute(opt, gpu_batch, batch);
		for (j = 0; j < batch->n_tasks; j++) {
			align_task_t* t = &batch->tasks[j];
			int ri = t->reg_idx >> 16, rgi = t->reg_idx & 0xFFFF;
			t->reg_idx = rgi;
			mm_process_left_ext_result(&ctxs[ri].states[rgi], t);
		}
	}

	/* Phase 4: GAP_FILL batched across all accumulated reads */
#if GPU_GAP_FILL_ENABLED
	align_batch_reset(batch);
	for (r = 0; r < n_ctxs; r++) {
		read_align_ctx_t* ctx = &ctxs[r];
		if (!ctx->use_gpu_batch) continue;
		for (i = 0; i < ctx->n_regs0; i++) {
			if (ctx->states[i].completed) continue;
			ctx->states[i].re1 = ctx->states[i].rs;
			ctx->states[i].qe1 = ctx->states[i].qs;
			int ret = mm_batch_collect_gaps(km, opt, mi, ctx->qlen, ctx->qseq0,
			    ctx->a, &ctx->states[i], batch,
			    i | (r << 16));
			if (ret == -1) {
				align_batch_execute(opt, gpu_batch, batch);
				for (int rr = 0; rr <= r; rr++) {
					if (!ctxs[rr].use_gpu_batch) continue;
					int ilim = (rr < r) ? ctxs[rr].n_regs0 : i;
					for (int ii = 0; ii < ilim; ii++) {
						if (ctxs[rr].states[ii].completed || ctxs[rr].states[ii].gap_task_count == 0) continue;
						mm_process_gap_fill_results(km, opt, mi, ctxs[rr].qlen,
						    ctxs[rr].qseq0, ctxs[rr].a, &ctxs[rr].states[ii], batch, NULL, core_ctx);
					}
				}
				align_batch_reset(batch);
				mm_batch_collect_gaps(km, opt, mi, ctx->qlen, ctx->qseq0,
				    ctx->a, &ctx->states[i], batch,
				    i | (r << 16));
			}
		}
	}
	if (batch->n_tasks > 0) {
		align_batch_execute(opt, gpu_batch, batch);
		for (r = 0; r < n_ctxs; r++) {
			read_align_ctx_t* ctx = &ctxs[r];
			if (!ctx->use_gpu_batch) continue;
			for (i = 0; i < ctx->n_regs0; i++) {
				if (ctx->states[i].completed || ctx->states[i].gap_task_count == 0) continue;
				mm_process_gap_fill_results(km, opt, mi, ctx->qlen,
				    ctx->qseq0, ctx->a, &ctx->states[i], batch, NULL, core_ctx);
			}
		}
	}
#else
	{
		ksw_extz_t ez_gf;
		memset(&ez_gf, 0, sizeof(ksw_extz_t));
		for (r = 0; r < n_ctxs; r++) {
			read_align_ctx_t* ctx = &ctxs[r];
			if (!ctx->use_gpu_batch) continue;
			for (i = 0; i < ctx->n_regs0; i++) {
				if (ctx->states[i].completed) continue;
				ctx->states[i].re1 = ctx->states[i].rs;
				ctx->states[i].qe1 = ctx->states[i].qs;
				mm_region_gap_fill(km, opt, mi, ctx->qlen, ctx->qseq0,
				    ctx->a, &ctx->states[i], &ez_gf, 0, core_ctx);
			}
		}
	}
#endif

	/* Phase 5: RIGHT_EXT from all accumulated reads */
	align_batch_reset(batch);
	for (r = 0; r < n_ctxs; r++) {
		read_align_ctx_t* ctx = &ctxs[r];
		if (!ctx->use_gpu_batch) continue;
		for (i = 0; i < ctx->n_regs0; i++) {
			if (ctx->states[i].completed || ctx->states[i].dropped) continue;
			int ret = mm_batch_add_right_ext(opt, mi, ctx->qlen, ctx->qseq0,
			    &ctx->states[i], batch,
			    i | (r << 16));
			if (ret == -2) {
				align_batch_execute(opt, gpu_batch, batch);
				for (j = 0; j < batch->n_tasks; j++) {
					align_task_t* t = &batch->tasks[j];
					int ri = t->reg_idx >> 16, rgi = t->reg_idx & 0xFFFF;
					t->reg_idx = rgi;
					mm_process_right_ext_result(ctxs[ri].qlen, &ctxs[ri].states[rgi], t);
				}
				align_batch_reset(batch);
				mm_batch_add_right_ext(opt, mi, ctx->qlen, ctx->qseq0,
				    &ctx->states[i], batch,
				    i | (r << 16));
			}
		}
	}
	if (batch->n_tasks > 0) {
		align_batch_execute(opt, gpu_batch, batch);
		for (j = 0; j < batch->n_tasks; j++) {
			align_task_t* t = &batch->tasks[j];
			int ri = t->reg_idx >> 16, rgi = t->reg_idx & 0xFFFF;
			t->reg_idx = rgi;
			mm_process_right_ext_result(ctxs[ri].qlen, &ctxs[ri].states[rgi], t);
		}
	}

	/* Phase 7: Finalize all accumulated reads */
	for (r = 0; r < n_ctxs; r++) {
		read_align_ctx_t* ctx = &ctxs[r];
		if (!ctx->use_gpu_batch) continue;

		for (i = 0; i < ctx->n_regs0; i++) {
			if (!ctx->states[i].completed)
				mm_region_finalize(km, opt, mi, ctx->qlen, ctx->qseq0, &ctx->states[i]);
		}

		int n_states = ctx->n_regs0;
		for (i = ctx->n_regs0 - 1; i >= 0; --i) {
			if (ctx->states[i].r2.cnt > 0)
				ctx->regs0 = mm_insert_reg(&ctx->states[i].r2, i, &ctx->n_regs0, ctx->regs0);
		}

		{
			mm_mapopt_t opt_cpu = *opt;
			opt_cpu.flag &= ~MM_F_GPU_ALIGN;
			ksw_extz_t ez;
			memset(&ez, 0, sizeof(ksw_extz_t));
			for (i = 0; i < ctx->n_regs0; ++i) {
				if (ctx->regs0[i].p == NULL && ctx->regs0[i].cnt > 0 && (ctx->regs0[i].split & 2)) {
					mm_reg1_t r2_of_r2;
					mm_align1(km, &opt_cpu, mi, ctx->qlen, ctx->qseq0, &ctx->regs0[i], &r2_of_r2, ctx->n_a, ctx->a, &ez, opt->flag, core_ctx, NULL, 0);
					if (r2_of_r2.cnt > 0)
						ctx->regs0 = mm_insert_reg(&r2_of_r2, i, &ctx->n_regs0, ctx->regs0);
				}
			}
			kfree(km, ez.cigar);
		}

		for (i = 0; i < n_states; i++)
			mm_state_free(km, &ctx->states[i]);

		for (i = 1; i < ctx->n_regs0; i++) {
			if (ctx->regs0[i].split_inv && !(opt->flag & MM_F_NO_INV)) {
				ksw_extz_t ez;
				memset(&ez, 0, sizeof(ksw_extz_t));
				mm_reg1_t r2;
				if (mm_align1_inv(km, opt, mi, ctx->qlen, ctx->qseq0,
					&ctx->regs0[i - 1], &ctx->regs0[i], &r2, &ez, core_ctx)) {
					ctx->regs0 = mm_insert_reg(&r2, i, &ctx->n_regs0, ctx->regs0);
					i++;
				}
				kfree(km, ez.cigar);
			}
		}

		mm_filter_regs(opt, ctx->qlen, &ctx->n_regs0, ctx->regs0);
		if (!(opt->flag & (MM_F_SR | MM_F_SR_RNA | MM_F_ALL_CHAINS)) && !opt->split_prefix && ctx->qlen >= opt->rank_min_len) {
			mm_update_dp_max(ctx->qlen, ctx->n_regs0, ctx->regs0, opt->rank_frac, opt->a, opt->b);
			mm_filter_regs(opt, ctx->qlen, &ctx->n_regs0, ctx->regs0);
		}
		mm_hit_sort(km, &ctx->n_regs0, ctx->regs0, opt->alt_drop);

		if (!(opt->flag & MM_F_ALL_CHAINS)) {
			mm_set_parent(km, opt->mask_level, opt->mask_len, ctx->n_regs0, ctx->regs0,
			    opt->a * 2 + opt->b, opt->flag & MM_F_HARD_MLEVEL, opt->alt_drop);
			mm_select_sub(km, opt->pri_ratio, mi->k * 2, opt->best_n, 0, opt->max_gap * 0.8,
			    &ctx->n_regs0, ctx->regs0);
			mm_set_sam_pri(ctx->n_regs0, ctx->regs0);
		}

		ctx->regs0 = (mm_reg1_t*)realloc(ctx->regs0, sizeof(mm_reg1_t) * ctx->n_regs0);
		mm_set_mapq2(km, ctx->n_regs0, ctx->regs0, opt->min_chain_score, opt->a,
		    ctx->rep_len, is_sr, is_splice);

		*ctx->out_n_regs = ctx->n_regs0;
		*ctx->out_regs = ctx->regs0;

		mm_dump_alignments(core_ctx, ctx->qname, 1, ctx->qlen, ctx->out_n_regs, ctx->out_regs, mi);

		kfree(km, ctx->states);
		kfree(km, ctx->qseq0[0]);
		kfree(km, ctx->a);
	}

	align_batch_destroy(km, batch);

	km_destroy(accum->km);
	accum->km = km_init();
	accum->n_ctxs = 0;
	accum->total_tasks = 0;

	if (t_align) mm_stage_timer_record(t_align, realtime() - t1);
}

void mm_gpu_accum_destroy(gpu_align_accum_t* accum)
{
	if (!accum) return;
	/* Multi-GPU: set device context before cleanup to ensure hipFree and
	 * hipDeviceSynchronize target the device where memory was allocated. */
	hipSetDevice(accum->device_id);
	if (accum->n_ctxs > 0) {
		for (int r = 0; r < accum->n_ctxs; r++) {
			if (accum->ctxs[r].regs0) free(accum->ctxs[r].regs0);
		}
	}
	if (accum->gpu_batch) ksw_extd2_gpu_batch_destroy(accum->gpu_batch);
	if (accum->km) km_destroy(accum->km);
	free(accum->ctxs);
	free(accum);
}

#endif /* HAVE_GPU_ALIGNMENT */

#ifdef HAVE_GPU_ALIGNMENT

#define ALIGN_BATCH_MAX_TASKS 4096

align_batch_t* align_batch_init(void* km, int64_t max_tasks)
{
	align_batch_t* batch = (align_batch_t*)kcalloc(km, 1, sizeof(align_batch_t));
	batch->tasks = (align_task_t*)kcalloc(km, max_tasks, sizeof(align_task_t));
	batch->max_tasks = max_tasks;
	batch->n_tasks = 0;
	batch->seq_buf_size = (size_t)max_tasks * 10240;
	batch->seq_buf = (uint8_t*)kmalloc(km, batch->seq_buf_size);
	batch->seq_buf_used = 0;
	return batch;
}

void align_batch_reset(align_batch_t* batch)
{
	batch->n_tasks = 0;
	batch->seq_buf_used = 0;
}

void align_batch_destroy(void* km, align_batch_t* batch)
{
	if (batch) {
		kfree(km, batch->tasks);
		kfree(km, batch->seq_buf);
		kfree(km, batch);
	}
}

static uint8_t* align_batch_alloc_seq(align_batch_t* batch, int len)
{
	if (batch->seq_buf_used + len > batch->seq_buf_size) {
		return NULL;
	}
	uint8_t* ptr = batch->seq_buf + batch->seq_buf_used;
	batch->seq_buf_used += len;
	return ptr;
}

static void align_batch_cpu_fallback_range(const mm_mapopt_t* opt,
    align_batch_t* batch, int start, int end)
{
	int8_t mat[25];
	ksw_gen_ts_mat(5, mat, opt->a, opt->b, opt->transition, opt->sc_ambi);

	for (int i = start; i < end; i++) {
		align_task_t* t = &batch->tasks[i];
		memset(&t->ez, 0, sizeof(ksw_extz_t));
		ksw_extd2_sse(NULL, t->qlen, t->qseq, t->tlen, t->tseq,
		    5, mat, opt->q, opt->e, opt->q2, opt->e2,
		    t->w, t->zdrop, t->end_bonus, t->flag, &t->ez);
	}
}

/**
 * Try adding tasks [start, end) to gpu_batch and flushing.
 * Returns: end on success, or the index of the first failed task.
 *          On flush failure after all adds succeed, returns start
 *          (all results are invalid).
 */
static int gpu_try_range(gpu_batch_t* gpu_batch, align_batch_t* batch,
    int start, int end)
{
	int batch_start = start;
	for (int i = start; i < end; i++) {
		align_task_t* t = &batch->tasks[i];
		int ret = ksw_extd2_gpu_batch_add(gpu_batch, t->qlen, t->qseq,
		    t->tlen, t->tseq, t->w, t->end_bonus, t->zdrop,
		    t->flag, &t->ez);
		if (ret < 0) return batch_start;
		if (ret == 1) batch_start = i;
	}
	int ret = ksw_extd2_gpu_batch_flush(gpu_batch);
	return (ret < 0) ? batch_start : end;
}

int align_batch_execute(const mm_mapopt_t* opt, gpu_batch_t* gpu_batch, align_batch_t* batch)
{
	int n_tasks = batch->n_tasks;

	/* Route tasks in order. An alignment whose traceback matrix can never fit
	 * the GPU batch's VRAM budget (e.g. whole-assembly asm5 contigs, whose `p`
	 * buffer is tens of GB) must NOT be added to the GPU batch: the device
	 * grow() would OOM, and under concurrency the recovery path can corrupt the
	 * heap. Run those on the CPU directly, and GPU-batch the runs in between. */
	int i = 0;
	while (i < n_tasks) {
		align_task_t* t = &batch->tasks[i];
		if (!ksw_extd2_gpu_task_fits(gpu_batch, t->qlen, t->tlen, t->w, t->flag)) {
			static std::atomic<bool> cap_hinted{false};
			bool first_hint = !cap_hinted.exchange(true);
			mm_log_warn("[GPU align] alignment too large for GPU VRAM budget "
				    "(qlen={}, tlen={}); falling back to CPU for this task{}",
			    t->qlen, t->tlen,
			    first_hint
				? " -- raise the per-thread GPU budget cap with "
				  "--gpu-batch-max-mem-cap (e.g. --gpu-batch-max-mem-cap 16G, "
				  "or lower the thread count) to keep larger alignments on the GPU"
				: "");
			align_batch_cpu_fallback_range(opt, batch, i, i + 1);
			i++;
			continue;
		}
		/* Gather a maximal run of GPU-eligible tasks. */
		int j = i;
		while (j < n_tasks) {
			align_task_t* tj = &batch->tasks[j];
			if (!ksw_extd2_gpu_task_fits(gpu_batch, tj->qlen, tj->tlen, tj->w, tj->flag))
				break;
			j++;
		}
		int done = gpu_try_range(gpu_batch, batch, i, j);
		if (done < j) {
			mm_log_warn("[GPU align] GPU batch failed at task {}, "
				    "falling back to CPU for {} tasks",
			    done, j - done);
			align_batch_cpu_fallback_range(opt, batch, done, j);
		}
		i = j;
	}

	return n_tasks;
}

/* Forward declaration */
mm_reg1_t* mm_insert_reg(const mm_reg1_t* r, int i, int* n_regs, mm_reg1_t* regs);

int mm_align1_prepare_state(void* km, const mm_mapopt_t* opt, const mm_idx_t* mi,
    int qlen, uint8_t* qseq0[2], mm_reg1_t* r,
    int n_a, mm128_t* a, int splice_flag, align_state_t* st)
{
	int is_sr = !!(opt->flag & MM_F_SR), is_splice = !!(opt->flag & MM_F_SPLICE);
	int32_t i, l;

	if (r->cnt == 0) return -1;

	st->r = r;
	memset(&st->r2, 0, sizeof(mm_reg1_t));
	st->rid = a[r->as].x << 1 >> 33;
	st->rev = a[r->as].x >> 63;
	st->dropped = 0;
	st->completed = 0;

	ksw_gen_ts_mat(5, st->mat, opt->a, opt->b, opt->transition, opt->sc_ambi);
	st->bw = (int)(opt->bw * 1.5 + 1.);
	st->bw_long = (int)(opt->bw_long * 1.5 + 1.);
	if (st->bw_long < st->bw) st->bw_long = st->bw;

	if (is_sr && !(mi->flag & MM_I_HPC)) {
		mm_max_stretch(r, a, &st->as1, &st->cnt1);
		st->rs = (int32_t)a[st->as1].x + 1 - (int32_t)(a[st->as1].y >> 32 & 0xff);
		st->qs = (int32_t)a[st->as1].y + 1 - (int32_t)(a[st->as1].y >> 32 & 0xff);
		st->re = (int32_t)a[st->as1 + st->cnt1 - 1].x + 1;
		st->qe = (int32_t)a[st->as1 + st->cnt1 - 1].y + 1;
	} else {
		if (!(opt->flag & MM_F_NO_END_FLT)) {
			if (is_splice)
				mm_fix_bad_ends_splice(km, opt, mi, r, st->mat, qlen, qseq0, a, &st->as1, &st->cnt1);
			else
				mm_fix_bad_ends(r, a, opt->bw, opt->min_chain_score * 2, &st->as1, &st->cnt1);
		} else
			st->as1 = r->as, st->cnt1 = r->cnt;
		mm_filter_bad_seeds(km, st->as1, st->cnt1, a, 10, 40, opt->max_gap >> 1, 10);
		mm_filter_bad_seeds_alt(km, st->as1, st->cnt1, a, 30, opt->max_gap >> 1);
		mm_adjust_minier(mi, qseq0, &a[st->as1], &st->rs, &st->qs);
		mm_adjust_minier(mi, qseq0, &a[st->as1 + st->cnt1 - 1], &st->re, &st->qe);
	}
	if (st->cnt1 <= 0) return -1;

	st->extra_flag = 0;
	if (is_splice) {
		if (splice_flag & MM_F_SPLICE_FOR) st->extra_flag |= st->rev ? KSW_EZ_SPLICE_REV : KSW_EZ_SPLICE_FOR;
		if (splice_flag & MM_F_SPLICE_REV) st->extra_flag |= st->rev ? KSW_EZ_SPLICE_FOR : KSW_EZ_SPLICE_REV;
		if (opt->flag & MM_F_SPLICE_FLANK) st->extra_flag |= KSW_EZ_SPLICE_FLANK;
		if (!(opt->flag & MM_F_SPLICE_OLD)) st->extra_flag |= KSW_EZ_SPLICE_CMPLX;
	}
	// Transition scoring (-b, e.g. map-iclr) requires matrix-based DP; mirrors mm_align_pair in align.c.
	if (opt->transition != 0 && opt->b != opt->transition) st->extra_flag |= KSW_EZ_GENERIC_SC;

	/* Compute region boundaries */
	if (is_sr) {
		st->qs0 = 0;
		st->qe0 = qlen;
		l = st->qs;
		l += l * opt->a + opt->end_bonus > opt->q ? (l * opt->a + opt->end_bonus - opt->q) / opt->e : 0;
		st->rs0 = st->rs - l > 0 ? st->rs - l : 0;
		l = qlen - st->qe;
		l += l * opt->a + opt->end_bonus > opt->q ? (l * opt->a + opt->end_bonus - opt->q) / opt->e : 0;
		st->re0 = st->re + l < (int32_t)mi->seq[st->rid].len ? st->re + l : mi->seq[st->rid].len;
	} else {
		st->rs0 = (int32_t)a[r->as].x + 1 - (int32_t)(a[r->as].y >> 32 & 0xff);
		st->qs0 = (int32_t)a[r->as].y + 1 - (int32_t)(a[r->as].y >> 32 & 0xff);
		if (st->rs0 < 0) st->rs0 = 0;
		st->rs1 = st->qs1 = 0;
		for (i = r->as - 1, l = 0; i >= 0 && a[i].x >> 32 == a[r->as].x >> 32; --i) {
			int32_t x = (int32_t)a[i].x + 1 - (int32_t)(a[i].y >> 32 & 0xff);
			int32_t y = (int32_t)a[i].y + 1 - (int32_t)(a[i].y >> 32 & 0xff);
			if (x < st->rs0 && y < st->qs0) {
				if (++l > opt->min_cnt) {
					l = st->rs0 - x > st->qs0 - y ? st->rs0 - x : st->qs0 - y;
					st->rs1 = st->rs0 - l;
					st->qs1 = st->qs0 - l;
					if (st->rs1 < 0) st->rs1 = 0;
					break;
				}
			}
		}
		if (st->qs > 0 && st->rs > 0) {
			l = st->qs < opt->max_gap ? st->qs : opt->max_gap;
			st->qs1 = st->qs1 > st->qs - l ? st->qs1 : st->qs - l;
			st->qs0 = st->qs0 < st->qs1 ? st->qs0 : st->qs1;
			l += l * opt->a > opt->q ? (l * opt->a - opt->q) / opt->e : 0;
			l = l < opt->max_gap ? l : opt->max_gap;
			l = l < st->rs ? l : st->rs;
			st->rs1 = st->rs1 > st->rs - l ? st->rs1 : st->rs - l;
			st->rs0 = st->rs0 < st->rs1 ? st->rs0 : st->rs1;
			st->rs0 = st->rs0 < st->rs ? st->rs0 : st->rs;
		} else
			st->rs0 = st->rs, st->qs0 = st->qs;

		st->re0 = (int32_t)a[r->as + r->cnt - 1].x + 1;
		st->qe0 = (int32_t)a[r->as + r->cnt - 1].y + 1;
		st->re1 = mi->seq[st->rid].len;
		st->qe1 = qlen;
		for (i = r->as + r->cnt, l = 0; i < n_a && a[i].x >> 32 == a[r->as].x >> 32; ++i) {
			int32_t x = (int32_t)a[i].x + 1;
			int32_t y = (int32_t)a[i].y + 1;
			if (x > st->re0 && y > st->qe0) {
				if (++l > opt->min_cnt) {
					l = x - st->re0 > y - st->qe0 ? x - st->re0 : y - st->qe0;
					st->re1 = st->re0 + l;
					st->qe1 = st->qe0 + l;
					break;
				}
			}
		}
		if (st->qe < qlen && st->re < (int32_t)mi->seq[st->rid].len) {
			l = qlen - st->qe < opt->max_gap ? qlen - st->qe : opt->max_gap;
			st->qe1 = st->qe1 < st->qe + l ? st->qe1 : st->qe + l;
			st->qe0 = st->qe0 > st->qe1 ? st->qe0 : st->qe1;
			l += l * opt->a > opt->q ? (l * opt->a - opt->q) / opt->e : 0;
			l = l < opt->max_gap ? l : opt->max_gap;
			l = l < (int32_t)mi->seq[st->rid].len - st->re ? l : mi->seq[st->rid].len - st->re;
			st->re1 = st->re1 < st->re + l ? st->re1 : st->re + l;
			st->re0 = st->re0 > st->re1 ? st->re0 : st->re1;
		} else
			st->re0 = st->re, st->qe0 = st->qe;
	}

	if (a[r->as].y & MM_SEED_SELF) {
		int max_ext = r->qs > r->rs ? r->qs - r->rs : r->rs - r->qs;
		if (r->rs - st->rs0 > max_ext) st->rs0 = r->rs - max_ext;
		if (r->qs - st->qs0 > max_ext) st->qs0 = r->qs - max_ext;
		max_ext = r->qe > r->re ? r->qe - r->re : r->re - r->qe;
		if (st->re0 - r->re > max_ext) st->re0 = r->re + max_ext;
		if (st->qe0 - r->qe > max_ext) st->qe0 = r->qe + max_ext;
	}

	if (st->re0 <= st->rs0) return -1;

	st->tseq = (uint8_t*)kmalloc(km, st->re0 - st->rs0);
	st->junc = (uint8_t*)kmalloc(km, st->re0 - st->rs0);

	return 0;
}

int mm_batch_add_left_ext(const mm_mapopt_t* opt, const mm_idx_t* mi,
    int qlen, uint8_t* qseq0[2], mm128_t* a,
    align_state_t* st, align_batch_t* batch, int reg_idx)
{
	uint8_t* qseq;
	int32_t qlen_ext, tlen_ext;

	(void)qlen;

	if (st->qs <= 0 || st->rs <= 0) return -1;

	qlen_ext = st->qs - st->qs0;
	tlen_ext = st->rs - st->rs0;

	
	/* Fall back to CPU SSE when tlen >> qlen + bw: the GPU anti-diagonal kernel's
	 * effective band is ~half of SSE's for the same bw, so highly asymmetric
	 * extensions (large indel ratio) zdrop prematurely on GPU. */
	if (tlen_ext > qlen_ext + st->bw) {
		uint8_t* tbuf_sse = (uint8_t*)malloc(tlen_ext);
		if (tbuf_sse) {
			if (opt->flag & MM_F_QSTRAND) {
				qseq = &qseq0[0][st->qs0];
				mm_idx_getseq2(mi, st->rev, st->rid, st->rs0, st->rs, tbuf_sse);
			} else {
				qseq = &qseq0[st->rev][st->qs0];
				mm_idx_getseq(mi, st->rid, st->rs0, st->rs, tbuf_sse);
			}
			mm_seq_rev(qlen_ext, qseq);
			mm_seq_rev(tlen_ext, tbuf_sse);
			ksw_extz_t ez;
			memset(&ez, 0, sizeof(ksw_extz_t));
			int zdrop = st->r->split_inv ? opt->zdrop_inv : opt->zdrop;
			int flag = st->extra_flag | KSW_EZ_EXTZ_ONLY | KSW_EZ_RIGHT | KSW_EZ_REV_CIGAR;
			if (opt->q == opt->q2 && opt->e == opt->e2)
				ksw_extz2_sse(NULL, qlen_ext, qseq, tlen_ext, tbuf_sse, 5, st->mat,
				    opt->q, opt->e, st->bw, zdrop, opt->end_bonus, flag, &ez);
			else
				ksw_extd2_sse(NULL, qlen_ext, qseq, tlen_ext, tbuf_sse, 5, st->mat,
				    opt->q, opt->e, opt->q2, opt->e2, st->bw, zdrop, opt->end_bonus, flag, &ez);
			mm_seq_rev(qlen_ext, qseq);
			free(tbuf_sse);
			align_task_t fake_t;
			memset(&fake_t, 0, sizeof(fake_t));
			fake_t.ez = ez;
			mm_process_left_ext_result(st, &fake_t);
			return 0;
		}
	}

	uint8_t* qbuf = align_batch_alloc_seq(batch, qlen_ext);
	uint8_t* tbuf = align_batch_alloc_seq(batch, tlen_ext);
	if (!qbuf || !tbuf) return -2;

	if (opt->flag & MM_F_QSTRAND) {
		qseq = &qseq0[0][st->qs0];
		mm_idx_getseq2(mi, st->rev, st->rid, st->rs0, st->rs, tbuf);
	} else {
		qseq = &qseq0[st->rev][st->qs0];
		mm_idx_getseq(mi, st->rid, st->rs0, st->rs, tbuf);
	}
	memcpy(qbuf, qseq, qlen_ext);

	mm_seq_rev(tlen_ext, tbuf);

	if (batch->n_tasks >= batch->max_tasks) return -2;
	align_task_t* t = &batch->tasks[batch->n_tasks++];
	t->task_type = ALIGN_TASK_LEFT_EXT;
	t->reg_idx = reg_idx;
	t->qlen = qlen_ext;
	t->tlen = tlen_ext;
	t->qseq = qbuf;
	t->tseq = tbuf;
	t->w = st->bw;
	t->end_bonus = opt->end_bonus;
	t->zdrop = st->r->split_inv ? opt->zdrop_inv : opt->zdrop;
	t->flag = st->extra_flag | KSW_EZ_EXTZ_ONLY | KSW_EZ_RIGHT | KSW_EZ_REV_CIGAR;
	t->rs = st->rs;
	t->qs = st->qs;
	t->rs0 = st->rs0;
	t->qs0 = st->qs0;
	t->need_rev = 1;
	memset(&t->ez, 0, sizeof(ksw_extz_t));

	return 0;
}

void mm_process_left_ext_result(align_state_t* st, align_task_t* t)
{
	ksw_extz_t* ez = &t->ez;

	if (ez->n_cigar > 0 && ez->cigar != NULL) {
		mm_append_cigar(st->r, ez->n_cigar, ez->cigar);
		st->r->p->dp_score += ez->max;
	}

	if (ez->reach_end) {
		st->rs1 = st->rs - (ez->mqe_t + 1);
		st->qs1 = st->qs0;
	} else if (ez->max_t >= 0 && ez->max_q >= 0) {
		st->rs1 = st->rs - (ez->max_t + 1);
		st->qs1 = st->qs - (ez->max_q + 1);
	} else {
		st->rs1 = st->rs;
		st->qs1 = st->qs;
	}

	if (st->rs1 < st->rs0) st->rs1 = st->rs0;
	if (st->qs1 < st->qs0) st->qs1 = st->qs0;
	if (st->rs1 < 0) st->rs1 = 0;
	if (st->qs1 < 0) st->qs1 = 0;

	if (ez->cigar) {
		free(ez->cigar);
		ez->cigar = NULL;
	}
}

void mm_region_gap_fill(void* km, const mm_mapopt_t* opt, const mm_idx_t* mi,
    int qlen, uint8_t* qseq0[2], mm128_t* a, align_state_t* st,
    ksw_extz_t* ez, int n_a, const mm_core_ctx_t* core_ctx)
{
	int is_sr = !!(opt->flag & MM_F_SR);
	int32_t i, j;
	int32_t rs = st->rs, qs = st->qs;
	int32_t re, qe, re1 = st->rs, qe1 = st->qs;
	uint8_t* qseq;

	(void)n_a;
	(void)qlen;

	for (i = is_sr ? st->cnt1 - 1 : 1; i < st->cnt1; ++i) {
		if ((a[st->as1 + i].y & (MM_SEED_IGNORE | MM_SEED_TANDEM)) && i != st->cnt1 - 1) continue;
		if (is_sr && !(mi->flag & MM_I_HPC)) {
			re = (int32_t)a[st->as1 + i].x + 1;
			qe = (int32_t)a[st->as1 + i].y + 1;
		} else
			mm_adjust_minier(mi, qseq0, &a[st->as1 + i], &re, &qe);
		re1 = re;
		qe1 = qe;

		if (i == st->cnt1 - 1 || (a[st->as1 + i].y & MM_SEED_LONG_JOIN) ||
		    (qe - qs >= opt->min_ksw_len && re - rs >= opt->min_ksw_len)) {
			int bw1 = st->bw_long, zdrop_code;
			if (a[st->as1 + i].y & MM_SEED_LONG_JOIN)
				bw1 = qe - qs > re - rs ? qe - qs : re - rs;

			if (opt->flag & MM_F_QSTRAND) {
				qseq = &qseq0[0][qs];
				mm_idx_getseq2(mi, st->rev, st->rid, rs, re, st->tseq);
			} else {
				qseq = &qseq0[st->rev][qs];
				mm_idx_getseq(mi, st->rid, rs, re, st->tseq);
			}
			mm_idx_bed_junc(mi, st->rid, rs, re, st->junc);

			if (is_sr) {
				if (qe - qs != re - rs) continue;
				int32_t max_gapped_score = (qe - qs - 2) * opt->a - 2 * (opt->q + opt->e);
				ksw_reset_extz(ez);
				for (j = 0, ez->score = 0; j < qe - qs; ++j) {
					if (qseq[j] >= 4 || st->tseq[j] >= 4)
						ez->score += opt->sc_ambi > 0 ? -opt->sc_ambi : opt->sc_ambi;
					else
						ez->score += qseq[j] == st->tseq[j] ? opt->a : -opt->b;
				}
				if (ez->score > max_gapped_score)
					ez->cigar = ksw_push_cigar(km, &ez->n_cigar, &ez->m_cigar, ez->cigar, MM_CIGAR_MATCH, qe - qs);
				else
					mm_align_pair(km, opt, qe - qs, qseq, re - rs, st->tseq, st->junc, st->mat, bw1, -1, opt->zdrop, st->extra_flag | KSW_EZ_APPROX_MAX, ez, core_ctx, NULL);
			} else {
				mm_align_pair(km, opt, qe - qs, qseq, re - rs, st->tseq, st->junc, st->mat, bw1, -1, opt->zdrop, st->extra_flag | KSW_EZ_APPROX_MAX, ez, core_ctx, NULL);
			}

			if ((zdrop_code = mm_test_zdrop(km, opt, qseq, st->tseq, ez->n_cigar, ez->cigar, st->mat)) != 0) {
				mm_align_pair(km, opt, qe - qs, qseq, re - rs, st->tseq, st->junc, st->mat, bw1, -1, zdrop_code == 2 ? opt->zdrop_inv : opt->zdrop, st->extra_flag, ez, core_ctx, NULL);
			}

			if (ez->n_cigar > 0)
				mm_append_cigar(st->r, ez->n_cigar, ez->cigar);

			if (ez->zdropped) {
				if (!st->r->p) {
					uint32_t capacity = sizeof(mm_extra_t) / 4;
					kroundup32(capacity);
					st->r->p = (mm_extra_t*)calloc(capacity, 4);
					st->r->p->capacity = capacity;
				}
				for (j = i - 1; j >= 0; --j)
					if ((int32_t)a[st->as1 + j].x <= rs + ez->max_t)
						break;
				st->dropped = 1;
				if (j < 0) j = 0;
				st->r->p->dp_score += ez->max;
				re1 = rs + (ez->max_t + 1);
				qe1 = qs + (ez->max_q + 1);
				if (st->cnt1 - (j + 1) >= opt->min_cnt) {
					mm_split_reg(st->r, &st->r2, st->as1 + j + 1 - st->r->as, qlen, a, !!(opt->flag & MM_F_QSTRAND));
					if (zdrop_code == 2) st->r2.split_inv = 1;
				}
				break;
			} else {
				st->r->p->dp_score += ez->score;
			}
			rs = re;
			qs = qe;
		}
	}

	st->re1 = re1;
	st->qe1 = qe1;
	st->rs = rs;
	st->qs = qs;
}

int mm_batch_collect_gaps(void* km, const mm_mapopt_t* opt, const mm_idx_t* mi,
    int qlen, uint8_t* qseq0[2], mm128_t* a, align_state_t* st,
    align_batch_t* batch, int reg_idx)
{
	int is_sr = !!(opt->flag & MM_F_SR);
	int32_t i;
	int32_t rs = st->rs, qs = st->qs;
	int32_t re, qe;
	uint8_t* qseq;
	int n_gaps = 0;

	(void)km;
	(void)qlen;

	st->gap_task_start = batch->n_tasks;
	st->gap_task_count = 0;

	if (is_sr) return 0;

	for (i = 1; i < st->cnt1; ++i) {
		if ((a[st->as1 + i].y & (MM_SEED_IGNORE | MM_SEED_TANDEM)) && i != st->cnt1 - 1) continue;
		mm_adjust_minier(mi, qseq0, &a[st->as1 + i], &re, &qe);

		if (i == st->cnt1 - 1 || (a[st->as1 + i].y & MM_SEED_LONG_JOIN) ||
		    (qe - qs >= opt->min_ksw_len && re - rs >= opt->min_ksw_len)) {
			int bw1 = st->bw_long;
			if (a[st->as1 + i].y & MM_SEED_LONG_JOIN)
				bw1 = qe - qs > re - rs ? qe - qs : re - rs;

			if (batch->n_tasks >= batch->max_tasks) return -1;

			int qlen_gap = qe - qs;
			int tlen_gap = re - rs;
			if (qlen_gap <= 0 || tlen_gap <= 0) {
				rs = re;
				qs = qe;
				continue;
			}

			uint8_t* qbuf = align_batch_alloc_seq(batch, qlen_gap);
			uint8_t* tbuf = align_batch_alloc_seq(batch, tlen_gap);
			if (!qbuf || !tbuf) return -1;

			if (opt->flag & MM_F_QSTRAND) {
				qseq = &qseq0[0][qs];
				mm_idx_getseq2(mi, st->rev, st->rid, rs, re, tbuf);
			} else {
				qseq = &qseq0[st->rev][qs];
				mm_idx_getseq(mi, st->rid, rs, re, tbuf);
			}
			memcpy(qbuf, qseq, qlen_gap);

			mm_seq_rev(qlen_gap, qbuf);

			align_task_t* t = &batch->tasks[batch->n_tasks++];
			t->task_type = ALIGN_TASK_GAP_FILL;
			t->reg_idx = reg_idx;
			t->qlen = qlen_gap;
			t->tlen = tlen_gap;
			t->qseq = qbuf;
			t->tseq = tbuf;
			t->w = bw1;
			t->end_bonus = -1;
			t->zdrop = opt->zdrop;
			t->flag = st->extra_flag | KSW_EZ_APPROX_MAX;
			t->rs = rs;
			t->qs = qs;
			t->re = re;
			t->qe = qe;
			t->gap_idx = n_gaps;
			t->anchor_idx = i;
			t->bw1 = bw1;
			t->is_last_gap = (i == st->cnt1 - 1) ? 1 : 0;
			memset(&t->ez, 0, sizeof(ksw_extz_t));

			n_gaps++;
			rs = re;
			qs = qe;
		}
	}

	st->gap_task_count = n_gaps;
	return n_gaps;
}

void mm_process_gap_fill_results(void* km, const mm_mapopt_t* opt, const mm_idx_t* mi,
    int qlen, uint8_t* qseq0[2], mm128_t* a,
    align_state_t* st, align_batch_t* batch, ksw_extz_t* ez_scratch, const mm_core_ctx_t* core_ctx)
{
	int is_sr = !!(opt->flag & MM_F_SR);
	int32_t rs = st->rs, qs = st->qs;
	int32_t re1 = st->rs, qe1 = st->qs;
	uint8_t* qseq;
	int32_t j;

	if (is_sr) {
		ksw_extz_t ez_local;
		memset(&ez_local, 0, sizeof(ksw_extz_t));
		mm_region_gap_fill(km, opt, mi, qlen, qseq0, a, st, ez_scratch ? ez_scratch : &ez_local, 0, core_ctx);
		if (!ez_scratch && ez_local.cigar) free(ez_local.cigar);
		return;
	}

	for (int g = 0; g < st->gap_task_count; ++g) {
		align_task_t* t = &batch->tasks[st->gap_task_start + g];
		ksw_extz_t* ez = &t->ez;
		int zdrop_code;

		re1 = t->re;
		qe1 = t->qe;

		if (opt->flag & MM_F_QSTRAND) {
			qseq = &qseq0[0][t->qs];
		} else {
			qseq = &qseq0[st->rev][t->qs];
		}

		int qlen_gap = t->qe - t->qs;
		int tlen_gap = t->re - t->rs;

		if (opt->max_sw_mat > 0 && (int64_t)tlen_gap * qlen_gap > opt->max_sw_mat) {
			if (ez->cigar) {
				free(ez->cigar);
				ez->cigar = NULL;
			}
			ksw_reset_extz(ez);
			ez->zdropped = 1;
		}

		/* If GPU kernel failed, redo with CPU SSE */
		if (ez->n_cigar == 0 && (ez->zdropped || ez->score == (int32_t)0x80000000)) {
			// int qlen_gap = t->qe - t->qs;
			// int tlen_gap = t->re - t->rs;
			if (ez->cigar) {
				free(ez->cigar);
				ez->cigar = NULL;
			}
			ksw_reset_extz(ez);
			if (opt->max_sw_mat > 0 && (int64_t)tlen_gap * qlen_gap > opt->max_sw_mat) {
				ez->zdropped = 1;
			} else {
				if (opt->q == opt->q2 && opt->e == opt->e2)
					ksw_extz2_sse(NULL, qlen_gap, qseq, tlen_gap, t->tseq, 5, st->mat,
					    opt->q, opt->e, st->bw_long, opt->zdrop, -1, st->extra_flag | KSW_EZ_APPROX_MAX, ez);
				else
					ksw_extd2_sse(NULL, qlen_gap, qseq, tlen_gap, t->tseq, 5, st->mat,
					    opt->q, opt->e, opt->q2, opt->e2, st->bw_long, opt->zdrop, -1, st->extra_flag | KSW_EZ_APPROX_MAX, ez);
			}
		}

		zdrop_code = mm_test_zdrop(km, opt, qseq, t->tseq, ez->n_cigar, ez->cigar, st->mat);

		if (zdrop_code != 0) {
			uint32_t* orig_cigar = ez->cigar;
			int orig_n_cigar = ez->n_cigar;
			int orig_m_cigar = ez->m_cigar;
			int orig_score = ez->score;
			ez->cigar = NULL;
			ez->n_cigar = 0;
			ez->m_cigar = 0;

			int qlen_gap = t->qe - t->qs;
			int tlen_gap = t->re - t->rs;
			int bw1 = st->bw_long;
			int zdrop_val = (zdrop_code == 2) ? opt->zdrop_inv : opt->zdrop;
			int redo_flag = st->extra_flag & ~KSW_EZ_APPROX_MAX;

			if (opt->max_sw_mat > 0 && (int64_t)tlen_gap * qlen_gap > opt->max_sw_mat) {
				ksw_reset_extz(ez);
				ez->zdropped = 1;
			} else {
				if (opt->q == opt->q2 && opt->e == opt->e2)
					ksw_extz2_sse(NULL, qlen_gap, qseq, tlen_gap, t->tseq, 5, st->mat,
					    opt->q, opt->e, bw1, zdrop_val, -1, redo_flag, ez);
				else
					ksw_extd2_sse(NULL, qlen_gap, qseq, tlen_gap, t->tseq, 5, st->mat,
					    opt->q, opt->e, opt->q2, opt->e2, bw1, zdrop_val, -1, redo_flag, ez);
			}

			if (ez->score == (int32_t)0x80000000 || ez->score == INT32_MIN) {
				if (ez->cigar) {
					free(ez->cigar);
					ez->cigar = NULL;
				}
				ez->cigar = orig_cigar;
				ez->n_cigar = orig_n_cigar;
				ez->m_cigar = orig_m_cigar;
				ez->score = orig_score;
				ez->zdropped = 0;
			} else {
				if (orig_cigar) free(orig_cigar);
			}
		}

		if (ez->n_cigar > 0)
			mm_append_cigar(st->r, ez->n_cigar, ez->cigar);

		if (ez->zdropped) {
			if (!st->r->p) {
				uint32_t capacity = sizeof(mm_extra_t) / 4;
				kroundup32(capacity);
				st->r->p = (mm_extra_t*)calloc(capacity, 4);
				st->r->p->capacity = capacity;
			}
			for (j = t->anchor_idx - 1; j >= 0; --j)
				if ((int32_t)a[st->as1 + j].x <= t->rs + ez->max_t)
					break;
			st->dropped = 1;
			if (j < 0) j = 0;
			st->r->p->dp_score += ez->max;
			re1 = t->rs + (ez->max_t + 1);
			qe1 = t->qs + (ez->max_q + 1);
			if (st->cnt1 - (j + 1) >= opt->min_cnt) {
				mm_split_reg(st->r, &st->r2, st->as1 + j + 1 - st->r->as, qlen, a, !!(opt->flag & MM_F_QSTRAND));
				if (zdrop_code == 2) st->r2.split_inv = 1;
			}
			break;
		} else {
			if (st->r->p)
				st->r->p->dp_score += ez->score;
		}

		if (ez->cigar) {
			free(ez->cigar);
			ez->cigar = NULL;
		}

		rs = t->re;
		qs = t->qe;
	}

	st->re1 = re1;
	st->qe1 = qe1;
	st->rs = rs;
	st->qs = qs;

	st->gap_task_count = 0;
}

int mm_batch_add_right_ext(const mm_mapopt_t* opt, const mm_idx_t* mi,
    int qlen, uint8_t* qseq0[2],
    align_state_t* st, align_batch_t* batch, int reg_idx)
{
	uint8_t* qseq;
	int32_t qe = st->qs;
	int32_t re = st->rs;
	int32_t qlen_ext, tlen_ext;

	(void)qlen;

	if (st->dropped || qe >= st->qe0 || re >= st->re0) return -1;

	qlen_ext = st->qe0 - qe;
	tlen_ext = st->re0 - re;

	uint8_t* qbuf = align_batch_alloc_seq(batch, qlen_ext);
	uint8_t* tbuf = align_batch_alloc_seq(batch, tlen_ext);
	if (!qbuf || !tbuf) return -2;

	if (opt->flag & MM_F_QSTRAND) {
		qseq = &qseq0[0][qe];
		mm_idx_getseq2(mi, st->rev, st->rid, re, st->re0, tbuf);
	} else {
		qseq = &qseq0[st->rev][qe];
		mm_idx_getseq(mi, st->rid, re, st->re0, tbuf);
	}
	memcpy(qbuf, qseq, qlen_ext);

	mm_seq_rev(qlen_ext, qbuf);

	if (batch->n_tasks >= batch->max_tasks) return -2;
	align_task_t* t = &batch->tasks[batch->n_tasks++];
	t->task_type = ALIGN_TASK_RIGHT_EXT;
	t->reg_idx = reg_idx;
	t->qlen = qlen_ext;
	t->tlen = tlen_ext;
	t->qseq = qbuf;
	t->tseq = tbuf;
	t->w = st->bw;
	t->end_bonus = opt->end_bonus;
	t->zdrop = opt->zdrop;
	t->flag = st->extra_flag | KSW_EZ_EXTZ_ONLY;
	t->rs = re;
	t->qs = qe;
	t->rs0 = st->rs0;
	t->qs0 = st->qs0;
	t->need_rev = 0;
	memset(&t->ez, 0, sizeof(ksw_extz_t));

	return 0;
}

void mm_process_right_ext_result(int qlen, align_state_t* st, align_task_t* t)
{
	ksw_extz_t* ez = &t->ez;
	int32_t qe = t->qs;
	int32_t re = t->rs;

	if (ez->n_cigar > 0 && ez->cigar != NULL) {
		mm_append_cigar(st->r, ez->n_cigar, ez->cigar);
		st->r->p->dp_score += ez->max;
	}

	if (ez->reach_end) {
		st->re1 = re + (ez->mqe_t + 1);
		st->qe1 = st->qe0;
	} else if (ez->max_t >= 0 && ez->max_q >= 0) {
		st->re1 = re + (ez->max_t + 1);
		st->qe1 = qe + (ez->max_q + 1);
	} else {
		st->re1 = re;
		st->qe1 = qe;
	}

	if (st->re1 > st->re0) st->re1 = st->re0;
	if (st->qe1 > st->qe0) st->qe1 = st->qe0;
	if (st->qe1 > qlen) st->qe1 = qlen;

	if (ez->cigar) {
		free(ez->cigar);
		ez->cigar = NULL;
	}
}

void mm_region_finalize(void* km, const mm_mapopt_t* opt, const mm_idx_t* mi,
    int qlen, uint8_t* qseq0[2], align_state_t* st)
{
	uint8_t* qseq;

	if (st->rs1 < 0) st->rs1 = 0;
	if (st->qs1 < 0) st->qs1 = 0;
	if (st->re1 <= st->rs1) st->re1 = st->rs1 + 1;
	if (st->qe1 <= st->qs1) st->qe1 = st->qs1 + 1;
	if (st->qe1 > qlen) st->qe1 = qlen;
	if (st->qs1 >= qlen) st->qs1 = qlen > 0 ? qlen - 1 : 0;

	st->r->rs = st->rs1;
	st->r->re = st->re1;
	if (!st->rev || (opt->flag & MM_F_QSTRAND)) {
		st->r->qs = st->qs1;
		st->r->qe = st->qe1;
	} else {
		st->r->qs = qlen - st->qe1;
		st->r->qe = qlen - st->qs1;
	}

	if (st->r->p && st->re1 > st->rs1 && st->qe1 > st->qs1) {
		int64_t tseq_size = st->re0 - st->rs0;
		int64_t needed_size = st->re1 - st->rs1;
		if (needed_size > tseq_size) {
			kfree(km, st->tseq);
			st->tseq = (uint8_t*)kmalloc(km, needed_size);
		}
		if (opt->flag & MM_F_QSTRAND) {
			mm_idx_getseq2(mi, st->r->rev, st->rid, st->rs1, st->re1, st->tseq);
			qseq = &qseq0[0][st->qs1];
		} else {
			mm_idx_getseq(mi, st->rid, st->rs1, st->re1, st->tseq);
			qseq = &qseq0[st->r->rev][st->qs1];
		}
		mm_update_extra(st->r, qseq, st->tseq, st->mat, opt->q, opt->e, opt->flag & MM_F_EQX, !(opt->flag & MM_F_SR));
		if (st->rev && st->r->p->trans_strand)
			st->r->p->trans_strand ^= 3;
	}

	st->completed = 1;
}

void mm_state_free(void* km, align_state_t* st)
{
	if (st->tseq) {
		kfree(km, st->tseq);
		st->tseq = NULL;
	}
	if (st->junc) {
		kfree(km, st->junc);
		st->junc = NULL;
	}
}

mm_reg1_t* mm_align_batched(void* km, const mm_mapopt_t* opt, const mm_idx_t* mi,
    int qlen, uint8_t* qseq0[2], int* n_regs_,
    mm_reg1_t* regs, mm128_t* a, int n_a,
    gpu_batch_t* gpu_batch, align_batch_t* batch,
    int splice_flag, const mm_core_ctx_t* core_ctx)
{
	int i, j, n_regs = *n_regs_;
	align_state_t* states;
	ksw_extz_t ez;

	if (n_regs == 0) return regs;

	states = (align_state_t*)kcalloc(km, n_regs, sizeof(align_state_t));
	memset(&ez, 0, sizeof(ksw_extz_t));

	/* Phase 1: Prepare all regions */
	for (i = 0; i < n_regs; ++i) {
		if (mm_align1_prepare_state(km, opt, mi, qlen, qseq0, &regs[i],
			n_a, a, splice_flag, &states[i]) < 0) {
			states[i].completed = 1;
		}
	}

	/* Phase 2: Batch LEFT_EXT */
	align_batch_reset(batch);
	for (i = 0; i < n_regs; ++i) {
		if (states[i].completed) continue;
		int ret = mm_batch_add_left_ext(opt, mi, qlen, qseq0, a, &states[i], batch, i);
		if (ret == -2) {
			align_batch_execute(opt, gpu_batch, batch);
			for (j = 0; j < batch->n_tasks; ++j) {
				align_task_t* t = &batch->tasks[j];
				if (t->task_type == ALIGN_TASK_LEFT_EXT)
					mm_process_left_ext_result(&states[t->reg_idx], t);
			}
			align_batch_reset(batch);
			mm_batch_add_left_ext(opt, mi, qlen, qseq0, a, &states[i], batch, i);
		} else if (ret == -1) {
			states[i].rs1 = states[i].rs;
			states[i].qs1 = states[i].qs;
		}
	}
	if (batch->n_tasks > 0) {
		align_batch_execute(opt, gpu_batch, batch);
		for (j = 0; j < batch->n_tasks; ++j) {
			align_task_t* t = &batch->tasks[j];
			if (t->task_type == ALIGN_TASK_LEFT_EXT)
				mm_process_left_ext_result(&states[t->reg_idx], t);
		}
	}

	/* Phase 4: GAP_FILL */
#if GPU_GAP_FILL_ENABLED
	align_batch_reset(batch);
	for (i = 0; i < n_regs; ++i) {
		if (states[i].completed) continue;
		states[i].re1 = states[i].rs;
		states[i].qe1 = states[i].qs;
		int ret = mm_batch_collect_gaps(km, opt, mi, qlen, qseq0, a, &states[i], batch, i);
		if (ret == -1) {
			align_batch_execute(opt, gpu_batch, batch);
			for (j = 0; j < i; ++j) {
				if (!states[j].completed && states[j].gap_task_count > 0)
					mm_process_gap_fill_results(km, opt, mi, qlen, qseq0, a, &states[j], batch, NULL, core_ctx);
			}
			align_batch_reset(batch);
			mm_batch_collect_gaps(km, opt, mi, qlen, qseq0, a, &states[i], batch, i);
		}
	}
	if (batch->n_tasks > 0) {
		align_batch_execute(opt, gpu_batch, batch);
		for (i = 0; i < n_regs; ++i) {
			if (states[i].completed || states[i].gap_task_count == 0) continue;
			mm_process_gap_fill_results(km, opt, mi, qlen, qseq0, a, &states[i], batch, NULL, core_ctx);
		}
	}
#else
	for (i = 0; i < n_regs; ++i) {
		if (states[i].completed) continue;
		states[i].re1 = states[i].rs;
		states[i].qe1 = states[i].qs;
		mm_region_gap_fill(km, opt, mi, qlen, qseq0, a, &states[i], &ez, 0, core_ctx);
	}
#endif

	/* Phase 5: Batch RIGHT_EXT */
	align_batch_reset(batch);
	for (i = 0; i < n_regs; ++i) {
		if (states[i].completed || states[i].dropped) continue;
		int ret = mm_batch_add_right_ext(opt, mi, qlen, qseq0, &states[i], batch, i);
		if (ret == -2) {
			align_batch_execute(opt, gpu_batch, batch);
			for (j = 0; j < batch->n_tasks; ++j) {
				align_task_t* t = &batch->tasks[j];
				if (t->task_type == ALIGN_TASK_RIGHT_EXT)
					mm_process_right_ext_result(qlen, &states[t->reg_idx], t);
			}
			align_batch_reset(batch);
			mm_batch_add_right_ext(opt, mi, qlen, qseq0, &states[i], batch, i);
		}
	}
	if (batch->n_tasks > 0) {
		align_batch_execute(opt, gpu_batch, batch);
		for (j = 0; j < batch->n_tasks; ++j) {
			align_task_t* t = &batch->tasks[j];
			if (t->reg_idx < 0 || t->reg_idx >= n_regs) continue;
			if (t->task_type == ALIGN_TASK_RIGHT_EXT)
				mm_process_right_ext_result(qlen, &states[t->reg_idx], t);
		}
	}

	/* Phase 7: Finalize all regions */
	for (i = 0; i < n_regs; ++i) {
		if (!states[i].completed)
			mm_region_finalize(km, opt, mi, qlen, qseq0, &states[i]);
		mm_state_free(km, &states[i]);
	}

	for (i = n_regs - 1; i >= 0; --i) {
		if (states[i].r2.cnt > 0)
			regs = mm_insert_reg(&states[i].r2, i, &n_regs, regs);
	}

	{
		mm_mapopt_t opt_cpu = *opt;
		opt_cpu.flag &= ~MM_F_GPU_ALIGN;
		for (i = 0; i < n_regs; ++i) {
			if (regs[i].p == NULL && regs[i].cnt > 0 && (regs[i].split & 2)) {
				mm_reg1_t r2_of_r2;
				mm_align1(km, &opt_cpu, mi, qlen, qseq0, &regs[i], &r2_of_r2, n_a, a, &ez, splice_flag, core_ctx, NULL, 0);
				if (r2_of_r2.cnt > 0)
					regs = mm_insert_reg(&r2_of_r2, i, &n_regs, regs);
			}
		}
	}

	for (i = 1; i < n_regs; ++i) {
		if (regs[i].split_inv && !(opt->flag & MM_F_NO_INV)) {
			mm_reg1_t r2;
			if (mm_align1_inv(km, opt, mi, qlen, qseq0, &regs[i - 1], &regs[i], &r2, &ez, core_ctx)) {
				regs = mm_insert_reg(&r2, i, &n_regs, regs);
				++i;
			}
		}
	}

	*n_regs_ = n_regs;

	kfree(km, states);
	if (ez.cigar) kfree(km, ez.cigar);

	return regs;
}

#endif /* HAVE_GPU_ALIGNMENT */
