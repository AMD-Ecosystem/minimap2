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
 * @file ksw2_gpu_batch.cpp
 * @brief Implementation of GPU batch alignment interface
 */

#include <hip/hip_runtime.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>

#include "gpu/ksw2_gpu_batch.h"
#include "gpu/hip_raii.h"
#include "mm_log.h"
#include "ksw2.h"

extern "C" void ksw_gen_simple_mat(int m, int8_t* mat, int8_t a, int8_t b, int8_t sc_ambi);
extern "C" void ksw_gen_ts_mat(int m, int8_t* mat, int8_t a, int8_t b, int8_t transition, int8_t sc_ambi);

/* External kernel declaration (defined in ksw2_extd2_gpu_multialign.cpp) */
__global__ void ksw_extd2_gpu_multialign(
    int8_t* u, int8_t* v, int8_t* x, int8_t* y, int8_t* x2, int8_t* y2,
    uint8_t* p, int32_t* H, int* off, int* off_end,
    const int* qlen, const uint8_t* qseq, const int* tlen, const uint8_t* tseq,
    const int8_t* mat,
    int8_t q, int8_t e, int8_t q2, int8_t e2,
    const int* w, int zdrop, int end_bonus, int flag,
    ksw_extz_t* ez,
    const uint32_t* tlen_ps, const uint32_t* qlen_ps, const uint64_t* p_ps, const uint32_t* off_ps,
    int8_t min_sc);

/* External backtracking kernel declaration */
__global__ void ksw_backtrack_batch_kernel(
    const uint8_t* p,
    const int* off,
    const int* off_end,
    const uint64_t* p_ps,
    const uint32_t* off_ps,
    const ksw_extz_t* ez,
    const int* qlen,
    const int* tlen,
    const int* w,
    int flag,
    int end_bonus,
    uint32_t* cigar_buf,
    const uint32_t* cigar_off,
    int* n_cigar_out,
    int* reach_end_out,
    int num_align);

static inline bool hip_check_impl(hipError_t err, const char* file, int line, const char* function)
{
	if (err == hipSuccess) return true;
	mm_logger_t* _lg = mm_tl_logger();
	_lg->log_warn(_lg, file, line, function,
	    fmt::format("[GPU Batch] HIP error {}: {}", (int)err, hipGetErrorString(err)).c_str());
	return false;
}

#define HIP_OK(x) hip_check_impl((x), __FILE__, __LINE__, __FUNCTION__)

/**
 * Per-alignment task stored in the batch
 */
struct batch_task_t {
	std::vector<uint8_t> qseq;
	std::vector<uint8_t> tseq;
	int qlen;
	int tlen;
	int w;
	int end_bonus;
	int zdrop;
	int flag;
	ksw_extz_t* ez;
};

/**
 * Internal batch state structure (opaque from C side)
 */
struct gpu_batch_s {
	int8_t q, e, q2, e2;
	int w, zdrop;
	int8_t mat[25];

	hip_stream stream;
	hipMemPool_t mem_pool = nullptr;

	hip_device_buf<int8_t> mat_d;
	hip_device_buf<ksw_extz_t> ez_d;
	hip_device_buf<int> qlen_d;
	hip_device_buf<int> tlen_d;
	hip_device_buf<int> w_d;
	hip_device_buf<uint32_t> qlen_ps_d;
	hip_device_buf<uint32_t> tlen_ps_d;
	hip_device_buf<uint64_t> p_ps_d;
	hip_device_buf<uint8_t> qseq_d;
	hip_device_buf<uint8_t> tseq_d;
	hip_device_buf<uint8_t> p_d;
	hip_device_buf<int32_t> H_d;
	hip_device_buf<int> off_d;
	hip_device_buf<int8_t> u_d, v_d, x_d, y_d, x2_d, y2_d;
	hip_device_buf<int> off_r_d;
	hip_device_buf<uint32_t> off_ps_d;

	/* Persistent backtrack temporaries (reused across flushes) */
	hip_device_buf<uint32_t> cigar_buf_d;
	hip_device_buf<uint32_t> cigar_off_d;
	hip_device_buf<int> n_cigar_d;
	hip_device_buf<int> reach_end_d;

	hip_pinned_buf<int> h_qlen, h_tlen, h_w;
	hip_pinned_buf<uint32_t> h_qlen_ps, h_tlen_ps;
	hip_pinned_buf<uint64_t> h_p_ps;
	hip_pinned_buf<uint32_t> h_off_ps;
	hip_pinned_buf<ksw_extz_t> h_ez;
	hip_pinned_buf<uint8_t> h_qseq, h_tseq;
	hip_pinned_buf<uint8_t> h_p;
	hip_pinned_buf<int> h_off, h_off_r;

	/* Pre-allocated per-flush vectors (reused across flushes) */
	std::vector<uint32_t> qlen_rounded_v;
	std::vector<uint32_t> tlen_rounded_v;
	std::vector<uint64_t> p_rounded;
	std::vector<uint32_t> off_v;
	std::vector<uint32_t> cigar_max;
	std::vector<uint32_t> cigar_offsets;
	std::vector<int> n_cigar_host;
	std::vector<int> reach_end_host;
	hip_pinned_buf<uint32_t> cigar_host;

	size_t avail_mem = 0;
	size_t current_mem = 0;
	int max_alignments = 0;

	std::vector<batch_task_t> tasks;

	void bind_all()
	{
		hipStream_t s = stream.get();
		hipMemPool_t p = mem_pool;
		mat_d.bind(s, p);
		ez_d.bind(s, p);
		qlen_d.bind(s, p);
		tlen_d.bind(s, p);
		w_d.bind(s, p);
		qlen_ps_d.bind(s, p);
		tlen_ps_d.bind(s, p);
		p_ps_d.bind(s, p);
		qseq_d.bind(s, p);
		tseq_d.bind(s, p);
		p_d.bind(s, p);
		H_d.bind(s, p);
		off_d.bind(s, p);
		u_d.bind(s, p);
		v_d.bind(s, p);
		x_d.bind(s, p);
		y_d.bind(s, p);
		x2_d.bind(s, p);
		y2_d.bind(s, p);
		off_r_d.bind(s, p);
		off_ps_d.bind(s, p);
		cigar_buf_d.bind(s, p);
		cigar_off_d.bind(s, p);
		n_cigar_d.bind(s, p);
		reach_end_d.bind(s, p);
	}
};

/**
 * Calculate GPU memory needed for one alignment
 */
static size_t estimate_alignment_memory(int qlen, int tlen, int w, int flag)
{
	int w_tmp = w;
	if (w < 0) w_tmp = (tlen > qlen) ? tlen : qlen;

	int tlen_ = (tlen + 15) / 16 + 1;
	int qlen_ = (qlen + 15) / 16 + 1;
	int n_col_ = (qlen < tlen) ? qlen : tlen;
	n_col_ = ((n_col_ < w_tmp + 1 ? n_col_ : w_tmp + 1) + 15) / 16 + 1;

	size_t mem = 0;
	mem += (tlen_ * 16) * sizeof(int32_t); // H
	mem += (qlen_ * 16) * sizeof(uint8_t); // qseq
	mem += (tlen_ * 16) * sizeof(uint8_t); // tseq
	mem += (tlen_ * 16) * 7 * sizeof(uint8_t); // u, v, x, y, x2, y2, s
	mem += (((uint64_t)(qlen + tlen - 1)) * n_col_ + 1) * 16 * sizeof(uint8_t); // p
	mem += 4 * sizeof(uint64_t); // prefix sums
	if (!(flag & KSW_EZ_SCORE_ONLY)) {
		mem += (qlen + tlen - 1) * 2 * sizeof(int); // off, off_end
		mem += (uint64_t)(qlen + tlen) * sizeof(uint32_t); // CIGAR buffer
	}

	return mem;
}

extern "C" {

gpu_batch_t* ksw_extd2_gpu_batch_init(size_t avail_mem, int max_alignments,
    int8_t q, int8_t e, int8_t q2, int8_t e2,
    int w, int zdrop, int8_t a, int8_t b, int8_t transition, int8_t sc_ambi,
    void* stream, void* mem_pool)
{
	auto batch = std::make_unique<gpu_batch_t>();

	batch->q = q;
	batch->e = e;
	batch->q2 = q2;
	batch->e2 = e2;
	batch->w = w;
	batch->zdrop = zdrop;
	batch->mem_pool = (hipMemPool_t)mem_pool;
	batch->max_alignments = max_alignments;

	ksw_gen_ts_mat(5, batch->mat, a, b, transition, sc_ambi);

	int ma = batch->max_alignments;
	batch->avail_mem = avail_mem;
	batch->current_mem = 25 * sizeof(int8_t) + ma * sizeof(ksw_extz_t);

	if (stream) {
		batch->stream.adopt((hipStream_t)stream);
	} else {
		if (!HIP_OK(batch->stream.create()))
			return NULL;
	}

	if (!batch->mem_pool)
		hipDeviceGetDefaultMemPool(&batch->mem_pool, 0);

	batch->bind_all();
	batch->tasks.reserve(1024);

	/* Pre-allocate flush vectors */
	batch->qlen_rounded_v.reserve(ma);
	batch->tlen_rounded_v.reserve(ma);
	batch->p_rounded.reserve(ma);
	batch->off_v.reserve(ma);
	batch->cigar_max.reserve(ma);
	batch->cigar_offsets.reserve(ma + 1);
	batch->n_cigar_host.reserve(ma);
	batch->reach_end_host.reserve(ma);

	if (!HIP_OK(batch->mat_d.grow(25)) ||
	    !HIP_OK(batch->ez_d.grow(ma)) ||
	    !HIP_OK(batch->qlen_d.grow(ma)) ||
	    !HIP_OK(batch->tlen_d.grow(ma)) ||
	    !HIP_OK(batch->w_d.grow(ma)) ||
	    !HIP_OK(batch->qlen_ps_d.grow(ma)) ||
	    !HIP_OK(batch->tlen_ps_d.grow(ma)) ||
	    !HIP_OK(batch->p_ps_d.grow(ma)))
		return NULL;

	if (!HIP_OK(hipMemcpyAsync(batch->mat_d.get(), batch->mat, 25 * sizeof(int8_t),
		hipMemcpyHostToDevice, batch->stream)))
		return NULL;

	if (!HIP_OK(batch->h_qlen.alloc(ma)) ||
	    !HIP_OK(batch->h_tlen.alloc(ma)) ||
	    !HIP_OK(batch->h_w.alloc(ma)) ||
	    !HIP_OK(batch->h_qlen_ps.alloc(ma)) ||
	    !HIP_OK(batch->h_tlen_ps.alloc(ma)) ||
	    !HIP_OK(batch->h_p_ps.alloc(ma)) ||
	    !HIP_OK(batch->h_off_ps.alloc(ma)) ||
	    !HIP_OK(batch->h_ez.alloc(ma)))
		return NULL;

	if (!HIP_OK(hipStreamSynchronize(batch->stream)))
		return NULL;

	return batch.release();
}

int ksw_extd2_gpu_batch_add(gpu_batch_t* batch,
    int qlen, const uint8_t* qseq,
    int tlen, const uint8_t* tseq,
    int w, int end_bonus, int zdrop, int flag,
    ksw_extz_t* ez)
{
	if (!batch || !qseq || !tseq || !ez || qlen <= 0 || tlen <= 0) {
		return -1;
	}

	size_t task_mem = estimate_alignment_memory(qlen, tlen, w, flag);

	int flushed = 0;

	if (!batch->tasks.empty()) {
		const batch_task_t& first = batch->tasks.front();
		if (first.flag != flag || first.end_bonus != end_bonus || first.zdrop != zdrop) {
			int ret = ksw_extd2_gpu_batch_flush(batch);
			if (ret < 0) return -1;
			flushed = 1;
		}
	}

	if ((batch->current_mem + task_mem > batch->avail_mem ||
		(int)batch->tasks.size() >= batch->max_alignments) &&
	    !batch->tasks.empty()) {
		int ret = ksw_extd2_gpu_batch_flush(batch);
		if (ret < 0) return -1;
		flushed = 1;
	}

	batch_task_t task;
	task.qlen = qlen;
	task.tlen = tlen;
	task.w = w;
	task.end_bonus = end_bonus;
	task.zdrop = zdrop;
	task.flag = flag;
	task.ez = ez;
	task.qseq.assign(qseq, qseq + qlen);
	task.tseq.assign(tseq, tseq + tlen);

	batch->tasks.push_back(std::move(task));
	batch->current_mem += task_mem;

	return flushed ? 1 : 0;
}

int ksw_extd2_gpu_task_fits(gpu_batch_t* batch, int qlen, int tlen, int w, int flag)
{
	if (!batch || qlen <= 0 || tlen <= 0)
		return 0;
	size_t task_mem = estimate_alignment_memory(qlen, tlen, w, flag);
	return task_mem <= batch->avail_mem ? 1 : 0;
}

/* Diagnostics for a batch that just failed on the GPU (e.g. HIP error 700).
 * The preceding HIP_OK warning already names the failing file:line; this adds
 * the batch shape (sizes, the largest task's traceback-matrix footprint, and
 * VRAM) so an oversized / size-overflowing alignment can be distinguished from a
 * genuine kernel fault. Cheap and only runs on the error path. */
static void log_batch_failure_diag(gpu_batch_t* batch)
{
	uint32_t n = batch ? (uint32_t)batch->tasks.size() : 0;
	if (n == 0) {
		mm_log_warn("[GPU align] batch failed with no tasks");
		return;
	}
	int max_q = 0, max_t = 0, max_w = 0;
	uint64_t sum_qt = 0, max_p = 0, sum_p = 0;
	uint32_t big_i = 0;
	for (uint32_t i = 0; i < n; i++) {
		const batch_task_t& tk = batch->tasks[i];
		if (tk.qlen > max_q) max_q = tk.qlen;
		if (tk.tlen > max_t) max_t = tk.tlen;
		if (tk.w > max_w) max_w = tk.w;
		sum_qt += (uint64_t)tk.qlen + (uint64_t)tk.tlen;
		int wi = tk.w;
		if (wi < 0) wi = (tk.tlen > tk.qlen) ? tk.tlen : tk.qlen;
		int n_col = (tk.qlen < tk.tlen) ? tk.qlen : tk.tlen;
		n_col = ((n_col < wi + 1 ? n_col : wi + 1) + 15) / 16 + 1;
		uint64_t p = ((uint64_t)(tk.qlen + tk.tlen - 1) * (uint64_t)n_col + 1) * 16;
		sum_p += p;
		if (p > max_p) {
			max_p = p;
			big_i = i;
		}
	}
	size_t vfree = 0, vtotal = 0;
	(void)hipMemGetInfo(&vfree, &vtotal);
	mm_log_warn("[GPU align] batch failed: n_align={} max_qlen={} max_tlen={} max_w={} "
		    "largest_task[{}] p={}MB sum_p={}MB est_mem={}MB budget={}MB "
		    "vram_free={:.1f}GB/{:.1f}GB",
	    n, max_q, max_t, max_w, big_i,
	    max_p / (1024 * 1024), sum_p / (1024 * 1024),
	    batch->current_mem / (1024 * 1024), batch->avail_mem / (1024 * 1024),
	    vfree / (1024.0 * 1024 * 1024), vtotal / (1024.0 * 1024 * 1024));
}

int ksw_extd2_gpu_batch_flush(gpu_batch_t* batch)
{
	if (!batch || batch->tasks.empty()) {
		return 0;
	}

	uint32_t num_align = batch->tasks.size();
	int unified_flag = batch->tasks[0].flag;
	int unified_end_bonus = batch->tasks[0].end_bonus;
	int unified_zdrop = batch->tasks[0].zdrop;

	int* qlen_v = batch->h_qlen.get();
	int* tlen_v = batch->h_tlen.get();
	int* w_v = batch->h_w.get();

	batch->qlen_rounded_v.resize(num_align);
	batch->tlen_rounded_v.resize(num_align);

	uint32_t* qlen_rounded_ps = batch->h_qlen_ps.get();
	memset(qlen_rounded_ps, 0, num_align * sizeof(uint32_t));
	uint32_t* tlen_rounded_ps = batch->h_tlen_ps.get();
	memset(tlen_rounded_ps, 0, num_align * sizeof(uint32_t));

	batch->p_rounded.resize(num_align);
	uint64_t* p_rounded_ps = batch->h_p_ps.get();
	memset(p_rounded_ps, 0, num_align * sizeof(uint64_t));

	batch->off_v.resize(num_align);
	uint32_t* off_ps = batch->h_off_ps.get();
	memset(off_ps, 0, num_align * sizeof(uint32_t));

	ksw_extz_t* ez_v = batch->h_ez.get();
	memset(ez_v, 0, num_align * sizeof(ksw_extz_t));

	for (uint32_t i = 0; i < num_align; i++) {
		qlen_v[i] = batch->tasks[i].qlen;
		tlen_v[i] = batch->tasks[i].tlen;
		w_v[i] = batch->tasks[i].w;
		batch->qlen_rounded_v[i] = (uint32_t)std::floor((qlen_v[i] + 15) / 16 + 1) * 16;
		batch->tlen_rounded_v[i] = (uint32_t)std::floor((tlen_v[i] + 15) / 16 + 1) * 16;
	}

	uint64_t cumulative_qlen = std::accumulate(batch->qlen_rounded_v.begin(), batch->qlen_rounded_v.end(), (uint64_t)0);
	uint64_t cumulative_tlen = std::accumulate(batch->tlen_rounded_v.begin(), batch->tlen_rounded_v.end(), (uint64_t)0);

	const uint64_t KERNEL_PAD = 1024;
	uint64_t cumulative_qlen_padded = cumulative_qlen + KERNEL_PAD;
	uint64_t cumulative_tlen_padded = cumulative_tlen + KERNEL_PAD;

	if (!HIP_OK(batch->h_qseq.grow(cumulative_qlen)))
		return -1;
	if (!HIP_OK(batch->h_tseq.grow(cumulative_tlen)))
		return -1;
	uint8_t* contiguous_qseq = batch->h_qseq.get();
	uint8_t* contiguous_tseq = batch->h_tseq.get();
	memset(contiguous_qseq, 0, cumulative_qlen);
	memset(contiguous_tseq, 0, cumulative_tlen);

	uint64_t k = 0;
	for (uint32_t i = 0; i < num_align; i++) {
		memcpy(&contiguous_qseq[k], batch->tasks[i].qseq.data(), qlen_v[i]);
		k += batch->qlen_rounded_v[i];
	}

	k = 0;
	for (uint32_t i = 0; i < num_align; i++) {
		memcpy(&contiguous_tseq[k], batch->tasks[i].tseq.data(), tlen_v[i]);
		k += batch->tlen_rounded_v[i];
	}

	std::partial_sum(batch->tlen_rounded_v.begin(), batch->tlen_rounded_v.end() - 1, tlen_rounded_ps + 1);
	std::partial_sum(batch->qlen_rounded_v.begin(), batch->qlen_rounded_v.end() - 1, qlen_rounded_ps + 1);

	for (uint32_t i = 0; i < num_align; i++) {
		int wi = w_v[i];
		if (wi < 0) wi = (tlen_v[i] > qlen_v[i]) ? tlen_v[i] : qlen_v[i];
		int n_col_ = (qlen_v[i] < tlen_v[i]) ? qlen_v[i] : tlen_v[i];
		n_col_ = ((n_col_ < wi + 1 ? n_col_ : wi + 1) + 15) / 16 + 1;
		batch->p_rounded[i] = ((uint64_t)(qlen_v[i] + tlen_v[i] - 1) * n_col_ + 1) * 16;
		batch->off_v[i] = qlen_v[i] + tlen_v[i] - 1;
	}

	uint64_t cumulative_p = std::accumulate(batch->p_rounded.begin(), batch->p_rounded.end(), (uint64_t)0) + KERNEL_PAD;
	uint64_t cumulative_off = std::accumulate(batch->off_v.begin(), batch->off_v.end(), (uint64_t)0);

	std::partial_sum(batch->p_rounded.begin(), batch->p_rounded.end() - 1, p_rounded_ps + 1);
	std::partial_sum(batch->off_v.begin(), batch->off_v.end() - 1, off_ps + 1);

	bool need_cigar = false;
	for (uint32_t i = 0; i < num_align && !need_cigar; i++) {
		if (!(batch->tasks[i].flag & KSW_EZ_SCORE_ONLY))
			need_cigar = true;
	}

	hipStream_t stream = batch->stream.get();

#ifdef DEBUG_PRINT
	size_t pre_free = 0, pre_total = 0;
	(void)hipMemGetInfo(&pre_free, &pre_total);
	mm_log_debug("[GPU Batch] pre-flush: n_align={}, estimated_mem={:.1f}MB, "
		     "budget={:.1f}MB, vram_free={:.1f}GB/{:.1f}GB",
	    num_align,
	    batch->current_mem / (1024.0 * 1024),
	    batch->avail_mem / (1024.0 * 1024),
	    pre_free / (1024.0 * 1024 * 1024),
	    pre_total / (1024.0 * 1024 * 1024));
#endif

#define FAIL_FLUSH() \
	do { \
		(void)hipStreamSynchronize(stream); \
		(void)hipGetLastError(); \
		log_batch_failure_diag(batch); \
		batch->tasks.clear(); \
		batch->current_mem = 25 * sizeof(int8_t) + batch->max_alignments * sizeof(ksw_extz_t); \
		return -1; \
	} while (0)

#ifdef DEBUG_PRINT
	{
		static int flush_count = 0;
		static void* oom_poison = nullptr;
		/* Free any previous poison block so retry/next batch can succeed */
		if (oom_poison) {
			(void)hipFree(oom_poison);
			oom_poison = nullptr;
		}
		const char* sim = getenv("MM_GPU_SIMULATE_OOM");
		if (sim) {
			int every_n = atoi(sim);
			if (every_n > 0 && (++flush_count % every_n) == 0) {
				/* Grab most free VRAM to pressure the allocator, then
				 * force-fail.  Pool-async allocs would recycle cached pages
				 * and survive even with 0 free VRAM, so we must also force
				 * the FAIL_FLUSH to exercise the recovery path. */
				size_t oom_free = 0, oom_total = 0;
				(void)hipMemGetInfo(&oom_free, &oom_total);
				size_t grab = oom_free > (64ULL << 20) ? oom_free - (64ULL << 20) : 0;
				if (grab > 0)
					(void)hipMalloc(&oom_poison, grab);
				mm_log_warn("[GPU Batch] SIMULATED OOM on flush #{}: "
					    "poisoned {:.1f}GB, free was {:.1f}GB, forcing FAIL_FLUSH",
				    flush_count,
				    grab / (1024.0 * 1024 * 1024),
				    oom_free / (1024.0 * 1024 * 1024));
				FAIL_FLUSH();
			}
		}
	}
#endif

	if (!HIP_OK(batch->w_d.grow(num_align)))
		FAIL_FLUSH();

	if (!HIP_OK(batch->qseq_d.grow(cumulative_qlen_padded)) ||
	    !HIP_OK(batch->tseq_d.grow(cumulative_tlen_padded)))
		FAIL_FLUSH();

	if (!HIP_OK(batch->u_d.grow(cumulative_tlen_padded)) ||
	    !HIP_OK(batch->v_d.grow(cumulative_tlen_padded)) ||
	    !HIP_OK(batch->x_d.grow(cumulative_tlen_padded)) ||
	    !HIP_OK(batch->y_d.grow(cumulative_tlen_padded)) ||
	    !HIP_OK(batch->x2_d.grow(cumulative_tlen_padded)) ||
	    !HIP_OK(batch->y2_d.grow(cumulative_tlen_padded)))
		FAIL_FLUSH();

	if (!HIP_OK(batch->p_d.grow(cumulative_p)) ||
	    !HIP_OK(batch->H_d.grow(cumulative_tlen_padded)))
		FAIL_FLUSH();

	if (need_cigar) {
		if (!HIP_OK(batch->off_ps_d.grow(num_align)) ||
		    !HIP_OK(batch->off_d.grow(cumulative_off)) ||
		    !HIP_OK(batch->off_r_d.grow(cumulative_off)))
			FAIL_FLUSH();
	}

#ifdef DEBUG_PRINT
	{
		size_t alloc_free = 0, alloc_total = 0;
		(void)hipMemGetInfo(&alloc_free, &alloc_total);
		mm_log_debug("[GPU Batch] post-device-alloc: vram_free={:.1f}GB "
			     "(consumed={:.1f}MB for DP buffers)",
		    alloc_free / (1024.0 * 1024 * 1024),
		    (pre_free - alloc_free) / (1024.0 * 1024));
	}
#endif

	/* Zero ez_d and sequence buffers (data region + padding) */
	if (!HIP_OK(hipMemsetAsync(batch->ez_d.get(), 0, num_align * sizeof(ksw_extz_t), stream)))
		FAIL_FLUSH();
	if (!HIP_OK(hipMemsetAsync(batch->qseq_d.get(), 0, cumulative_qlen_padded * sizeof(uint8_t), stream)))
		FAIL_FLUSH();
	if (!HIP_OK(hipMemsetAsync(batch->tseq_d.get(), 0, cumulative_tlen_padded * sizeof(uint8_t), stream)))
		FAIL_FLUSH();

	/* Only zero the KERNEL_PAD tail for scratch buffers — kernel overwrites the data region */
	if (!HIP_OK(hipMemsetAsync(batch->u_d.get() + cumulative_tlen, 0, KERNEL_PAD * sizeof(int8_t), stream)))
		FAIL_FLUSH();
	if (!HIP_OK(hipMemsetAsync(batch->v_d.get() + cumulative_tlen, 0, KERNEL_PAD * sizeof(int8_t), stream)))
		FAIL_FLUSH();
	if (!HIP_OK(hipMemsetAsync(batch->x_d.get() + cumulative_tlen, 0, KERNEL_PAD * sizeof(int8_t), stream)))
		FAIL_FLUSH();
	if (!HIP_OK(hipMemsetAsync(batch->y_d.get() + cumulative_tlen, 0, KERNEL_PAD * sizeof(int8_t), stream)))
		FAIL_FLUSH();
	if (!HIP_OK(hipMemsetAsync(batch->x2_d.get() + cumulative_tlen, 0, KERNEL_PAD * sizeof(int8_t), stream)))
		FAIL_FLUSH();
	if (!HIP_OK(hipMemsetAsync(batch->y2_d.get() + cumulative_tlen, 0, KERNEL_PAD * sizeof(int8_t), stream)))
		FAIL_FLUSH();
	if (!HIP_OK(hipMemsetAsync(batch->H_d.get() + cumulative_tlen, 0, KERNEL_PAD * sizeof(int32_t), stream)))
		FAIL_FLUSH();
	if (!HIP_OK(hipMemsetAsync(batch->p_d.get() + (cumulative_p - KERNEL_PAD), 0, KERNEL_PAD * sizeof(uint8_t), stream)))
		FAIL_FLUSH();

	if (need_cigar) {
		if (!HIP_OK(hipMemsetAsync(batch->off_d.get(), 0, cumulative_off * sizeof(int), stream)))
			FAIL_FLUSH();
		if (!HIP_OK(hipMemsetAsync(batch->off_r_d.get(), 0, cumulative_off * sizeof(int), stream)))
			FAIL_FLUSH();
	}

	if (!HIP_OK(hipMemcpyAsync(batch->qseq_d.get(), contiguous_qseq, cumulative_qlen * sizeof(uint8_t), hipMemcpyHostToDevice, stream)))
		FAIL_FLUSH();
	if (!HIP_OK(hipMemcpyAsync(batch->tseq_d.get(), contiguous_tseq, cumulative_tlen * sizeof(uint8_t), hipMemcpyHostToDevice, stream)))
		FAIL_FLUSH();
	if (!HIP_OK(hipMemcpyAsync(batch->qlen_d.get(), qlen_v, num_align * sizeof(int), hipMemcpyHostToDevice, stream)))
		FAIL_FLUSH();
	if (!HIP_OK(hipMemcpyAsync(batch->tlen_d.get(), tlen_v, num_align * sizeof(int), hipMemcpyHostToDevice, stream)))
		FAIL_FLUSH();
	if (!HIP_OK(hipMemcpyAsync(batch->w_d.get(), w_v, num_align * sizeof(int), hipMemcpyHostToDevice, stream)))
		FAIL_FLUSH();
	if (!HIP_OK(hipMemcpyAsync(batch->qlen_ps_d.get(), qlen_rounded_ps, num_align * sizeof(uint32_t), hipMemcpyHostToDevice, stream)))
		FAIL_FLUSH();
	if (!HIP_OK(hipMemcpyAsync(batch->tlen_ps_d.get(), tlen_rounded_ps, num_align * sizeof(uint32_t), hipMemcpyHostToDevice, stream)))
		FAIL_FLUSH();
	if (!HIP_OK(hipMemcpyAsync(batch->p_ps_d.get(), p_rounded_ps, num_align * sizeof(uint64_t), hipMemcpyHostToDevice, stream)))
		FAIL_FLUSH();

	if (need_cigar) {
		if (!HIP_OK(hipMemcpyAsync(batch->off_ps_d.get(), off_ps, num_align * sizeof(uint32_t), hipMemcpyHostToDevice, stream)))
			FAIL_FLUSH();
	}

	/* Launch batched kernel */
	{
		int8_t min_sc = batch->mat[0];
		for (int mk = 0; mk < 25; mk++) {
			if (batch->mat[mk] < min_sc) min_sc = batch->mat[mk];
		}
		ksw_extd2_gpu_multialign<<<num_align, 512, 0, stream>>>(
		    batch->u_d.get(), batch->v_d.get(), batch->x_d.get(),
		    batch->y_d.get(), batch->x2_d.get(), batch->y2_d.get(),
		    batch->p_d.get(), batch->H_d.get(),
		    batch->off_d.get(), batch->off_r_d.get(),
		    batch->qlen_d.get(), batch->qseq_d.get(),
		    batch->tlen_d.get(), batch->tseq_d.get(),
		    batch->mat_d.get(),
		    batch->q, batch->e, batch->q2, batch->e2,
		    batch->w_d.get(), unified_zdrop, unified_end_bonus, unified_flag,
		    batch->ez_d.get(),
		    batch->tlen_ps_d.get(), batch->qlen_ps_d.get(),
		    batch->p_ps_d.get(), batch->off_ps_d.get(),
		    min_sc);

		if (!HIP_OK(hipGetLastError()))
			FAIL_FLUSH();
	}

	/* Copy results back */
	if (need_cigar) {
		batch->cigar_max.resize(num_align);
		batch->cigar_offsets.resize(num_align + 1);
		batch->cigar_offsets[0] = 0;
		for (uint32_t i = 0; i < num_align; i++) {
			batch->cigar_max[i] = qlen_v[i] + tlen_v[i];
		}
		std::partial_sum(batch->cigar_max.begin(), batch->cigar_max.end(), batch->cigar_offsets.begin() + 1);
		uint64_t total_cigar_buf = batch->cigar_offsets[num_align];

		if (!HIP_OK(batch->cigar_buf_d.grow(total_cigar_buf)) ||
		    !HIP_OK(batch->cigar_off_d.grow(num_align + 1)) ||
		    !HIP_OK(batch->n_cigar_d.grow(num_align)) ||
		    !HIP_OK(batch->reach_end_d.grow(num_align)))
			FAIL_FLUSH();

#ifdef DEBUG_PRINT
		{
			size_t bt_free = 0, bt_total = 0;
			(void)hipMemGetInfo(&bt_free, &bt_total);
			mm_log_debug("[GPU Batch] post-backtrack-alloc: "
				     "cigar_buf={}MB, vram_free={:.1f}GB",
			    total_cigar_buf * sizeof(uint32_t) / (1024 * 1024),
			    bt_free / (1024.0 * 1024 * 1024));
		}
#endif

		if (!HIP_OK(hipMemcpyAsync(batch->cigar_off_d.get(), batch->cigar_offsets.data(), (num_align + 1) * sizeof(uint32_t), hipMemcpyHostToDevice, stream)))
			FAIL_FLUSH();

		{
			int bt_threads = 256;
			int bt_blocks = (num_align + bt_threads - 1) / bt_threads;
			ksw_backtrack_batch_kernel<<<bt_blocks, bt_threads, 0, stream>>>(
			    batch->p_d.get(), batch->off_d.get(), batch->off_r_d.get(),
			    batch->p_ps_d.get(), batch->off_ps_d.get(),
			    batch->ez_d.get(),
			    batch->qlen_d.get(), batch->tlen_d.get(),
			    batch->w_d.get(),
			    unified_flag, unified_end_bonus,
			    batch->cigar_buf_d.get(), batch->cigar_off_d.get(),
			    batch->n_cigar_d.get(), batch->reach_end_d.get(),
			    num_align);

			if (!HIP_OK(hipGetLastError()))
				FAIL_FLUSH();
		}

		if (!HIP_OK(hipMemcpyAsync(ez_v, batch->ez_d.get(), num_align * sizeof(ksw_extz_t), hipMemcpyDeviceToHost, stream)))
			FAIL_FLUSH();

		batch->n_cigar_host.resize(num_align);
		batch->reach_end_host.resize(num_align);

		if (!HIP_OK(hipMemcpyAsync(batch->n_cigar_host.data(), batch->n_cigar_d.get(), num_align * sizeof(int), hipMemcpyDeviceToHost, stream)))
			FAIL_FLUSH();
		if (!HIP_OK(hipMemcpyAsync(batch->reach_end_host.data(), batch->reach_end_d.get(), num_align * sizeof(int), hipMemcpyDeviceToHost, stream)))
			FAIL_FLUSH();

		if (!HIP_OK(batch->cigar_host.grow(total_cigar_buf)) || !batch->cigar_host.get())
			FAIL_FLUSH();
		if (!HIP_OK(hipMemcpyAsync(batch->cigar_host.get(), batch->cigar_buf_d.get(), total_cigar_buf * sizeof(uint32_t), hipMemcpyDeviceToHost, stream)))
			FAIL_FLUSH();

		if (!HIP_OK(hipStreamSynchronize(stream)))
			FAIL_FLUSH();

		for (uint32_t i = 0; i < num_align; i++) {
			ksw_extz_t* ez = batch->tasks[i].ez;
			if (!ez) continue;
			ksw_extz_t* ez_gpu = &ez_v[i];

			ez->max = ez_gpu->max;
			ez->max_q = ez_gpu->max_q;
			ez->max_t = ez_gpu->max_t;
			ez->mqe = ez_gpu->mqe;
			ez->mqe_t = ez_gpu->mqe_t;
			ez->mte = ez_gpu->mte;
			ez->mte_q = ez_gpu->mte_q;
			ez->score = ez_gpu->score;
			ez->zdropped = ez_gpu->zdropped;
			ez->reach_end = batch->reach_end_host[i];

			int nc = batch->n_cigar_host[i];
			if (nc > 0) {
				ez->n_cigar = nc;
				ez->m_cigar = nc;
				ez->cigar = (uint32_t*)malloc(nc * sizeof(uint32_t));
				memcpy(ez->cigar, &batch->cigar_host.get()[batch->cigar_offsets[i]], nc * sizeof(uint32_t));
			} else {
				ez->n_cigar = 0;
				ez->m_cigar = 0;
				ez->cigar = NULL;
			}
		}

	} else {
		if (!HIP_OK(hipMemcpyAsync(ez_v, batch->ez_d.get(), num_align * sizeof(ksw_extz_t), hipMemcpyDeviceToHost, stream)))
			FAIL_FLUSH();
		if (!HIP_OK(hipStreamSynchronize(stream)))
			FAIL_FLUSH();
		for (uint32_t i = 0; i < num_align; i++) {
			ksw_extz_t* ez = batch->tasks[i].ez;
			ksw_extz_t* ez_gpu = &ez_v[i];

			ez->max = ez_gpu->max;
			ez->max_q = ez_gpu->max_q;
			ez->max_t = ez_gpu->max_t;
			ez->mqe = ez_gpu->mqe;
			ez->mqe_t = ez_gpu->mqe_t;
			ez->mte = ez_gpu->mte;
			ez->mte_q = ez_gpu->mte_q;
			ez->score = ez_gpu->score;
			ez->zdropped = ez_gpu->zdropped;
			ez->reach_end = ez_gpu->reach_end;
			ez->n_cigar = 0;
			ez->m_cigar = 0;
			ez->cigar = NULL;
		}
	}

#ifdef DEBUG_PRINT
	{
		size_t post_free = 0, post_total = 0;
		(void)hipMemGetInfo(&post_free, &post_total);
		mm_log_debug("[GPU Batch] post-flush-cleanup: vram_free={:.1f}GB "
			     "(reclaimed after temporaries released)",
		    post_free / (1024.0 * 1024 * 1024));
	}
#endif

	int result = (int)num_align;
	batch->tasks.clear();
	batch->current_mem = 25 * sizeof(int8_t) + batch->max_alignments * sizeof(ksw_extz_t);

#undef FAIL_FLUSH

	return result;
}

int ksw_extd2_gpu_batch_pending(gpu_batch_t* batch)
{
	if (!batch) return 0;
	return (int)batch->tasks.size();
}

void ksw_extd2_gpu_batch_destroy(gpu_batch_t* batch)
{
	delete batch;
}

} /* extern "C" */

extern "C" void ksw_gpu_warmup(void)
{
	void* dummy = NULL;
	if (hipMalloc(&dummy, 1) == hipSuccess)
		hipFree(dummy);
}
