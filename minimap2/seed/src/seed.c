/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2018-2022 Heng Li; Dana-Farber Cancer Institute; Broad Institute
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

#include <string.h>
#include "kalloc.h"
#include "ksort.h"
#include "misc.h"
#include "index.h"
#include "sdust.h"
#include "seed_priv.h"
#include "mm_profiler.h"

#ifdef MM_ENABLE_SEED_GPU
#include "seed_gpu.h"
#endif

#define heap_lt(a, b) ((a).x > (b).x)
KSORT_INIT(heap, mm128_t, heap_lt)

/**
 * Filter minimizers that occur too frequently in the query sequence
 * Removes minimizers that appear more than q_occ_max times AND more than q_occ_frac of total minimizers
 * @param km           Thread-local memory pool; using NULL falls back to malloc()
 * @param mv           Vector of minimizers (mm128_t); filtered in place
 * @param q_occ_max    Maximum occurrences threshold (absolute count)
 * @param q_occ_frac   Maximum occurrences threshold (fraction of total minimizers)
 */
void mm_seed_mz_flt(void* km, mm128_v* mv, int32_t q_occ_max, float q_occ_frac)
{
	MM_PROFILE_SEED("seed_mz_flt");
	mm128_t* a;
	size_t i, j, st;
	if (mv->n <= q_occ_max || q_occ_frac <= 0.0f || q_occ_max <= 0) {
		MM_PROFILE_RANGE_POP();
		return;
	}
	a = Kmalloc(km, mm128_t, mv->n);
#ifdef MM_ENABLE_SEED_GPU
	mm_copy_mz_gpu(a, mv->a, mv->n);
#else
	for (i = 0; i < mv->n; ++i)
		a[i].x = mv->a[i].x, a[i].y = i;
#endif
	radix_sort_128x(a, a + mv->n); //sort
	for (st = 0, i = 1; i <= mv->n; ++i) { //filter minimizers
		if (i == mv->n || a[i].x != a[st].x) {
			int32_t cnt = i - st;
			if (cnt > q_occ_max && cnt > mv->n * q_occ_frac)
				for (j = st; j < i; ++j)
					mv->a[a[j].y].x = 0;
			st = i;
		}
	}
	kfree(km, a);
	for (i = j = 0; i < mv->n; ++i)
		if (mv->a[i].x != 0)
			mv->a[j++] = mv->a[i];
	mv->n = j;
	MM_PROFILE_RANGE_POP();
}

/**
 * Collect all seed matches from the index for each minimizer
 * For each minimizer in mv, look up its hits in the index and create a seed entry
 * @param km    Thread-local memory pool; using NULL falls back to malloc()
 * @param mi    Minimap2 index containing reference k-mer positions
 * @param mv    Vector of minimizers from the query sequence
 * @param n_m_  Output: number of seeds collected
 * @return      Array of mm_seed_t entries with index lookup results
 */
mm_seed_t* mm_seed_collect_all(void* km, const mm_idx_t* mi, const mm128_v* mv, int32_t* n_m_)
{
	mm_seed_t* m;
	size_t i;
	int32_t k;
	m = (mm_seed_t*)kmalloc(km, mv->n * sizeof(mm_seed_t));
	for (i = k = 0; i < mv->n; ++i) {
		const uint64_t* cr;
		mm_seed_t* q;
		mm128_t* p = &mv->a[i];
		uint32_t q_pos = (uint32_t)p->y, q_span = p->x & 0xff;
		int t;
		cr = mm_idx_get(mi, p->x >> 8, &t);
		if (t == 0) continue;
		q = &m[k++];
		q->q_pos = q_pos, q->q_span = q_span, q->cr = cr, q->n = t, q->seg_id = p->y >> 32;
		q->is_tandem = q->flt = 0;
		if (i > 0 && p->x >> 8 == mv->a[i - 1].x >> 8) q->is_tandem = 1;
		if (i < mv->n - 1 && p->x >> 8 == mv->a[i + 1].x >> 8) q->is_tandem = 1;
	}
	*n_m_ = k;
	return m;
}

#define MAX_MAX_HIGH_OCC 128

/**
 * Selectively mark high-occurrence seeds for filtering using a binomial heap
 * For each streak of high-occ minimizers, retain only the top max_high_occ seeds with lowest occurrence
 * @param n            Number of seeds in array
 * @param a            Array of seeds to filter; flt field is set for seeds to discard
 * @param len          Query sequence length
 * @param max_occ      Occurrence threshold; seeds with n > max_occ are considered high-occ
 * @param max_max_occ  Hard threshold; seeds with n > max_max_occ are always filtered
 * @param dist         Distance parameter to calculate max_high_occ = (pe - ps) / dist
 */
void mm_seed_select(int32_t n, mm_seed_t* a, int len, int max_occ, int max_max_occ, int dist)
{ // for high-occ minimizers, choose up to max_high_occ in each high-occ streak
	extern void ks_heapdown_uint64_t(size_t i, size_t n, uint64_t*);
	extern void ks_heapmake_uint64_t(size_t n, uint64_t*);
	int32_t i, last0, m;
	uint64_t b[MAX_MAX_HIGH_OCC]; // this is to avoid a heap allocation

	if (n == 0 || n == 1) return;
	for (i = m = 0; i < n; ++i)
		if (a[i].n > max_occ) ++m;
	if (m == 0) return; // no high-frequency k-mers; do nothing
	for (i = 0, last0 = -1; i <= n; ++i) {
		if (i == n || a[i].n <= max_occ) {
			if (i - last0 > 1) {
				int32_t ps = last0 < 0 ? 0 : (uint32_t)a[last0].q_pos >> 1;
				int32_t pe = i == n ? len : (uint32_t)a[i].q_pos >> 1;
				int32_t j, k, st = last0 + 1, en = i;
				int32_t max_high_occ = (int32_t)((double)(pe - ps) / dist + .499);
				if (max_high_occ > 0) {
					if (max_high_occ > MAX_MAX_HIGH_OCC)
						max_high_occ = MAX_MAX_HIGH_OCC;
					for (j = st, k = 0; j < en && k < max_high_occ; ++j, ++k)
						b[k] = (uint64_t)a[j].n << 32 | j;
					ks_heapmake_uint64_t(k, b); // initialize the binomial heap
					for (; j < en; ++j) { // if there are more, choose top max_high_occ
						if (a[j].n < (int32_t)(b[0] >> 32)) { // then update the heap
							b[0] = (uint64_t)a[j].n << 32 | j;
							ks_heapdown_uint64_t(0, k, b);
						}
					}
					for (j = 0; j < k; ++j) a[(uint32_t)b[j]].flt = 1;
				}
				for (j = st; j < en; ++j) a[j].flt ^= 1;
				for (j = st; j < en; ++j)
					if (a[j].n > max_max_occ)
						a[j].flt = 1;
			}
			last0 = i;
		}
	}
}

/**
 * Collect seed matches and filter high-occurrence seeds
 * Main function that collects all seeds, applies occurrence filtering, and tracks statistics
 * @param km            Thread-local memory pool; using NULL falls back to malloc()
 * @param _n_m          Output: number of seeds after filtering
 * @param qlen          Query sequence length
 * @param max_occ       Occurrence threshold for filtering
 * @param max_max_occ   Hard occurrence threshold (always filter if exceeded)
 * @param dist          Distance parameter for selective filtering
 * @param mi            Minimap2 index
 * @param mv            Vector of minimizers from query
 * @param n_a           Output: total number of anchors (sum of seed occurrences)
 * @param rep_len       Output: total length of repetitive regions (filtered seeds)
 * @param n_mini_pos    Output: number of minimizer positions retained
 * @param mini_pos      Output: array of minimizer positions encoded as (span<<32)|pos
 * @return              Array of filtered seeds
 */
mm_seed_t* mm_collect_matches(void* km, int* _n_m, int qlen, int max_occ, int max_max_occ, int dist, const mm_idx_t* mi, const mm128_v* mv, int64_t* n_a, int* rep_len, int* n_mini_pos, uint64_t** mini_pos)
{
	int rep_st = 0, rep_en = 0, n_m, n_m0;
	size_t i;
	mm_seed_t* m;
	*n_mini_pos = 0;
	*mini_pos = (uint64_t*)kmalloc(km, mv->n * sizeof(uint64_t));
	m = mm_seed_collect_all(km, mi, mv, &n_m0);
	if (dist > 0 && max_max_occ > max_occ) {
		mm_seed_select(n_m0, m, qlen, max_occ, max_max_occ, dist);
	} else {
		for (i = 0; i < n_m0; ++i)
			if (m[i].n > max_occ)
				m[i].flt = 1;
	}
	for (i = 0, n_m = 0, *rep_len = 0, *n_a = 0; i < n_m0; ++i) {
		mm_seed_t* q = &m[i];
		if (mm_dbg_flag & MM_DBG_SEED_FREQ)
			fprintf(stderr, "SF\t%d\t%d\t%d\n", q->q_pos >> 1, q->n, q->flt);
		if (q->flt) {
			int en = (q->q_pos >> 1) + 1, st = en - q->q_span;
			if (st > rep_en) {
				*rep_len += rep_en - rep_st;
				rep_st = st, rep_en = en;
			} else
				rep_en = en;
		} else {
			*n_a += q->n;
			(*mini_pos)[(*n_mini_pos)++] = (uint64_t)q->q_span << 32 | q->q_pos >> 1;
			m[n_m++] = *q;
		}
	}
	*rep_len += rep_en - rep_st;
	*_n_m = n_m;
	return m;
}

/**
 * mm_dust_minier: Filter out minimizers that overlap with low-complexity regions (DUST)
 * @param km memory pool for allocation
 * @param n number of minimizers
 * @param a array of mm128_t minimizers
 * @param l_seq sequence length
 * @param seq sequence string
 * @param sdust_thres SDUST threshold (<=0 disables filtering)
 * @return: number of minimizers kept after filtering
 *
 * Uses SDUST algorithm to identify low-complexity regions and removes minimizers
 * that have >50% overlap with these regions.
 */
int mm_dust_minier(void* km, int n, mm128_t* a, int l_seq, const char* seq, int sdust_thres)
{
	MM_PROFILE_SEED("dust_minier");
	int n_dreg, j, k, u = 0;
	const uint64_t* dreg;
	sdust_buf_t* sdb;
	if (sdust_thres <= 0) {
		MM_PROFILE_RANGE_POP();
		return n;
	}
	sdb = sdust_buf_init(km);
	dreg = sdust_core((const uint8_t*)seq, l_seq, sdust_thres, 64, &n_dreg, sdb);
	for (j = k = 0; j < n; ++j) { // squeeze out minimizers that significantly overlap with LCRs
		int32_t qpos = (uint32_t)a[j].y >> 1, span = a[j].x & 0xff;
		int32_t s = qpos - (span - 1), e = s + span;
		while (u < n_dreg && (int32_t)dreg[u] <= s)
			++u;
		if (u < n_dreg && (int32_t)(dreg[u] >> 32) < e) {
			int v, l = 0;
			for (v = u; v < n_dreg && (int32_t)(dreg[v] >> 32) < e; ++v) {
				int ss = s > (int32_t)(dreg[v] >> 32) ? s : dreg[v] >> 32;
				int ee = e < (int32_t)dreg[v] ? e : (uint32_t)dreg[v];
				l += ee - ss;
			}
			if (l <= span >> 1)
				a[k++] = a[j]; // keep the minimizer if less than half of it falls in masked region
		} else
			a[k++] = a[j];
	}
	sdust_buf_destroy(sdb);
	MM_PROFILE_RANGE_POP();
	return k;
}

/**
 * Collect all k-mer minimizers from query sequence(s).
 * @param km memory pool for allocation
 * @param mi minimap2 index (provides k, w, hpc flag)
 * @param n_segs number of segments (query sequences)
 * @param qlens array of query lengths
 * @param seqs array of query sequences
 * @param sdust_thres SDUST threshold (0 to disable)
 * @param mv output vector of mm128_t minimizers
 */
void mm_collect_minimizers(void* km, const mm_idx_t* mi, int n_segs, const int* qlens,
    const char** seqs, int sdust_thres, mm128_v* mv)
{
	MM_PROFILE_SEED("collect_minimizers");
	int i, n, sum = 0;
	mv->n = 0;
	for (i = n = 0; i < n_segs; ++i) {
		size_t j;
		mm_sketch(km, seqs[i], qlens[i], mi->w, mi->k, i, mi->flag & MM_I_HPC, mv);
		for (j = n; j < mv->n; ++j)
			mv->a[j].y += sum << 1;
		if (sdust_thres > 0) // mask low-complexity minimizers
			mv->n = n + mm_dust_minier(km, mv->n - n, mv->a + n, qlens[i], seqs[i], sdust_thres);
		sum += qlens[i], n = mv->n;
	}
	MM_PROFILE_RANGE_POP();
}

/**
 * skip_seed: Decide whether to skip a seed match based on flags
 * @param flag mapping flags (NO_DIAG, NO_DUAL, FOR_ONLY, REV_ONLY)
 * @param r reference anchor position
 * @param q query seed information
 * @param qname query name
 * @param qlen query length
 * @param mi minimap2 index
 * @param is_self output flag indicating self-chain
 * @return: 1 if seed should be skipped, 0 if should be kept
 */
static inline int skip_seed(int flag, uint64_t r, const mm_seed_t* q, const char* qname, int qlen, const mm_idx_t* mi, int* is_self)
{
	*is_self = 0;
	if (qname && (flag & (MM_F_NO_DIAG | MM_F_NO_DUAL))) {
		const mm_idx_seq_t* s = &mi->seq[r >> 32];
		int cmp;
		cmp = strcmp(qname, s->name);
		if ((flag & MM_F_NO_DIAG) && cmp == 0 && (int)s->len == qlen) {
			if ((uint32_t)r >> 1 == (q->q_pos >> 1))
				return 1;
			if ((r & 1) == (q->q_pos & 1))
				*is_self = 1;
		}
		if ((flag & MM_F_NO_DUAL) && cmp > 0)
			return 1;
	}
	if (flag & (MM_F_FOR_ONLY | MM_F_REV_ONLY)) {
		if ((r & 1) == (q->q_pos & 1)) {
			if (flag & MM_F_REV_ONLY)
				return 1;
		} else {
			if (flag & MM_F_FOR_ONLY)
				return 1;
		}
	}
	return 0;
}

/**
 * Collect seed hits using heap-based sorting.
 * @param km memory pool for allocation
 * @param opt mapping options (flag, max_max_occ, occ_dist extracted from here)
 * @param max_occ maximum seed occurrence threshold
 * @param mi minimap2 index
 * @param qname query name
 * @param mv minimizers vector
 * @param qlen query length
 * @param n_a output number of anchors collected
 * @param rep_len output repetitive length
 * @param n_mini_pos output number of minimizer positions
 * @param mini_pos output minimizer positions
 * @return mm128_t array of seed anchors
 */
mm128_t* collect_seed_hits_heap(void* km, const mm_mapopt_t* opt, int max_occ,
    const mm_idx_t* mi, const char* qname, const mm128_v* mv,
    int qlen, int64_t* n_a, int* rep_len,
    int* n_mini_pos, uint64_t** mini_pos)
{
	MM_PROFILE_SEED("collect_hits_heap");
	int i, n_m, heap_size = 0;
	int64_t j, n_for = 0, n_rev = 0;
	mm_seed_t* m;
	mm128_t *a, *heap;
	int64_t flag = opt->flag;

	m = mm_collect_matches(km, &n_m, qlen, max_occ, opt->max_max_occ, opt->occ_dist, mi, mv, n_a, rep_len, n_mini_pos, mini_pos);

	heap = (mm128_t*)kmalloc(km, n_m * sizeof(mm128_t));
	a = (mm128_t*)kmalloc(km, *n_a * sizeof(mm128_t));

	for (i = 0, heap_size = 0; i < n_m; ++i) {
		if (m[i].n > 0) {
			heap[heap_size].x = m[i].cr[0];
			heap[heap_size].y = (uint64_t)i << 32;
			++heap_size;
		}
	}
	ks_heapmake_heap(heap_size, heap);
	while (heap_size > 0) {
		mm_seed_t* q = &m[heap->y >> 32];
		mm128_t* p;
		uint64_t r = heap->x;
		int32_t is_self, rpos = (uint32_t)r >> 1;
		if (!skip_seed(flag, r, q, qname, qlen, mi, &is_self)) {
			if ((r & 1) == (q->q_pos & 1)) { // forward strand
				p = &a[n_for++];
				p->x = (r & 0xffffffff00000000ULL) | rpos;
				p->y = (uint64_t)q->q_span << 32 | q->q_pos >> 1;
			} else { // reverse strand
				p = &a[(*n_a) - (++n_rev)];
				p->x = 1ULL << 63 | (r & 0xffffffff00000000ULL) | rpos;
				p->y = (uint64_t)q->q_span << 32 | (qlen - ((q->q_pos >> 1) + 1 - q->q_span) - 1);
			}
			p->y |= (uint64_t)q->seg_id << MM_SEED_SEG_SHIFT;
			if (q->is_tandem)
				p->y |= MM_SEED_TANDEM;
			if (is_self)
				p->y |= MM_SEED_SELF;
		}
		// update the heap
		if ((uint32_t)heap->y < q->n - 1) {
			++heap[0].y;
			heap[0].x = m[heap[0].y >> 32].cr[(uint32_t)heap[0].y];
		} else {
			heap[0] = heap[heap_size - 1];
			--heap_size;
		}
		ks_heapdown_heap(0, heap_size, heap);
	}
	kfree(km, m);
	kfree(km, heap);

	for (j = 0; j < n_rev >> 1; ++j) {
		mm128_t t = a[(*n_a) - 1 - j];
		a[(*n_a) - 1 - j] = a[(*n_a) - (n_rev - j)];
		a[(*n_a) - (n_rev - j)] = t;
	}
	if (*n_a > n_for + n_rev) {
		memmove(a + n_for, a + (*n_a) - n_rev, n_rev * sizeof(mm128_t));
		*n_a = n_for + n_rev;
	}
	MM_PROFILE_RANGE_POP();
	return a;
}

/**
 * Collect all seed hits and sort them using radix sort.
 * @param km memory pool for allocation
 * @param opt mapping options (flag, max_max_occ, occ_dist extracted from here)
 * @param max_occ maximum seed occurrence threshold
 * @param mi minimap2 index
 * @param qname query name
 * @param mv minimizers vector
 * @param qlen query length
 * @param n_a output number of anchors collected
 * @param rep_len output repetitive length
 * @param n_mini_pos output number of minimizer positions
 * @param mini_pos output minimizer positions
 * @return mm128_t array of sorted seed anchors
 */
mm128_t* collect_seed_hits(void* km, const mm_mapopt_t* opt, int max_occ,
    const mm_idx_t* mi, const char* qname, const mm128_v* mv,
    int qlen, int64_t* n_a, int* rep_len,
    int* n_mini_pos, uint64_t** mini_pos)
{
	MM_PROFILE_SEED("collect_hits_radix");
	int i, n_m;
	int64_t flag = opt->flag;
	mm_seed_t* m;
	mm128_t* a;
	m = mm_collect_matches(km, &n_m, qlen, max_occ, opt->max_max_occ, opt->occ_dist, mi, mv, n_a, rep_len, n_mini_pos, mini_pos);
	a = (mm128_t*)kmalloc(km, *n_a * sizeof(mm128_t));
	for (i = 0, *n_a = 0; i < n_m; ++i) {
		mm_seed_t* q = &m[i];
		const uint64_t* r = q->cr;
		uint32_t k;
		for (k = 0; k < q->n; ++k) {
			int32_t is_self, rpos = (uint32_t)r[k] >> 1;
			mm128_t* p;
			if (skip_seed(flag, r[k], q, qname, qlen, mi, &is_self))
				continue;
			p = &a[(*n_a)++];
			if ((r[k] & 1) == (q->q_pos & 1)) { // forward strand
				p->x = (r[k] & 0xffffffff00000000ULL) | rpos;
				p->y = (uint64_t)q->q_span << 32 | q->q_pos >> 1;
			} else if (!(flag & MM_F_QSTRAND)) { // reverse strand
				p->x = 1ULL << 63 | (r[k] & 0xffffffff00000000ULL) | rpos;
				p->y = (uint64_t)q->q_span << 32 | (qlen - ((q->q_pos >> 1) + 1 - q->q_span) - 1);
			} else { // reverse strand; query-strand mode
				int32_t len = mi->seq[r[k] >> 32].len;
				p->x = 1ULL << 63 | (r[k] & 0xffffffff00000000ULL) | (len - (rpos + 1 - q->q_span) - 1);
				p->y = (uint64_t)q->q_span << 32 | q->q_pos >> 1;
			}
			p->y |= (uint64_t)q->seg_id << MM_SEED_SEG_SHIFT;
			if (q->is_tandem)
				p->y |= MM_SEED_TANDEM;
			if (is_self)
				p->y |= MM_SEED_SELF;
		}
	}
	kfree(km, m);
	radix_sort_128x(a, a + (*n_a));
	MM_PROFILE_RANGE_POP();
	return a;
}
