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

/* GPU memory management  */


#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include "plmem.cuh"
#include "plrange.cuh"
#include "plscore.cuh"
#include <time.h>
#include <limits.h>
#include "mm_log.h"
#include "gpu_device_descriptor.h"

// Helper function for integer ceiling division with overflow checking
// Uses formula: ceil(a/b) = 1 + (a-1)/b to avoid overflow
inline int ceil_div(int numerator, int denominator)
{
	// Check for division by zero
	if (denominator == 0) {
		mm_log_error("ceil_div: division by zero (denominator=0)");
		exit(1);
	}

	// Handle negative values
	if (numerator <= 0 || denominator < 0) {
		mm_log_error("ceil_div: expected positive values, got numerator={}, denominator={}",
		    numerator, denominator);
		exit(1);
	}

	return 1 + (numerator - 1) / denominator;
}

void plmem_malloc_host_mem(hostMemPtr* host_mem, size_t anchor_per_batch,
    int range_grid_size)
{
	// Align each sub-buffer to 256 bytes for optimal DMA alignment.
	const size_t ALIGN = 256;
#define ALIGN_UP(x) (((x) + ALIGN - 1) & ~(ALIGN - 1))

	size_t off = 0;
	size_t sz_f = ALIGN_UP(anchor_per_batch * sizeof(int32_t));
	size_t sz_p = ALIGN_UP(anchor_per_batch * sizeof(uint16_t));
	size_t sz_start = ALIGN_UP(range_grid_size * sizeof(size_t));
	size_t sz_rend = ALIGN_UP(range_grid_size * sizeof(size_t));
	size_t sz_cut = ALIGN_UP(range_grid_size * sizeof(size_t));
	size_t sz_lsnum = ALIGN_UP(sizeof(unsigned int));
	size_t total = sz_f + sz_p + sz_start + sz_rend + sz_cut + sz_lsnum;

#ifdef DEBUG_PRINT
	mm_gpu_log_info("Host Malloc Pinned Memory Size {:.2f} GB (consolidated)", (float)total / OneG);
#endif

	char* base;
	cudaMallocHost((void**)&base, total);
	cudaCheck();
	host_mem->pinned_base = base;

	host_mem->f = (int32_t*)(base + off);
	off += sz_f;
	host_mem->p = (uint16_t*)(base + off);
	off += sz_p;
	host_mem->start_idx = (size_t*)(base + off);
	off += sz_start;
	host_mem->read_end_idx = (size_t*)(base + off);
	off += sz_rend;
	host_mem->cut_start_idx = (size_t*)(base + off);
	off += sz_cut;
	host_mem->long_segs_num = (unsigned int*)(base + off);

#undef ALIGN_UP
}

void plmem_malloc_long_mem(const gpu_chain_ctx_t* ctx, longMemPtr* long_mem, size_t buffer_size_long)
{
	const int long_seg_cutoff = ctx->score_config.long_seg_cutoff;
	const int cut_unit = ctx->score_config.cut_unit;
	size_t max_segs = buffer_size_long / (long_seg_cutoff * cut_unit);

	const size_t ALIGN = 256;
#define ALIGN_UP(x) (((x) + ALIGN - 1) & ~(ALIGN - 1))

	size_t sz_og = ALIGN_UP(max_segs * sizeof(seg_t));
	size_t sz_buf = ALIGN_UP(max_segs * sizeof(seg_t));
	size_t sz_fl = ALIGN_UP(buffer_size_long * sizeof(int32_t));
	size_t sz_pl = ALIGN_UP(buffer_size_long * sizeof(uint16_t));
	size_t sz_num = ALIGN_UP(sizeof(unsigned int));
	size_t sz_n = ALIGN_UP(sizeof(size_t));
	size_t total = sz_og + sz_buf + sz_fl + sz_pl + sz_num + sz_n;

#ifdef DEBUG_PRINT
	mm_gpu_log_info("Host Malloc Pinned Memory Size {:.2f} GB (long seg, consolidated)", (float)total / OneG);
#endif

	char* base;
	cudaMallocHost((void**)&base, total);
	cudaCheck();
	long_mem->pinned_base = base;

	size_t off = 0;
	long_mem->long_segs_og_idx = (seg_t*)(base + off);
	off += sz_og;
	long_mem->long_segs_buf_idx = (seg_t*)(base + off);
	off += sz_buf;
	long_mem->f_long = (int32_t*)(base + off);
	off += sz_fl;
	long_mem->p_long = (uint16_t*)(base + off);
	off += sz_pl;
	long_mem->total_long_segs_num = (unsigned int*)(base + off);
	off += sz_num;
	long_mem->total_long_segs_n = (size_t*)(base + off);

#undef ALIGN_UP
}

void plmem_free_host_mem(hostMemPtr* host_mem)
{
	if (host_mem->pinned_base) {
		cudaFreeHost(host_mem->pinned_base);
		host_mem->pinned_base = NULL;
	}
	cudaCheck();
}

void plmem_free_long_mem(longMemPtr* long_mem)
{
	if (long_mem->pinned_base) {
		cudaFreeHost(long_mem->pinned_base);
		long_mem->pinned_base = NULL;
	}
	cudaCheck();
}

void plmem_malloc_device_mem(const gpu_chain_ctx_t* ctx, hipMemPool_t mempool, deviceMemPtr* dev_mem, size_t anchor_per_batch, int range_grid_size, int num_cut, cudaStream_t stream)
{
	const int long_seg_cutoff = ctx->score_config.long_seg_cutoff;
	const int cut_unit = ctx->score_config.cut_unit;
	const int mid_seg_cutoff = ctx->score_config.mid_seg_cutoff;
	size_t max_segs = dev_mem->buffer_size_long / (long_seg_cutoff * cut_unit);
	size_t mid_segs = num_cut / (mid_seg_cutoff + 1);

	// Compute sub-buffer sizes with 256-byte alignment for coalesced access.
	const size_t ALIGN = 256;
#define ALIGN_UP(x) (((x) + ALIGN - 1) & ~(ALIGN - 1))

	// Short/mid segment arrays
	size_t sz_raw = ALIGN_UP(anchor_per_batch * 2 * sizeof(uint64_t));
	size_t sz_ax = ALIGN_UP(anchor_per_batch * sizeof(int32_t));
	size_t sz_ay = ALIGN_UP(anchor_per_batch * sizeof(int32_t));
	size_t sz_sid = ALIGN_UP(anchor_per_batch * sizeof(int8_t));
	size_t sz_qspan = ALIGN_UP(anchor_per_batch * sizeof(uint8_t));
	size_t sz_xrev = ALIGN_UP(anchor_per_batch * sizeof(int32_t));
	size_t sz_range = ALIGN_UP(anchor_per_batch * sizeof(int32_t));
	size_t sz_f = ALIGN_UP(anchor_per_batch * sizeof(int32_t));
	size_t sz_p = ALIGN_UP(anchor_per_batch * sizeof(uint16_t));
	// Index arrays
	size_t sz_start = ALIGN_UP(range_grid_size * sizeof(size_t));
	size_t sz_rend = ALIGN_UP(range_grid_size * sizeof(size_t));
	size_t sz_cstart = ALIGN_UP(range_grid_size * sizeof(size_t));
	// Cut arrays
	size_t sz_cut = ALIGN_UP(num_cut * sizeof(size_t));
	size_t sz_lsc = ALIGN_UP(sizeof(unsigned int));
	size_t sz_lseg = ALIGN_UP(max_segs * sizeof(seg_t));
	size_t sz_lsog = ALIGN_UP(max_segs * sizeof(seg_t));
	size_t sz_msc = ALIGN_UP(sizeof(unsigned int));
	size_t sz_mseg = ALIGN_UP(mid_segs * sizeof(seg_t));
	// Long seg buffer
	size_t sz_map = ALIGN_UP(max_segs * sizeof(unsigned));
	size_t sz_axl = ALIGN_UP(dev_mem->buffer_size_long * sizeof(int32_t));
	size_t sz_ayl = ALIGN_UP(dev_mem->buffer_size_long * sizeof(int32_t));
	size_t sz_sidl = ALIGN_UP(dev_mem->buffer_size_long * sizeof(int8_t));
	size_t sz_qsl = ALIGN_UP(dev_mem->buffer_size_long * sizeof(uint8_t));
	size_t sz_rl = ALIGN_UP(dev_mem->buffer_size_long * sizeof(int32_t));
	size_t sz_xrl = ALIGN_UP(dev_mem->buffer_size_long * sizeof(int32_t));
	size_t sz_tnl = ALIGN_UP(sizeof(size_t));
	size_t sz_fl = ALIGN_UP(dev_mem->buffer_size_long * sizeof(int32_t));
	size_t sz_pl = ALIGN_UP(dev_mem->buffer_size_long * sizeof(uint16_t));
	size_t sz_lsid = ALIGN_UP(sizeof(unsigned));

	// d_anchors_raw aliases {d_range, d_f, d_p}: raw is only live during
	// H2D + extract; range/f/p are written afterwards on the same stream.
	size_t sz_shared = sz_raw > (sz_range + sz_f + sz_p)
	    ? sz_raw
	    : (sz_range + sz_f + sz_p);

	size_t total = sz_ax + sz_ay + sz_sid + sz_qspan + sz_xrev + sz_shared +
	    sz_start + sz_rend + sz_cstart + sz_cut +
	    sz_lsc + sz_lseg + sz_lsog + sz_msc + sz_mseg +
	    sz_map +
	    sz_axl + sz_ayl + sz_sidl + sz_qsl + sz_rl + sz_xrl +
	    sz_tnl + sz_fl + sz_pl + sz_lsid;

	char* base;
	if (mempool && stream) {
		cudaMallocFromPoolAsync((void**)&base, total, mempool, stream);
	} else {
		cudaMalloc((void**)&base, total);
	}
	cudaCheck();
	dev_mem->dev_base = base;

	size_t off = 0;
	dev_mem->d_ax = (int32_t*)(base + off);
	off += sz_ax;
	dev_mem->d_ay = (int32_t*)(base + off);
	off += sz_ay;
	dev_mem->d_sid = (int8_t*)(base + off);
	off += sz_sid;
	dev_mem->d_qspan = (uint8_t*)(base + off);
	off += sz_qspan;
	dev_mem->d_xrev = (int32_t*)(base + off);
	off += sz_xrev;
	// Shared region: d_anchors_raw overlaps d_range + d_f + d_p
	char* shared_base = base + off;
	dev_mem->d_anchors_raw = (uint64_t*)shared_base;
	dev_mem->d_range = (int32_t*)shared_base;
	dev_mem->d_f = (int32_t*)(shared_base + sz_range);
	dev_mem->d_p = (uint16_t*)(shared_base + sz_range + sz_f);
	off += sz_shared;
	dev_mem->d_start_idx = (size_t*)(base + off);
	off += sz_start;
	dev_mem->d_read_end_idx = (size_t*)(base + off);
	off += sz_rend;
	dev_mem->d_cut_start_idx = (size_t*)(base + off);
	off += sz_cstart;
	dev_mem->d_cut = (size_t*)(base + off);
	off += sz_cut;
	dev_mem->d_long_seg_count = (unsigned int*)(base + off);
	off += sz_lsc;
	dev_mem->d_long_seg = (seg_t*)(base + off);
	off += sz_lseg;
	dev_mem->d_long_seg_og = (seg_t*)(base + off);
	off += sz_lsog;
	dev_mem->d_mid_seg_count = (unsigned int*)(base + off);
	off += sz_msc;
	dev_mem->d_mid_seg = (seg_t*)(base + off);
	off += sz_mseg;
	dev_mem->d_map = (unsigned*)(base + off);
	off += sz_map;
	dev_mem->d_ax_long = (int32_t*)(base + off);
	off += sz_axl;
	dev_mem->d_ay_long = (int32_t*)(base + off);
	off += sz_ayl;
	dev_mem->d_sid_long = (int8_t*)(base + off);
	off += sz_sidl;
	dev_mem->d_qspan_long = (uint8_t*)(base + off);
	off += sz_qsl;
	dev_mem->d_range_long = (int32_t*)(base + off);
	off += sz_rl;
	dev_mem->d_xrev_long = (int32_t*)(base + off);
	off += sz_xrl;
	dev_mem->d_total_n_long = (size_t*)(base + off);
	off += sz_tnl;
	dev_mem->d_f_long = (int32_t*)(base + off);
	off += sz_fl;
	dev_mem->d_p_long = (uint16_t*)(base + off);
	off += sz_pl;
	dev_mem->d_long_segid = (unsigned*)(base + off);

#undef ALIGN_UP

#ifdef DEBUG_PRINT
	size_t gpu_free_mem, gpu_total_mem;
	cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);
	mm_gpu_log_info("Device alloc: {:.2f} GB consolidated, GPU free: {:.2f} GB",
	    (float)total / OneG, (float)gpu_free_mem / OneG);
#endif
}

void plmem_free_device_mem(deviceMemPtr* dev_mem, cudaStream_t stream)
{
	if (dev_mem->dev_base) {
		hipError_t err;
		if (stream) {
			err = cudaFreeAsync(dev_mem->dev_base, stream);
			if (err != hipSuccess)
				mm_log_error("[gpu-chain] cudaFreeAsync({:p}, {:p}) failed: {} ({})",
				    dev_mem->dev_base, (void*)stream, (int)err, hipGetErrorString(err));
		} else {
			err = cudaFree(dev_mem->dev_base);
			if (err != hipSuccess)
				mm_log_error("[gpu-chain] cudaFree({:p}) failed: {} ({})",
				    dev_mem->dev_base, (int)err, hipGetErrorString(err));
		}
		dev_mem->dev_base = NULL;
	}
	cudaCheck();
}


/**
 * Input
 *  reads[]:    array
 *  n_reads
 *  config:     range kernel configuartions
 * Output
 *  *host_mem   populate host_mem
*/
void plmem_reorg_input_arr(const gpu_chain_ctx_t* ctx,
    chain_read_t* reads, int n_read,
    hostMemPtr* host_mem, uint64_t* staging_raw)
{
	const range_kernel_config_t config = ctx->range_config;
	size_t total_n = 0, cut_num = 0;
	size_t griddim = 0;

#ifdef DEBUG_VERBOSE
	mm_log_debug("plmem_reorg_input_arr: n_read={}, config.anchor_per_block={}, config.blockdim={}",
	    n_read, config.anchor_per_block, config.blockdim);
#endif

	host_mem->size = n_read;
	for (int i = 0; i < n_read; i++) {
		total_n += reads[i].n;
	}
	host_mem->total_n = total_n;

#ifdef DEBUG_VERBOSE
	mm_log_debug("plmem_reorg_input_arr: total_n={}", total_n);
#endif

	size_t idx = 0;
	for (int i = 0; i < n_read; i++) {
		int n = reads[i].n;
		// Skip reads with no anchors (can occur if no seeds found or filtering removed all anchors)
		if (n <= 0) continue;

		int block_num = ceil_div(n, config.anchor_per_block);

		host_mem->start_idx[griddim] = idx;
		size_t end_idx = idx + config.anchor_per_block;
		host_mem->read_end_idx[griddim] = idx + n;
		host_mem->cut_start_idx[griddim] = cut_num;

		// Process full blocks (all except the last)
		for (int j = 1; j < block_num; j++) {
			// Each full block has exactly (anchor_per_block / blockdim) segments
			cut_num += config.anchor_per_block / config.blockdim;
			host_mem->start_idx[griddim + j] = end_idx;
			end_idx = host_mem->start_idx[griddim + j] + config.anchor_per_block;
			host_mem->read_end_idx[griddim + j] = idx + n;
			host_mem->cut_start_idx[griddim + j] = cut_num;
		}

		// Process the last block (may be partial) using ceiling division
		int remaining_anchors = n - (block_num - 1) * config.anchor_per_block;
		int last_block_cuts = ceil_div(remaining_anchors, config.blockdim);
		cut_num += last_block_cuts;

#ifdef DEBUG_VERBOSE
		mm_log_debug("plmem_reorg_input_arr: read {}: n={}, block_num={}, remaining={}, cuts_added={}, griddim={}, cut_num={}",
		    i, n, block_num, remaining_anchors, last_block_cuts, griddim, cut_num);
#endif

		griddim += block_num;

		memcpy(&staging_raw[2 * idx], reads[i].a, n * sizeof(mm128_t));
		idx += n;
	}
	host_mem->cut_num = cut_num;
	host_mem->griddim = griddim;

#ifdef DEBUG_VERBOSE
	mm_log_debug("plmem_reorg_input_arr: FINAL cut_num={}, griddim={}", cut_num, griddim);
#endif
}

void plmem_async_h2d_short_memcpy(gpu_worker_t* worker, size_t uid)
{
	hostMemPtr* host_mem = &worker->host_mems[uid];
	deviceMemPtr* dev_mem = &worker->dev_mem;
	cudaStream_t* stream = &worker->cudastream;
	cudaMemcpyAsync(dev_mem->d_anchors_raw, worker->staging_raw,
	    sizeof(uint64_t) * 2 * host_mem->total_n, cudaMemcpyHostToDevice,
	    *stream);
	// start_idx, read_end_idx, cut_start_idx are contiguous on both host
	// and device (same ALIGN_UP layout).  One H2D covers all three.
	cudaMemcpyAsync(dev_mem->d_start_idx, host_mem->start_idx,
	    (size_t)((char*)host_mem->long_segs_num - (char*)host_mem->start_idx),
	    cudaMemcpyHostToDevice, *stream);
	cudaMemsetAsync(dev_mem->d_cut, 0xff,
	    sizeof(size_t) * host_mem->cut_num, *stream);
	cudaCheck();
	dev_mem->total_n = host_mem->total_n;
	dev_mem->num_cut = host_mem->cut_num;
	dev_mem->size = host_mem->size;
	dev_mem->griddim = host_mem->griddim;
}

void plmem_async_d2h_short_memcpy(gpu_worker_t* worker, size_t uid)
{
	hostMemPtr* host_mem = &worker->host_mems[uid];
	deviceMemPtr* dev_mem = &worker->dev_mem;
	cudaStream_t* stream = &worker->cudastream;
	cudaMemcpyAsync(host_mem->f, dev_mem->d_f,
	    sizeof(int32_t) * host_mem->total_n, cudaMemcpyDeviceToHost,
	    *stream);
	cudaMemcpyAsync(host_mem->p, dev_mem->d_p,
	    sizeof(uint16_t) * host_mem->total_n,
	    cudaMemcpyDeviceToHost, *stream);
	// copy back d_long_seg_count to long_segs_num, this is an accumulative value
	cudaMemcpyAsync(host_mem->long_segs_num, dev_mem->d_long_seg_count,
	    sizeof(unsigned int), cudaMemcpyDeviceToHost, *stream);
	cudaCheck();

#ifdef DEBUG_VERBOSE
	// Sync and check scores after copy
	cudaStreamSynchronize(*stream);
	if (host_mem->total_n > 0) {
		mm_log_debug("GPU Scores after D2H copy (uid={}, total_n={}):", uid, host_mem->total_n);
		char score_buf[256];
		int pos = 0;
		pos = snprintf(score_buf, sizeof(score_buf), "  First 10 scores:");
		for (size_t i = 0; i < host_mem->total_n && i < 10; i++) {
			pos += snprintf(score_buf + pos, sizeof(score_buf) - pos, " %d", host_mem->f[i]);
		}
		mm_log_debug("{}", score_buf);
		if (host_mem->total_n > 10) {
			pos = snprintf(score_buf, sizeof(score_buf), "  Last 10 scores:");
			for (size_t i = (host_mem->total_n > 10 ? host_mem->total_n - 10 : 0); i < host_mem->total_n; i++) {
				pos += snprintf(score_buf + pos, sizeof(score_buf) - pos, " %d", host_mem->f[i]);
			}
			mm_log_debug("{}", score_buf);
		}
	}
#endif
}

void plmem_async_d2h_long_memcpy(gpu_worker_t* worker)
{
	longMemPtr* long_mem = &worker->long_mem;
	deviceMemPtr* dev_mem = &worker->dev_mem;
	cudaStream_t* stream = &worker->cudastream;
	const int long_seg_cutoff = worker->parent_ctx->score_config.long_seg_cutoff;
	const int cut_unit = worker->parent_ctx->score_config.cut_unit;
	cudaMemcpyAsync(long_mem->long_segs_og_idx, dev_mem->d_long_seg_og,
	    dev_mem->buffer_size_long / (long_seg_cutoff * cut_unit) * sizeof(seg_t),
	    cudaMemcpyDeviceToHost, *stream);
	cudaMemcpyAsync(long_mem->long_segs_buf_idx, dev_mem->d_long_seg,
	    dev_mem->buffer_size_long / (long_seg_cutoff * cut_unit) * sizeof(seg_t),
	    cudaMemcpyDeviceToHost, *stream);
	cudaMemcpyAsync(long_mem->f_long, dev_mem->d_f_long, sizeof(int32_t) * dev_mem->buffer_size_long,
	    cudaMemcpyDeviceToHost, *stream);
	cudaMemcpyAsync(long_mem->p_long, dev_mem->d_p_long, sizeof(uint16_t) * dev_mem->buffer_size_long,
	    cudaMemcpyDeviceToHost, *stream);
	cudaMemcpyAsync(long_mem->total_long_segs_n, dev_mem->d_total_n_long, sizeof(size_t),
	    cudaMemcpyDeviceToHost, *stream);
	cudaMemcpyAsync(long_mem->total_long_segs_num, dev_mem->d_long_seg_count, sizeof(unsigned int),
	    cudaMemcpyDeviceToHost, *stream);
	cudaCheck();
}

#ifdef USEHIP
/* Lightweight kernel that writes identity map values for long segments.
 * Only threads with index < *long_seg_count write (the long kernel reads
 * at most that many entries from d_map). */
__global__ void long_seg_identity_map_kernel(unsigned* __restrict__ map,
    const unsigned int* __restrict__ long_seg_count,
    unsigned int max_segs)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < max_segs) {
		unsigned int n = *long_seg_count;
		if (i < n) map[i] = i;
	}
}

/* Extract SoA fields from interleaved (x,y) anchor pairs on device.
 * mm128_t layout: x = {xrev[31:0], ax[31:0]}, y = {.., sid[55:48], .., qspan[39:32], ay[31:0]} */
__global__ void soa_extract_kernel(const uint64_t* __restrict__ raw,
    int32_t* __restrict__ ax, int32_t* __restrict__ ay,
    int8_t* __restrict__ sid, uint8_t* __restrict__ qspan,
    int32_t* __restrict__ xrev, size_t total_n)
{
	size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= total_n) return;
	uint64_t x = raw[2 * i];
	uint64_t y = raw[2 * i + 1];
	ax[i] = (int32_t)x;
	ay[i] = (int32_t)y;
	xrev[i] = (int32_t)(x >> 32);
	sid[i] = (int8_t)((y >> 48) & 0xff);
	qspan[i] = (uint8_t)((y >> 32) & 0xff);
}
#endif

void plmem_async_sort_long_segs(gpu_worker_t* worker, unsigned int max_long_segs)
{
#ifdef USEHIP
	if (max_long_segs == 0) return;

	deviceMemPtr* dev_mem = &worker->dev_mem;
	cudaStream_t stream = worker->cudastream;

	/* With typically ~36 long segments the sort overhead outweighs its
	 * benefit.  Use identity map and let the work-stealing long kernel
	 * handle any load imbalance naturally. */
	unsigned int blocks = (max_long_segs + 255) / 256;
	long_seg_identity_map_kernel<<<blocks, 256, 0, stream>>>(
	    dev_mem->d_map, dev_mem->d_long_seg_count, max_long_segs);
	cudaCheck();
#else
	(void)worker;
	(void)max_long_segs;
#endif
}

void plmem_async_soa_extract(gpu_worker_t* worker, size_t uid)
{
#ifdef USEHIP
	hostMemPtr* host_mem = &worker->host_mems[uid];
	deviceMemPtr* dev_mem = &worker->dev_mem;
	size_t total_n = host_mem->total_n;
	if (total_n == 0) return;

	unsigned int blocks = (unsigned int)((total_n + 255) / 256);
	soa_extract_kernel<<<blocks, 256, 0, worker->cudastream>>>(
	    dev_mem->d_anchors_raw, dev_mem->d_ax, dev_mem->d_ay,
	    dev_mem->d_sid, dev_mem->d_qspan, dev_mem->d_xrev, total_n);
	// d_f and d_p alias d_anchors_raw and are contiguous in the shared
	// device region.  One memset zeros both after extract makes raw dead.
	size_t fp_clear = (size_t)((char*)dev_mem->d_p - (char*)dev_mem->d_f) + sizeof(uint16_t) * total_n;
	cudaMemsetAsync(dev_mem->d_f, 0, fp_clear, worker->cudastream);
	cudaCheck();
#else
	(void)worker;
	(void)uid;
#endif
}

//////////////////// Initialization and Cleanup ////////////////////////
// The former extern g_gpu_chain_ctx has been retired. All callers now
// own their gpu_chain_ctx_t (production code via mm_gpu_chain_ctx_create,
// tests via GpuEnvironment::s_default_ctx in test_gpu_common.h).

#include "cJSON.h"
cJSON* plmem_parse_gpu_config(const char filename[])
{
	// read json file to cstring
	char* buffer = 0;
	long length;
	FILE* f = fopen(filename, "rb");

	if (f) {
		fseek(f, 0, SEEK_END);
		length = ftell(f);
		fseek(f, 0, SEEK_SET);
		buffer = (char*)malloc(length + 1);
		if (buffer) {
			size_t nread = fread(buffer, 1, length, f);
			buffer[nread] = '\0';
		}
		fclose(f);
	}

	if (!buffer) {
		mm_log_error("fail to open gpu config file {}", filename);
		exit(1);
	}

	cJSON* json = cJSON_Parse(buffer);
	if (!json) {
		const char* error_ptr = cJSON_GetErrorPtr();
		if (error_ptr != NULL) {
			mm_log_error("cJSON error before {}", error_ptr);
		}
		exit(1);
	}

	return json;
}

int get_json_int(cJSON* json, const char name[])
{
	cJSON* elt = cJSON_GetObjectItem(json, name);
	if (!cJSON_IsNumber(elt)) {
		mm_log_error("cJSON error failed to get field {}", name);
		exit(1);
	}
	return elt->valueint;
}

// Return the integer value of a JSON field, or fallback if absent/not a number.
static int get_json_int_opt(cJSON* json, const char name[], int fallback)
{
	if (!json) return fallback;
	cJSON* elt = cJSON_GetObjectItem(json, name);
	if (!elt || !cJSON_IsNumber(elt)) return fallback;
	return elt->valueint;
}

// plmem_config_stream: caller passes ctx so we read max_range_grid bounds from
// the cached device_prop instead of querying again.
void plmem_config_stream(const gpu_chain_ctx_t* ctx,
    size_t* max_range_grid_, size_t* max_num_cut_,
    size_t max_total_n, size_t max_read, size_t min_n)
{
	size_t max_range_grid, max_num_cut;
	max_range_grid =
	    (max_total_n - 1) / ctx->range_config.anchor_per_block + 1 + max_read;
	max_num_cut = (max_total_n - 1) / ctx->range_config.blockdim + 1 + max_read;
	*max_range_grid_ = max_range_grid;
	*max_num_cut_ = max_num_cut;

	if (*max_range_grid_ > (size_t)ctx->device_list[0]->prop.maxGridSize[0]) {
		mm_log_error("Invalid memory config");
		exit(1);
	}
}

// plmem_initialize / plmem_config_kernels / plmem_config_batch have been
// retired. They were the legacy non-streaming init path which referenced
// the now-removed file-scope globals score_kernel_config /
// range_kernel_config and was never called by any caller after the streaming
// pipeline became the default.
#if 0 // legacy init disabled: kept here as a comment for archaeology.
void plmem_initialize(size_t *max_total_n_, int *max_read_,
                      int *min_anchors_) {
#ifndef GPU_CONFIG
    cJSON *json = plmem_parse_gpu_config("gpu/gpu_config.json");
#else
    cJSON *json = plmem_parse_gpu_config(GPU_CONFIG);
#endif
    plmem_config_kernels(json);
    int num_streams;
    size_t buffer_size_long;
    plmem_config_batch<true>(json, &num_streams, min_anchors_, max_total_n_,
                             max_read_, &buffer_size_long);
}
#endif // legacy init disabled

// Populate a gpu_stream_config_t with sensible defaults derived from GPU
// hardware properties. Called before any JSON overrides so every field has
// a valid baseline. Batch-size fields (max_total_n, max_read,
// long_seg_buffer_size) are left at 0 to trigger the auto-detect path in
// plmem_derive_batch_from_config.
gpu_stream_config_t plmem_config_defaults(const cudaDeviceProp* prop)
{
	gpu_stream_config_t c;
	memset(&c, 0, sizeof(c));

	int n_cu = prop->multiProcessorCount;

	// -- algorithm constants (same across all known configs) --
	c.min_n = 512;
	c.avg_read_n = 10000;

	// -- range kernel --
	c.range_blockdim = 512;
	c.range_cut_check_anchors = 10;
	c.range_anchor_per_block = c.range_blockdim * 64; // 32768

	// -- score kernel --
	// Batch accumulation (map.c) targets max_anchors_stream per dispatch,
	// which fits in a single micro-batch. micro_batch=1 avoids allocating
	// unused host buffers (saves ~1.4 GB pinned memory with 8 workers).
	c.score_micro_batch = 1;
	c.score_mid_blockdim = 512;
	c.score_short_griddim = n_cu * 8;
	c.score_long_griddim = n_cu * 2;
	c.score_mid_griddim = n_cu * 8;
	c.score_long_seg_cutoff = 20;
	c.score_mid_seg_cutoff = 3;

	// -- batch sizes left at 0 → auto-derived from VRAM in
	//    plmem_derive_batch_from_config --
	// c.max_total_n = 0;
	// c.max_read = 0;
	// c.long_seg_buffer_size = 0;

	return c;
}

// Parse a GPU config JSON file into a gpu_stream_config_t struct.
// Starts from hardware-derived defaults, then overrides with any fields
// present in the JSON.
gpu_stream_config_t plmem_parse_stream_config(const char* gpu_config_file)
{
	gpu_stream_config_t config;
	memset(&config, 0, sizeof(config));

	cJSON* json = plmem_parse_gpu_config(gpu_config_file);

	// batch config — 0 means "auto-detect from VRAM"
	config.min_n = get_json_int_opt(json, "min_n", 0);

	cJSON* max_total_n_json = cJSON_GetObjectItem(json, "max_total_n");
	cJSON* max_read_json = cJSON_GetObjectItem(json, "max_read");
	cJSON* long_seg_buffer_size_json = cJSON_GetObjectItem(json, "long_seg_buffer_size");
	if (max_total_n_json && cJSON_IsNumber(max_total_n_json))
		config.max_total_n = (size_t)max_total_n_json->valuedouble;
	if (max_read_json && cJSON_IsNumber(max_read_json))
		config.max_read = max_read_json->valueint;
	if (long_seg_buffer_size_json && cJSON_IsNumber(long_seg_buffer_size_json))
		config.long_seg_buffer_size = (size_t)long_seg_buffer_size_json->valuedouble;

	cJSON* avg_read_n_json = cJSON_GetObjectItem(json, "avg_read_n");
	if (avg_read_n_json && cJSON_IsNumber(avg_read_n_json))
		config.avg_read_n = (size_t)avg_read_n_json->valueint;

	// range kernel config — 0 means "use default"
	cJSON* range_json = cJSON_GetObjectItem(json, "range_kernel");
	config.range_blockdim = get_json_int_opt(range_json, "blockdim", 0);
	config.range_cut_check_anchors = get_json_int_opt(range_json, "cut_check_anchors", 0);
	config.range_anchor_per_block = get_json_int_opt(range_json, "anchor_per_block", 0);

	// score kernel config — 0 means "use default"
	cJSON* score_json = cJSON_GetObjectItem(json, "score_kernel");
	config.score_micro_batch = get_json_int_opt(score_json, "micro_batch", 0);
	config.score_mid_blockdim = get_json_int_opt(score_json, "mid_blockdim", 0);
	config.score_short_griddim = get_json_int_opt(score_json, "short_griddim", 0);
	config.score_long_griddim = get_json_int_opt(score_json, "long_griddim", 0);
	config.score_mid_griddim = get_json_int_opt(score_json, "mid_griddim", 0);
	config.score_long_seg_cutoff = get_json_int_opt(score_json, "long_seg_cutoff", 0);
	config.score_mid_seg_cutoff = get_json_int_opt(score_json, "mid_seg_cutoff", 0);

	cJSON_Delete(json);
	return config;
}

// Apply a gpu_stream_config_t to the per-ctx kernel config fields.
static void plmem_apply_stream_config(gpu_chain_ctx_t* ctx, const cudaDeviceProp* prop, const gpu_stream_config_t* config)
{
	// range kernel
	ctx->range_config.blockdim = config->range_blockdim;
	ctx->range_config.cut_check_anchors = config->range_cut_check_anchors;
	ctx->range_config.anchor_per_block = config->range_anchor_per_block;

	// Validate range config
	if (ctx->range_config.blockdim <= 0) {
		mm_log_error("[gpu config] range_kernel:blockdim must be positive, got {}",
		    ctx->range_config.blockdim);
		exit(1);
	}
	if (ctx->range_config.anchor_per_block <= 0) {
		mm_log_error("[gpu config] range_kernel:anchor_per_block must be positive, got {}",
		    ctx->range_config.anchor_per_block);
		exit(1);
	}
	if (ctx->range_config.anchor_per_block % ctx->range_config.blockdim != 0) {
		mm_log_warn("[gpu config] anchor_per_block ({}) is not evenly divisible by blockdim ({}). "
			    "This may cause uneven workload distribution across GPU threads.",
		    ctx->range_config.anchor_per_block, ctx->range_config.blockdim);
	}

	// score kernel: use the reference device properties instead of a ctx cache.
	ctx->score_config.short_blockdim = prop->warpSize;
	ctx->score_config.long_blockdim = prop->maxThreadsPerBlock;
	ctx->score_config.mid_blockdim = config->score_mid_blockdim;
	ctx->score_config.short_griddim = config->score_short_griddim;
	ctx->score_config.long_griddim = config->score_long_griddim;
	ctx->score_config.mid_griddim = config->score_mid_griddim;
	ctx->score_config.long_seg_cutoff = config->score_long_seg_cutoff;
	ctx->score_config.mid_seg_cutoff = config->score_mid_seg_cutoff;
	ctx->score_config.cut_unit = ctx->range_config.blockdim;
	ctx->score_config.micro_batch = config->score_micro_batch;
	if (ctx->score_config.micro_batch > MAX_MICRO_BATCH) {
		mm_log_error("[gpu config] score_kernel:micro_batch should be less than {}"
			     " or recompile with MAX_MICRO_BATCH=%d",
		    MAX_MICRO_BATCH, ctx->score_config.micro_batch);
		exit(1);
	}
}

// Per-anchor byte cost for device memory (one set per worker).
// d_ax(4) + d_ay(4) + d_sid(1) + d_qspan(1) + d_xrev(4) + d_range(4) +
// d_f(4) + d_p(2) = 24 bytes per anchor.
static const int DEVICE_BYTES_PER_ANCHOR = 4 + 4 + 1 + 1 + 4 + 4 + 4 + 2;

// Per-anchor byte cost for one host micro-batch (pinned memory).
// ax(4) + ay(4) + sid(1) + qspan(1) + xrev(4) + f(4) + p(2) = 20 bytes.
static const int HOST_BYTES_PER_ANCHOR = 4 + 4 + 1 + 1 + 4 + 4 + 2;

// Per-anchor byte cost for long-segment buffers (host + device each).
// Host: f_long(4) + p_long(2) = 6.  Device: 8 arrays ≈ 20.
static const int HOST_LONG_BYTES_PER_ANCHOR = 4 + 2;
static const int DEVICE_LONG_BYTES_PER_ANCHOR = 4 + 4 + 1 + 1 + 4 + 4 + 2;

// Compute the optimal number of anchors per kernel launch from GPU compute
// capacity.  The range kernel launches grid = total_n / anchor_per_block, so
// we target enough blocks to give each CU `occupancy_waves` waves of work,
// with 1.5× headroom for uneven anchor distribution across reads.
static size_t plmem_compute_optimal_anchors(int n_cu, int anchor_per_block)
{
	const int occupancy_waves = 4;
	return (size_t)n_cu * occupancy_waves * anchor_per_block * 3 / 2;
}

static float plmem_dev_cost_per_anchor(const gpu_chain_ctx_t* ctx)
{
	return DEVICE_BYTES_PER_ANCHOR + DEVICE_LONG_BYTES_PER_ANCHOR * 0.2f + 32.0f / ctx->range_config.blockdim;
}

static float plmem_host_cost_per_anchor(const gpu_chain_ctx_t* ctx)
{
	int micro_batch = ctx->score_config.micro_batch > 0
	    ? ctx->score_config.micro_batch
	    : 4;
	return HOST_BYTES_PER_ANCHOR * micro_batch + HOST_LONG_BYTES_PER_ANCHOR * 0.2f + 24.0f * micro_batch / ctx->range_config.anchor_per_block;
}

// Derive batch sizes from GPU compute capacity (not VRAM fill).
// Primary sizing: CU count × occupancy waves × anchor_per_block.
// VRAM and host RAM serve as upper-bound caps only.
static void plmem_derive_batch_from_config(const gpu_chain_ctx_t* ctx,
    const gpu_stream_config_t* config,
    int n_workers, int* min_n_,
    size_t* max_total_n_, int* max_read_,
    size_t* long_seg_buffer_size_)
{
	*min_n_ = config->min_n;

	if (config->max_total_n > 0 && config->max_read > 0) {
		*max_total_n_ = config->max_total_n / n_workers;
		*max_read_ = config->max_read / n_workers;
		*long_seg_buffer_size_ = config->long_seg_buffer_size / n_workers;
		return;
	}

	// --- Primary: GPU compute capacity ---
	const hipDeviceProp_t& ref_prop = ctx->device_list[0]->prop;
	size_t compute_optimal = plmem_compute_optimal_anchors(
	    ref_prop.multiProcessorCount,
	    ctx->range_config.anchor_per_block);

	// --- Memory caps (upper bounds — should rarely limit) ---
	float dev_cost = plmem_dev_cost_per_anchor(ctx);
	float host_cost = plmem_host_cost_per_anchor(ctx);

	size_t dev_total = ref_prop.totalGlobalMem;
	size_t dev_reserve = config->dev_mem_reserve;
	if (dev_reserve >= dev_total) dev_reserve = dev_total / 2;
	size_t dev_budget = (size_t)((double)(dev_total - dev_reserve) / n_workers * 0.9);

	long pages = sysconf(_SC_PHYS_PAGES);
	long page_size = sysconf(_SC_PAGESIZE);
	size_t phys_ram = (pages > 0 && page_size > 0)
	    ? (size_t)pages * (size_t)page_size
	    : (size_t)64 * OneG;
	size_t host_cap = phys_ram / 2 / n_workers;

	size_t max_n_dev = (size_t)(dev_budget / dev_cost);
	size_t max_n_host = (size_t)(host_cap / host_cost);

	size_t max_total_n = compute_optimal;
	if (max_total_n > max_n_dev) max_total_n = max_n_dev;
	if (max_total_n > max_n_host) max_total_n = max_n_host;

	size_t avg_n = config->avg_read_n > 0 ? config->avg_read_n : 5000;
	*max_read_ = (int)(max_total_n / avg_n);
	if (*max_read_ < 1) *max_read_ = 1;
	*max_total_n_ = (size_t)(*max_read_) * avg_n;

	*long_seg_buffer_size_ = *max_total_n_ / 5;

	mm_log_debug("[gpu] auto-detect: compute_optimal={:.1f}M ({} CUs × 4 waves"
		     " × {} apb × 1.5), dev_cap={:.1f}M, host_cap={:.1f}M, "
		     "max_total_n={}, max_read={}, n_workers={}, long_seg_buf={}",
	    (double)compute_optimal / 1e6,
	    ref_prop.multiProcessorCount,
	    ctx->range_config.anchor_per_block,
	    (double)max_n_dev / 1e6, (double)max_n_host / 1e6,
	    *max_total_n_, *max_read_, n_workers, *long_seg_buffer_size_);
}

// Merge a user-provided config into a defaults config. Non-zero fields
// in 'user' override the corresponding field in 'defaults'.
static gpu_stream_config_t plmem_merge_config(const gpu_stream_config_t* defaults,
    const gpu_stream_config_t* user)
{
	gpu_stream_config_t m = *defaults;
	if (user->min_n) m.min_n = user->min_n;
	if (user->max_total_n) m.max_total_n = user->max_total_n;
	if (user->max_read) m.max_read = user->max_read;
	if (user->long_seg_buffer_size) m.long_seg_buffer_size = user->long_seg_buffer_size;
	if (user->avg_read_n) m.avg_read_n = user->avg_read_n;
	if (user->range_blockdim) m.range_blockdim = user->range_blockdim;
	if (user->range_cut_check_anchors) m.range_cut_check_anchors = user->range_cut_check_anchors;
	if (user->range_anchor_per_block) m.range_anchor_per_block = user->range_anchor_per_block;
	if (user->score_micro_batch) m.score_micro_batch = user->score_micro_batch;
	if (user->score_mid_blockdim) m.score_mid_blockdim = user->score_mid_blockdim;
	if (user->score_short_griddim) m.score_short_griddim = user->score_short_griddim;
	if (user->score_long_griddim) m.score_long_griddim = user->score_long_griddim;
	if (user->score_mid_griddim) m.score_mid_griddim = user->score_mid_griddim;
	if (user->score_long_seg_cutoff) m.score_long_seg_cutoff = user->score_long_seg_cutoff;
	if (user->score_mid_seg_cutoff) m.score_mid_seg_cutoff = user->score_mid_seg_cutoff;
	// dev_mem_reserve is always taken from user (set by caller, not from JSON).
	if (user->dev_mem_reserve) m.dev_mem_reserve = user->dev_mem_reserve;
	return m;
}

// initialize the caller-provided gpu_chain_ctx from a config struct
void plmem_stream_initialize(gpu_chain_ctx_t* ctx,
    size_t* max_total_n_,
    int* max_read_, int* min_anchors_,
    const gpu_stream_config_t* config, int n_threads,
    int n_gpu_workers, gpu_device_list_t device_list)
{
	assert(ctx != NULL);

	ctx->device_list = std::move(device_list);
	if (ctx->device_list.empty()) {
		mm_log_error("[gpu] No GPU devices available for chaining");
		exit(1);
	}

	const int selected_device_id = ctx->device_list[0]->device_id;
	hipSetDevice(selected_device_id);

	// Use the first device descriptor as the sizing reference.
	const cudaDeviceProp& reference_prop = ctx->device_list[0]->prop;

	// Compute hardware-derived defaults, then merge with caller-provided
	// config (non-zero fields in config override the defaults).
	gpu_stream_config_t defaults = plmem_config_defaults(&reference_prop);
	gpu_stream_config_t merged = plmem_merge_config(&defaults, config);
	ctx->stream_config = merged;

	// Apply kernel configs before deriving batch sizes (range_config needed).
	plmem_apply_stream_config(ctx, &reference_prop, &merged);

	// --- Determine worker count ---
	int n_workers;
	if (n_gpu_workers > 0) {
		n_workers = n_gpu_workers;
	} else if (n_threads <= 1) {
		n_workers = 1;
	} else {
		// Size from compute-optimal batch and memory budgets.
		size_t optimal_n = plmem_compute_optimal_anchors(
		    reference_prop.multiProcessorCount,
		    ctx->range_config.anchor_per_block);
		float dev_cost = plmem_dev_cost_per_anchor(ctx);
		float host_cost = plmem_host_cost_per_anchor(ctx);
		size_t per_worker_dev = (size_t)(optimal_n * dev_cost);
		size_t per_worker_host = (size_t)(optimal_n * host_cost);

		size_t dev_total = reference_prop.totalGlobalMem;
		size_t dev_reserve = merged.dev_mem_reserve;
		if (dev_reserve >= dev_total) dev_reserve = dev_total / 2;
		size_t dev_avail = (size_t)((dev_total - dev_reserve) * 0.9);

		long pages = sysconf(_SC_PHYS_PAGES);
		long page_sz = sysconf(_SC_PAGESIZE);
		size_t phys_ram = (pages > 0 && page_sz > 0)
		    ? (size_t)pages * (size_t)page_sz
		    : (size_t)64 * OneG;
		size_t host_avail = phys_ram / 2;

		int n_from_dev = per_worker_dev > 0
		    ? (int)(dev_avail / per_worker_dev)
		    : n_threads;
		int n_from_host = per_worker_host > 0
		    ? (int)(host_avail / per_worker_host)
		    : n_threads;
		n_workers = n_from_dev < n_from_host ? n_from_dev : n_from_host;
		if (n_workers > n_threads) n_workers = n_threads;
		// Cap workers by XCD/XCC topology: target ~2 workers per compute
		// die (XCC). MI300X has 8 XCCs -> 16. Floored at 8 so single-die
		// parts (RDNA) and the no-attribute fallback keep the historical
		// cap. Empirically (map-ont, MI300X) 2/XCC is the knee: 16 workers
		// beat the old flat cap of 8 by 6-13% on gpu_chain and gpu_both,
		// while >2/XCC (24, 32) regressed from GPU-queue contention.
		int n_xcc = 0;
		if (hipDeviceGetAttribute(&n_xcc, hipDeviceAttributeNumberOfXccs, 0) != hipSuccess || n_xcc < 1)
			n_xcc = 4; // fallback -> worker_cap 8 (legacy behavior)
		int worker_cap = 2 * n_xcc;
		if (worker_cap < 8) worker_cap = 8;
		if (n_workers > worker_cap) n_workers = worker_cap;
		if (n_workers < 1) n_workers = 1;
		mm_log_debug("[gpu] auto workers: optimal_n={:.1f}M, "
			     "per_worker_dev={:.2f} GB, per_worker_host={:.2f} GB, "
			     "n_from_dev={}, n_from_host={}, n_xcc={}, "
			     "worker_cap={} → n_workers={}",
		    (double)optimal_n / 1e6,
		    (double)per_worker_dev / OneG,
		    (double)per_worker_host / OneG,
		    n_from_dev, n_from_host, n_xcc, worker_cap, n_workers);
	}
	if (n_workers > n_threads) n_workers = n_threads;

	size_t max_anchors_stream, max_range_grid, max_num_cut, long_seg_buffer_size;
	int max_read_local = 0, min_anchors_local = 0;

	size_t gpu_free_mem, gpu_total_mem;
	cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);

	plmem_derive_batch_from_config(ctx, &merged, n_workers, &min_anchors_local,
	    &max_anchors_stream, &max_read_local,
	    &long_seg_buffer_size);
	plmem_config_stream(ctx, &max_range_grid, &max_num_cut, max_anchors_stream,
	    max_read_local, min_anchors_local);

	// If the user explicitly set --gpu-chain-workers N, validate that the
	// requested worker count fits in available VRAM before allocating.  The
	// auto path already enforces this; the override path intentionally skips
	// it so expert users can exceed the conservative auto estimate — but we
	// still need to catch values that would actually OOM, and give an
	// actionable error rather than a mid-run crash or silent segfault.
	//
	// Note: max_anchors_stream is derived with the current n_workers already
	// baked in (plmem_derive_batch_from_config divides the budget by n_workers),
	// so per_worker_bytes × n_workers would cancel.  Instead, estimate using
	// the compute-optimal batch size at n_workers=1 as the per-worker ceiling —
	// this is what each worker would allocate if unconstrained by VRAM.
	if (n_gpu_workers > 0) {
		size_t optimal_n_1w = plmem_compute_optimal_anchors(
		    ctx->device_list[0]->prop.multiProcessorCount,
		    ctx->range_config.anchor_per_block);
		float dev_cost = plmem_dev_cost_per_anchor(ctx);
		size_t per_worker_ceil = (size_t)(optimal_n_1w * dev_cost);
		size_t total_needed = per_worker_ceil * (size_t)n_workers;
		if (total_needed > gpu_free_mem) {
			mm_log_error("[gpu] --gpu-chain-workers {} requires ~{:.2f} GB device "
				     "memory (≤{:.2f} GB/worker × {} workers) but only {:.2f} GB "
				     "is free. Lower --gpu-chain-workers or use 0 for auto-detect.",
			    n_workers,
			    (double)total_needed / OneG,
			    (double)per_worker_ceil / OneG,
			    n_workers,
			    (double)gpu_free_mem / OneG);
			exit(EXIT_FAILURE);
		}
	}

	ctx->n_workers = n_workers;

	ctx->workers = new gpu_worker_t[n_workers];
#ifdef DEBUG_PRINT
	mm_gpu_log_info("max anchors per stream: {}, max range grid {} "
			"max_num_cut {} long_seg_buffer_size {}",
	    max_anchors_stream, max_range_grid, max_num_cut,
	    long_seg_buffer_size);
#endif // DEBUG_PRINT

	for (int i = 0; i < n_workers; i++) {
		gpu_worker_t* w = &ctx->workers[i];
		w->parent_ctx = ctx; // back-pointer used by worker-taking helpers
		w->busy = false;
		const auto& device = ctx->device_list[i % ctx->device_list.size()];
		w->device_id = device ? device->device_id : 0;
		w->device_prop = device ? device->prop : hipDeviceProp_t{};
		w->device_mempool = device ? device->mempool : NULL;
		hipSetDevice(w->device_id);
		cudaStreamCreate(&w->cudastream);
		cudaCheck();
		w->dev_mem.buffer_size_long = long_seg_buffer_size;

		size_t sz_staging = ((max_anchors_stream * 2 * sizeof(uint64_t)) + 255) & ~(size_t)255;
		cudaMallocHost((void**)&w->staging_raw, sz_staging);
		cudaCheck();

		for (int j = 0; j < ctx->score_config.micro_batch; j++) {
			plmem_malloc_host_mem(&w->host_mems[j], max_anchors_stream,
			    max_range_grid);
		}
		plmem_malloc_long_mem(ctx, &w->long_mem, long_seg_buffer_size);
		plmem_malloc_device_mem(ctx, w->device_mempool, &w->dev_mem, max_anchors_stream,
		    max_range_grid, max_num_cut, w->cudastream);
		mm_log_debug("[gpu] Worker {}: device={}, stream+memory initialized", i, w->device_id);
		// Pre-compute max long segment count for sort buffer sizing
		w->max_long_segs = long_seg_buffer_size / (ctx->score_config.long_seg_cutoff * ctx->score_config.cut_unit);
		// Use async memset on the worker stream so it orders after the
		// async pool allocation above (both on w->cudastream).
		cudaMemsetAsync(w->dev_mem.d_long_seg_count, 0, sizeof(unsigned int), w->cudastream);
		cudaMemsetAsync(w->dev_mem.d_mid_seg_count, 0, sizeof(unsigned int), w->cudastream);
		cudaMemsetAsync(w->dev_mem.d_total_n_long, 0, sizeof(size_t), w->cudastream);
		cudaCheck();
	}

	// Sync all worker streams so async allocs + memsets are complete before use.
	for (int i = 0; i < n_workers; i++)
		cudaStreamSynchronize(ctx->workers[i].cudastream);

	cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);
#ifdef DEBUG_PRINT
	mm_gpu_log_info("GPU free mem: {:.2f} GB, total mem: {:.2f} GB", (float)gpu_free_mem / OneG, (float)gpu_total_mem / OneG);
#endif

	// Validate micro_batch before using it in calculations
	if (ctx->score_config.micro_batch <= 0) {
		mm_log_warn("Invalid micro_batch (%d), defaulting to 1",
		    ctx->score_config.micro_batch);
		ctx->score_config.micro_batch = 1;
	}

	const size_t resolved_max_total_n = max_anchors_stream * ctx->score_config.micro_batch;
	const int resolved_max_read = max_read_local * ctx->score_config.micro_batch;

	ctx->max_anchors_per_worker = max_anchors_stream;
	ctx->max_range_grid = max_range_grid;
	ctx->max_num_cut = max_num_cut;
	ctx->long_seg_buffer_size = long_seg_buffer_size;

	// Persist post-derivation batch sizing on the ctx (replaces the writes
	// that used to go back into mm_mapopt_t::gpu_chain_*).
	ctx->max_anchors_per_batch = resolved_max_total_n;
	ctx->max_reads_per_batch = resolved_max_read;
	ctx->min_n_anchors = min_anchors_local;

	// Heuristic: Derive optimal batch size from max_total_n and micro_batch
	// Each host batch should accumulate max_total_n / micro_batch anchors
	// to saturate the GPU when micro_batch host batches are aggregated
	ctx->target_batch_anchors = resolved_max_total_n / ctx->score_config.micro_batch;

	// Enforce minimum threshold to prevent starvation with pathological configs
	if (ctx->target_batch_anchors < 1000) {
		mm_log_warn("Auto-derived target_batch_anchors (%zu) too small, using minimum 1000",
		    ctx->target_batch_anchors);
		ctx->target_batch_anchors = 1000;
	}

#ifdef DEBUG_PRINT
	mm_gpu_log_info("Auto-derived target_batch_anchors: %zu (max_total_n=%zu, micro_batch=%d)",
	    ctx->target_batch_anchors, resolved_max_total_n, ctx->score_config.micro_batch);
#endif

	// Optional out-params: preserved so existing tests can read sizing without
	// going through the getters. NULL = uninterested.
	if (max_total_n_) *max_total_n_ = resolved_max_total_n;
	if (max_read_) *max_read_ = resolved_max_read;
	if (min_anchors_) *min_anchors_ = min_anchors_local;

	// Initialize worker pool (O(1) free-stack)
	ctx->n_threads = n_threads;
	ctx->thread_worker_map = new int[n_threads];
	for (int i = 0; i < n_threads; i++)
		ctx->thread_worker_map[i] = -1;
	ctx->free_stack = new int[n_workers];
	ctx->free_top = n_workers;
	for (int i = 0; i < n_workers; i++)
		ctx->free_stack[i] = i;
	pthread_mutex_init(&ctx->pool_mutex, NULL);
	pthread_cond_init(&ctx->pool_cond, NULL);

	mm_log_debug("[gpu] Worker pool: {} GPU workers for {} threads", n_workers, n_threads);
	cudaCheck();
}

void plmem_stream_cleanup(gpu_chain_ctx_t* ctx)
{
	if (ctx == NULL || ctx->n_workers == 0) return;
	mm_log_debug("[gpu-chain] stream_cleanup: {} workers", ctx->n_workers);
	for (int i = 0; i < ctx->n_workers; i++) {
		gpu_worker_t* w = &ctx->workers[i];
		hipSetDevice(w->device_id);
		mm_log_debug("[gpu-chain] cleanup worker[{}]: stream={:p} dev_base={:p}",
		    i, (void*)w->cudastream, w->dev_mem.dev_base);
		plmem_free_device_mem(&w->dev_mem, w->cudastream);
		hipError_t err;
		err = cudaStreamSynchronize(w->cudastream);
		if (err != hipSuccess)
			mm_log_error("[gpu-chain] cudaStreamSynchronize worker[{}] failed: {} ({})",
			    i, (int)err, hipGetErrorString(err));
		err = cudaStreamDestroy(w->cudastream);
		if (err != hipSuccess)
			mm_log_error("[gpu-chain] cudaStreamDestroy worker[{}] failed: {} ({})",
			    i, (int)err, hipGetErrorString(err));
		cudaCheck();
		if (w->staging_raw) {
			err = cudaFreeHost(w->staging_raw);
			if (err != hipSuccess)
				mm_log_error("[gpu-chain] cudaFreeHost worker[{}] staging failed: {} ({})",
				    i, (int)err, hipGetErrorString(err));
			w->staging_raw = NULL;
		}
		for (int j = 0; j < ctx->score_config.micro_batch; j++) {
			plmem_free_host_mem(&w->host_mems[j]);
		}
		plmem_free_long_mem(&w->long_mem);
	}
	mm_log_debug("[gpu-chain] all workers cleaned up");
	delete[] ctx->workers;
	delete[] ctx->thread_worker_map;
	delete[] ctx->free_stack;
	pthread_mutex_destroy(&ctx->pool_mutex);
	pthread_cond_destroy(&ctx->pool_cond);

	// Leave the struct in a zeroed state so a follow-up gpu_chain_init
	// (or NULL-safe checks like in gpu_chain_free) sees an empty ctx.
	memset(ctx, 0, sizeof(*ctx));
}

/**
 * Acquire a GPU worker for the given thread.
 * If the thread already holds a worker, returns it immediately.
 * Otherwise pops from the free-stack (O(1)), blocking if empty.
 * Returns the worker index.
 */
int plmem_pool_acquire(gpu_chain_ctx_t* ctx, int tid)
{
	pthread_mutex_lock(&ctx->pool_mutex);

	// Already holding a worker?
	if (ctx->thread_worker_map[tid] >= 0) {
		int wid = ctx->thread_worker_map[tid];
		pthread_mutex_unlock(&ctx->pool_mutex);
		return wid;
	}

	// Pop from free-stack (block if empty)
	while (ctx->free_top == 0)
		pthread_cond_wait(&ctx->pool_cond, &ctx->pool_mutex);

	int wid = ctx->free_stack[--ctx->free_top];
	ctx->thread_worker_map[tid] = wid;
	pthread_mutex_unlock(&ctx->pool_mutex);
	return wid;
}

/**
 * Release the GPU worker held by the given thread back to the pool.
 * Pushes onto the free-stack (O(1)) and signals waiting threads.
 */
void plmem_pool_release(gpu_chain_ctx_t* ctx, int tid)
{
	pthread_mutex_lock(&ctx->pool_mutex);
	int wid = ctx->thread_worker_map[tid];
	if (wid >= 0) {
		ctx->thread_worker_map[tid] = -1;
		ctx->free_stack[ctx->free_top++] = wid;
		pthread_cond_signal(&ctx->pool_cond);
	}
	pthread_mutex_unlock(&ctx->pool_mutex);
}

extern "C" int gpu_chain_ctx_worker_device_id(const gpu_chain_ctx_t* ctx, int worker_id)
{
	if (ctx == NULL || worker_id < 0 || worker_id >= ctx->n_workers) return -1;
	return ctx->workers[worker_id].device_id;
}
