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
 * GPU mapping-context concurrency tests.
 *
 * Reproduces the scenario exercised by the Python mappy bindings when several
 * Aligner instances are built concurrently (e.g. one per ThreadPoolExecutor
 * worker): each Aligner creates its own mm_mapctx_t, which spins up the GPU
 * device list and the chain/align worker pools. Concurrent creation and
 * destruction of these contexts from multiple host threads must be safe.
 */

#include "test_gpu_common.h"

#include "gpu/gpu_device_queries.h"

#include <atomic>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace
{

#define MM_TEST_REF_FA TEST_SUITE_DIR "/original/MT-human.fa"
#define MM_TEST_QRY_FA TEST_SUITE_DIR "/original/MT-orang.fa"

// Build a GPU-enabled option set pinned to device 0. `dev` must outlive the
// returned opt (mm_mapopt_t::gpu_device_ids is a non-owning pointer).
static void makeGpuOpt(mm_mapopt_t* opt, int* dev, bool gpu_chain, bool gpu_align)
{
	mm_idxopt_t io;
	mm_set_opt(nullptr, &io, opt);
	mm_set_opt("map-ont", &io, opt);
	opt->flag |= MM_F_CIGAR;
	if (gpu_chain) opt->flag |= MM_F_GPU_CHAIN;
	if (gpu_align) opt->flag |= MM_F_GPU_ALIGN;
	*dev = 0;
	opt->gpu_device_ids = dev;
	opt->gpu_num_devices = 1;
}

// Read the first sequence from a plain (uncompressed) FASTA file.
static std::string readFirstSeq(const char* path)
{
	std::ifstream in(path);
	std::string line, seq;
	bool in_seq = false;
	while (std::getline(in, line)) {
		if (!line.empty() && line[0] == '>') {
			if (in_seq) break; // stop at the second record
			in_seq = true;
			continue;
		}
		if (in_seq) {
			while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
				line.pop_back();
			seq += line;
		}
	}
	return seq;
}

// Fixture: load one shared, read-only index (mirrors a single mappy Aligner
// reference shared across workers) plus one query sequence.
class GpuMapctxConcurrency : public ::testing::Test
{
      protected:
	static mm_idx_t* s_idx;
	static std::string s_query;

	static void SetUpTestSuite()
	{
		mm_idxopt_t io;
		mm_mapopt_t mo;
		mm_set_opt(nullptr, &io, &mo);
		mm_set_opt("map-ont", &io, &mo);
		io.batch_size = 0x7fffffffffffffffLL;
		mm_idx_reader_t* r = mm_idx_reader_open(MM_TEST_REF_FA, &io, nullptr);
		ASSERT_NE(r, nullptr);
		s_idx = mm_idx_reader_read(r, 1);
		mm_idx_reader_close(r);
		ASSERT_NE(s_idx, nullptr);
		mm_idx_index_name(s_idx);
		s_query = readFirstSeq(MM_TEST_QRY_FA);
		ASSERT_FALSE(s_query.empty());
	}

	static void TearDownTestSuite()
	{
		if (s_idx) mm_idx_destroy(s_idx);
		s_idx = nullptr;
	}

	// Map a small batch through an existing context; returns total hits, or -1.
	int mapBatch(mm_mapctx_t* ctx, mm_mapopt_t* opt)
	{
		constexpr int n = 4;
		const char* names[n] = {"q0", "q1", "q2", "q3"};
		const char* seqs[n] = {s_query.c_str(), s_query.c_str(), s_query.c_str(), s_query.c_str()};
		int n_regs[n] = {0, 0, 0, 0};
		mm_reg1_t* regs[n] = {nullptr, nullptr, nullptr, nullptr};

		int rc = mm_map_batch_ctx(s_idx, n, names, seqs, opt, 3, ctx, n_regs, regs);
		int total = 0;
		for (int i = 0; i < n; ++i) {
			total += n_regs[i];
			for (int j = 0; j < n_regs[i]; ++j)
				free(regs[i][j].p);
			free(regs[i]);
		}
		return rc == 0 ? total : -1;
	}

	// One full worker cycle: create a private context, map a small batch,
	// destroy it. Returns total hits, or -1 on failure.
	int runWorker(bool gpu_chain, bool gpu_align)
	{
		int dev = 0;
		mm_mapopt_t opt;
		makeGpuOpt(&opt, &dev, gpu_chain, gpu_align);
		mm_mapopt_update(&opt, s_idx);

		mm_logger_t* lg = mm_logger_create(0, nullptr);
		mm_mapctx_t* ctx = mm_mapctx_create(lg, &opt, 3);
		if (ctx == nullptr) {
			mm_logger_destroy(lg);
			return -1;
		}
		int total = mapBatch(ctx, &opt);
		mm_mapctx_destroy(ctx);
		mm_logger_destroy(lg);
		return total;
	}

	// Mirror the Python mappy + ThreadPoolExecutor pattern under load: many
	// workers concurrently create their own context, map, and destroy it, so a
	// context teardown on one thread overlaps with mapping/creation on others.
	// Each thread records its per-iteration results in its own vector slot
	// (no shared atomics on the hot path).
	void runConcurrent(bool gpu_chain, bool gpu_align)
	{
		if (mm_gpu_visible_device_count() < 1)
			GTEST_SKIP() << "requires at least 1 visible GPU";

		constexpr int kThreads = 8;
		constexpr int kIters = 4;

		std::vector<std::vector<int>> results(kThreads);
		std::vector<std::thread> threads;
		threads.reserve(kThreads);
		for (int t = 0; t < kThreads; ++t) {
			threads.emplace_back([&, t]() {
				results[t].reserve(kIters);
				for (int iter = 0; iter < kIters; ++iter)
					results[t].push_back(runWorker(gpu_chain, gpu_align));
			});
		}
		for (auto& th : threads)
			th.join();

		int failures = 0;
		for (int t = 0; t < kThreads; ++t)
			for (int got : results[t])
				if (got <= 0) ++failures;
		EXPECT_EQ(failures, 0);
	}
};

mm_idx_t* GpuMapctxConcurrency::s_idx = nullptr;
std::string GpuMapctxConcurrency::s_query;

// Concurrent creation + mapping + destruction of independent GPU contexts, the
// mappy ThreadPoolExecutor-with-multiple-Aligners pattern. Pre-fix these
// reliably segfault at >= 8 threads.
TEST_F(GpuMapctxConcurrency, ConcurrentGpuChain) { runConcurrent(true, false); }
TEST_F(GpuMapctxConcurrency, ConcurrentGpuAlign) { runConcurrent(false, true); }
TEST_F(GpuMapctxConcurrency, ConcurrentGpuBoth) { runConcurrent(true, true); }

} // namespace
