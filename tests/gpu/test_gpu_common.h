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
 * Shared GPU test infrastructure.
 *
 * Provides:
 * - make_gpu_config(): Hardcoded GPU stream config for tests
 * - GpuEnvironment: Global test environment for GPU init/teardown lifecycle
 * - runGpuChaining(): Free function to run GPU chaining on a batch of reads
 * - GpuChainParamTest: Parameterized fixture for GPU chain verification tests
 * - GpuChainBehavioralTest: Behavioral fixture for GPU chain tests
 * - GpuAlignBehavioralTest: Behavioral fixture for GPU align tests
 * - GpuChainCpuAlignParamTest: Parameterized fixture for GPU chain + CPU align
 * - GpuFullPipelineParamTest: Parameterized fixture for full GPU pipeline tests
 *
 * GPU initialization is expensive (~9.5s, allocates ~35GB pinned memory),
 * so it is done once via GpuEnvironment (registered with
 * ::testing::AddGlobalTestEnvironment) and torn down at process exit.
 */
#ifndef TEST_GPU_COMMON_H
#define TEST_GPU_COMMON_H

#include "test_map_common.h"
#include "gpu/plutils.h"
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

// ============================================================================
// Shared GPU config
// ============================================================================

/**
 * Hardcoded GPU config for tests (mirrors configs/gpu_config.json).
 * Constructed directly as a struct — no file I/O or JSON parsing needed.
 */
inline gpu_stream_config_t make_gpu_config()
{
	gpu_stream_config_t c = {};
	// batch config
	c.min_n = 512;
	c.max_total_n = 500000000;
	c.max_read = 500000;
	c.long_seg_buffer_size = 100000000;
	c.avg_read_n = 0; // not used when max_total_n/max_read are set
	// range kernel
	c.range_blockdim = 512;
	c.range_cut_check_anchors = 10;
	c.range_anchor_per_block = 32768;
	// score kernel
	c.score_micro_batch = 1;
	c.score_mid_blockdim = 512;
	c.score_short_griddim = 2688;
	c.score_long_griddim = 144;
	c.score_mid_griddim = 2688;
	c.score_long_seg_cutoff = 20;
	c.score_mid_seg_cutoff = 3;
	return c;
}

// ============================================================================
// Global GPU lifecycle management
// ============================================================================

/**
 * Global test environment for GPU init/teardown.
 *
 * Registered once via ::testing::AddGlobalTestEnvironment so
 * gpu_chain_init and gpu_chain_free are called exactly once per
 * test process, regardless of how many fixtures use the GPU.
 *
 * Owns a heap-allocated default gpu_chain_ctx_t (s_default_ctx) that
 * the legacy GPU tests (test_gpu_chain.cpp, test_gpu_align.cpp) share
 * via runGpuChaining(). The newer per-test-owned ctx pattern (used by
 * test_gpu_chain_context.cpp) bypasses this singleton entirely; both
 * coexist on the same device through the per-device const-mem cache.
 *
 * The default ctx used to live as a process-wide static
 * (s_default_chain_ctx) inside the chain library itself, exposed
 * via gpu_chain_default_ctx(). It is now scoped to test code because
 * production code (main.c, cli/mm2_chain.cpp, map.c) creates its own
 * ctx through mm_gpu_chain_ctx_create() and the static singleton has
 * no remaining production callers.
 */
class GpuEnvironment : public ::testing::Environment
{
      public:
	static size_t s_max_total_n;
	static int s_max_reads;
	static int s_min_n;
	static bool s_initialized;
	// Heap-allocated default ctx, owned by this environment. Allocated
	// in ensureInitialized(), freed in TearDown(). NULL before init.
	static gpu_chain_ctx_t* s_default_ctx;

	void SetUp() override
	{
		// Actual GPU init is deferred to ensureInitialized() which needs an index.
	}

	/**
     * Lazy-initialize the GPU with the given index.
     * Thread-safe for single-threaded GTest execution.
     */
	static void ensureInitialized(const mm_idx_t* mi, const mm_mapopt_t* opt)
	{
		if (s_initialized) return;
		(void)mi;
		(void)opt; // Misc is uploaded lazily on first submit;
		// init no longer needs it.
		s_default_ctx = gpu_chain_alloc();
		gpu_stream_config_t config = make_gpu_config();
		gpu_chain_init(s_default_ctx, &config, 1, 0, mm_gpu_create_device_list(NULL, 0));
		// Mirror the resolved batch sizing into the static vars existing tests
		// read; gpu_chain_init no longer writes back to mm_mapopt_t.
		s_max_total_n = gpu_chain_ctx_max_anchors(s_default_ctx);
		s_max_reads = gpu_chain_ctx_max_reads(s_default_ctx);
		s_min_n = gpu_chain_ctx_min_n(s_default_ctx);
		s_initialized = true;
	}

	void TearDown() override
	{
		if (s_initialized) {
			gpu_chain_free(s_default_ctx, 1); // 1 thread
			gpu_chain_dealloc(s_default_ctx);
			s_default_ctx = nullptr;
			s_initialized = false;
		}
	}
};

// Static member definitions (inline for header-only)
inline size_t GpuEnvironment::s_max_total_n = 0;
inline int GpuEnvironment::s_max_reads = 0;
inline int GpuEnvironment::s_min_n = 0;
inline bool GpuEnvironment::s_initialized = false;
inline gpu_chain_ctx_t* GpuEnvironment::s_default_ctx = nullptr;

// ============================================================================
// Shared GPU chaining helper
// ============================================================================

/**
 * Run GPU chaining on a batch of reads.
 * Mimics the streaming GPU chaining pipeline:
 *   1. Submit reads via gpu_chain_submit (launches async GPU work)
 *   2. Finish remaining batches via gpu_chain_finish
 *   3. Copy results back to caller's reads array
 */
inline void runGpuChaining(const mm_idx_t* mi, const mm_mapopt_t* opt,
    chain_read_t* reads, int n_reads, void* km)
{
	chain_read_t* batch = reads;
	int batch_size = n_reads;
	int thread_id = 0;

	// Submit the batch for GPU chaining
	gpu_chain_submit(GpuEnvironment::s_default_ctx, mi, opt, &batch, &batch_size, thread_id, km);

	// Finish any remaining GPU work
	chain_read_t* out_reads = nullptr;
	int out_n = 0;
	gpu_chain_finish(GpuEnvironment::s_default_ctx, mi, opt, &out_reads, &out_n, thread_id, km);

	// If finish returned results, copy chain data back to original reads
	if (out_reads && out_n > 0) {
		for (int i = 0; i < std::min(out_n, n_reads); i++) {
			reads[i].n_u = out_reads[i].n_u;
			reads[i].u = out_reads[i].u;
			reads[i].a = out_reads[i].a;
			reads[i].n = out_reads[i].n;
			reads[i].frag_gap = out_reads[i].frag_gap;
			reads[i].rep_len = out_reads[i].rep_len;
		}
	}
}

// ============================================================================
// GPU align helper — single-read accumulator path
// ============================================================================

inline void runGpuAlign(const mm_idx_t* mi, const mm_mapopt_t* opt,
    chain_read_t* read, mm_reg1_t** regs, int* n_regs, void* km = NULL)
{
	mm_align_ctx_t* align_ctx = mm_align_ctx_create(opt, 1,
	    mm_gpu_create_device_list(NULL, 0));
	gpu_align_accum_t* accum = mm_gpu_align_ctx_acquire_accum(
	    align_ctx->gpu_align, opt, mi);
	mm_gpu_accum_add_read(accum, read, regs, n_regs, NULL, km, NULL);
	mm_gpu_accum_flush(accum, NULL, NULL);
	mm_gpu_align_ctx_release_accum(align_ctx->gpu_align, accum);
	mm_align_ctx_destroy(align_ctx);
}

// ============================================================================
// GPU test fixtures
// ============================================================================

/**
 * Find a representative dataset suitable for GPU behavioral tests.
 * Requirements:
 * - Has anchors, chains, and alignments
 * - Has enough anchors (>= 512) for GPU chaining
 * - Avoids "hg00438" which has known GPU/CPU non-determinism
 * Falls back to the first dataset if no ideal match is found.
 */
inline const MappingTestData* findGpuRepresentativeDataset()
{
	const auto& all = getAllTestData();
	// Phase 1: ideal — enough anchors + not hg00438
	for (auto* td : all) {
		if (td->n_expected_anchors >= 512 &&
		    td->n_expected_chains > 0 &&
		    td->n_expected_alignments > 0 &&
		    std::string(td->name).find("hg00438") == std::string::npos) {
			return td;
		}
	}
	// Phase 2: any dataset with anchors/chains/alignments
	for (auto* td : all) {
		if (td->n_expected_anchors > 0 &&
		    td->n_expected_chains > 0 &&
		    td->n_expected_alignments > 0) {
			return td;
		}
	}
	return all[0];
}

/**
 * GPU behavioral test fixture base.
 * Uses a single representative dataset that is suitable for GPU testing
 * (enough anchors for GPU chaining, avoids datasets with known non-determinism).
 */
class GpuBehavioralFixture : public MapTestFixture
{
      protected:
	const MappingTestData& getTestData() const override
	{
		static const MappingTestData* rep = findGpuRepresentativeDataset();
		return *rep;
	}
};

/**
 * Parameterized GPU chain test fixture.
 * For exact-match verification tests that run against all test datasets.
 */
class GpuChainParamTest : public MapParamTestFixture
{
      protected:
	void SetUp() override
	{
		MapParamTestFixture::SetUp();
		ASSERT_NE(mi, nullptr) << "Index must be loaded for GPU chain tests";
		GpuEnvironment::ensureInitialized(mi, &opt);
	}
};

/**
 * Behavioral GPU chain test fixture.
 * Uses a single GPU-suitable representative dataset for non-verification tests
 * (determinism, edge cases, batch processing, etc.).
 */
class GpuChainBehavioralTest : public GpuBehavioralFixture
{
      protected:
	void SetUp() override
	{
		GpuBehavioralFixture::SetUp();
		ASSERT_NE(mi, nullptr) << "Index must be loaded for GPU chain tests";
		GpuEnvironment::ensureInitialized(mi, &opt);
	}
};

/**
 * Parameterized GPU align test fixture.
 * For exact-match verification tests comparing GPU alignment vs CPU baseline.
 */
class GpuAlignParamTest : public MapParamTestFixture
{
      protected:
	void SetUp() override
	{
		MapParamTestFixture::SetUp();
		ASSERT_NE(mi, nullptr) << "Index must be loaded for GPU align tests";
		opt.flag |= MM_F_CIGAR;
		opt.flag |= MM_F_GPU_ALIGN;
	}

	void runAlignPipeline(chain_read_t& read, mm_reg1_t** regs, int* n_regs, void* km)
	{
		mm_map_seed(mi, &opt, &read, &tbuf->timers.seed, km, NULL);
		mm_map_chain(mi, &opt, &read, &tbuf->timers.chain, km, NULL);
		runGpuAlign(mi, &opt, &read, regs, n_regs, km);
	}
};

/**
 * Behavioral GPU align test fixture.
 * Uses a single GPU-suitable representative dataset for non-verification tests.
 */
class GpuAlignBehavioralTest : public GpuBehavioralFixture
{
      protected:
	void SetUp() override
	{
		GpuBehavioralFixture::SetUp();
		ASSERT_NE(mi, nullptr) << "Index must be loaded for GPU align tests";
		opt.flag |= MM_F_CIGAR;
		opt.flag |= MM_F_GPU_ALIGN;
	}

	void runAlignPipeline(chain_read_t& read, mm_reg1_t** regs, int* n_regs, void* km)
	{
		mm_map_seed(mi, &opt, &read, &tbuf->timers.seed, km, NULL);
		mm_map_chain(mi, &opt, &read, &tbuf->timers.chain, km, NULL);
		runGpuAlign(mi, &opt, &read, regs, n_regs, km);
	}

	struct ConstantChainAlignResult {
		mm_reg1_t* regs = nullptr;
		int n_regs = 0;
		ScopedMemPool km;
	};

	std::unique_ptr<ConstantChainAlignResult> runConstantChainAlign(mm_mapopt_t* use_opt = nullptr)
	{
		auto res = std::make_unique<ConstantChainAlignResult>();
		const MappingTestData& tc = getTestData();
		chain_read_t read;
		createReadWithConstantChain(read, res->km, tc, getPrimaryQuery());
		if (read.n > 0) {
			mm_reg1_t* r[1] = {nullptr};
			int nr[1] = {0};
			runGpuAlign(mi, use_opt ? use_opt : &opt, &read, r, nr, res->km);
			res->regs = r[0];
			res->n_regs = nr[0];
		}
		return res;
	}
};

/**
 * Parameterized GPU chain + CPU align test fixture.
 * Isolates GPU chaining's effect on downstream alignment:
 *   seed(CPU) -> chain(GPU) -> align(CPU).
 * When VerifyFullGpuPipelineVsCpuBaseline fails, comparing with this test
 * reveals whether divergence comes from GPU chaining or GPU alignment.
 */
class GpuChainCpuAlignParamTest : public MapParamTestFixture
{
      protected:
	void SetUp() override
	{
		MapParamTestFixture::SetUp();
		ASSERT_NE(mi, nullptr) << "Index must be loaded";
		opt.flag |= MM_F_CIGAR;
		// MM_F_GPU_ALIGN intentionally NOT set — alignment runs on CPU.
		GpuEnvironment::ensureInitialized(mi, &opt);
	}
};

/**
 * Parameterized full GPU pipeline test fixture (GPU chain + GPU align).
 * Mimics --gpu-chain --gpu-align: seed(CPU) -> chain(GPU) -> align(GPU).
 */
class GpuFullPipelineParamTest : public MapParamTestFixture
{
      protected:
	void SetUp() override
	{
		MapParamTestFixture::SetUp();
		ASSERT_NE(mi, nullptr) << "Index must be loaded";
		opt.flag |= MM_F_CIGAR;
		opt.flag |= MM_F_GPU_ALIGN;
		GpuEnvironment::ensureInitialized(mi, &opt);
	}

	void runGpuAlignAfterChain(chain_read_t& read, mm_reg1_t** regs, int* n_regs, void* km = NULL)
	{
		runGpuAlign(mi, &opt, &read, regs, n_regs, km);
	}
};

#endif // TEST_GPU_COMMON_H
