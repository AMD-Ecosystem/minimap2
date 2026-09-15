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
 * GPU Chaining Context Tests
 *
 * Verifies the mm_gpu_chain_ctx_t API:
 *   - mm_gpu_chain_ctx_create / mm_gpu_chain_ctx_destroy lifecycle
 *   - NULL-safe submit/finish/destroy on the CPU-only path
 *   - Context reuse across sequential runs
 *   - Context isolation across sequential ctxs
 *   - In-place embedded storage (simulates a future mm_run_ctx_t parent)
 *   - Concurrent submit-paths from independent ctxs sharing one device
 *   - RMQ fallback: ctx is valid but MM_F_RMQ routes through CPU chain
 */

#include "test_gpu_common.h"
#include "gpu/gpu_device_queries.h"

#include <atomic>
#include <thread>
#include <vector>
#include <cstring>

namespace
{

// Build the default device list for tests: auto-discover all visible GPUs.
static gpu_device_list_t make_test_device_list()
{
	return mm_gpu_create_device_list(NULL, 0);
}

// Run a single GPU chaining batch using an explicit ctx (vs. the singleton
// default ctx used by runGpuChaining in test_gpu_common.h). All submit/finish
// calls go through the same ctx so the per-device const-mem cache and per-ctx
// worker pool are exercised.
void runGpuChainingWithCtx(mm_gpu_chain_ctx_t* ctx,
    const mm_idx_t* mi, const mm_mapopt_t* opt,
    chain_read_t* reads, int n_reads, void* km)
{
	chain_read_t* batch = reads;
	int batch_size = n_reads;
	int thread_id = 0;

	gpu_chain_submit(ctx, mi, opt, &batch, &batch_size, thread_id, km);

	chain_read_t* out_reads = nullptr;
	int out_n = 0;
	gpu_chain_finish(ctx, mi, opt, &out_reads, &out_n, thread_id, km);

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

// Same as runGpuChainingWithCtx but uses an explicit thread id (tid). The
// concurrency tests below need each std::thread to pass its own tid into the
// worker-pool acquire / release path so the per-ctx thread_worker_map stays
// consistent. tid must be < ctx->n_threads.
void runGpuChainingWithCtxTid(gpu_chain_ctx_t* ctx,
    const mm_idx_t* mi, const mm_mapopt_t* opt,
    chain_read_t* read, int tid, void* km)
{
	chain_read_t* batch = read;
	int n = 1;
	gpu_chain_submit(ctx, mi, opt, &batch, &n, tid, km);

	chain_read_t* out = nullptr;
	int out_n = 0;
	gpu_chain_finish(ctx, mi, opt, &out, &out_n, tid, km);

	if (out && out_n > 0) {
		read->n_u = out[0].n_u;
		read->u = out[0].u;
		read->a = out[0].a;
		read->n = out[0].n;
		read->frag_gap = out[0].frag_gap;
		read->rep_len = out[0].rep_len;
	}
}

// Smaller per-worker batch sizes than make_gpu_config() so that the
// multi-worker / multi-ctx stress tests can run several workers at once
// without exhausting GPU memory. The test data has at most a few thousand
// anchors per query, so 5M/worker is generous.
inline gpu_stream_config_t make_small_gpu_config()
{
	gpu_stream_config_t c = make_gpu_config();
	c.max_total_n = 5000000; // anchors per micro batch
	c.max_read = 5000; // reads per micro batch
	c.long_seg_buffer_size = 1000000;
	return c;
}

} // namespace

// =============================================================================
// NullCtxNoOp
//
// Verifies that all NULL-ctx paths are safe and silent. Important because the
// non-_ctx wrappers in map.c rely on mm_gpu_chain_ctx_create returning NULL
// on the CPU-only path, and on gpu_chain_submit/finish/destroy being no-ops.
// =============================================================================
TEST(GpuChainContextNullPath, NullCtxIsSafe)
{
	// destroy(NULL) is documented as no-op
	mm_gpu_chain_ctx_destroy(nullptr);

	// submit(NULL, ...) bails immediately, sets out-params to (NULL, 0)
	chain_read_t* batch = nullptr;
	int batch_size = 99;
	gpu_chain_submit(nullptr, /*mi=*/nullptr, /*opt=*/nullptr,
	    &batch, &batch_size, /*tid=*/0, /*km=*/nullptr);
	EXPECT_EQ(batch, nullptr);
	EXPECT_EQ(batch_size, 0);

	// finish(NULL, ...) likewise
	chain_read_t* out_reads = (chain_read_t*)0xdeadbeef;
	int out_n = 99;
	gpu_chain_finish(nullptr, /*mi=*/nullptr, /*opt=*/nullptr,
	    &out_reads, &out_n, /*tid=*/0, /*km=*/nullptr);
	EXPECT_EQ(out_reads, nullptr);
	EXPECT_EQ(out_n, 0);

	// free(NULL, ...) is also documented as no-op
	gpu_chain_free(nullptr, /*n_threads=*/0);

	// The ctx-* getters return 0 on NULL
	EXPECT_EQ((size_t)0, gpu_chain_ctx_max_anchors(nullptr));
	EXPECT_EQ(0, gpu_chain_ctx_max_reads(nullptr));
	EXPECT_EQ(0, gpu_chain_ctx_min_n(nullptr));
}

TEST(GpuChainContextNullPath, CreateReturnsNullWhenFlagUnset)
{
	// mm_gpu_chain_ctx_create must return NULL when MM_F_GPU_CHAIN is unset,
	// so the non-_ctx wrappers fall through to the CPU path with no GPU work.
	mm_mapopt_t opt;
	mm_mapopt_init(&opt);
	opt.flag &= ~MM_F_GPU_CHAIN;
	mm_gpu_chain_ctx_t* ctx = mm_gpu_chain_ctx_create(&opt, /*n_threads=*/1, make_test_device_list());
	EXPECT_EQ(ctx, nullptr);
	// Also verify NULL opt is safe
	EXPECT_EQ(mm_gpu_chain_ctx_create(nullptr, 1), nullptr);
}

// =============================================================================
// Public-API lifecycle on the GPU path
//
// These exercise mm_gpu_chain_ctx_create / mm_gpu_chain_ctx_destroy with
// MM_F_GPU_CHAIN set, including reuse across sequential submits and
// isolation across sequential ctxs.
// =============================================================================
class GpuChainContextLifecycleTest : public GpuBehavioralFixture
{
      protected:
	// We deliberately inherit from GpuBehavioralFixture (NOT
	// GpuChainBehavioralTest), because the latter calls
	// GpuEnvironment::ensureInitialized which spins up a process-wide
	// singleton ctx. Each lifecycle test in this file creates its OWN
	// ctx; coexisting with the singleton would double-allocate GPU
	// worker buffers and potentially cause cross-ctx contention on the
	// shared device that is unrelated to what we are testing.
	//
	// We also need MM_F_GPU_CHAIN on opt and a valid gpu_config_file
	// for mm_gpu_chain_ctx_create to do real work; point it at the
	// in-tree configs/gpu_config.json.
	void SetUp() override
	{
		GpuBehavioralFixture::SetUp();
		opt.flag |= MM_F_GPU_CHAIN;
		/* opt.gpu_config_file is a non-owning const char *; TEST_GPU_CONFIG_FILE
         * is a string literal injected via target_compile_definitions, so its
         * lifetime is the whole process. */
		opt.gpu_config_file = TEST_GPU_CONFIG_FILE;
	}
};

TEST_F(GpuChainContextLifecycleTest, ContextReuse_SequentialRuns)
{
	mm_gpu_chain_ctx_t* ctx = mm_gpu_chain_ctx_create(&opt, /*n_threads=*/1, make_test_device_list());
	ASSERT_NE(ctx, nullptr) << "ctx_create with MM_F_GPU_CHAIN should succeed on a GPU build";

	const MappingTestData& tc = getTestData();
	if (tc.n_expected_anchors < (size_t)gpu_chain_ctx_min_n(ctx)) {
		mm_gpu_chain_ctx_destroy(ctx);
		GTEST_SKIP() << "Dataset too small for GPU chaining";
	}

	// Run #1
	{
		ScopedMemPool km;
		chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);
		populateAnchorsFromConstants(read, km, tc);
		runGpuChainingWithCtx(ctx, mi, &opt, &read, 1, km);
		EXPECT_GT(read.n_u, 0) << "first run should produce chains";
	}

	// Run #2 — same ctx, fresh read. Must not crash, must produce chains.
	{
		ScopedMemPool km;
		chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);
		populateAnchorsFromConstants(read, km, tc);
		runGpuChainingWithCtx(ctx, mi, &opt, &read, 1, km);
		EXPECT_GT(read.n_u, 0) << "ctx reuse must not break subsequent runs";
	}

	mm_gpu_chain_ctx_destroy(ctx);
}

TEST_F(GpuChainContextLifecycleTest, ContextIsolation_SequentialContexts)
{
	const MappingTestData& tc = getTestData();
	if (tc.n_expected_anchors == 0) GTEST_SKIP() << "no anchors";

	// Ctx A: create -> run -> destroy.
	mm_gpu_chain_ctx_t* ctxA = mm_gpu_chain_ctx_create(&opt, /*n_threads=*/1, make_test_device_list());
	ASSERT_NE(ctxA, nullptr);
	if (tc.n_expected_anchors < (size_t)gpu_chain_ctx_min_n(ctxA)) {
		mm_gpu_chain_ctx_destroy(ctxA);
		GTEST_SKIP() << "Dataset too small for GPU chaining";
	}
	{
		ScopedMemPool km;
		chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);
		populateAnchorsFromConstants(read, km, tc);
		runGpuChainingWithCtx(ctxA, mi, &opt, &read, 1, km);
		EXPECT_GT(read.n_u, 0);
	}
	mm_gpu_chain_ctx_destroy(ctxA);

	// Ctx B: create -> run -> destroy. No leftover device state from ctxA must
	// affect ctxB; if the per-worker d_long_segid (commit 5) or any other
	// state was static-shared, multi-batch differences would show here.
	mm_gpu_chain_ctx_t* ctxB = mm_gpu_chain_ctx_create(&opt, /*n_threads=*/1, make_test_device_list());
	ASSERT_NE(ctxB, nullptr);
	{
		ScopedMemPool km;
		chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);
		populateAnchorsFromConstants(read, km, tc);
		runGpuChainingWithCtx(ctxB, mi, &opt, &read, 1, km);
		EXPECT_GT(read.n_u, 0);
	}
	mm_gpu_chain_ctx_destroy(ctxB);
}

TEST_F(GpuChainContextLifecycleTest, MultiGpuWorkerAssignmentRoundRobin)
{
	int device_count = mm_gpu_visible_device_count();
	if (device_count < 2)
		GTEST_SKIP() << "requires at least 2 visible GPUs";
	opt.gpu_device_ids = (int*)malloc(2 * sizeof(int));
	ASSERT_NE(opt.gpu_device_ids, nullptr);
	opt.gpu_device_ids[0] = 0;
	opt.gpu_device_ids[1] = 1;
	opt.gpu_num_devices = 2;

	mm_gpu_chain_ctx_t* ctx = mm_gpu_chain_ctx_create(
	    &opt, /*n_threads=*/4,
	    mm_gpu_create_device_list(opt.gpu_device_ids, opt.gpu_num_devices));
	ASSERT_NE(ctx, nullptr);
	ASSERT_GE(gpu_chain_ctx_n_workers(ctx), 2);

	for (int i = 0; i < gpu_chain_ctx_n_workers(ctx); i++) {
		EXPECT_EQ(gpu_chain_ctx_worker_device_id(ctx, i), i % 2);
	}

	mm_gpu_chain_ctx_destroy(ctx);
	free(opt.gpu_device_ids);
	opt.gpu_device_ids = nullptr;
	opt.gpu_num_devices = 0;
}

// =============================================================================
// Embedded_StorageWorks (heap-owned variant)
//
// Drives the gpu_chain_alloc / gpu_chain_init / gpu_chain_free /
// gpu_chain_dealloc lifecycle directly. The plan's "embedded" goal is that
// gpu_chain_ctx_t is safe as an in-place by-value member of a future
// mm_run_ctx_t parent. Tests living in the chain TU can stack-alloc the
// struct; here in C++ test code outside the chain TU we use the heap helper
// pair, which exercises the same init / first-use / deinit path with no
// hidden assumption about static state. Subsequent commits introducing the
// parent ctx will add a stack-alloc variant inside the chain unit tests.
// =============================================================================
TEST_F(GpuChainContextLifecycleTest, Embedded_StorageWorks_HeapHelpers)
{
	gpu_chain_ctx_t* ctx = gpu_chain_alloc();
	ASSERT_NE(ctx, nullptr);

	gpu_stream_config_t cfg = make_gpu_config();
	gpu_chain_init(ctx, &cfg, /*n_threads=*/1, /*n_gpu_workers=*/0, make_test_device_list());

	const MappingTestData& tc = getTestData();
	if (tc.n_expected_anchors < (size_t)gpu_chain_ctx_min_n(ctx)) {
		gpu_chain_free(ctx, 1);
		gpu_chain_dealloc(ctx);
		GTEST_SKIP() << "Dataset too small for GPU chaining";
	}

	{
		ScopedMemPool km;
		chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);
		populateAnchorsFromConstants(read, km, tc);
		runGpuChainingWithCtx(ctx, mi, &opt, &read, 1, km);
		EXPECT_GT(read.n_u, 0) << "alloc/init/run/free/dealloc lifecycle must work";
	}

	gpu_chain_free(ctx, 1); // returns the ctx to a zeroed state
	gpu_chain_dealloc(ctx); // frees the struct itself
}

// =============================================================================
// Concurrent_IdenticalParams
//
// Two ctxs, each submitting on its own thread with identical Misc/configs.
// Verifies that concurrent submits with the same params produce correct output.
// =============================================================================
TEST_F(GpuChainContextLifecycleTest, Concurrent_IdenticalParams)
{
	const MappingTestData& tc = getTestData();
	if (tc.n_expected_anchors == 0) GTEST_SKIP() << "no anchors";

	mm_gpu_chain_ctx_t* ctxA = mm_gpu_chain_ctx_create(&opt, /*n_threads=*/1, make_test_device_list());
	mm_gpu_chain_ctx_t* ctxB = mm_gpu_chain_ctx_create(&opt, /*n_threads=*/1, make_test_device_list());
	ASSERT_NE(ctxA, nullptr);
	ASSERT_NE(ctxB, nullptr);
	if (tc.n_expected_anchors < (size_t)gpu_chain_ctx_min_n(ctxA)) {
		mm_gpu_chain_ctx_destroy(ctxA);
		mm_gpu_chain_ctx_destroy(ctxB);
		GTEST_SKIP() << "Dataset too small";
	}

	// Snapshot the upload counter -- some prior tests on the same process may
	// have already pushed constants for this device.

	auto run = [&](mm_gpu_chain_ctx_t* ctx) {
		ScopedMemPool km;
		chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);
		populateAnchorsFromConstants(read, km, tc);
		runGpuChainingWithCtx(ctx, mi, &opt, &read, 1, km);
	};

	std::thread t1(run, ctxA);
	std::thread t2(run, ctxB);
	t1.join();
	t2.join();

	mm_gpu_chain_ctx_destroy(ctxA);
	mm_gpu_chain_ctx_destroy(ctxB);
}

// =============================================================================
// RmqFallback
//
// MM_F_RMQ requires CPU chain (mg_lchain_rmq); even with a valid ctx, the
// chaining pipeline must take the CPU branch. We verify by direct call:
// gpu_chain_submit asserts !(opt->flag & MM_F_RMQ) so callers MUST gate. The
// ctx itself remains usable for non-RMQ work afterwards.
// =============================================================================
TEST_F(GpuChainContextLifecycleTest, RmqFlag_DoesNotInvalidateCtx)
{
	mm_gpu_chain_ctx_t* ctx = mm_gpu_chain_ctx_create(&opt, /*n_threads=*/1, make_test_device_list());
	ASSERT_NE(ctx, nullptr);

	// Toggle MM_F_RMQ (don't actually call gpu_chain_submit with it -- the
	// chain submit asserts on RMQ, mirroring map.c's MM_F_RMQ gate).
	mm_mapopt_t opt_rmq = opt;
	opt_rmq.flag |= MM_F_RMQ;

	// Sanity: gpu_chain_ctx_min_n still returns the configured value; the ctx
	// is valid regardless of opt flags.
	EXPECT_GT(gpu_chain_ctx_min_n(ctx), 0);

	// Without RMQ the same ctx still works.
	const MappingTestData& tc = getTestData();
	if (tc.n_expected_anchors >= (size_t)gpu_chain_ctx_min_n(ctx)) {
		ScopedMemPool km;
		chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);
		populateAnchorsFromConstants(read, km, tc);
		runGpuChainingWithCtx(ctx, mi, &opt, &read, 1, km);
		EXPECT_GT(read.n_u, 0);
	}

	mm_gpu_chain_ctx_destroy(ctx);
}

// =============================================================================
// SingleCtx_MultiThread_SameCtx
//
// One ctx with n_threads=2, n_gpu_workers=0 (auto -> 2 workers). Two threads
// share the ctx with distinct tids (0 and 1) and submit + finish concurrently.
// With one worker per thread there is no pool contention; this test verifies
// the basic multi-thread acquire/release path and that two simultaneous
// submits on independent workers produce correct output.
// =============================================================================
TEST_F(GpuChainContextLifecycleTest, SingleCtx_MultiThread_SameCtx)
{
	const MappingTestData& tc = getTestData();
	if (tc.n_expected_anchors == 0) GTEST_SKIP() << "no anchors";

	gpu_chain_ctx_t* ctx = gpu_chain_alloc();
	ASSERT_NE(ctx, nullptr);
	gpu_stream_config_t cfg = make_small_gpu_config();
	gpu_chain_init(ctx, &cfg, /*n_threads=*/2, /*n_gpu_workers=*/0, make_test_device_list());

	if (tc.n_expected_anchors < (size_t)gpu_chain_ctx_min_n(ctx)) {
		gpu_chain_free(ctx, 2);
		gpu_chain_dealloc(ctx);
		GTEST_SKIP() << "Dataset too small for GPU chaining";
	}

	auto worker_fn = [&](int tid, int* out_nu) {
		ScopedMemPool km;
		chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);
		populateAnchorsFromConstants(read, km, tc);
		runGpuChainingWithCtxTid(ctx, mi, &opt, &read, tid, km);
		*out_nu = read.n_u;
	};

	int nu0 = 0, nu1 = 0;
	std::thread t0(worker_fn, 0, &nu0);
	std::thread t1(worker_fn, 1, &nu1);
	t0.join();
	t1.join();

	EXPECT_GT(nu0, 0) << "thread 0 should produce chains";
	EXPECT_GT(nu1, 0) << "thread 1 should produce chains";
	// Same input data on both threads -> same number of chains.
	EXPECT_EQ(nu0, nu1) << "concurrent threads on the same input must agree";

	gpu_chain_free(ctx, 2);
	gpu_chain_dealloc(ctx);
}

// =============================================================================
// SingleCtx_MultiWorker_Saturated
//
// One ctx with n_threads=4 and explicit n_gpu_workers=4. Four threads submit
// concurrently; every thread immediately gets its own worker (saturated, no
// pthread_cond_wait wait). Verifies the full-fanout path through the per-ctx
// worker pool: no double-acquire, no deadlock, all four threads return real
// chains, and ctx->n_workers really is 4.
// =============================================================================
TEST_F(GpuChainContextLifecycleTest, SingleCtx_MultiWorker_Saturated)
{
	const MappingTestData& tc = getTestData();
	if (tc.n_expected_anchors == 0) GTEST_SKIP() << "no anchors";

	gpu_chain_ctx_t* ctx = gpu_chain_alloc();
	ASSERT_NE(ctx, nullptr);
	gpu_stream_config_t cfg = make_small_gpu_config();
	gpu_chain_init(ctx, &cfg, /*n_threads=*/4, /*n_gpu_workers=*/4, make_test_device_list());

	EXPECT_EQ(gpu_chain_ctx_n_workers(ctx), 4) << "explicit n_gpu_workers=4 should stick";

	if (tc.n_expected_anchors < (size_t)gpu_chain_ctx_min_n(ctx)) {
		gpu_chain_free(ctx, 4);
		gpu_chain_dealloc(ctx);
		GTEST_SKIP() << "Dataset too small for GPU chaining";
	}

	constexpr int N = 4;
	std::vector<int> nu(N, 0);

	auto worker_fn = [&](int tid) {
		ScopedMemPool km;
		chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);
		populateAnchorsFromConstants(read, km, tc);
		runGpuChainingWithCtxTid(ctx, mi, &opt, &read, tid, km);
		nu[tid] = read.n_u;
	};

	std::vector<std::thread> threads;
	threads.reserve(N);
	for (int i = 0; i < N; i++) threads.emplace_back(worker_fn, i);
	for (auto& t : threads) t.join();

	int expected = nu[0];
	EXPECT_GT(expected, 0) << "thread 0 should produce chains";
	for (int i = 1; i < N; i++) {
		EXPECT_EQ(nu[i], expected)
		    << "thread " << i << " disagrees on n_u (concurrency bug?)";
	}

	gpu_chain_free(ctx, 4);
	gpu_chain_dealloc(ctx);
}

// =============================================================================
// SingleCtx_MoreThreadsThanWorkers
//
// n_threads=4 but n_gpu_workers=2. Four threads submit concurrently; two get
// workers immediately, the other two block on pthread_cond_wait until the
// first two release. Verifies the over-subscription path: cond signalling,
// no double-acquire under release, and final output correctness even when
// half the submits queued behind the lock.
// =============================================================================
TEST_F(GpuChainContextLifecycleTest, SingleCtx_MoreThreadsThanWorkers)
{
	const MappingTestData& tc = getTestData();
	if (tc.n_expected_anchors == 0) GTEST_SKIP() << "no anchors";

	gpu_chain_ctx_t* ctx = gpu_chain_alloc();
	ASSERT_NE(ctx, nullptr);
	gpu_stream_config_t cfg = make_small_gpu_config();
	gpu_chain_init(ctx, &cfg, /*n_threads=*/4, /*n_gpu_workers=*/2, make_test_device_list());

	EXPECT_EQ(gpu_chain_ctx_n_workers(ctx), 2);
	EXPECT_EQ(gpu_chain_ctx_n_threads(ctx), 4);

	if (tc.n_expected_anchors < (size_t)gpu_chain_ctx_min_n(ctx)) {
		gpu_chain_free(ctx, 4);
		gpu_chain_dealloc(ctx);
		GTEST_SKIP() << "Dataset too small for GPU chaining";
	}

	constexpr int N = 4;
	std::vector<int> nu(N, 0);

	auto worker_fn = [&](int tid) {
		ScopedMemPool km;
		chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);
		populateAnchorsFromConstants(read, km, tc);
		runGpuChainingWithCtxTid(ctx, mi, &opt, &read, tid, km);
		nu[tid] = read.n_u;
	};

	std::vector<std::thread> threads;
	threads.reserve(N);
	for (int i = 0; i < N; i++) threads.emplace_back(worker_fn, i);
	for (auto& t : threads) t.join();

	int expected = nu[0];
	EXPECT_GT(expected, 0);
	for (int i = 1; i < N; i++) {
		EXPECT_EQ(nu[i], expected)
		    << "thread " << i << " (over-subscribed pool) produced wrong n_u";
	}

	gpu_chain_free(ctx, 4);
	gpu_chain_dealloc(ctx);
}

// =============================================================================
// Concurrent_DifferentParams
//
// Two ctxs, each with n_gpu_workers=2, submitting concurrently with DIFFERENT
// Misc parameters (different opt->max_gap -> different misc.max_dist_x).
// Verifies that concurrent submits with different params each produce correct
// output (no cross-contamination between kernel invocations).
// =============================================================================
TEST_F(GpuChainContextLifecycleTest, Concurrent_DifferentParams)
{
	const MappingTestData& tc = getTestData();
	if (tc.n_expected_anchors == 0) GTEST_SKIP() << "no anchors";

	gpu_chain_ctx_t* ctxA = gpu_chain_alloc();
	gpu_chain_ctx_t* ctxB = gpu_chain_alloc();
	ASSERT_NE(ctxA, nullptr);
	ASSERT_NE(ctxB, nullptr);
	gpu_stream_config_t cfg = make_small_gpu_config();
	gpu_chain_init(ctxA, &cfg, /*n_threads=*/1, /*n_gpu_workers=*/2, make_test_device_list());
	gpu_chain_init(ctxB, &cfg, /*n_threads=*/1, /*n_gpu_workers=*/2, make_test_device_list());

	if (tc.n_expected_anchors < (size_t)gpu_chain_ctx_min_n(ctxA)) {
		gpu_chain_free(ctxA, 1);
		gpu_chain_dealloc(ctxA);
		gpu_chain_free(ctxB, 1);
		gpu_chain_dealloc(ctxB);
		GTEST_SKIP() << "Dataset too small";
	}

	// ctxA and ctxB each use a copy of opt with DIFFERENT max_gap values, both
	// distinct from `opt.max_gap` itself. The third sentinel value below is
	// also distinct from both -- it primes the cache to a state that neither
	// optA nor optB matches, so when the two ctxs submit they BOTH miss.
	// Without the sentinel primer, prior tests in the same process may have
	// already left the cache holding `opt.max_gap`'s Misc; ctxA would then
	// hit and we'd only see one upload from this test.
	mm_mapopt_t optA = opt;
	mm_mapopt_t optB = opt;
	optA.max_gap = (opt.max_gap > 1000 ? opt.max_gap / 2 : opt.max_gap * 2);
	optB.max_gap = (opt.max_gap > 1000 ? opt.max_gap / 4 : opt.max_gap * 4);

	auto run = [&](gpu_chain_ctx_t* ctx, const mm_mapopt_t* o, int* out_nu) {
		ScopedMemPool km;
		chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);
		populateAnchorsFromConstants(read, km, tc);
		runGpuChainingWithCtxTid(ctx, mi, o, &read, /*tid=*/0, km);
		*out_nu = read.n_u;
	};

	int nuA = 0, nuB = 0;
	std::thread tA(run, ctxA, &optA, &nuA);
	std::thread tB(run, ctxB, &optB, &nuB);
	tA.join();
	tB.join();

	EXPECT_GT(nuA, 0) << "ctxA produced no chains";
	EXPECT_GT(nuB, 0) << "ctxB produced no chains";

	gpu_chain_free(ctxA, 1);
	gpu_chain_dealloc(ctxA);
	gpu_chain_free(ctxB, 1);
	gpu_chain_dealloc(ctxB);
}

// =============================================================================
// Concurrent_MultiWorker_Stress
//
// 2 ctxs * 4 workers/ctx, each ctx fed by 4 threads = 8 total in-flight
// submits worst case. Identical Misc/configs across all submits. Verifies:
//   - no deadlocks (test joins cleanly)
//   - all 8 results agree on n_u (same input, same params -> same output).
// =============================================================================
TEST_F(GpuChainContextLifecycleTest, Concurrent_MultiWorker_Stress)
{
	const MappingTestData& tc = getTestData();
	if (tc.n_expected_anchors == 0) GTEST_SKIP() << "no anchors";

	gpu_chain_ctx_t* ctxA = gpu_chain_alloc();
	gpu_chain_ctx_t* ctxB = gpu_chain_alloc();
	ASSERT_NE(ctxA, nullptr);
	ASSERT_NE(ctxB, nullptr);
	gpu_stream_config_t cfg = make_small_gpu_config();
	gpu_chain_init(ctxA, &cfg, /*n_threads=*/4, /*n_gpu_workers=*/4, make_test_device_list());
	gpu_chain_init(ctxB, &cfg, /*n_threads=*/4, /*n_gpu_workers=*/4, make_test_device_list());

	if (tc.n_expected_anchors < (size_t)gpu_chain_ctx_min_n(ctxA)) {
		gpu_chain_free(ctxA, 4);
		gpu_chain_dealloc(ctxA);
		gpu_chain_free(ctxB, 4);
		gpu_chain_dealloc(ctxB);
		GTEST_SKIP() << "Dataset too small";
	}

	constexpr int N = 4; // threads per ctx
	std::vector<int> nuA(N, 0), nuB(N, 0);

	// Warm the cache once with the same params before snapshotting the
	// counter, so the >= 2 ctxs concurrent below cannot also be the very
	// first submit ever (which would be an unavoidable miss).
	{
		ScopedMemPool km;
		chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);
		populateAnchorsFromConstants(read, km, tc);
		runGpuChainingWithCtxTid(ctxA, mi, &opt, &read, /*tid=*/0, km);
	}

	auto worker_fn = [&](gpu_chain_ctx_t* ctx, int tid, std::vector<int>& dst) {
		ScopedMemPool km;
		chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);
		populateAnchorsFromConstants(read, km, tc);
		runGpuChainingWithCtxTid(ctx, mi, &opt, &read, tid, km);
		dst[tid] = read.n_u;
	};

	std::vector<std::thread> threads;
	threads.reserve(2 * N);
	for (int i = 0; i < N; i++) threads.emplace_back(worker_fn, ctxA, i, std::ref(nuA));
	for (int i = 0; i < N; i++) threads.emplace_back(worker_fn, ctxB, i, std::ref(nuB));
	for (auto& t : threads) t.join();

	int expected = nuA[0];
	EXPECT_GT(expected, 0);
	for (int i = 0; i < N; i++) {
		EXPECT_EQ(nuA[i], expected) << "ctxA tid=" << i << " disagrees";
		EXPECT_EQ(nuB[i], expected) << "ctxB tid=" << i << " disagrees";
	}

	gpu_chain_free(ctxA, 4);
	gpu_chain_dealloc(ctxA);
	gpu_chain_free(ctxB, 4);
	gpu_chain_dealloc(ctxB);
}

// =============================================================================
// Concurrent_HitDuringMiss
//
// Concurrent correctness test: two ctxs with DIFFERENT Misc params run
// concurrent submits in a tight loop.  With kernel-argument-based Misc (no
// shared __constant__ memory) every kernel launch carries its own params, so
// outputs must always match the serial reference for each ctx regardless of
// scheduling order.
// =============================================================================
TEST_F(GpuChainContextLifecycleTest, Concurrent_HitDuringMiss)
{
	const MappingTestData& tc = getTestData();
	if (tc.n_expected_anchors == 0) GTEST_SKIP() << "no anchors";

	gpu_chain_ctx_t* ctxA = gpu_chain_alloc();
	gpu_chain_ctx_t* ctxB = gpu_chain_alloc();
	ASSERT_NE(ctxA, nullptr);
	ASSERT_NE(ctxB, nullptr);
	gpu_stream_config_t cfg = make_small_gpu_config();
	gpu_chain_init(ctxA, &cfg, /*n_threads=*/1, /*n_gpu_workers=*/1, make_test_device_list());
	gpu_chain_init(ctxB, &cfg, /*n_threads=*/1, /*n_gpu_workers=*/1, make_test_device_list());

	if (tc.n_expected_anchors < (size_t)gpu_chain_ctx_min_n(ctxA)) {
		gpu_chain_free(ctxA, 1);
		gpu_chain_dealloc(ctxA);
		gpu_chain_free(ctxB, 1);
		gpu_chain_dealloc(ctxB);
		GTEST_SKIP() << "Dataset too small";
	}

	// Two opts with DIFFERENT max_gap -> different misc.max_dist_x /
	// misc.max_iter -> chain results for long segments diverge.
	mm_mapopt_t optA = opt;
	mm_mapopt_t optB = opt;
	optA.max_gap = (opt.max_gap > 1000 ? opt.max_gap / 2 : opt.max_gap * 2);
	optB.max_gap = (opt.max_gap > 1000 ? opt.max_gap / 4 : opt.max_gap * 4);

	auto run_one = [&](gpu_chain_ctx_t* ctx, const mm_mapopt_t* o) -> int {
		ScopedMemPool km;
		chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);
		populateAnchorsFromConstants(read, km, tc);
		runGpuChainingWithCtxTid(ctx, mi, o, &read, /*tid=*/0, km);
		return read.n_u;
	};

	// Capture serial reference for each ctx (these can't race with anything).
	const int ref_nuA = run_one(ctxA, &optA);
	const int ref_nuB = run_one(ctxB, &optB);
	ASSERT_GT(ref_nuA, 0) << "serial reference for ctxA produced no chains";
	ASSERT_GT(ref_nuB, 0) << "serial reference for ctxB produced no chains";

	// Stress loop. Each iteration spawns one ctxA and one ctxB submit
	// concurrently, with the cache holding whichever params won the previous
	// race. After the warmup iteration this means every iteration has one
	// HIT and one MISS interleaved -- exactly the racy mixed cohort.
	constexpr int N_ITER = 32;
	std::vector<int> nuA(N_ITER, 0), nuB(N_ITER, 0);

	for (int it = 0; it < N_ITER; it++) {
		std::thread tA([&] { nuA[it] = run_one(ctxA, &optA); });
		std::thread tB([&] { nuB[it] = run_one(ctxB, &optB); });
		tA.join();
		tB.join();
	}

	// EVERY concurrent submit must produce the same chains as its own ctx's
	// serial reference. With kernel-argument-based Misc, each kernel carries
	// its own params so there is no shared constant-memory state to corrupt.
	for (int it = 0; it < N_ITER; it++) {
		EXPECT_EQ(nuA[it], ref_nuA)
		    << "ctxA iteration " << it << " produced n_u=" << nuA[it]
		    << " (expected " << ref_nuA << ")";
		EXPECT_EQ(nuB[it], ref_nuB)
		    << "ctxB iteration " << it << " produced n_u=" << nuB[it]
		    << " (expected " << ref_nuB << ")";
	}

	gpu_chain_free(ctxA, 1);
	gpu_chain_dealloc(ctxA);
	gpu_chain_free(ctxB, 1);
	gpu_chain_dealloc(ctxB);
}
