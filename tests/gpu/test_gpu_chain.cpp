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
 * GPU Chaining Unit Tests
 *
 * Tests for the GPU chaining pipeline (gpu_chain_submit).
 * Mirrors the CPU chaining tests in tests/core/test_map_chain.cpp,
 * comparing GPU chaining results against CPU chaining for correctness.
 *
 * Exact-match tests (parameterized across all datasets):
 *   GpuChainFromConstantAnchors, VerifyAllChainsVsCpuBaseline
 *
 * Behavioral tests (single representative dataset):
 *   GpuChainDeterminismAndBasics, GpuChainStructureAndOrdering
 *
 * Requirements:
 * - GPU hardware (AMD ROCm) must be available
 * - Built with GPU=AMD
 */

#include "test_gpu_common.h"
#include <tuple>

// ============================================================================
// Parameterized tests - run against all test datasets
// ============================================================================

// Test GPU chaining with constant anchor input (bypasses seeding).
// Uses known anchors from test JSON for reproducibility.
// NOTE: Basic chain properties (score>0, length>0) overlap with
// GpuChainStructureAndOrdering — this test's value is the constant-anchor
// input path which isolates GPU chaining from seed variability.
TEST_P(GpuChainParamTest, GpuChainFromConstantAnchors)
{
	const MappingTestData& tc = getTestData();

	const ExpectedChain* expected_chain = tc.getPrimaryChain();
	if (expected_chain == nullptr || tc.n_expected_anchors == 0) {
		GTEST_SKIP() << "Test data has no expected chains or anchors";
	}

	ScopedMemPool km;
	chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);

	// Populate anchors from constants (skip seeding)
	populateAnchorsFromConstants(read, km, tc);
	ASSERT_EQ(read.n, (int64_t)tc.n_expected_anchors)
	    << "Should have expected number of anchors";

	// Skip if too few anchors for GPU
	if (read.n < GpuEnvironment::s_min_n) {
		GTEST_SKIP() << "Too few anchors (" << read.n << ") for GPU chaining (min: " << GpuEnvironment::s_min_n << ")";
	}

	runGpuChaining(mi, &opt, &read, 1, km);

	// Verify chain outputs
	ASSERT_GT(read.n_u, 0) << "GPU should produce at least one chain";
	EXPECT_NE(read.u, nullptr) << "Chain array u should be allocated";
	EXPECT_NE(read.a, nullptr) << "Anchor array a should still exist";

	// Verify primary chain properties
	int32_t chain0_score = getChainScore(read.u[0]);
	int32_t chain0_len = getChainLen(read.u[0]);
	EXPECT_GT(chain0_score, 0) << "GPU primary chain score should be positive";
	EXPECT_GT(chain0_len, 0) << "GPU primary chain length should be positive";
}

// Test exact GPU chain output vs CPU baseline from test JSON.
// Uses the shared verifyChains() helper which checks per-chain score, length,
// and anchor positions, reporting up to 10 mismatches per field with a summary.
// This flags any divergence between GPU chaining and the CPU baseline -
// e.g. hg00438 datasets are known to have GPU/CPU chain differences.
TEST_P(GpuChainParamTest, VerifyAllChainsVsCpuBaseline)
{
	const MappingTestData& tc = getTestData();

	if (tc.n_expected_chains == 0 || tc.n_expected_anchors == 0) {
		GTEST_SKIP() << "Test data has no expected chains or anchors";
	}

	// Seed the read, then run GPU chaining
	ScopedMemPool km;
	chain_read_t read = createReadFromSeq(km, getPrimaryQuery());
	mm_map_seed(mi, &opt, &read, &tbuf->timers.seed, km, NULL);

	if (read.n < GpuEnvironment::s_min_n) {
		GTEST_SKIP() << "Too few anchors (" << read.n << ") for GPU chaining (min: " << GpuEnvironment::s_min_n << ")";
	}

	runGpuChaining(mi, &opt, &read, 1, km);

	verifyChains(read, tc, "GPU chain vs CPU baseline");
}

// Both GPU chain TEST_P cases compare against expected chains and anchors;
// only instantiate for datasets that provide that baseline. Datasets that
// have too few anchors for the GPU are still skipped at runtime since the
// threshold comes from GpuEnvironment after initialization.
INSTANTIATE_TEST_SUITE_P(
    AllTestData,
    GpuChainParamTest,
    ::testing::ValuesIn(getTestDataWithChains()),
    testDataNameGenerator);

// ============================================================================
// Behavioral tests - single representative dataset
// ============================================================================

// Combines: GpuInitialization + GpuChainDeterminism + GpuChainEmptyInput +
//           GpuChainMinAnchorThreshold + GpuChainBatchProcessing
TEST_F(GpuChainBehavioralTest, GpuChainDeterminismAndBasics)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr);

	// --- GPU initialization check ---
	EXPECT_TRUE(GpuEnvironment::s_initialized) << "GPU should be initialized";
	EXPECT_GT(GpuEnvironment::s_max_total_n, 0u) << "max_total_n should be set by init";
	EXPECT_GT(GpuEnvironment::s_max_reads, 0) << "max_reads should be set by init";
	EXPECT_GT(GpuEnvironment::s_min_n, 0) << "min_n should be set by init";

	// --- Determinism (3 iterations with seeded pipeline) ---
	{
		std::vector<std::tuple<int, int32_t, int32_t>> results;

		for (int iter = 0; iter < 3; iter++) {
			ScopedMemPool km;
			chain_read_t read = createReadFromSeq(km, getPrimaryQuery());
			mm_map_seed(mi, &opt, &read, &tbuf->timers.seed, km, NULL);

			if (read.n < GpuEnvironment::s_min_n) {
				GTEST_SKIP() << "Too few anchors for GPU chaining";
			}

			runGpuChaining(mi, &opt, &read, 1, km);

			if (read.n_u > 0) {
				results.emplace_back(
				    read.n_u,
				    getChainLen(read.u[0]),
				    getChainScore(read.u[0]));
			}
		}

		if (!results.empty()) {
			for (size_t i = 1; i < results.size(); i++) {
				EXPECT_EQ(std::get<0>(results[i]), std::get<0>(results[0]))
				    << "GPU chain count should be deterministic (iter " << i << ")";
				EXPECT_EQ(std::get<1>(results[i]), std::get<1>(results[0]))
				    << "GPU first chain length should be deterministic (iter " << i << ")";
				EXPECT_EQ(std::get<2>(results[i]), std::get<2>(results[0]))
				    << "GPU first chain score should be deterministic (iter " << i << ")";
			}
		}
	}

	// --- Empty input ---
	{
		ScopedMemPool km;
		chain_read_t read = createRead("no_match", "NNNNNNNN", km, 8);
		mm_map_seed(mi, &opt, &read, &tbuf->timers.seed, km, NULL);

		if (read.n >= GpuEnvironment::s_min_n) {
			runGpuChaining(mi, &opt, &read, 1, km);
		}
		EXPECT_GE(read.n_u, 0) << "Chain count should be non-negative";
	}

	// --- Short subsequence (may fall below min_n) ---
	{
		const mm_bseq1_t& seq = getPrimaryQuery();
		int qlen = std::min(500, seq.l_seq);
		std::vector<char> subseq(seq.seq, seq.seq + qlen);
		subseq.push_back('\0');

		ScopedMemPool km;
		chain_read_t read = createRead("short_query", subseq.data(), km, qlen);
		mm_map_seed(mi, &opt, &read, &tbuf->timers.seed, km, NULL);

		if (read.n > 0 && read.n >= GpuEnvironment::s_min_n) {
			runGpuChaining(mi, &opt, &read, 1, km);
			EXPECT_GE(read.n_u, 0) << "GPU chain count should be non-negative";
		}
	}

	// --- Batch processing (3 identical reads) ---
	if (tc.n_expected_anchors > 0) {
		const int batch_size = 3;
		ScopedMemPool km;

		std::vector<chain_read_t> reads(batch_size);

		for (int i = 0; i < batch_size; i++) {
			reads[i] = createReadFromSeq(km, getPrimaryQuery());
			mm_map_seed(mi, &opt, &reads[i], &tbuf->timers.seed, km, NULL);
		}

		// All reads are identical, so either all or none meet the threshold
		if (reads[0].n >= GpuEnvironment::s_min_n) {
			runGpuChaining(mi, &opt, reads.data(), batch_size, km);

			// All identical reads should produce same number of chains
			if (reads[0].n_u > 0) {
				for (int i = 1; i < batch_size; i++) {
					EXPECT_EQ(reads[i].n_u, reads[0].n_u)
					    << "Identical reads should produce same chain count in batch";
				}
			}
		}
	}
}

// Combines: GpuChainFromSeededReads + GpuChainColinearity +
//           GpuChainScoreOrdering + GpuChainMinimumThresholds
TEST_F(GpuChainBehavioralTest, GpuChainStructureAndOrdering)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr);

	if (tc.n_expected_anchors == 0) {
		GTEST_SKIP() << "Test data has no expected anchors";
	}

	ScopedMemPool km;
	chain_read_t read = createReadFromSeq(km, getPrimaryQuery());
	mm_map_seed(mi, &opt, &read, &tbuf->timers.seed, km, NULL);

	if (read.n < GpuEnvironment::s_min_n) {
		GTEST_SKIP() << "Too few anchors for GPU chaining";
	}

	runGpuChaining(mi, &opt, &read, 1, km);

	if (read.n_u == 0) {
		GTEST_SKIP() << "No GPU chains produced";
	}

	// --- Basic chain structure ---
	EXPECT_NE(read.u, nullptr) << "Chain array u should be allocated";
	EXPECT_NE(read.a, nullptr) << "Anchor array a should still exist";
	EXPECT_GT(read.frag_gap, 0) << "Fragment gap should be set";

	int64_t total_chained_anchors = 0;
	for (int i = 0; i < read.n_u; i++) {
		int32_t chain_len = getChainLen(read.u[i]);
		int32_t chain_score = getChainScore(read.u[i]);

		EXPECT_GT(chain_len, 0) << "GPU chain " << i << " should have positive length";
		EXPECT_GT(chain_score, 0) << "GPU chain " << i << " should have positive score";

		total_chained_anchors += chain_len;
	}
	EXPECT_GT(total_chained_anchors, 0) << "GPU should chain at least some anchors";

	// --- Colinearity: anchors within chains sorted by query position ---
	{
		int64_t anchor_offset = 0;
		for (int i = 0; i < read.n_u; i++) {
			int32_t chain_len = getChainLen(read.u[i]);
			for (int32_t j = 1; j < chain_len; j++) {
				uint32_t prev_qpos = getAnchorQPos(read.a[anchor_offset + j - 1].y);
				uint32_t curr_qpos = getAnchorQPos(read.a[anchor_offset + j].y);
				EXPECT_LE(prev_qpos, curr_qpos)
				    << "GPU chain " << i << " anchors not colinear at position " << j
				    << ": prev_qpos=" << prev_qpos << " curr_qpos=" << curr_qpos;
			}
			anchor_offset += chain_len;
		}
	}

	// --- Score ordering: chains sorted by score descending ---
	if (read.n_u > 1) {
		for (int i = 1; i < read.n_u; i++) {
			int32_t prev_score = getChainScore(read.u[i - 1]);
			int32_t curr_score = getChainScore(read.u[i]);
			EXPECT_GE(prev_score, curr_score)
			    << "GPU chains should be sorted by score descending at index " << i;
		}
	}

	// --- Minimum thresholds ---
	for (int i = 0; i < read.n_u; i++) {
		int32_t chain_len = getChainLen(read.u[i]);
		int32_t chain_score = getChainScore(read.u[i]);

		EXPECT_GE(chain_score, opt.min_chain_score)
		    << "GPU chain " << i << " score should meet minimum threshold";
		EXPECT_GE(chain_len, opt.min_cnt)
		    << "GPU chain " << i << " length should meet minimum count";
	}
}
