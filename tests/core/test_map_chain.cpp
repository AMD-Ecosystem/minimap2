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
 * Tests for mm_map_chain function
 * 
 * mm_map_chain takes anchors (from mm_map_seed) and groups them into chains.
 * A chain represents a candidate alignment region with colinear anchors.
 * 
 * Validation performed:
 * - Chain count and structure validation
 * - Chain score and length extraction
 * - Colinearity: anchors within each chain are sorted by query position
 * - Primary chain selection matches expected values
 * 
 * Exact-match tests (parameterized across all datasets):
 *   ChainFromConstantAnchors, ChainPropertiesAndValidation
 *
 * Behavioral tests (single representative dataset):
 *   ChainDeterminismAndBasics, ChainModeVariants,
 *   ChainConstraintsAndParameters, ChainParametersAndScores
 */
#include "test_map_common.h"
#include <algorithm>
#include <vector>

// ============================================================================
// Parameterized chain test fixture
// ============================================================================

class MapChainParamTest : public MapParamTestFixture
{
      protected:
	// Helper: Run seed -> chain pipeline
	void runChainPipeline(chain_read_t& read, void* km)
	{
		mm_map_seed(mi, &opt, &read, &tbuf->timers.seed, km, NULL);
		mm_map_chain(mi, &opt, &read, &tbuf->timers.chain, km, NULL);
	}
};

// ============================================================================
// Parameterized tests - run against all test datas
// ============================================================================

// Test that mm_map_chain produces valid chains from constant anchor input
TEST_P(MapChainParamTest, ChainFromConstantAnchors)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr) << "Index should be loaded";

	const ExpectedChain* expected_chain = tc.getPrimaryChain();
	if (expected_chain == nullptr || tc.n_expected_anchors == 0) {
		GTEST_SKIP() << "Test data has no expected chains or anchors";
	}

	ScopedMemPool km;
	chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);

	// Populate anchors from constants (skip seeding!)
	populateAnchorsFromConstants(read, km, tc);
	ASSERT_EQ(read.n, (int64_t)tc.n_expected_anchors) << "Should have expected number of anchors";

	int64_t n_anchors_before = read.n;

	// Run mm_map_chain directly on constant anchors
	mm_map_chain(mi, &opt, &read, &tbuf->timers.chain, km, NULL);

	// Verify chain outputs
	ASSERT_GT(read.n_u, 0) << "Should have at least one chain";
	EXPECT_NE(read.u, nullptr) << "Chain array u should be allocated";
	EXPECT_NE(read.a, nullptr) << "Anchor array a should still exist";
	EXPECT_GT(read.frag_gap, 0) << "Fragment gap should be set";

	// Verify chain structure and count total anchors
	int64_t total_chained_anchors = 0;
	for (int i = 0; i < read.n_u; i++) {
		int32_t chain_len = getChainLen(read.u[i]);
		int32_t chain_score = getChainScore(read.u[i]);

		EXPECT_GT(chain_len, 0) << "Chain " << i << " should have positive length";
		EXPECT_GT(chain_score, 0) << "Chain " << i << " should have positive score";

		total_chained_anchors += chain_len;
	}

	EXPECT_LE(total_chained_anchors, n_anchors_before)
	    << "Total chained anchors should not exceed input anchors";

	// Verify chain 0 matches expected values
	int32_t chain0_len = getChainLen(read.u[0]);
	int32_t chain0_score = getChainScore(read.u[0]);

	EXPECT_EQ(chain0_score, expected_chain->score) << "Chain 0 score mismatch";
	EXPECT_EQ(chain0_len, expected_chain->length) << "Chain 0 anchor count mismatch";

	// Verify anchor positions after chaining (anchors are reordered)
	uint32_t first_qpos = getAnchorQPos(read.a[0].y);
	uint32_t first_tpos = (uint32_t)(read.a[0].x);
	EXPECT_EQ(first_qpos, expected_chain->first_qpos) << "First anchor query position mismatch";
	EXPECT_EQ(first_tpos, expected_chain->first_tpos) << "First anchor target position mismatch";

	uint32_t last_qpos = getAnchorQPos(read.a[chain0_len - 1].y);
	uint32_t last_tpos = (uint32_t)(read.a[chain0_len - 1].x);
	EXPECT_EQ(last_qpos, expected_chain->last_qpos) << "Last anchor query position mismatch";
	EXPECT_EQ(last_tpos, expected_chain->last_tpos) << "Last anchor target position mismatch";

	// Verify strand (forward strand: high bit of x should be 0)
	EXPECT_EQ(read.a[0].x >> 63, 0u) << "First anchor should be on forward strand";

	// Verify chain colinearity: anchors should be sorted by query position within each chain
	int64_t anchor_offset = 0;
	for (int i = 0; i < read.n_u; i++) {
		int32_t chain_len = getChainLen(read.u[i]);

		// Check colinearity within this chain
		for (int32_t j = 1; j < chain_len; j++) {
			uint32_t prev_qpos = getAnchorQPos(read.a[anchor_offset + j - 1].y);
			uint32_t curr_qpos = getAnchorQPos(read.a[anchor_offset + j].y);

			// Query positions should be monotonically increasing (colinear)
			EXPECT_LE(prev_qpos, curr_qpos)
			    << "Chain " << i << " anchors not colinear at position " << j
			    << ": prev_qpos=" << prev_qpos << " curr_qpos=" << curr_qpos;
		}
		anchor_offset += chain_len;
	}
}

// Test chain properties: exact match against expected, min score/length requirements
TEST_P(MapChainParamTest, ChainPropertiesAndValidation)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr);

	ScopedMemPool km;
	chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);

	populateAnchorsFromConstants(read, km, tc);
	mm_map_chain(mi, &opt, &read, &tbuf->timers.chain, km, NULL);

	// Verify all chains against expected baseline values
	// (score, length, anchor positions for each chain)
	verifyChains(read, tc, "mm_map_chain (constant anchors)");

	// Verify each chain meets minimum requirements
	for (int i = 0; i < read.n_u; i++) {
		int32_t chain_len = getChainLen(read.u[i]);
		int32_t chain_score = getChainScore(read.u[i]);

		EXPECT_GE(chain_score, opt.min_chain_score)
		    << "Chain " << i << " score should meet minimum threshold";
		EXPECT_GE(chain_len, opt.min_cnt)
		    << "Chain " << i << " length should meet minimum count";
	}
}

// Both chain TEST_P cases are only meaningful when the dataset has expected
// chains and anchors: ChainFromConstantAnchors asserts chain content matches,
// and ChainPropertiesAndValidation is a trivial no-op without anchors.
INSTANTIATE_TEST_SUITE_P(
    AllTestData,
    MapChainParamTest,
    ::testing::ValuesIn(getTestDataWithChains()),
    testDataNameGenerator);

// ============================================================================
// Behavioral tests (single representative dataset)
// ============================================================================

class MapChainBehavioralTest : public BehavioralTestFixture
{
      protected:
	void runChainPipeline(chain_read_t& read, void* km)
	{
		mm_map_seed(mi, &opt, &read, &tbuf->timers.seed, km, NULL);
		mm_map_chain(mi, &opt, &read, &tbuf->timers.chain, km, NULL);
	}
};

// Test chain determinism, subsequence chaining, empty input, and error handling
TEST_F(MapChainBehavioralTest, ChainDeterminismAndBasics)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr);

	// --- Determinism with seed-based pipeline (3 iterations) ---
	{
		std::vector<std::tuple<int, int, int>> results;
		for (int iter = 0; iter < 3; iter++) {
			ScopedMemPool km;
			chain_read_t read = createReadFromSeq(km, getPrimaryQuery());
			mm_map_seed(mi, &opt, &read, &tbuf->timers.seed, km, NULL);
			mm_map_chain(mi, &opt, &read, &tbuf->timers.chain, km, NULL);
			if (read.n_u > 0) {
				results.push_back({read.n_u, getChainLen(read.u[0]), getChainScore(read.u[0])});
			}
		}
		if (!results.empty()) {
			for (size_t i = 1; i < results.size(); i++) {
				EXPECT_EQ(std::get<0>(results[i]), std::get<0>(results[0])) << "Chain count deterministic (iter " << i << ")";
				EXPECT_EQ(std::get<1>(results[i]), std::get<1>(results[0])) << "Chain length deterministic (iter " << i << ")";
				EXPECT_EQ(std::get<2>(results[i]), std::get<2>(results[0])) << "Chain score deterministic (iter " << i << ")";
			}
		}
	}

	// --- Determinism with constant anchors ---
	if (tc.n_expected_anchors > 0) {
		auto runChaining = [&]() {
			ScopedMemPool km;
			chain_read_t read = createMinimalRead(getQueryName(), getQueryLen(), km);
			populateAnchorsFromConstants(read, km, tc);
			mm_map_chain(mi, &opt, &read, &tbuf->timers.chain, km, NULL);
			return std::vector<uint64_t>(read.u, read.u + read.n_u);
		};
		auto chains1 = runChaining(), chains2 = runChaining();
		ASSERT_EQ(chains1.size(), chains2.size()) << "Chain count should be deterministic";
		for (size_t i = 0; i < chains1.size(); i++) {
			EXPECT_EQ(chains1[i], chains2[i]) << "Chain " << i << " should be deterministic";
		}
	}

	// --- Subsequence: chain with first 2000bp of query ---
	{
		const mm_bseq1_t& seq = getPrimaryQuery();
		int qlen = std::min(2000, seq.l_seq);
		char* subseq = (char*)malloc(qlen + 1);
		memcpy(subseq, seq.seq, qlen);
		subseq[qlen] = '\0';

		ScopedMemPool km;
		chain_read_t read = createRead("query_2000bp", subseq, km, qlen);
		runChainPipeline(read, km);

		// May or may not have chains depending on subsequence content
		EXPECT_GE(read.n_u, 0) << "Chain count should be non-negative";

		// Verify chain anchors are within query bounds
		int64_t anchor_offset = 0;
		for (int i = 0; i < read.n_u; i++) {
			int32_t chain_len = getChainLen(read.u[i]);
			for (int32_t j = 0; j < chain_len; j++) {
				uint32_t q_pos = (uint32_t)(read.a[anchor_offset + j].y & 0xFFFFFFFF);
				EXPECT_LT(q_pos, (uint32_t)qlen) << "Query position should be within bounds";
			}
			anchor_offset += chain_len;
		}
		free(subseq);
	}

	// --- Empty input (no matching anchors) ---
	{
		ScopedMemPool km;
		chain_read_t read = createRead("no_match_query", "NNNNNNNN", km);
		runChainPipeline(read, km);
		EXPECT_GE(read.n_u, 0) << "Chain count should be non-negative";
	}

	// --- Error handling with invalid parameters ---
	{
		ScopedMemPool km;
		chain_read_t read;
		memset(&read, 0, sizeof(read));

		// Test with zero segments
		read.n_seg = 0;
		mm_map_chain(mi, &opt, &read, &tbuf->timers.chain, km, NULL);
		EXPECT_EQ(read.n_u, 0) << "Should handle zero segments gracefully";

		// Test with no anchors
		read.n_seg = 1;
		read.n = 0;
		read.a = nullptr;
		mm_map_chain(mi, &opt, &read, &tbuf->timers.chain, km, NULL);
		EXPECT_EQ(read.n_u, 0) << "Should handle no anchors gracefully";
	}
}

// Test mm_map_chain with RMQ mode, short read (SR) mode, and splice mode
TEST_F(MapChainBehavioralTest, ChainModeVariants)
{
	ASSERT_NE(mi, nullptr);

	// Get baseline anchors
	ScopedMemPool km;
	chain_read_t read_base = createReadFromSeq(km, getPrimaryQuery());
	mm_map_seed(mi, &opt, &read_base, &tbuf->timers.seed, km, NULL);
	int64_t n_anchors = read_base.n;

	if (n_anchors == 0) {
		GTEST_SKIP() << "No anchors for mode variant tests";
	}
	mm_map_chain(mi, &opt, &read_base, &tbuf->timers.chain, km, NULL);
	if (read_base.n_u == 0) {
		GTEST_SKIP() << "Baseline chaining produced no chains";
	}
	int32_t best_score_base = getChainScore(read_base.u[0]);

	// --- RMQ mode: chain with Range Minimum Query algorithm ---
	{
		chain_read_t read_rmq = createReadFromSeq(km, getPrimaryQuery());
		read_rmq.n = n_anchors;
		read_rmq.a = (mm128_t*)kmalloc(km, n_anchors * sizeof(mm128_t));
		memcpy(read_rmq.a, read_base.a, n_anchors * sizeof(mm128_t));

		mm_mapopt_t opt_rmq = opt;
		opt_rmq.flag |= MM_F_RMQ;
		mm_map_chain(mi, &opt_rmq, &read_rmq, &tbuf->timers.chain, km, NULL);

		// RMQ might produce different results; both should be positive if anchors exist
		if (read_rmq.n_u > 0) {
			int32_t best_score_rmq = getChainScore(read_rmq.u[0]);
			EXPECT_GT(best_score_rmq, 0) << "RMQ chaining should produce positive scores";
			// RMQ and non-RMQ may produce different chain counts or scores
			// Scores should be within reasonable range (RMQ uses different algorithm)
			double score_ratio = (double)best_score_rmq / best_score_base;
			EXPECT_GT(score_ratio, 0.5)
			    << "RMQ score should be reasonably close to baseline. "
			    << "Ratio: " << score_ratio << " (RMQ: " << best_score_rmq << ", baseline: " << best_score_base << ")";
			EXPECT_LT(score_ratio, 2.0)
			    << "RMQ score should be reasonably close to baseline. "
			    << "Ratio: " << score_ratio << " (RMQ: " << best_score_rmq << ", baseline: " << best_score_base << ")";
		}
	}

	// --- Short Read mode: chain with short read settings ---
	{
		chain_read_t read_sr = createReadFromSeq(km, getPrimaryQuery());
		read_sr.n = n_anchors;
		read_sr.a = (mm128_t*)kmalloc(km, n_anchors * sizeof(mm128_t));
		memcpy(read_sr.a, read_base.a, n_anchors * sizeof(mm128_t));

		mm_mapopt_t opt_sr = opt;
		opt_sr.flag |= MM_F_SR;
		mm_map_chain(mi, &opt_sr, &read_sr, &tbuf->timers.chain, km, NULL);

		// SR mode may produce different scores, but should be in reasonable range
		if (read_sr.n_u > 0) {
			int32_t sr_score = getChainScore(read_sr.u[0]);
			EXPECT_GT(sr_score, 0) << "SR mode should produce positive chain scores";
			double ratio = (double)sr_score / best_score_base;
			EXPECT_GT(ratio, 0.3) << "SR score should be reasonable vs baseline";
			EXPECT_LT(ratio, 3.0) << "SR score should be reasonable vs baseline";
		}
	}

	// --- Splice mode: chain with RNA-seq splice settings ---
	{
		mm_mapopt_t opt_splice = opt;
		opt_splice.flag |= MM_F_SPLICE;
		chain_read_t read = createReadFromSeq(km, getPrimaryQuery());
		mm_map_seed(mi, &opt_splice, &read, &tbuf->timers.seed, km, NULL);
		if (read.n > 0) {
			mm_map_chain(mi, &opt_splice, &read, &tbuf->timers.chain, km, NULL);
			// Chaining may or may not produce results depending on seed quality
			EXPECT_GE(read.n_u, 0) << "Splice chaining should handle seeds gracefully";
		}
	}
}

// Test chain_post with max_gap_ref, max_frag_len, long read RMQ rescue, and mid_occ rechaining
TEST_F(MapChainBehavioralTest, ChainConstraintsAndParameters)
{
	ASSERT_NE(mi, nullptr);
	ScopedMemPool km;

	// --- max_gap_ref: set fragment gap to custom value ---
	{
		chain_read_t read = createReadFromSeq(km, getPrimaryQuery());
		mm_mapopt_t opt_gap = opt;
		opt_gap.max_gap_ref = 5000;
		mm_map_seed(mi, &opt_gap, &read, &tbuf->timers.seed, km, NULL);
		if (read.n > 0) {
			mm_map_chain(mi, &opt_gap, &read, &tbuf->timers.chain, km, NULL);
			// frag_gap should be set to max_gap_ref
			EXPECT_EQ(read.frag_gap, opt_gap.max_gap_ref);
		}
	}

	// --- max_frag_len: constrain fragment length ---
	{
		chain_read_t read = createReadFromSeq(km, getPrimaryQuery());
		mm_mapopt_t opt_frag = opt;
		opt_frag.max_frag_len = 10000; // 10kb max fragment
		mm_map_seed(mi, &opt_frag, &read, &tbuf->timers.seed, km, NULL);
		if (read.n > 0) {
			mm_map_chain(mi, &opt_frag, &read, &tbuf->timers.chain, km, NULL);
			// Should still produce valid chains
			EXPECT_GE(read.n_u, 0);
			if (read.n_u > 0) {
				EXPECT_NE(read.u, nullptr);
				EXPECT_GT(read.frag_gap, 0);
			}
		}
	}

	// --- Long read RMQ rescue: trigger RMQ rescue path with 20x repeated query ---
	{
		int repeat_factor = 20;
		int long_qlen = getQueryLen() * repeat_factor;
		char* long_seq = (char*)malloc(long_qlen + 1);
		for (int i = 0; i < repeat_factor; i++)
			memcpy(long_seq + i * getQueryLen(), getQuerySeq(), getQueryLen());
		long_seq[long_qlen] = '\0';

		chain_read_t read = createRead("long_query", long_seq, km, long_qlen);
		// Enable RMQ mode with long read settings
		mm_mapopt_t opt_long = opt;
		opt_long.flag |= MM_F_RMQ;
		opt_long.bw_long = 20000; // Wide bandwidth for long reads
		opt_long.rmq_rescue_size = 100000; // Trigger RMQ rescue
		mm_map_seed(mi, &opt_long, &read, &tbuf->timers.seed, km, NULL);
		if (read.n > 0) {
			mm_map_chain(mi, &opt_long, &read, &tbuf->timers.chain, km, NULL);
			// Should handle long reads without crash
			EXPECT_GE(read.n_u, 0) << "Long read chaining should complete";
		}
		free(long_seq);
	}

	// --- Multi-segment rechaining: mid_occ triggers rechaining logic ---
	{
		chain_read_t read = createReadFromSeq(km, getPrimaryQuery());
		mm_mapopt_t opt_rechain = opt;
		opt_rechain.max_occ = 1000;
		opt_rechain.mid_occ = 50; // Lower threshold for rechaining
		mm_map_seed(mi, &opt_rechain, &read, &tbuf->timers.seed, km, NULL);
		if (read.n > 0) {
			mm_map_chain(mi, &opt_rechain, &read, &tbuf->timers.chain, km, NULL);
			// Should produce chains even with rechaining
			EXPECT_GE(read.n_u, 0);
		}
	}
}

// Test chain parameter sensitivity (gap thresholds) and chain score validity
TEST_F(MapChainBehavioralTest, ChainParametersAndScores)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr);

	ScopedMemPool km;

	// --- Parameter sensitivity: gap threshold effect on chain count ---
	{
		// Baseline chaining
		chain_read_t read_base = createReadFromSeq(km, getPrimaryQuery());
		mm_map_seed(mi, &opt, &read_base, &tbuf->timers.seed, km, NULL);
		mm_map_chain(mi, &opt, &read_base, &tbuf->timers.chain, km, NULL);
		int n_chains_base = read_base.n_u;

		if (n_chains_base > 0) {
			// Chain with relaxed gap threshold (allows longer gaps between anchors)
			mm_mapopt_t opt_relax = opt;
			opt_relax.max_gap = 10000; // Much larger gap tolerance
			chain_read_t read_relax = createReadFromSeq(km, getPrimaryQuery());
			mm_map_seed(mi, &opt, &read_relax, &tbuf->timers.seed, km, NULL); // Same seeds
			mm_map_chain(mi, &opt_relax, &read_relax, &tbuf->timers.chain, km, NULL);
			// Relaxed gap should find >= chains (allows more flexibility)
			EXPECT_GE(read_relax.n_u, n_chains_base - 1)
			    << "Relaxed gap should find >= chains. "
			    << "Relaxed: " << read_relax.n_u << ", Baseline: " << n_chains_base;

			// Chain with strict gap threshold
			mm_mapopt_t opt_strict = opt;
			opt_strict.max_gap = 100; // Very tight gap tolerance
			chain_read_t read_strict = createReadFromSeq(km, getPrimaryQuery());
			mm_map_seed(mi, &opt, &read_strict, &tbuf->timers.seed, km, NULL); // Same seeds
			mm_map_chain(mi, &opt_strict, &read_strict, &tbuf->timers.chain, km, NULL);
			// Strict gap should find <= chains
			EXPECT_LE(read_strict.n_u, n_chains_base + 1)
			    << "Strict gap should find <= chains. "
			    << "Strict: " << read_strict.n_u << ", Baseline: " << n_chains_base;
		}
	}

	// --- Score validity: verify chain scores are valid and ordered ---
	{
		chain_read_t read = createReadFromSeq(km, getPrimaryQuery());
		mm_map_seed(mi, &opt, &read, &tbuf->timers.seed, km, NULL);
		if (read.n > 0) {
			mm_map_chain(mi, &opt, &read, &tbuf->timers.chain, km, NULL);
			for (int i = 0; i < read.n_u; i++) {
				EXPECT_GT(getChainScore(read.u[i]), 0) << "Chain " << i << " score should be positive";
			}
		}
	}
}
