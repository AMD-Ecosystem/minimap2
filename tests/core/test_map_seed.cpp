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
 * Tests for mm_map_seed function
 * 
 * mm_map_seed takes a query sequence and finds anchors (k-mer matches) against
 * the indexed reference. Anchors are the starting point for chaining and alignment.
 * 
 * Validation performed:
 * - Anchor count matches expected
 * - Anchor x (reference) and y (query) values match expected
 * - Anchor bounds: q_pos < query_length, q_span > 0
 * - Reference position validity
 * 
 * Note: Anchor q_pos + q_span can exceed query length for minimizers near the
 * query end (the k-mer window extends past the boundary), so this is not checked.
 * 
 * Exact-match tests (parameterized across all datasets):
 *   LoadTestData, MapRealSequence
 *
 * Behavioral tests (single representative dataset):
 *   SeedDeterminismAndSubsequence, SeedEdgeCases,
 *   SeedFlagEffects, SeedOccupancyAndParameters
 */
#include "test_map_common.h"
#include <vector>
#include <utility>

// ============================================================================
// Test Data Validation: Ensure test data is properly configured
// ============================================================================

// Sanity check that test data is configured

TEST(TestCaseValidation, AllTestDataNotEmpty)
{
	ASSERT_GT(getAllTestData().size(), 0u) << "getAllTestData() must return at least one test data";
}

// ============================================================================
// Parameterized exact-match tests (run against ALL datasets)
// ============================================================================

class MapSeedParamTest : public MapParamTestFixture
{
};

// Test that we can load the test data
TEST_P(MapSeedParamTest, LoadTestData)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr) << "Failed to load reference index";

	EXPECT_GT(mi->n_seq, 0) << "Index should have at least one sequence";
	if (mi->n_seq > 0) {
		EXPECT_GT(mi->seq[0].len, 0) << "Reference sequence should have non-zero length";
		EXPECT_STRNE(mi->seq[0].name, nullptr);
		// Verify ref name matches expected if provided
		if (tc.ref_name) {
			EXPECT_STREQ(mi->seq[0].name, tc.ref_name) << "Reference name mismatch";
		}
	}
}

// Test that we can map a real sequence and get anchors
TEST_P(MapSeedParamTest, MapRealSequence)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr);
	ASSERT_GT(getNumQuerySeqs(), 0) << "Should have loaded query sequences";

	// Use cached query sequence from fixture
	const mm_bseq1_t& seq = getPrimaryQuery();

	EXPECT_GT(seq.l_seq, 0) << "Query sequence should have non-zero length";
	EXPECT_EQ(seq.l_seq, getQueryLen()) << "Query length mismatch";

	ScopedMemPool km;
	chain_read_t read = createRead(seq.name, seq.seq, km, seq.l_seq);

	EXPECT_EQ(read.n, 0) << "Initially should have no anchors";

	mm_map_seed(mi, &opt, &read, &tbuf->timers.seed, km, NULL);

	EXPECT_EQ(read.seq.qlen_sum, getQueryLen()) << "qlen_sum should be set";
	if (tc.n_expected_anchors > 0) {
		EXPECT_GT(read.n_mini_pos, 0) << "Should have minimizer positions when anchors expected";
	}

	// Verify exact number of anchors
	ASSERT_EQ(read.n, (int64_t)tc.n_expected_anchors)
	    << "Should find exactly " << tc.n_expected_anchors << " anchors";
	if (tc.n_expected_anchors > 0) {
		EXPECT_NE(read.a, nullptr) << "Anchor array should be allocated";
	}

	// Check ALL seed positions against expected values
	for (int64_t i = 0; i < read.n; i++) {
		const ExpectedAnchor* expected = tc.getAnchor(i);
		ASSERT_NE(expected, nullptr) << "Expected anchor " << i << " should exist";
		EXPECT_EQ(read.a[i].x, expected->x) << "Anchor " << i << " x value mismatch";
		uint32_t q_pos = getAnchorQPos(read.a[i].y);
		EXPECT_EQ(q_pos, expected->y) << "Anchor " << i << " y value mismatch";

		// Validate anchor bounds
		uint32_t q_span = getAnchorQSpan(read.a[i].y);
		EXPECT_GT(q_span, 0u) << "Anchor " << i << " q_span should be positive";

		// Query position should be within query bounds
		// Note: q_pos + q_span can exceed query length for minimizers near the end
		// (the k-mer window extends past the query end), so we only check q_pos
		EXPECT_LT(q_pos, (uint32_t)getQueryLen())
		    << "Anchor " << i << " query position should be within query length";

		// Extract reference position and ID from anchor x value
		uint64_t x_val = read.a[i].x;
		uint32_t ref_id = (uint32_t)(x_val >> 32); // Upper 32 bits contain ref ID (with strand in bit 31)
		uint32_t ref_pos = (uint32_t)x_val; // Lower 32 bits contain position

		// Mask out strand bit from ref_id (bit 31 of upper 32 bits = bit 63 of x)
		ref_id = ref_id & 0x7FFFFFFF;

		// Reference position should be within the reference sequence bounds
		ASSERT_LT(ref_id, mi->n_seq) << "Anchor " << i << " reference ID out of bounds";
		EXPECT_LT(ref_pos, mi->seq[ref_id].len) << "Anchor " << i << " reference position exceeds sequence length";
	}
}

INSTANTIATE_TEST_SUITE_P(
    AllTestData,
    MapSeedParamTest,
    ::testing::ValuesIn(getAllTestData()),
    testDataNameGenerator);

// ============================================================================
// Behavioral tests (single representative dataset)
// ============================================================================

class MapSeedBehavioralTest : public BehavioralTestFixture
{
};

// Test that mm_map_seed produces deterministic results and handles subsequences
TEST_F(MapSeedBehavioralTest, SeedDeterminismAndSubsequence)
{
	ASSERT_NE(mi, nullptr);
	ASSERT_GT(getNumQuerySeqs(), 0);

	// --- Determinism: run seeding twice and verify identical results ---
	{
		// First run
		ScopedMemPool km1;
		chain_read_t read1 = createRead(getQueryName(), getQuerySeq(), km1, getQueryLen());
		mm_map_seed(mi, &opt, &read1, &tbuf->timers.seed, km1, NULL);

		std::vector<std::pair<uint64_t, uint64_t>> anchors1;
		for (int64_t i = 0; i < read1.n; i++) {
			anchors1.push_back({read1.a[i].x, read1.a[i].y});
		}

		// Second run
		ScopedMemPool km2;
		chain_read_t read2 = createRead(getQueryName(), getQuerySeq(), km2, getQueryLen());
		mm_map_seed(mi, &opt, &read2, &tbuf->timers.seed, km2, NULL);

		ASSERT_EQ(read2.n, read1.n) << "Anchor count should be deterministic";
		for (int64_t i = 0; i < read2.n; i++) {
			EXPECT_EQ(read2.a[i].x, anchors1[i].first) << "Anchor " << i << " x should be deterministic";
			EXPECT_EQ(read2.a[i].y, anchors1[i].second) << "Anchor " << i << " y should be deterministic";
		}
	}

	// --- Subsequence: test mm_map_seed with first 1000bp of query ---
	{
		const mm_bseq1_t& seq = getPrimaryQuery();
		int qlen = std::min(1000, seq.l_seq);
		char* subseq = (char*)malloc(qlen + 1);
		memcpy(subseq, seq.seq, qlen);
		subseq[qlen] = '\0';

		ScopedMemPool km;
		chain_read_t read = createRead("query_1000bp", subseq, km, qlen);
		mm_map_seed(mi, &opt, &read, &tbuf->timers.seed, km, NULL);

		EXPECT_EQ(read.seq.qlen_sum, qlen) << "qlen_sum should match substring length";
		// Only expect anchors if query is long enough and has similarity to reference
		EXPECT_GE(read.n, 0) << "Should handle substring without crash";
		EXPECT_GE(read.n_mini_pos, 0) << "Should have non-negative minimizer positions";

		// Verify anchor positions are within bounds
		for (int64_t i = 0; i < read.n; i++) {
			uint32_t query_pos = (uint32_t)(read.a[i].y & 0xFFFFFFFF);
			EXPECT_LT(query_pos, 1000u) << "Query position should be within substring length";
		}
		free(subseq);
	}
}

// Test seeding with edge case sequences (Ns, homopolymers, low complexity)
TEST_F(MapSeedBehavioralTest, SeedEdgeCases)
{
	ASSERT_NE(mi, nullptr);
	ScopedMemPool km;

	// Test 1: Sequence with Ns inserted at regular intervals
	std::string seq_with_ns = getQuerySeq();
	if (seq_with_ns.size() > 100) {
		for (size_t i = 100; i < seq_with_ns.size(); i += 100) {
			seq_with_ns[i] = 'N';
		}
	}

	chain_read_t read_ns = createRead("query_with_ns", seq_with_ns.c_str(), km, seq_with_ns.length());
	mm_map_seed(mi, &opt, &read_ns, &tbuf->timers.seed, km, NULL);

	// Ambiguous bases (N) should not crash, may reduce anchor count
	EXPECT_GE(read_ns.n, 0) << "N-containing sequences should be handled without crash";
	// Note: inserting Ns may reduce anchors to 0, which is valid behavior

	// Test 2: Homopolymer sequence (low information content)
	std::string homopolymer(1000, 'A');
	chain_read_t read_homo = createRead("homopolymer", homopolymer.c_str(), km, homopolymer.length());
	mm_map_seed(mi, &opt, &read_homo, &tbuf->timers.seed, km, NULL);

	EXPECT_GE(read_homo.n, 0) << "Homopolymer should not crash";
	// Note: may have 0 anchors if reference doesn't contain matching homopolymers

	// Test 3: Low complexity sequence (alternating dinucleotide)
	std::string low_complexity;
	low_complexity.reserve(1000);
	for (int i = 0; i < 500; i++) {
		low_complexity += "AT";
	}

	chain_read_t read_lowcplx = createRead("low_complexity", low_complexity.c_str(), km, low_complexity.length());
	mm_map_seed(mi, &opt, &read_lowcplx, &tbuf->timers.seed, km, NULL);

	EXPECT_GE(read_lowcplx.n, 0) << "Low complexity should not crash";
}

// Test collect_minimizers with sdust filtering, strand filtering, heap-based collection,
// and seed skipping modes (MM_F_NO_DIAG and MM_F_NO_DUAL flags)
TEST_F(MapSeedBehavioralTest, SeedFlagEffects)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr);
	ScopedMemPool km;

	// --- SDust filtering: enabled vs disabled ---
	{
		mm_mapopt_t opt_with_sdust = opt;
		opt_with_sdust.sdust_thres = 10;
		chain_read_t read1 = createRead(getQueryName(), getQuerySeq(), km, getQueryLen());
		mm_map_seed(mi, &opt_with_sdust, &read1, &tbuf->timers.seed, km, NULL);
		int64_t n_with = read1.n;

		mm_mapopt_t opt_no_sdust = opt;
		opt_no_sdust.sdust_thres = 0;
		chain_read_t read2 = createRead(getQueryName(), getQuerySeq(), km, getQueryLen());
		mm_map_seed(mi, &opt_no_sdust, &read2, &tbuf->timers.seed, km, NULL);
		int64_t n_without = read2.n;
		// SDUST should filter or maintain anchor count
		EXPECT_LE(n_with, n_without);
	}

	// --- Strand filtering (MM_F_FOR_ONLY, MM_F_REV_ONLY) ---
	{
		// Baseline: no strand filtering
		chain_read_t read_both = createRead(nullptr, getQuerySeq(), km, getQueryLen());
		mm_map_seed(mi, &opt, &read_both, &tbuf->timers.seed, km, NULL);
		int64_t n_both_strands = read_both.n;

		// Forward strand only
		mm_mapopt_t opt_fwd = opt;
		opt_fwd.flag |= MM_F_FOR_ONLY;
		chain_read_t read_fwd = createRead(nullptr, getQuerySeq(), km, getQueryLen());
		mm_map_seed(mi, &opt_fwd, &read_fwd, &tbuf->timers.seed, km, NULL);

		// Reverse strand only
		mm_mapopt_t opt_rev = opt;
		opt_rev.flag |= MM_F_REV_ONLY;
		chain_read_t read_rev = createRead(nullptr, getQuerySeq(), km, getQueryLen());
		mm_map_seed(mi, &opt_rev, &read_rev, &tbuf->timers.seed, km, NULL);

		// Verify that with strand filtering, we get different results
		if (n_both_strands > 0) {
			EXPECT_TRUE((read_fwd.n != n_both_strands) || (read_rev.n != n_both_strands))
			    << "Strand filtering flags should affect anchor collection. "
			    << "Baseline: " << n_both_strands << ", FWD: " << read_fwd.n << ", REV: " << read_rev.n;
		}
		// Combined anchors should not exceed baseline
		EXPECT_LE(read_fwd.n + read_rev.n, n_both_strands * 2)
		    << "Total anchors from both filters should be reasonable";
	}

	// --- MM_F_HEAP_SORT flag (heap-based priority processing) ---
	{
		mm_mapopt_t opt_normal = opt;
		opt_normal.flag &= ~MM_F_HEAP_SORT;
		chain_read_t read1 = createRead(getQueryName(), getQuerySeq(), km, getQueryLen());
		mm_map_seed(mi, &opt_normal, &read1, &tbuf->timers.seed, km, NULL);
		int64_t n_normal = read1.n;

		mm_mapopt_t opt_heap = opt;
		opt_heap.flag |= MM_F_HEAP_SORT;
		chain_read_t read2 = createRead(getQueryName(), getQuerySeq(), km, getQueryLen());
		mm_map_seed(mi, &opt_heap, &read2, &tbuf->timers.seed, km, NULL);
		int64_t n_heap = read2.n;
		// Both methods should find the same anchors (just different collection order)
		EXPECT_EQ(n_heap, n_normal)
		    << "Heap-based collection should find same number of anchors. "
		    << "Heap: " << n_heap << ", Normal: " << n_normal;
	}

	// --- MM_F_NO_DIAG (skips exact diagonal hits, self-mappings) ---
	// --- MM_F_NO_DUAL (symmetric all-vs-all, map once per pair) ---
	{
		chain_read_t read_baseline = createRead(getQueryName(), getQuerySeq(), km, getQueryLen());
		mm_map_seed(mi, &opt, &read_baseline, &tbuf->timers.seed, km, NULL);
		int64_t n_baseline = read_baseline.n;

		// NO_DIAG: should still find anchors unless query exactly matches reference
		mm_mapopt_t opt_no_diag = opt;
		opt_no_diag.flag |= MM_F_NO_DIAG;
		chain_read_t read_no_diag = createRead(getQueryName(), getQuerySeq(), km, getQueryLen());
		mm_map_seed(mi, &opt_no_diag, &read_no_diag, &tbuf->timers.seed, km, NULL);
		EXPECT_GE(read_no_diag.n, 0) << "MM_F_NO_DIAG should not crash";

		// NO_DUAL: skips pairs where query name > reference name lexicographically
		mm_mapopt_t opt_no_dual = opt;
		opt_no_dual.flag |= MM_F_NO_DUAL;
		chain_read_t read_no_dual = createRead(getQueryName(), getQuerySeq(), km, getQueryLen());
		mm_map_seed(mi, &opt_no_dual, &read_no_dual, &tbuf->timers.seed, km, NULL);
		// MM_F_NO_DUAL should find same or fewer anchors (symmetric filtering)
		EXPECT_LE(read_no_dual.n, n_baseline)
		    << "MM_F_NO_DUAL should not increase anchors. "
		    << "With NO_DUAL: " << read_no_dual.n << ", Without: " << n_baseline;
	}
}

// Test q_occ_frac parameter (seed occupancy filtering) and parameter impact on seeding
TEST_F(MapSeedBehavioralTest, SeedOccupancyAndParameters)
{
	ASSERT_NE(mi, nullptr);
	ScopedMemPool km;

	// --- Occupancy filtering: filter seeds occurring in >1% of minimizers ---
	{
		// Without occupancy filtering
		mm_mapopt_t opt_no_flt = opt;
		opt_no_flt.q_occ_frac = 0.0f;
		chain_read_t read1 = createRead(getQueryName(), getQuerySeq(), km, getQueryLen());
		mm_map_seed(mi, &opt_no_flt, &read1, &tbuf->timers.seed, km, NULL);
		int64_t n_no_filter = read1.n;

		// With aggressive occupancy filtering
		mm_mapopt_t opt_with_flt = opt;
		opt_with_flt.q_occ_frac = 0.01f;
		chain_read_t read2 = createRead(getQueryName(), getQuerySeq(), km, getQueryLen());
		mm_map_seed(mi, &opt_with_flt, &read2, &tbuf->timers.seed, km, NULL);
		int64_t n_with_filter = read2.n;
		// Filtering should reduce or maintain anchor count (remove high-occupancy seeds)
		EXPECT_LE(n_with_filter, n_no_filter)
		    << "Occupancy filtering should not increase anchors. "
		    << "Filtered: " << n_with_filter << ", Unfiltered: " << n_no_filter;
	}

	// --- Parameter impact: stricter filtering should find <= anchors ---
	{
		// Baseline seeding
		chain_read_t read_baseline = createRead(getQueryName(), getQuerySeq(), km, getQueryLen());
		mm_map_seed(mi, &opt, &read_baseline, &tbuf->timers.seed, km, NULL);
		int64_t n_baseline = read_baseline.n;

		if (n_baseline > 0) {
			// Seeding with higher occurrence threshold (more strict filtering)
			mm_mapopt_t opt_strict = opt;
			opt_strict.q_occ_frac = 0.5; // Filter seeds that are too common
			chain_read_t read_strict = createRead(getQueryName(), getQuerySeq(), km, getQueryLen());
			mm_map_seed(mi, &opt_strict, &read_strict, &tbuf->timers.seed, km, NULL);
			EXPECT_LE(read_strict.n, n_baseline)
			    << "Stricter filtering should find <= anchors. "
			    << "Strict: " << read_strict.n << ", Baseline: " << n_baseline;
		}
	}
}
