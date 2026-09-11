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
 * Tests for mm_map function (full pipeline)
 *
 * mm_map is the main entry point for mapping a single query sequence to a reference index.
 * It orchestrates all three mapping stages:
 * 1. Seed: finds k-mer matches (anchors)
 * 2. Chain: groups anchors into colinear chains
 * 3. Align: performs sequence alignment
 *
 * mm_map is a wrapper around mm_map_frag with n_segs=1 for single-segment reads.
 *
 * Validation performed:
 * - Full pipeline integration (seed -> chain -> align)
 * - Primary alignment validation (coordinates, CIGAR, score)
 * - Secondary alignment validation (valid coordinates, scores, reference IDs)
 * - Mapping quality consistency
 * - Fragment mode handling
 * - Parameter sensitivity tests
 *
 * Non-parameterized: TbufTest
 *
 * Exact-match tests (parameterized across all datasets):
 *   BasicMapping, VerifyCpuOutput
 *
 * Behavioral tests (single representative dataset):
 *   MapEdgeCasesAndErrors, MapDeterminismAndQuality,
 *   MapAlignmentVariants, MapModeTests, MapPairedEnd
 */
#include "test_map_common.h"
#include <vector>

// ============================================================================
// Buffer management tests (non-parameterized)
// ============================================================================

// Test mm_tbuf lifecycle (init, use, destroy) including NULL handling
TEST(TbufTest, BufferLifecycle)
{
	// Test NULL buffer destruction (should be safe and not crash)
	mm_tbuf_destroy(nullptr);
	SUCCEED() << "NULL buffer destruction is safe";

	// Test normal lifecycle
	mm_tbuf_t* b = mm_tbuf_init();
	ASSERT_NE(b, nullptr) << "Buffer initialization should succeed";

	// Use the buffer and verify it's functional
	void* km = mm_tbuf_get_km(b);
	ASSERT_NE(km, nullptr) << "Buffer should provide valid memory pool";

	// Test that we can allocate from the pool
	void* test_alloc = kmalloc(km, 100);
	EXPECT_NE(test_alloc, nullptr) << "Should be able to allocate from buffer's pool";

	// Destroy the buffer (cleanup is automatic)
	mm_tbuf_destroy(b);
}

// ============================================================================
// Parameterized exact-match tests (run against ALL datasets)
// ============================================================================

// ============================================================================
// Parameterized test fixture
// ============================================================================

class MapParamTest : public MapParamTestFixture
{
      protected:
	void SetUp() override
	{
		MapParamTestFixture::SetUp();
		opt.flag |= MM_F_CIGAR; // Enable CIGAR string generation
	}
};

// ============================================================================
// Basic functionality tests
// ============================================================================

// Test basic mapping functionality
TEST_P(MapParamTest, BasicMapping)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr) << "Index should be loaded";

	const ExpectedAlignment* expected = tc.getPrimaryAlignment();
	if (expected == nullptr || tc.n_expected_alignments == 0) {
		GTEST_SKIP() << "No expected alignments";
	}

	// Call mm_map with RAII cleanup
	ScopedAlignments result;
	result.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &result.count(), tbuf, &opt, getQueryName());

	if (result.size() == 0) {
		GTEST_SKIP() << "No alignments produced";
	}

	mm_reg1_t* r = &result[0]; // Primary alignment

	// Basic sanity checks
	EXPECT_GE(r->rid, 0) << "Reference ID should be valid";
	EXPECT_LT(r->rid, (int32_t)mi->n_seq) << "Reference ID should be within index range";
	EXPECT_GT(r->score, 0) << "Alignment score should be positive";
	EXPECT_GE(r->qs, 0) << "Query start should be non-negative";
	EXPECT_GT(r->qe, r->qs) << "Query end should be greater than start";
	EXPECT_GE(r->rs, 0) << "Reference start should be non-negative";
	EXPECT_GT(r->re, r->rs) << "Reference end should be greater than start";
	EXPECT_LE(r->re, (int32_t)mi->seq[r->rid].len) << "Reference end should not exceed ref length";
	EXPECT_LE(r->mapq, 60) << "Mapping quality should be <= 60";

	// Validate strand information
	EXPECT_LT(r->qs, r->qe) << "Query coords well-formed";
	EXPECT_LT(r->rs, r->re) << "Ref coords well-formed";

	// Verify alignment matches expected values
	if (expected != nullptr && tc.n_expected_alignments > 0) {
		// Score should match (same seeds and chains produce same scores)
		EXPECT_EQ(r->score, expected->score) << "Alignment score should match";
		// Strand should match
		EXPECT_EQ(r->rev, expected->rev) << "Strand should match";
	}

	// Verify CIGAR is present
	ASSERT_NE(r->p, nullptr) << "CIGAR information should be present";
	EXPECT_GT(r->p->n_cigar, 0) << "Should have CIGAR operations";
}

// Verify full mm_map() output against ALL expected values from JSON
TEST_P(MapParamTest, VerifyCpuOutput)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr) << "Index should be loaded";

	if (tc.n_expected_alignments == 0) {
		GTEST_SKIP() << "No expected alignments";
	}

	ScopedAlignments result;
	result.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &result.count(), tbuf, &opt, getQueryName());

	verifyAlignments(result.regs(), result.size(), tc, "mm_map");
}

// Both BasicMapping and VerifyCpuOutput require expected alignments, so only
// instantiate the suite for datasets that actually have them. This keeps the
// no-op cases out of the test report instead of reporting them as "Skipped".
INSTANTIATE_TEST_SUITE_P(
    AllTestData,
    MapParamTest,
    ::testing::ValuesIn(getTestDataWithAlignments()),
    testDataNameGenerator);

// ============================================================================
// Behavioral tests (single representative dataset)
// ============================================================================

class MapBehavioralTest : public BehavioralTestFixture
{
      protected:
	void SetUp() override
	{
		BehavioralTestFixture::SetUp();
		opt.flag |= MM_F_CIGAR;
	}
};

// Combines: EdgeCaseInputs + ErrorHandling + MappingWithoutCIGAR
TEST_F(MapBehavioralTest, MapEdgeCasesAndErrors)
{
	ASSERT_NE(mi, nullptr) << "Index should be loaded";

	// --- Edge cases (empty, short, NULL name) ---
	{
		// Empty query
		ScopedAlignments r;
		r.regs() = mm_map(mi, 0, "", &r.count(), tbuf, &opt, "empty_query");
		EXPECT_EQ(r.size(), 0) << "Empty query should produce no alignments";
	}
	{
		// Fixed short query (20bp)
		const char* short_seq = "ACGTACGTACGTACGTACGT";
		ScopedAlignments r;
		r.regs() = mm_map(mi, 20, short_seq, &r.count(), tbuf, &opt, "short_query");
		EXPECT_GE(r.size(), 0) << "Short query should not crash";
	}
	// Query shorter than k-mer from actual sequence
	if (getQueryLen() > mi->k + 10) {
		const int SHORT_LEN = mi->k - 5;
		char* short_seq = (char*)malloc(SHORT_LEN + 1);
		memcpy(short_seq, getQuerySeq(), SHORT_LEN);
		short_seq[SHORT_LEN] = '\0';
		ScopedAlignments r;
		r.regs() = mm_map(mi, SHORT_LEN, short_seq, &r.count(), tbuf, &opt, "too_short");
		EXPECT_GE(r.size(), 0) << "Query shorter than k-mer should not crash";
		free(short_seq);
	}
	// NULL query name
	{
		ScopedAlignments r;
		r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r.count(), tbuf, &opt, nullptr);
		EXPECT_GE(r.size(), 0) << "NULL query name should not crash";
	}

	// --- Error handling (NULL inputs, n_segs boundary, zero-length, max_qlen) ---
	{
		// Arrays sized MM_MAX_SEG + 2 to safely test out-of-bounds n_segs values
		// without causing buffer overflows during error checking.
		int n_regs[MM_MAX_SEG + 2] = {0};
		mm_reg1_t* regs[MM_MAX_SEG + 2] = {nullptr};
		const char* query_seq_ptr = getQuerySeq();
		int query_len = getQueryLen();

		mm_map_frag(nullptr, 1, &query_len, &query_seq_ptr, n_regs, regs, tbuf, &opt, getQueryName());
		EXPECT_EQ(n_regs[0], 0) << "Should handle NULL mi gracefully";

		mm_map_frag(mi, 1, nullptr, &query_seq_ptr, n_regs, regs, tbuf, &opt, getQueryName());
		EXPECT_EQ(n_regs[0], 0) << "Should handle NULL qlens gracefully";

		mm_map_frag(mi, 1, &query_len, nullptr, n_regs, regs, tbuf, &opt, getQueryName());
		EXPECT_EQ(n_regs[0], 0) << "Should handle NULL seqs gracefully";

		mm_map_frag(mi, 0, &query_len, &query_seq_ptr, n_regs, regs, tbuf, &opt, getQueryName());
		EXPECT_EQ(n_regs[0], 0) << "Should handle n_segs=0 gracefully";

		mm_map_frag(mi, MM_MAX_SEG + 1, &query_len, &query_seq_ptr, n_regs, regs, tbuf, &opt, getQueryName());
		EXPECT_EQ(n_regs[0], 0) << "Should handle n_segs > MM_MAX_SEG gracefully";

		int zero_len = 0;
		mm_map_frag(mi, 1, &zero_len, &query_seq_ptr, n_regs, regs, tbuf, &opt, getQueryName());
		EXPECT_EQ(n_regs[0], 0) << "Should handle zero-length query gracefully";

		// Test max_qlen restriction
		mm_mapopt_t opt_limited = opt;
		opt_limited.max_qlen = query_len / 2;
		mm_map_frag(mi, 1, &query_len, &query_seq_ptr, n_regs, regs, tbuf, &opt_limited, getQueryName());
		EXPECT_EQ(n_regs[0], 0) << "Should skip queries exceeding max_qlen";
	}

	// --- No CIGAR (mapping without CIGAR generation) ---
	{
		mm_mapopt_t opt_no_cigar = opt;
		opt_no_cigar.flag &= ~MM_F_CIGAR;
		ScopedAlignments r;
		r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r.count(), tbuf, &opt_no_cigar, getQueryName());
		if (r.size() > 0) {
			EXPECT_GT(r[0].score, 0) << "Alignment score should be positive";
			EXPECT_EQ(r[0].p, nullptr) << "CIGAR should be NULL when MM_F_CIGAR is disabled";
		}
	}
}

// Combines: MappingDeterminismAndConsistency + MappingQualityValidation + StrandInformation + AlignmentMetricsAndCIGAR
TEST_F(MapBehavioralTest, MapDeterminismAndQuality)
{
	ASSERT_NE(mi, nullptr) << "Index should be loaded";

	// --- Determinism (run mapping 3 times and verify full consistency) ---
	ScopedAlignments first_result;
	for (int iter = 0; iter < 3; iter++) {
		ScopedAlignments result;
		result.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &result.count(), tbuf, &opt, getQueryName());
		if (iter == 0) {
			if (result.size() == 0) GTEST_SKIP() << "No alignments";
			validateMAPQOrdering(result.regs(), result.size(), "First run");
			// Check secondary alignments don't exceed primary score
			for (int i = 1; i < result.size(); i++)
				EXPECT_LE(result[i].score, result[0].score) << "Secondary " << i << " should have <= score than primary";
			first_result.regs() = result.regs();
			first_result.count() = result.count();
			result.regs() = nullptr;
			result.count() = 0;
		} else {
			// Verify complete determinism across all fields
			EXPECT_EQ(result.size(), first_result.size()) << "Result count deterministic (iter " << iter << ")";
			if (result.size() == first_result.size()) {
				for (int i = 0; i < result.size(); i++) {
					AlignmentDifference diff = compareAlignments(&first_result[i], &result[i]);
					EXPECT_FALSE(diff.score_diff || diff.coord_diff || diff.strand_diff ||
					    diff.mapq_diff || diff.cigar_len_diff)
					    << "Iter " << iter << " alignment " << i << " differs: "
					    << "score_diff=" << diff.score_diff
					    << " coord_diff=" << diff.coord_diff
					    << " strand_diff=" << diff.strand_diff
					    << " mapq_diff=" << diff.mapq_diff
					    << " cigar_len_diff=" << diff.cigar_len_diff;
				}
			}
		}
	}

	// --- MAPQ validation (all alignments in valid range [0,60], primary >= secondary) ---
	{
		ScopedAlignments r;
		r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r.count(), tbuf, &opt, getQueryName());
		for (int i = 0; i < r.size(); i++) {
			EXPECT_GE(r[i].mapq, 0) << "MAPQ should be >= 0 for alignment " << i;
			EXPECT_LE(r[i].mapq, 60) << "MAPQ should be <= 60 for alignment " << i;
			if (i == 0) {
				// Primary alignment should have highest MAPQ
				for (int j = 1; j < r.size(); j++)
					EXPECT_GE(r[0].mapq, r[j].mapq) << "Primary MAPQ >= secondary " << j;
			}
		}
	}

	// --- Strand validation ---
	{
		const ExpectedAlignment* expected = getTestData().getPrimaryAlignment();
		if (expected) {
			ScopedAlignments r;
			r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r.count(), tbuf, &opt, getQueryName());
			if (r.size() > 0) {
				EXPECT_EQ(r[0].rev, expected->rev) << "Strand should match expected value";
				EXPECT_LT(r[0].qs, r[0].qe) << "Query start should always be less than end";
				EXPECT_LT(r[0].rs, r[0].re) << "Reference start should always be less than end";
			}
		}
	}

	// --- Metrics and CIGAR (coverage, match/block length, DP scores, CIGAR verification) ---
	{
		ScopedAlignments r;
		r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r.count(), tbuf, &opt, getQueryName());
		if (r.size() > 0) {
			mm_reg1_t* a = &r[0];
			// Check alignment spans
			EXPECT_GT(a->qe - a->qs, 0) << "Query span should be positive";
			EXPECT_GT(a->re - a->rs, 0) << "Reference span should be positive";
			// Check match/block length relationship
			EXPECT_GT(a->mlen, 0) << "Match length should be positive";
			EXPECT_GT(a->blen, 0) << "Block length should be positive";
			EXPECT_LE(a->mlen, a->blen) << "Match length should not exceed block length";
			// Verify score consistency
			EXPECT_GT(a->score, 0) << "Alignment score should be positive";
			// Check DP scores when CIGAR is enabled
			ASSERT_NE(a->p, nullptr) << "CIGAR should be present with MM_F_CIGAR";
			EXPECT_GT(a->p->dp_score, 0) << "DP score should be positive";
			EXPECT_GE(a->p->dp_max, a->p->dp_score) << "DP max should be >= DP score";
			// Validate CIGAR operations
			ASSERT_GT(a->p->n_cigar, 0u) << "Should have CIGAR operations";
			for (uint32_t i = 0; i < a->p->n_cigar; i++) {
				uint32_t op_len = a->p->cigar[i] >> 4;
				uint32_t op_type = a->p->cigar[i] & 0xf;
				EXPECT_GT(op_len, 0u) << "CIGAR operation " << i << " length should be positive";
				EXPECT_LE(op_type, 8u) << "CIGAR operation " << i << " type should be valid";
			}
			// Validate CIGAR consumption using helper
			auto [qc, rc] = calculateCigarConsumption(a);
			EXPECT_EQ(qc, a->qe - a->qs) << "CIGAR query consumption should match alignment span";
			EXPECT_EQ(rc, a->re - a->rs) << "CIGAR ref consumption should match alignment span";
		}
	}
}

// Combines: SecondaryAlignments + PrimaryAlignmentSelection + MultipleAlignmentsWithDPMax
//         + VerboseAndDebugOutput + MemoryPoolManagement
TEST_F(MapBehavioralTest, MapAlignmentVariants)
{
	ASSERT_NE(mi, nullptr) << "Index should be loaded";

	// --- Secondary alignments (sorted by score, valid coordinates) ---
	{
		mm_mapopt_t opt_multi = opt;
		opt_multi.best_n = 5; // Report up to 5 best chains
		ScopedAlignments r;
		r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r.count(), tbuf, &opt_multi, getQueryName());
		if (r.size() > 1) {
			// Check parent field
			EXPECT_EQ(r[0].parent, r[0].id) << "Primary alignment should be its own parent";
			// Verify secondary alignments have valid properties
			for (int i = 1; i < r.size(); i++) {
				EXPECT_GE(r[i].score, 0) << "Secondary alignment " << i << " should have non-negative score";
				EXPECT_GE(r[i].qs, 0) << "Secondary alignment " << i << " qs >= 0";
				EXPECT_GT(r[i].qe, r[i].qs) << "Secondary alignment " << i << " qe > qs";
				EXPECT_GE(r[i].rs, 0) << "Secondary alignment " << i << " rs >= 0";
				EXPECT_GT(r[i].re, r[i].rs) << "Secondary alignment " << i << " re > rs";
				EXPECT_LT(r[i].rid, mi->n_seq) << "Secondary alignment " << i << " rid in range";
			}
		}
	}

	// --- Primary selection + pri_ratio parameter effect ---
	{
		ScopedAlignments baseline;
		baseline.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &baseline.count(), tbuf, &opt, getQueryName());
		if (baseline.size() > 0) {
			// Test MM_F_ALL_CHAINS vs primary selection
			mm_mapopt_t opt_all = opt;
			opt_all.flag |= MM_F_ALL_CHAINS;
			opt_all.best_n = 10;
			ScopedAlignments all_r;
			all_r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &all_r.count(), tbuf, &opt_all, getQueryName());

			mm_mapopt_t opt_pri = opt;
			opt_pri.flag &= ~MM_F_ALL_CHAINS;
			opt_pri.best_n = 10;
			ScopedAlignments pri_r;
			pri_r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &pri_r.count(), tbuf, &opt_pri, getQueryName());
			EXPECT_LE(pri_r.size(), all_r.size()) << "Primary selection should filter some alignments";

			// Test pri_ratio: strict vs relaxed filtering
			mm_mapopt_t opt_strict = opt;
			opt_strict.pri_ratio = 0.95f;
			opt_strict.best_n = 10;
			ScopedAlignments strict_r;
			strict_r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &strict_r.count(), tbuf, &opt_strict, getQueryName());

			mm_mapopt_t opt_relaxed = opt;
			opt_relaxed.pri_ratio = 0.5f;
			opt_relaxed.best_n = 10;
			ScopedAlignments relaxed_r;
			relaxed_r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &relaxed_r.count(), tbuf, &opt_relaxed, getQueryName());
			EXPECT_GE(relaxed_r.size(), strict_r.size()) << "Relaxed pri_ratio should produce >= alignments";
		}
	}

	// --- dp_max tracking (triggers mm_update_dp_max in align.c) ---
	{
		mm_mapopt_t opt_multi = opt;
		opt_multi.best_n = 5; // Allow up to 5 alignments
		opt_multi.flag |= MM_F_ALL_CHAINS; // Report all chains
		ScopedAlignments r;
		r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r.count(), tbuf, &opt_multi, getQueryName());
		if (r.size() > 0) {
			// Verify dp_max is set (if alignment has CIGAR)
			int max_dp = -1;
			for (int i = 0; i < r.size(); i++) {
				EXPECT_GT(r[i].score, 0) << "Alignment " << i << " should have positive score";
				if (r[i].p && r[i].p->dp_max > max_dp)
					max_dp = r[i].p->dp_max;
			}
			if (max_dp >= 0) EXPECT_GT(max_dp, 0) << "At least one alignment should have positive dp_max";
		}
	}

	// --- Verbose/debug output (mm_verbose, mm_dbg_flag, MM_DBG_PRINT_SEED) ---
	{
		// First check if we get any alignments at all with default settings
		ScopedAlignments baseline;
		baseline.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &baseline.count(), tbuf, &opt, getQueryName());
		if (baseline.size() > 0) {
			// Save current state
			int old_verbose = mm_verbose;
			int old_dbg_flag = mm_dbg_flag;
			// Test verbose output and debug flags
			mm_verbose = 3;
			mm_dbg_flag |= MM_DBG_PRINT_SEED;
			ScopedAlignments r;
			r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r.count(), tbuf, &opt, getQueryName());
			EXPECT_GE(r.size(), 0) << "Mapping with verbose/debug output should succeed";
			// Restore state
			mm_verbose = old_verbose;
			mm_dbg_flag = old_dbg_flag;
		}
	}

	// --- Memory pool management (cap_kalloc, cap_kalloc_largest, km_stat leak detection, mm_map_frag) ---
	{
		// Test memory pool with moderate cap_kalloc (triggers periodic resets)
		mm_mapopt_t opt_low_cap = opt;
		opt_low_cap.cap_kalloc = 1024 * 1024; // 1MB limit
		// Map the query multiple times to accumulate memory usage
		for (int iter = 0; iter < 3; iter++) {
			ScopedAlignments r;
			r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r.count(), tbuf, &opt_low_cap, getQueryName());
		}
		// Get km after all operations (pool may have been reset)
		void* km = mm_tbuf_get_km(tbuf);
		ASSERT_NE(km, nullptr) << "Thread buffer should have valid memory pool after resets";
		// Verify memory pool statistics (leak detection)
		km_stat_t kmst;
		km_stat(km, &kmst);
		EXPECT_EQ(kmst.n_blocks, kmst.n_cores)
		    << "Memory leak detected: n_blocks != n_cores. "
		    << "Blocks: " << kmst.n_blocks << ", Cores: " << kmst.n_cores;

		// Extreme case - very small cap_kalloc with mm_map_frag
		mm_mapopt_t opt_very_small = opt;
		opt_very_small.cap_kalloc = 1024; // 1KB limit (triggers frequent resets)
		ScopedAlignments frag_result;
		const char* query_seq_ptr = getQuerySeq();
		int query_len = getQueryLen();
		mm_map_frag(mi, 1, &query_len, &query_seq_ptr, &frag_result.count(), &frag_result.regs(), tbuf, &opt_very_small, getQueryName());
		// Should work even with aggressive memory limit
		EXPECT_GE(frag_result.size(), 0) << "Should handle extreme cap_kalloc without crash";
	}

	// --- cap_kalloc_largest threshold tests ---
	{
		// Very small cap_kalloc_largest should trigger resets via largest-block check
		mm_mapopt_t opt_small_largest = opt;
		opt_small_largest.cap_kalloc = 0; // disable total-capacity check
		opt_small_largest.cap_kalloc_largest = 1024; // 1KB largest-block limit
		ScopedAlignments r_small;
		r_small.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r_small.count(), tbuf, &opt_small_largest, getQueryName());
		EXPECT_GE(r_small.size(), 0) << "Should handle small cap_kalloc_largest without crash";

		// Disabled cap_kalloc_largest (0) should not trigger largest-block resets
		mm_mapopt_t opt_disabled = opt;
		opt_disabled.cap_kalloc = 0; // disable total-capacity check
		opt_disabled.cap_kalloc_largest = 0; // disable largest-block check
		ScopedAlignments r_disabled;
		r_disabled.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r_disabled.count(), tbuf, &opt_disabled, getQueryName());
		EXPECT_GE(r_disabled.size(), 0) << "Should handle disabled thresholds without crash";

		// Large cap_kalloc_largest should not trigger resets
		mm_mapopt_t opt_large = opt;
		opt_large.cap_kalloc = 0; // disable total-capacity check
		opt_large.cap_kalloc_largest = 1LL << 40; // 1TB - should never trigger
		ScopedAlignments r_large;
		r_large.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r_large.count(), tbuf, &opt_large, getQueryName());
		EXPECT_GE(r_large.size(), 0) << "Should handle large cap_kalloc_largest without crash";
	}
}

// Combines: SpliceModeMapping + FragmentMode + ConflictingFlagCombinations + ParameterSensitivity
TEST_F(MapBehavioralTest, MapModeTests)
{
	ASSERT_NE(mi, nullptr) << "Index should be loaded";

	// --- Splice modes (basic, strand-specific, flanking, intron lengths) ---
	{
		// Basic splice
		mm_mapopt_t opt_splice = opt;
		opt_splice.flag |= MM_F_SPLICE | MM_F_CIGAR;
		ScopedAlignments r;
		r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r.count(), tbuf, &opt_splice, getQueryName());
		EXPECT_GE(r.size(), 0) << "Basic splice should not crash";
	}
	{
		// Forward-strand only
		mm_mapopt_t opt_for = opt;
		opt_for.flag |= MM_F_SPLICE | MM_F_SPLICE_FOR | MM_F_CIGAR;
		ScopedAlignments r;
		r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r.count(), tbuf, &opt_for, getQueryName());
		EXPECT_GE(r.size(), 0) << "Forward splice should not crash";
	}
	{
		// Reverse-strand only
		mm_mapopt_t opt_rev = opt;
		opt_rev.flag |= MM_F_SPLICE | MM_F_SPLICE_REV | MM_F_CIGAR;
		ScopedAlignments r;
		r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r.count(), tbuf, &opt_rev, getQueryName());
		EXPECT_GE(r.size(), 0) << "Reverse splice should not crash";
	}
	{
		// Splice with flanking
		mm_mapopt_t opt_flank = opt;
		opt_flank.flag |= MM_F_SPLICE | MM_F_SPLICE_FLANK | MM_F_CIGAR;
		ScopedAlignments r;
		r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r.count(), tbuf, &opt_flank, getQueryName());
		EXPECT_GE(r.size(), 0) << "Splice with flanking should not crash";
	}
	// Different intron lengths
	for (int max_intron : {1000, 10000, 200000}) {
		mm_mapopt_t opt_test = opt;
		opt_test.flag |= MM_F_SPLICE | MM_F_CIGAR;
		mm_mapopt_max_intron_len(&opt_test, max_intron);
		ScopedAlignments r;
		r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r.count(), tbuf, &opt_test, getQueryName());
		EXPECT_GE(r.size(), 0) << "Intron length " << max_intron << " should not crash";
	}

	// --- Fragment mode (MM_F_FRAG_MODE) ---
	{
		mm_mapopt_t opt_frag = opt;
		opt_frag.flag |= MM_F_FRAG_MODE | MM_F_CIGAR;
		ScopedAlignments r;
		r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r.count(), tbuf, &opt_frag, getQueryName());
		EXPECT_GE(r.size(), 0) << "Fragment mode should not crash";
	}

	// --- Conflicting flags (SR+SPLICE are orthogonal modes, should be handled gracefully) ---
	{
		mm_mapopt_t opt_conflict = opt;
		opt_conflict.flag |= MM_F_SR | MM_F_SPLICE;
		ScopedAlignments r;
		r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r.count(), tbuf, &opt_conflict, getQueryName());
		EXPECT_GE(r.size(), 0) << "Conflicting flags (SR+SPLICE) should not crash";
	}
	{
		// EQX mode (verify graceful handling)
		mm_mapopt_t opt_eqx = opt;
		opt_eqx.flag |= MM_F_EQX; // EQX requires CIGAR (CIGAR set by default)
		ScopedAlignments r;
		r.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &r.count(), tbuf, &opt_eqx, getQueryName());
		EXPECT_GE(r.size(), 0) << "EQX mode should not crash";
	}

	// --- Parameter sensitivity (bandwidth, filtering thresholds) ---
	{
		ScopedAlignments baseline;
		baseline.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &baseline.count(), tbuf, &opt, getQueryName());
		if (baseline.size() == 0) return;

		// Liberal filtering should find >= alignments
		mm_mapopt_t opt_liberal = opt;
		opt_liberal.mid_occ = 1000000;
		ScopedAlignments liberal;
		liberal.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &liberal.count(), tbuf, &opt_liberal, getQueryName());
		EXPECT_GT(liberal.size(), 0) << "Liberal filtering should still find alignments";

		// Narrow bandwidth still allows mapping
		mm_mapopt_t opt_narrow = opt;
		opt_narrow.bw = 10;
		ScopedAlignments narrow;
		narrow.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &narrow.count(), tbuf, &opt_narrow, getQueryName());
		EXPECT_GE(narrow.size(), 0) << "Narrow bandwidth should not crash";

		// Very large bandwidth doesn't break alignment
		mm_mapopt_t opt_large = opt;
		opt_large.bw = 100000;
		opt_large.bw_long = 200000;
		ScopedAlignments large;
		large.regs() = mm_map(mi, getQueryLen(), getQuerySeq(), &large.count(), tbuf, &opt_large, getQueryName());
		if (large.size() > 0) {
			EXPECT_GT(large[0].score, 0);
			EXPECT_GT(large[0].qe, large[0].qs);
		}
	}
}

// Combines: PairedEndMapping + PairedEndOrientations + PairedEndDisabled + PairedEndSpliceMode
TEST_F(MapBehavioralTest, MapPairedEnd)
{
	ASSERT_NE(mi, nullptr) << "Index should be loaded";

	// Use the same query as both R1 and R2 (simulating overlapping reads)
	const char* seqs[2] = {getQuerySeq(), getQuerySeq()};
	int qlens[2] = {getQueryLen(), getQueryLen()};

	auto cleanupPE = [](mm_reg1_t** regs, int* n_regs) {
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < n_regs[i]; j++) free(regs[i][j].p);
			free(regs[i]);
		}
	};

	// --- Basic PE mapping (FR orientation, typical Illumina) ---
	{
		int n_regs[2] = {0, 0};
		mm_reg1_t* regs[2] = {nullptr, nullptr};
		mm_mapopt_t opt_pe = opt;
		opt_pe.pe_ori = 0; // FR orientation
		opt_pe.pe_bonus = 33; // Default paired-end bonus
		mm_map_frag(mi, 2, qlens, seqs, n_regs, regs, tbuf, &opt_pe, getQueryName());
		EXPECT_GE(n_regs[0], 0) << "R1 should have results (or 0 if no mapping)";
		EXPECT_GE(n_regs[1], 0) << "R2 should have results (or 0 if no mapping)";
		if (n_regs[0] > 0) EXPECT_GT(regs[0][0].score, 0) << "R1 primary should have positive score";
		if (n_regs[1] > 0) EXPECT_GT(regs[1][0].score, 0) << "R2 primary should have positive score";
		cleanupPE(regs, n_regs);
	}

	// --- Different orientations (0=FR, 1=RF, 2=FF) ---
	for (int pe_ori : {0, 1, 2}) {
		int n_regs[2] = {0, 0};
		mm_reg1_t* regs[2] = {nullptr, nullptr};
		mm_mapopt_t opt_pe = opt;
		opt_pe.pe_ori = pe_ori;
		opt_pe.pe_bonus = 33;
		mm_map_frag(mi, 2, qlens, seqs, n_regs, regs, tbuf, &opt_pe, getQueryName());
		// Just verify no crash - results may vary by orientation
		EXPECT_GE(n_regs[0], 0) << "pe_ori=" << pe_ori << " R1 should not crash";
		EXPECT_GE(n_regs[1], 0) << "pe_ori=" << pe_ori << " R2 should not crash";
		cleanupPE(regs, n_regs);
	}

	// --- Disabled pairing (pe_ori=-1, treat as independent segments) ---
	{
		int n_regs[2] = {0, 0};
		mm_reg1_t* regs[2] = {nullptr, nullptr};
		mm_mapopt_t opt_pe = opt;
		opt_pe.pe_ori = -1; // Disabled - treat as independent segments
		mm_map_frag(mi, 2, qlens, seqs, n_regs, regs, tbuf, &opt_pe, getQueryName());
		// Both segments should map independently
		EXPECT_GE(n_regs[0], 0) << "Disabled pairing R1";
		EXPECT_GE(n_regs[1], 0);
		cleanupPE(regs, n_regs);
	}

	// --- PE + splice mode (RNA-seq paired-end) ---
	{
		int n_regs[2] = {0, 0};
		mm_reg1_t* regs[2] = {nullptr, nullptr};
		mm_mapopt_t opt_pe = opt;
		opt_pe.flag |= MM_F_SPLICE;
		opt_pe.pe_ori = 0; // FR orientation
		opt_pe.pe_bonus = 33;
		mm_map_frag(mi, 2, qlens, seqs, n_regs, regs, tbuf, &opt_pe, getQueryName());
		// Verify no crash and valid structure
		EXPECT_GE(n_regs[0], 0) << "PE+splice R1";
		EXPECT_GE(n_regs[1], 0) << "PE+splice R2";
		cleanupPE(regs, n_regs);
	}
}
