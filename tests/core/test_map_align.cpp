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
 * Tests for mm_map_align function
 *
 * mm_map_align takes the output from mm_map_chain (chained anchors) and performs
 * sequence alignment to produce final alignment results (mm_reg1_t).
 *
 * Validation performed:
 * - Alignment coordinate bounds (qs, qe, rs, re within valid ranges)
 * - Reference ID validity (rid < n_seq, re <= reference length)
 * - CIGAR string validity and consistency
 * - Query/reference span ratio (within reasonable bounds)
 * - Mapping quality range (0-60)
 * - Alignment score positivity
 *
 * Exact-match tests (parameterized across all datasets):
 *   AlignRealSequence
 *
 * Behavioral tests (single representative dataset):
 *   AlignBasicsAndDeterminism, AlignQualityMetrics, AlignFlagVariants
 */
#include "test_map_common.h"
#include <vector>

// ============================================================================
// Parameterized align test fixture
// ============================================================================

class MapAlignParamTest : public MapParamTestFixture
{
      protected:
	void SetUp() override
	{
		MapParamTestFixture::SetUp();
		opt.flag |= MM_F_CIGAR;
	}
};

// ============================================================================
// Parameterized tests - run against all test datas
// ============================================================================

// Test that mm_map_align produces valid alignments via the full pipeline
TEST_P(MapAlignParamTest, AlignRealSequence)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr) << "Index should be loaded";

	if (tc.getPrimaryAlignment() == nullptr || tc.n_expected_alignments == 0) {
		GTEST_SKIP() << "Test data has no expected alignment values";
	}

	ScopedMemPool km;
	const mm_bseq1_t& seq = getPrimaryQuery();
	chain_read_t read = createRead(seq.name, seq.seq, km, seq.l_seq);

	// Run full seed -> chain -> align pipeline
	mm_reg1_t* regs[1] = {nullptr};
	int n_regs[1] = {0};
	runAlignPipeline(read, regs, n_regs, km);

	ASSERT_GT(n_regs[0], 0) << "Should have at least one alignment";
	ASSERT_NE(regs[0], nullptr) << "Alignment array should be allocated";

	// Verify all alignments against expected baseline values
	// (score, qs, qe, rs, re, mapq, rev, dp_score, dp_max, blen, mlen, n_ambi, n_cigar_ops)
	verifyAlignments(regs[0], n_regs[0], tc, "mm_map_align (stage API)");

	freeAlignmentResults(regs[0], n_regs[0]);
}

// AlignRealSequence is the only test in this fixture and requires expected
// alignment baselines, so restrict instantiation to datasets that have them.
INSTANTIATE_TEST_SUITE_P(
    AllTestData,
    MapAlignParamTest,
    ::testing::ValuesIn(getTestDataWithAlignments()),
    testDataNameGenerator);

// ============================================================================
// Behavioral tests (single representative dataset)
// ============================================================================

class MapAlignBehavioralTest : public BehavioralTestFixture
{
      protected:
	void SetUp() override
	{
		BehavioralTestFixture::SetUp();
		opt.flag |= MM_F_CIGAR;
	}
};

// Combines: AlignSubsequence + AlignmentResultValidation + AlignmentDeterminism + CoordinateConsistency
TEST_F(MapAlignBehavioralTest, AlignBasicsAndDeterminism)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr);

	// --- Subsequence alignment (first 2000bp) ---
	{
		const mm_bseq1_t& seq = getPrimaryQuery();
		int qlen = std::min(2000, seq.l_seq);
		// Skip if query is too short for meaningful alignment
		if (qlen >= 100) {
			char* subseq = (char*)malloc(qlen + 1);
			memcpy(subseq, seq.seq, qlen);
			subseq[qlen] = '\0';

			ScopedMemPool km;
			chain_read_t read = createRead("query_2000bp", subseq, km, qlen);
			mm_reg1_t* regs[1] = {nullptr};
			int n_regs[1] = {0};
			runAlignPipeline(read, regs, n_regs, km);

			// Some test data may not produce alignments for subsequences
			if (n_regs[0] > 0) {
				EXPECT_GT(regs[0][0].score, 0) << "Should have positive score";
				EXPECT_LE(regs[0][0].qe, qlen) << "Query end should be within bounds";
			}
			freeAlignmentResults(regs[0], n_regs[0]);
			free(subseq);
		}
	}

	// --- Result validation (empty input, sorting) ---
	{
		// Test empty/no-match input
		ScopedMemPool km;
		chain_read_t read_empty = createRead("no_match_query", "NNNNNNNN", km, 8);
		mm_reg1_t* regs_empty[1] = {nullptr};
		int n_regs_empty[1] = {0};
		runAlignPipeline(read_empty, regs_empty, n_regs_empty, km);
		EXPECT_EQ(n_regs_empty[0], 0) << "No alignments expected for N-only sequence";
		freeAlignmentResults(regs_empty[0], n_regs_empty[0]);
	}

	if (tc.n_expected_chains == 0 || tc.n_expected_anchors == 0) return;

	// --- Sorting check (alignments should be sorted by score descending) ---
	{
		ScopedMemPool km;
		chain_read_t read;
		createReadWithConstantChain(read, km, tc, getPrimaryQuery());
		if (read.n > 0) {
			mm_reg1_t* regs[1] = {nullptr};
			int n_regs[1] = {0};
			mm_map_align(mi, &opt, &read, regs, n_regs, &tbuf->timers.align, km, NULL, NULL, 0);
			if (n_regs[0] > 1) {
				for (int i = 1; i < n_regs[0]; i++)
					EXPECT_GE(regs[0][i - 1].score, regs[0][i].score)
					    << "Alignments should be sorted by score descending at index " << i;
			}
			freeAlignmentResults(regs[0], n_regs[0]);
		}
	}

	// --- Coordinate consistency ---
	{
		ScopedMemPool km;
		chain_read_t read;
		createReadWithConstantChain(read, km, tc, getPrimaryQuery());
		if (read.n > 0) {
			mm_reg1_t* regs[1] = {nullptr};
			int n_regs[1] = {0};
			mm_map_align(mi, &opt, &read, regs, n_regs, &tbuf->timers.align, km, NULL, NULL, 0);
			for (int i = 0; i < n_regs[0]; i++) {
				mm_reg1_t* r = &regs[0][i];
				EXPECT_GE(r->qs, 0) << "Query start >= 0";
				EXPECT_LT(r->qs, r->qe) << "Query start < end";
				EXPECT_LE(r->qe, getQueryLen()) << "Query end <= qlen";
				EXPECT_GE(r->rs, 0) << "Reference start >= 0";
				EXPECT_LT(r->rs, r->re) << "Reference start < end";
				EXPECT_LE(r->re, (int32_t)mi->seq[r->rid].len) << "Ref end <= rlen";
				EXPECT_GT(r->mlen, 0) << "mlen > 0";
				EXPECT_GT(r->blen, 0) << "blen > 0";
				EXPECT_LE(r->mlen, r->blen) << "mlen <= blen";
			}
			freeAlignmentResults(regs[0], n_regs[0]);
		}
	}

	// --- Determinism ---
	{
		struct AlignResult {
			int32_t score, qs, qe, rs, re, mapq, dp_score;
			bool rev;
		};

		auto runAlignment = [&]() -> std::vector<AlignResult> {
			ScopedMemPool km;
			chain_read_t read;
			createReadWithConstantChain(read, km, tc, getPrimaryQuery());
			std::vector<AlignResult> results;
			if (read.n == 0) return results;
			mm_reg1_t* regs[1] = {nullptr};
			int n_regs[1] = {0};
			mm_map_align(mi, &opt, &read, regs, n_regs, &tbuf->timers.align, km, NULL, NULL, 0);
			for (int i = 0; i < n_regs[0]; i++) {
				results.push_back({regs[0][i].score, regs[0][i].qs, regs[0][i].qe,
				    regs[0][i].rs, regs[0][i].re, (int32_t)regs[0][i].mapq,
				    regs[0][i].p ? regs[0][i].p->dp_score : 0,
				    regs[0][i].rev != 0});
			}
			freeAlignmentResults(regs[0], n_regs[0]);
			return results;
		};

		auto r1 = runAlignment();
		if (!r1.empty()) {
			auto r2 = runAlignment();
			ASSERT_EQ(r1.size(), r2.size()) << "Alignment count should be deterministic";
			for (size_t i = 0; i < r1.size(); i++) {
				EXPECT_EQ(r1[i].score, r2[i].score) << "Score at " << i;
				EXPECT_EQ(r1[i].qs, r2[i].qs) << "qs at " << i;
				EXPECT_EQ(r1[i].qe, r2[i].qe) << "qe at " << i;
				EXPECT_EQ(r1[i].rs, r2[i].rs) << "rs at " << i;
				EXPECT_EQ(r1[i].re, r2[i].re) << "re at " << i;
				EXPECT_EQ(r1[i].mapq, r2[i].mapq) << "mapq at " << i;
				EXPECT_EQ(r1[i].dp_score, r2[i].dp_score) << "dp_score at " << i;
			}
		}
	}
}

// Combines: MappingQuality + CIGARVerification + ReverseStrandAlignment
TEST_F(MapAlignBehavioralTest, AlignQualityMetrics)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr);

	if (tc.n_expected_chains == 0 || tc.n_expected_anchors == 0)
		GTEST_SKIP() << "No chains/anchors";

	// --- MAPQ validation ---
	{
		ScopedMemPool km;
		chain_read_t read;
		createReadWithConstantChain(read, km, tc, getPrimaryQuery());
		if (read.n > 0) {
			mm_reg1_t* regs[1] = {nullptr};
			int n_regs[1] = {0};
			mm_map_align(mi, &opt, &read, regs, n_regs, &tbuf->timers.align, km, NULL, NULL, 0);
			if (n_regs[0] > 0) {
				validateMappingQuality(regs[0], n_regs[0]);
				const ExpectedAlignment* expected = tc.getPrimaryAlignment();
				if (expected && expected->dp_score > 0)
					EXPECT_EQ(regs[0][0].mapq, expected->mapq) << "Primary MAPQ should match";
			}
			freeAlignmentResults(regs[0], n_regs[0]);
		}
	}

	// --- CIGAR verification (use validateCigar helper) ---
	{
		ScopedMemPool km;
		chain_read_t read;
		createReadWithConstantChain(read, km, tc, getPrimaryQuery());
		if (read.n > 0) {
			mm_reg1_t* regs[1] = {nullptr};
			int n_regs[1] = {0};
			mm_map_align(mi, &opt, &read, regs, n_regs, &tbuf->timers.align, km, NULL, NULL, 0);
			if (n_regs[0] > 0 && regs[0][0].p) {
				mm_reg1_t* r = &regs[0][0];
				CigarStats stats = validateCigar(r);
				EXPECT_TRUE(stats.valid) << "CIGAR should contain only valid operations";
				EXPECT_EQ(stats.qlen, (uint32_t)(r->qe - r->qs)) << "CIGAR query length should match";
				EXPECT_EQ(stats.rlen, (uint32_t)(r->re - r->rs)) << "CIGAR ref length should match";
				EXPECT_GT(stats.n_match, stats.n_ins + stats.n_del) << "Matches should dominate over indels";
			}
			freeAlignmentResults(regs[0], n_regs[0]);
		}
	}

	// --- Reverse strand alignment ---
	{
		// Create reverse complement of cached query sequence
		const mm_bseq1_t& seq = getPrimaryQuery();
		std::string rc_seq = reverseComplement(seq.seq);
		ScopedMemPool km;
		chain_read_t read = createRead("query_revcomp", rc_seq.c_str(), km, rc_seq.length());
		mm_reg1_t* regs[1] = {nullptr};
		int n_regs[1] = {0};
		runAlignPipeline(read, regs, n_regs, km);
		// Check alignment properties (rev strand may or may not be primary)
		if (n_regs[0] > 0)
			EXPECT_GT(regs[0][0].score, 0) << "Score should be positive";
		freeAlignmentResults(regs[0], n_regs[0]);
	}
}

// Combines: AlignWithoutCIGAR + AlignWithQStrandMode + AlignWithAllChainsFlag
//         + AlignWithShortReadMode + AlignWithEQXFlag + AlignWithSpliceMode
TEST_F(MapAlignBehavioralTest, AlignFlagVariants)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr);

	if (tc.n_expected_chains == 0 || tc.n_expected_anchors == 0)
		GTEST_SKIP() << "No chains/anchors";

	// --- No CIGAR (alignment without CIGAR generation) ---
	{
		mm_mapopt_t opt_no_cigar = opt;
		opt_no_cigar.flag &= ~MM_F_CIGAR;
		const mm_bseq1_t& seq = getPrimaryQuery();
		ScopedMemPool km;
		chain_read_t read = createRead(seq.name, seq.seq, km, seq.l_seq);
		mm_reg1_t* regs[1] = {nullptr};
		int n_regs[1] = {0};
		mm_map_seed(mi, &opt_no_cigar, &read, &tbuf->timers.seed, km, NULL);
		mm_map_chain(mi, &opt_no_cigar, &read, &tbuf->timers.chain, km, NULL);
		mm_map_align(mi, &opt_no_cigar, &read, regs, n_regs, &tbuf->timers.align, km, NULL, NULL, 0);
		if (n_regs[0] > 0) {
			EXPECT_GT(regs[0][0].score, 0) << "Should have positive score";
			EXPECT_EQ(regs[0][0].p, nullptr) << "CIGAR should not be generated without MM_F_CIGAR flag";
		}
		freeAlignmentResults(regs[0], n_regs[0]);
	}

	// Helper: run alignment with constant chain + modified opts
	auto runWithOpt = [&](mm_mapopt_t& mopts) {
		ScopedMemPool km;
		chain_read_t read;
		createReadWithConstantChain(read, km, tc, getPrimaryQuery());
		if (read.n == 0) return std::make_pair(0, (mm_reg1_t*)nullptr);
		mm_reg1_t* regs[1] = {nullptr};
		int n_regs[1] = {0};
		mm_map_align(mi, &mopts, &read, regs, n_regs, &tbuf->timers.align, km, NULL, NULL, 0);
		return std::make_pair(n_regs[0], regs[0]);
	};

	// --- Baseline for comparison ---
	auto [n_base, regs_base] = runWithOpt(opt);

	// --- QSTRAND mode (query-strand alignment) ---
	{
		mm_mapopt_t opt_qs = opt;
		opt_qs.flag |= MM_F_QSTRAND;
		auto [n, r] = runWithOpt(opt_qs);
		if (n > 0 && n_base > 0) {
			EXPECT_GT(r[0].score, 0) << "QSTRAND should have positive score";
			EXPECT_GE(r[0].mapq, 0) << "QSTRAND MAPQ should be non-negative";
		}
		freeAlignmentResults(r, n);
	}

	// --- ALL_CHAINS (no filtering) ---
	{
		mm_mapopt_t opt_all = opt;
		opt_all.flag |= MM_F_ALL_CHAINS;
		auto [n, r] = runWithOpt(opt_all);
		// ALL_CHAINS should produce same or more alignments (no filtering)
		EXPECT_GE(n, n_base) << "MM_F_ALL_CHAINS should not filter any chains";
		if (n > 0) {
			// Verify all alignments have valid scores
			for (int i = 0; i < n; i++) {
				EXPECT_GT(r[i].score, 0) << "Alignment " << i << " should have positive score";
				EXPECT_LE(r[i].mapq, 60) << "Alignment " << i << " MAPQ should be <= 60";
			}
		}
		freeAlignmentResults(r, n);
	}

	// --- Short read mode (SR) ---
	{
		mm_mapopt_t opt_sr = opt;
		opt_sr.flag |= MM_F_SR;
		auto [n, r] = runWithOpt(opt_sr);
		if (n > 0) EXPECT_GE(r[0].mapq, 0) << "SR MAPQ should be non-negative";
		freeAlignmentResults(r, n);
	}

	// --- EQX mode: converts CIGAR M operations to =/X (match/mismatch) ---
	{
		mm_mapopt_t opt_eqx = opt;
		opt_eqx.flag |= MM_F_EQX;
		auto [n, r] = runWithOpt(opt_eqx);
		if (n > 0 && r[0].p && r[0].p->n_cigar > 0) {
			// Verify CIGAR has =/X operations instead of M
			bool has_eq_or_x = false, has_m = false;
			for (uint32_t i = 0; i < r[0].p->n_cigar; i++) {
				int op = r[0].p->cigar[i] & 0xf;
				if (op == 7 || op == 8) has_eq_or_x = true; // = or X
				if (op == 0) has_m = true; // M operation
			}
			EXPECT_TRUE(has_eq_or_x) << "EQX mode should produce = or X CIGAR operations";
			EXPECT_FALSE(has_m) << "EQX mode should NOT have M operations (should be converted to =/X)";
		}
		freeAlignmentResults(r, n);
	}

	// --- Splice mode (for RNA-seq) ---
	{
		mm_mapopt_t opt_splice = opt;
		opt_splice.flag |= MM_F_SPLICE | MM_F_SPLICE_FOR | MM_F_SPLICE_REV;
		auto [n, r] = runWithOpt(opt_splice);
		// Splice mode should complete without crash
		EXPECT_GE(n, 0) << "Splice mode alignment should complete";
		if (n > 0) {
			EXPECT_GE(r[0].score, 0);
			EXPECT_GT(r[0].qe, r[0].qs);
		}
		freeAlignmentResults(r, n);
	}

	freeAlignmentResults(regs_base, n_base);
}
