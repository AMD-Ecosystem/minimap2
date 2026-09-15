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
 * GPU Alignment Unit Tests
 *
 * Tests for the GPU alignment pipeline (gpu_align_dispatch / ksw2_gpu_wrapper).
 * Mirrors the CPU alignment tests in tests/core/test_map_align.cpp,
 * comparing GPU alignment results against CPU alignment for correctness.
 *
 * The GPU alignment is invoked through the standard mm_map_align pipeline
 * when MM_F_GPU_ALIGN is set. This test suite enables the GPU align flag
 * and validates the full seed -> chain -> align pipeline with GPU alignment.
 *
 * Exact-match + T5 count-tolerance tests (parameterized across all datasets):
 *   VerifyAllAlignmentsVsCpuBaseline
 *
 * Behavioral tests (single representative dataset):
 *   GpuAlignBasicsAndDeterminism, GpuAlignQualityMetrics, GpuAlignFlagVariants
 *
 * GPU chain + CPU align isolation tests (parameterized):
 *   VerifyGpuChainCpuAlignVsBaseline
 *
 * Full GPU pipeline tests (parameterized, GPU chain + GPU align):
 *   VerifyFullGpuPipelineVsCpuBaseline
 *
 * Requirements:
 * - GPU hardware (AMD ROCm) must be available
 * - Built with GPU=AMD and HAVE_GPU_ALIGNMENT
 */

#include "test_gpu_common.h"
#include "gpu/gpu_device_queries.h"

// ============================================================================
// Parameterized tests - run against all test datasets
// ============================================================================

// Test exact GPU alignment output vs CPU baseline from test JSON.
// Uses the shared verifyAlignments() helper which checks all alignment fields
// (score, qs, qe, rs, re, mapq, rev, blen, mlen, dp_score, dp_max, n_ambi,
// n_cigar_ops), reporting up to 10 mismatches per field with a summary.
// This flags any divergence between GPU alignment and the CPU baseline -
// e.g. hg00438 datasets are known to have GPU/CPU alignment differences.
//
// Additionally asserts (T5) that GPU alignment count does not exceed CPU by
// more than 2% for every query sequence that has a precomputed baseline.
// getAllQueryData() returns all MappingTestData entries for the same query
// file (one per query in the JSON "queries" array), each with its own
// query_index and n_expected_alignments, so no CPU re-run is needed.
TEST_P(GpuAlignParamTest, VerifyAllAlignmentsVsCpuBaseline)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr) << "Index should be loaded";

	if (tc.n_expected_alignments == 0) {
		GTEST_SKIP() << "Test data has no expected alignments";
	}

	// --- Exact field-match on representative read ---
	{
		ScopedMemPool km;
		const mm_bseq1_t& seq = getPrimaryQuery();
		chain_read_t read = createReadFromSeq(km, seq);

		mm_reg1_t* regs[1] = {nullptr};
		int n_regs[1] = {0};
		runAlignPipeline(read, regs, n_regs, km);

		verifyAlignments(regs[0], n_regs[0], tc, "GPU align vs CPU baseline");
		freeAlignmentResults(regs[0], n_regs[0]);
	}

	// --- T5: per-read GPU count vs precomputed baseline for all queries ---
	// getAllQueryData() returns one MappingTestData* per query sequence in
	// this dataset's JSON, each carrying its own precomputed n_expected_alignments.
	// This covers every read that has a CPU baseline — no CPU re-run needed,
	// no scaling. No lower bound: GPU DP divergence producing slightly fewer
	// alignments is a known, accepted property of the kernel.
	for (const MappingTestData* qd : getAllQueryData()) {
		ScopedMemPool km;
		chain_read_t read = createReadFromSeq(km, getQuerySeq(qd->query_index));
		mm_reg1_t* regs[1] = {nullptr};
		int n_regs[1] = {0};
		runAlignPipeline(read, regs, n_regs, km);

		const double ratio =
		    static_cast<double>(n_regs[0]) /
		    static_cast<double>(qd->n_expected_alignments);
		EXPECT_LE(ratio, 1.02)
		    << "T5: GPU alignment count (" << n_regs[0]
		    << ") exceeds CPU baseline (" << qd->n_expected_alignments
		    << ") by more than 2% on query " << qd->query_index
		    << " of dataset " << tc.name
		    << ". Possible bandwidth mismatch (AIOSS-5029) or false-positive inflation.";

		freeAlignmentResults(regs[0], n_regs[0]);
	}
}

// VerifyAllAlignmentsVsCpuBaseline requires an alignment baseline to diff
// against, so only instantiate the suite for datasets that have one.
INSTANTIATE_TEST_SUITE_P(
    AllTestData,
    GpuAlignParamTest,
    ::testing::ValuesIn(getTestDataWithAlignments()),
    testDataNameGenerator);

// ============================================================================
// Behavioral tests - single representative dataset
// ============================================================================

// Combines: GpuAlignSubsequence + GpuAlignEmptyInput + GpuAlignResultSorting +
//           GpuAlignCoordinateConsistency + GpuAlignDeterminism
TEST_F(GpuAlignBehavioralTest, GpuAlignBasicsAndDeterminism)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr);

	// --- Subsequence alignment (first 2000bp) ---
	{
		const mm_bseq1_t& seq = getPrimaryQuery();
		int qlen = std::min(2000, seq.l_seq);
		if (qlen >= 100) {
			std::vector<char> subseq(seq.seq, seq.seq + qlen);
			subseq.push_back('\0');

			ScopedMemPool km;
			chain_read_t read = createRead("gpu_query_2000bp", subseq.data(), km, qlen);

			mm_reg1_t* regs[1] = {nullptr};
			int n_regs[1] = {0};
			runAlignPipeline(read, regs, n_regs, km);

			if (n_regs[0] > 0) {
				EXPECT_GT(regs[0][0].score, 0) << "GPU should have positive score";
				EXPECT_LE(regs[0][0].qe, qlen) << "Query end should be within bounds";
			}
			freeAlignmentResults(regs[0], n_regs[0]);
		}
	}

	// --- Empty input ---
	{
		ScopedMemPool km;
		chain_read_t read = createRead("no_match_query", "NNNNNNNN", km, 8);

		mm_reg1_t* regs[1] = {nullptr};
		int n_regs[1] = {0};
		runAlignPipeline(read, regs, n_regs, km);

		EXPECT_EQ(n_regs[0], 0) << "No alignments expected for N-only sequence";
		freeAlignmentResults(regs[0], n_regs[0]);
	}

	if (tc.n_expected_chains == 0 || tc.n_expected_anchors == 0)
		GTEST_SKIP() << "No chains/anchors for behavioral tests";

	// --- Sorting check (alignments sorted by score descending) ---
	{
		auto res = runConstantChainAlign();
		if (res->n_regs > 1) {
			for (int i = 1; i < res->n_regs; i++) {
				EXPECT_GE(res->regs[i - 1].score, res->regs[i].score)
				    << "GPU alignments should be sorted by score descending at index " << i;
			}
		}
		freeAlignmentResults(res->regs, res->n_regs);
	}

	// --- Coordinate consistency ---
	{
		auto res = runConstantChainAlign();
		for (int i = 0; i < res->n_regs; i++) {
			mm_reg1_t* r = &res->regs[i];
			EXPECT_GE(r->qs, 0) << "Query start >= 0";
			EXPECT_GE(r->rs, 0) << "Reference start >= 0";
			EXPECT_LT(r->qs, r->qe) << "Query start < end";
			EXPECT_LT(r->rs, r->re) << "Reference start < end";
			EXPECT_LE(r->qe, getQueryLen()) << "Query end <= qlen";
			EXPECT_LE(r->re, (int32_t)mi->seq[r->rid].len) << "Ref end <= rlen";
			EXPECT_GT(r->mlen, 0) << "mlen > 0";
			EXPECT_GT(r->blen, 0) << "blen > 0";
			EXPECT_LE(r->mlen, r->blen) << "mlen <= blen";
		}
		freeAlignmentResults(res->regs, res->n_regs);
	}

	// --- Determinism ---
	{
		struct AlignResult {
			int32_t score, qs, qe, rs, re, mapq;
			bool rev;
			int32_t dp_score;
		};

		auto runAlignment = [&]() -> std::vector<AlignResult> {
			auto res = runConstantChainAlign();
			std::vector<AlignResult> results;
			for (int i = 0; i < res->n_regs; i++) {
				AlignResult ar;
				ar.score = res->regs[i].score;
				ar.qs = res->regs[i].qs;
				ar.qe = res->regs[i].qe;
				ar.rs = res->regs[i].rs;
				ar.re = res->regs[i].re;
				ar.mapq = res->regs[i].mapq;
				ar.rev = res->regs[i].rev;
				ar.dp_score = res->regs[i].p ? res->regs[i].p->dp_score : 0;
				results.push_back(ar);
			}
			freeAlignmentResults(res->regs, res->n_regs);
			return results;
		};

		auto r1 = runAlignment();
		if (!r1.empty()) {
			auto r2 = runAlignment();
			ASSERT_EQ(r1.size(), r2.size()) << "GPU alignment count should be deterministic";
			for (size_t i = 0; i < r1.size(); i++) {
				EXPECT_EQ(r1[i].score, r2[i].score) << "Score at " << i;
				EXPECT_EQ(r1[i].qs, r2[i].qs) << "qs at " << i;
				EXPECT_EQ(r1[i].qe, r2[i].qe) << "qe at " << i;
				EXPECT_EQ(r1[i].rs, r2[i].rs) << "rs at " << i;
				EXPECT_EQ(r1[i].re, r2[i].re) << "re at " << i;
				EXPECT_EQ(r1[i].mapq, r2[i].mapq) << "mapq at " << i;
				EXPECT_EQ(r1[i].rev, r2[i].rev) << "rev at " << i;
				EXPECT_EQ(r1[i].dp_score, r2[i].dp_score) << "dp_score at " << i;
			}
		}
	}
}

TEST_F(GpuAlignBehavioralTest, MultiGpuAccumulatorAssignmentRoundRobin)
{
	int device_count = mm_gpu_visible_device_count();
	if (device_count < 2)
		GTEST_SKIP() << "requires at least 2 visible GPUs";

	opt.gpu_accum_pool_size = 4;
	opt.gpu_device_ids = (int*)malloc(2 * sizeof(int));
	ASSERT_NE(opt.gpu_device_ids, nullptr);
	opt.gpu_device_ids[0] = 0;
	opt.gpu_device_ids[1] = 1;
	opt.gpu_num_devices = 2;

	mm_align_ctx_t* align_ctx = mm_align_ctx_create(
	    &opt, 4,
	    mm_gpu_create_device_list(opt.gpu_device_ids, opt.gpu_num_devices));
	ASSERT_NE(align_ctx, nullptr);
	ASSERT_NE(align_ctx->gpu_align, nullptr);
	ASSERT_EQ(mm_gpu_align_ctx_pool_size(align_ctx->gpu_align), 8);

	for (int i = 0; i < mm_gpu_align_ctx_pool_size(align_ctx->gpu_align); i++) {
		EXPECT_EQ(mm_gpu_align_ctx_pool_device_id(align_ctx->gpu_align, i), i % 2);
	}

	mm_align_ctx_destroy(align_ctx);
	free(opt.gpu_device_ids);
	opt.gpu_device_ids = nullptr;
	opt.gpu_num_devices = 0;
}

// Combines: GpuAlignMappingQuality + GpuAlignCIGARVerification
TEST_F(GpuAlignBehavioralTest, GpuAlignQualityMetrics)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr);

	if (tc.n_expected_chains == 0 || tc.n_expected_anchors == 0)
		GTEST_SKIP() << "No chains/anchors";

	// --- MAPQ validation ---
	{
		auto res = runConstantChainAlign();
		if (res->n_regs > 0) {
			validateMappingQuality(res->regs, res->n_regs, " (GPU alignment)");
		}
		freeAlignmentResults(res->regs, res->n_regs);
	}

	// --- CIGAR verification ---
	{
		auto res = runConstantChainAlign();
		if (res->n_regs > 0 && res->regs[0].p) {
			mm_reg1_t* r = &res->regs[0];
			CigarStats stats = validateCigar(r);
			EXPECT_TRUE(stats.valid) << "GPU CIGAR should contain only valid operations";

			int32_t expected_qlen = r->qe - r->qs;
			int32_t expected_rlen = r->re - r->rs;

			EXPECT_EQ(stats.qlen, (uint32_t)expected_qlen)
			    << "GPU CIGAR query length should match alignment coordinates";
			EXPECT_EQ(stats.rlen, (uint32_t)expected_rlen)
			    << "GPU CIGAR ref length should match alignment coordinates";
			EXPECT_GT(stats.n_match, stats.n_ins + stats.n_del)
			    << "Matches should dominate over indels in GPU alignment";
		}
		freeAlignmentResults(res->regs, res->n_regs);
	}
}

// Combines: GpuAlignWithoutCIGAR + GpuAlignWithEQXFlag
TEST_F(GpuAlignBehavioralTest, GpuAlignFlagVariants)
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
		runGpuAlign(mi, &opt_no_cigar, &read, regs, n_regs, km);

		if (n_regs[0] > 0) {
			EXPECT_GT(regs[0][0].score, 0) << "GPU should have positive score without CIGAR";
			EXPECT_EQ(regs[0][0].p, nullptr)
			    << "CIGAR should not be generated without MM_F_CIGAR flag";
		}
		freeAlignmentResults(regs[0], n_regs[0]);
	}

	// --- EQX CIGAR mode ---
	{
		ScopedMemPool km;
		chain_read_t read;
		createReadWithConstantChain(read, km, tc, getPrimaryQuery());

		if (read.n > 0) {
			mm_mapopt_t opt_eqx = opt;
			opt_eqx.flag |= MM_F_EQX;

			mm_reg1_t* regs[1] = {nullptr};
			int n_regs[1] = {0};
			runGpuAlign(mi, &opt_eqx, &read, regs, n_regs, km);

			if (n_regs[0] > 0) {
				mm_reg1_t* r = &regs[0][0];
				if (r->p && r->p->n_cigar > 0) {
					bool has_eq_or_x = false;
					bool has_m = false;
					for (uint32_t i = 0; i < r->p->n_cigar; i++) {
						int op = r->p->cigar[i] & 0xf;
						if (op == 7 || op == 8) has_eq_or_x = true; // = or X
						if (op == 0) has_m = true; // M
					}
					EXPECT_TRUE(has_eq_or_x) << "GPU EQX mode should produce = or X CIGAR operations";
					EXPECT_FALSE(has_m) << "GPU EQX mode should NOT have M operations";
				}
			}
			freeAlignmentResults(regs[0], n_regs[0]);
		}
	}
}

// ============================================================================
// GPU Chain + CPU Align Pipeline Tests (parameterized)
// ============================================================================

// Test GPU chain + CPU align vs CPU baseline.
// Isolates GPU chaining's effect on downstream alignment. When
// VerifyFullGpuPipelineVsCpuBaseline fails, comparing with this test
// reveals whether the divergence comes from GPU chaining or GPU alignment.
// Pipeline: seed(CPU) -> chain(GPU) -> align(CPU).
TEST_P(GpuChainCpuAlignParamTest, VerifyGpuChainCpuAlignVsBaseline)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr) << "Index should be loaded";

	if (tc.n_expected_alignments == 0 || tc.n_expected_anchors == 0) {
		GTEST_SKIP() << "Test data has no expected alignments or anchors";
	}

	// Step 1: Seed (CPU)
	ScopedMemPool km;
	const mm_bseq1_t& seq = getPrimaryQuery();
	chain_read_t read = createReadFromSeq(km, seq);
	mm_map_seed(mi, &opt, &read, &tbuf->timers.seed, km, NULL);

	if (read.n < GpuEnvironment::s_min_n) {
		GTEST_SKIP() << "Too few anchors (" << read.n << ") for GPU chaining (min: " << GpuEnvironment::s_min_n << ")";
	}

	// Step 2: Chain (GPU)
	runGpuChaining(mi, &opt, &read, 1, km);

	// Step 3: Align (CPU) - MM_F_GPU_ALIGN is NOT set
	mm_reg1_t* regs[1] = {nullptr};
	int n_regs[1] = {0};
	mm_map_align(mi, &opt, &read, regs, n_regs, &tbuf->timers.align, km, NULL, NULL, 0);

	verifyAlignments(regs[0], n_regs[0], tc, "GPU chain + CPU align vs baseline");

	freeAlignmentResults(regs[0], n_regs[0]);
}

// GPU chain + CPU align needs both expected anchors (to seed through the GPU
// chaining path) and expected alignments (as the final baseline).
INSTANTIATE_TEST_SUITE_P(
    AllTestData,
    GpuChainCpuAlignParamTest,
    ::testing::ValuesIn(getTestDataWithAlignmentsAndAnchors()),
    testDataNameGenerator);

// ============================================================================
// Combined GPU Chain + GPU Align Pipeline Tests (parameterized)
// ============================================================================

// Test full GPU pipeline (GPU chain + GPU align) vs CPU baseline.
// This reproduces exactly what --gpu-chain --gpu-align does and compares
// the alignment output against the CPU baseline from the test JSON.
// Uses verifyAlignments() to flag any divergence - e.g. hg00438 datasets
// are known to produce GPU/CPU differences that will be reported here.
TEST_P(GpuFullPipelineParamTest, VerifyFullGpuPipelineVsCpuBaseline)
{
	const MappingTestData& tc = getTestData();
	ASSERT_NE(mi, nullptr) << "Index should be loaded";

	if (tc.n_expected_alignments == 0 || tc.n_expected_anchors == 0) {
		GTEST_SKIP() << "Test data has no expected alignments or anchors";
	}

	// Step 1: Seed (CPU)
	ScopedMemPool km;
	const mm_bseq1_t& seq = getPrimaryQuery();
	chain_read_t read = createReadFromSeq(km, seq);
	mm_map_seed(mi, &opt, &read, &tbuf->timers.seed, km, NULL);

	if (read.n < GpuEnvironment::s_min_n) {
		GTEST_SKIP() << "Too few anchors (" << read.n << ") for GPU chaining (min: " << GpuEnvironment::s_min_n << ")";
	}

	// Step 2: Chain (GPU)
	runGpuChaining(mi, &opt, &read, 1, km);

	// Step 3: Align (GPU) - MM_F_GPU_ALIGN is already set in SetUp
	mm_reg1_t* regs[1] = {nullptr};
	int n_regs[1] = {0};
	runGpuAlignAfterChain(read, regs, n_regs, km);

	verifyAlignments(regs[0], n_regs[0], tc, "Full GPU pipeline vs CPU baseline");

	freeAlignmentResults(regs[0], n_regs[0]);
}

// Full GPU pipeline requires the same anchors + alignments baseline as the
// GPU chain + CPU align pipeline above.
INSTANTIATE_TEST_SUITE_P(
    AllTestData,
    GpuFullPipelineParamTest,
    ::testing::ValuesIn(getTestDataWithAlignmentsAndAnchors()),
    testDataNameGenerator);

// ============================================================================
// Regression tests
//
// Each test is self-contained: it derives the sequence from the representative
// dataset already loaded by GpuAlignBehavioralTest and creates its own reads
// inline, so no additional FASTA files or JSON fixtures are required.
// ============================================================================

class GpuAlignRegressionTest : public GpuAlignBehavioralTest
{
      protected:
	// Run the seed->chain->align pipeline with an explicitly supplied opt,
	// bypassing the fixture's this->opt (which has MM_F_GPU_ALIGN set).
	void runAlignWithOpt(const mm_mapopt_t& run_opt, chain_read_t& read,
	    mm_reg1_t** regs, int* n_regs, void* km)
	{
		mm_map_seed(mi, &run_opt, &read, &tbuf->timers.seed, km, NULL);
		mm_map_chain(mi, &run_opt, &read, &tbuf->timers.chain, km, NULL);
		if (run_opt.flag & MM_F_GPU_ALIGN) {
			runGpuAlign(mi, &run_opt, &read, regs, n_regs, km);
		} else {
			mm_map_align(mi, &run_opt, &read, regs, n_regs, &tbuf->timers.align, km, NULL, NULL, 0);
		}
	}

	// CPU-opt: identical to fixture opt but with GPU_ALIGN flag cleared.
	mm_mapopt_t cpuOpt() const
	{
		mm_mapopt_t o = opt;
		o.flag &= ~MM_F_GPU_ALIGN;
		return o;
	}
};

// Regression test: sc_N wildcard mask must use OR, not AND.
//
// Bug (present from first GPU kernel commit, Dec 2025):
//   GPU kernel computed sc_N substitution score only when BOTH target and
//   query bases were the wildcard m1 (AND logic). CPU uses OR: sc_N whenever
//   EITHER base is the wildcard. This caused a 3-point score difference per
//   wildcard position, which could accumulate to trigger a false z-drop on
//   the first pass of mm_align1's two-pass logic, splitting one contiguous
//   alignment into spurious supplementary records.
//
// Reproducer (1% map-hifi, read m64076_200211_192227/657093/ccs):
//   CPU: 15909M169I2632M3D2030M  AS:i:19931  NM:i:372  (1 record)
//   GPU: 15909M4831S (zd:i:1) + 16267H2443M3D2030M (zd:i:2)  (2 records)
//
// This test injects N characters at interior positions of the MT-orang query
// (which has no N), then verifies that GPU and CPU produce the same number
// of alignments and the same primary score.
TEST_F(GpuAlignRegressionTest, ScNWildcardMaskMatchesCpu)
{
	const mm_bseq1_t& seq = getPrimaryQuery();
	if (seq.l_seq < 500) GTEST_SKIP() << "Query too short for sc_N regression";

	// Inject N at interior positions 200..219 (query has no N naturally).
	// N encodes to the wildcard value (4) via seq_nt4_table; target (MT-human)
	// has no N at these reference positions, so only one side is wildcard.
	std::string query_with_n(seq.seq, seq.l_seq);
	for (int i = 200; i < 220; i++) query_with_n[i] = 'N';

	mm_mapopt_t cpu_opt = cpuOpt();

	// CPU alignment
	ScopedMemPool km_cpu;
	chain_read_t cpu_read = createRead("scn_cpu", query_with_n.c_str(), km_cpu,
	    query_with_n.size());
	mm_reg1_t* cpu_regs = nullptr;
	int cpu_n = 0;
	runAlignWithOpt(cpu_opt, cpu_read, &cpu_regs, &cpu_n, km_cpu);

	// GPU alignment
	ScopedMemPool km_gpu;
	chain_read_t gpu_read = createRead("scn_gpu", query_with_n.c_str(), km_gpu,
	    query_with_n.size());
	mm_reg1_t* gpu_regs = nullptr;
	int gpu_n = 0;
	runAlignWithOpt(opt, gpu_read, &gpu_regs, &gpu_n, km_gpu);

	ASSERT_GT(cpu_n, 0) << "CPU should produce at least one alignment";
	EXPECT_EQ(gpu_n, cpu_n)
	    << "GPU should produce the same number of alignments as CPU "
	       "(spurious z-drop split indicates AND-vs-OR sc_N mask bug)";
	if (gpu_n > 0) {
		EXPECT_EQ(gpu_regs[0].score, cpu_regs[0].score)
		    << "GPU primary alignment score must match CPU for query with N bases";
		EXPECT_EQ(gpu_regs[0].qs, cpu_regs[0].qs) << "query start must match";
		EXPECT_EQ(gpu_regs[0].qe, cpu_regs[0].qe) << "query end must match";
	}

	freeAlignmentResults(cpu_regs, cpu_n);
	freeAlignmentResults(gpu_regs, gpu_n);
}

// Regression test: p-buffer must be zeroed between alignments.
//
// Bug (AIOSS-4672): GpuWorker reused its device p-buffer across alignments
// without zeroing it first. When a shorter alignment followed a longer one
// on the same worker, ksw_backtrack_with_gpu read stale traceback bits from
// the previous (longer) alignment's p-buffer tail, producing a corrupt CIGAR
// and a wrong alignment score.
//
// Fix: hipMemsetAsync(dev.p, 0, r.p_bytes, dev.stream) before each kernel
// launch in GpuWorker::run() (gpu_align_dispatch.cpp).
//
// This test runs a long alignment followed by a short alignment through the
// GPU align path (same worker pool, high probability of buffer reuse), then
// checks that both scores match the CPU reference.
TEST_F(GpuAlignRegressionTest, PBufferReuseDoesNotCorruptSubsequentAlignment)
{
	const mm_bseq1_t& seq = getPrimaryQuery();
	if (seq.l_seq < 200) GTEST_SKIP() << "Query too short for p-buffer regression";

	std::string long_q(seq.seq, seq.l_seq);
	// Short query: first quarter of the sequence — guaranteed shorter p-buffer.
	std::string short_q(seq.seq, seq.l_seq / 4);

	mm_mapopt_t cpu_opt = cpuOpt();

	// CPU reference scores
	int cpu_long_score = 0, cpu_short_score = 0;
	{
		ScopedMemPool km;
		chain_read_t r = createRead("pbuf_long_cpu", long_q.c_str(), km, long_q.size());
		mm_reg1_t* regs = nullptr;
		int n = 0;
		runAlignWithOpt(cpu_opt, r, &regs, &n, km);
		if (n > 0) cpu_long_score = regs[0].score;
		freeAlignmentResults(regs, n);
	}
	{
		ScopedMemPool km;
		chain_read_t r = createRead("pbuf_short_cpu", short_q.c_str(), km, short_q.size());
		mm_reg1_t* regs = nullptr;
		int n = 0;
		runAlignWithOpt(cpu_opt, r, &regs, &n, km);
		if (n > 0) cpu_short_score = regs[0].score;
		freeAlignmentResults(regs, n);
	}

	// GPU: run long first, then short. With N_GPU_WORKERS=4 and sequential
	// submission, workers are assigned round-robin; repeating 5 times ensures
	// each worker sees at least one (long, short) pair.
	std::vector<int> gpu_long_scores, gpu_short_scores;
	for (int rep = 0; rep < 5; rep++) {
		{
			ScopedMemPool km;
			chain_read_t r = createRead("pbuf_long_gpu", long_q.c_str(), km, long_q.size());
			mm_reg1_t* regs = nullptr;
			int n = 0;
			runAlignWithOpt(opt, r, &regs, &n, km);
			gpu_long_scores.push_back(n > 0 ? regs[0].score : 0);
			freeAlignmentResults(regs, n);
		}
		{
			ScopedMemPool km;
			chain_read_t r = createRead("pbuf_short_gpu", short_q.c_str(), km, short_q.size());
			mm_reg1_t* regs = nullptr;
			int n = 0;
			runAlignWithOpt(opt, r, &regs, &n, km);
			gpu_short_scores.push_back(n > 0 ? regs[0].score : 0);
			freeAlignmentResults(regs, n);
		}
	}

	if (cpu_long_score > 0) {
		for (int i = 0; i < 5; i++) {
			EXPECT_EQ(gpu_long_scores[i], cpu_long_score)
			    << "Long GPU alignment score should match CPU (rep " << i << ")";
		}
	}
	if (cpu_short_score > 0) {
		for (int i = 0; i < 5; i++) {
			EXPECT_EQ(gpu_short_scores[i], cpu_short_score)
			    << "Short GPU alignment after long should match CPU "
			       "(stale p-buffer bits would corrupt backtracking) (rep "
			    << i << ")";
		}
	}
}

// Regression test: multi-gap batch alignment (AIOSS-4683 Option A).
// Single kernel per mm_align1 call must produce same result as CPU and be deterministic.
TEST_F(GpuAlignRegressionTest, MultiGapBatchMatchesCpuAndIsDeterministic)
{
	const mm_bseq1_t& seq = getPrimaryQuery();
	if (seq.l_seq < 100) GTEST_SKIP() << "Query too short for multi-gap regression";
	mm_mapopt_t cpu_opt = cpuOpt();
	int cpu_score = 0, cpu_qs = -1, cpu_qe = -1, cpu_n = 0;
	{
		ScopedMemPool km;
		chain_read_t r = createRead("multigap_cpu", seq.seq, km, seq.l_seq);
		mm_reg1_t* regs = nullptr;
		runAlignWithOpt(cpu_opt, r, &regs, &cpu_n, km);
		if (cpu_n > 0) {
			cpu_score = regs[0].score;
			cpu_qs = regs[0].qs;
			cpu_qe = regs[0].qe;
		}
		freeAlignmentResults(regs, cpu_n);
	}
	if (cpu_n == 0) GTEST_SKIP() << "No CPU alignments — cannot validate GPU batch";
	int gpu_score[2] = {}, gpu_qs[2] = {}, gpu_qe[2] = {}, gpu_n[2] = {};
	for (int rep = 0; rep < 2; rep++) {
		ScopedMemPool km;
		chain_read_t r = createRead("multigap_gpu", seq.seq, km, seq.l_seq);
		mm_reg1_t* regs = nullptr;
		runAlignWithOpt(opt, r, &regs, &gpu_n[rep], km);
		if (gpu_n[rep] > 0) {
			gpu_score[rep] = regs[0].score;
			gpu_qs[rep] = regs[0].qs;
			gpu_qe[rep] = regs[0].qe;
		}
		freeAlignmentResults(regs, gpu_n[rep]);
	}
	EXPECT_EQ(gpu_n[0], cpu_n) << "GPU alignment count must match CPU";
	if (gpu_n[0] > 0) {
		EXPECT_EQ(gpu_score[0], cpu_score) << "GPU primary score must match CPU (multi-gap batch)";
		EXPECT_EQ(gpu_qs[0], cpu_qs) << "GPU query start must match CPU";
		EXPECT_EQ(gpu_qe[0], cpu_qe) << "GPU query end must match CPU";
		EXPECT_EQ(gpu_score[1], gpu_score[0]) << "GPU batch result must be deterministic";
		EXPECT_EQ(gpu_qs[1], gpu_qs[0]) << "GPU query start must be deterministic";
		EXPECT_EQ(gpu_qe[1], gpu_qe[0]) << "GPU query end must be deterministic";
	}
}
