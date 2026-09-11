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
 * Regression tests for AIOSS-5026 / AIOSS-4486:
 *
 * Root cause: in v2.24.0a2, max_chain_skip was forced to INT32_MAX for ALL
 * runs with MM_F_SR set (the sr preset), not just --gpu-chain runs. This caused
 * ~60% of short reads to produce no alignment when mapped without --gpu-chain.
 *
 * Fix (commit 83824c6b, PR AIOSS-4486): the override is now scoped to
 * (MM_F_GPU_CHAIN set) && !(user explicitly set --max-chain-skip).
 *
 * Tests:
 *   T1b: Option-parsing guard — mm_set_opt("sr") must not set max_chain_skip
 *        to INT32_MAX. The only path that legitimately sets INT32_MAX is the
 *        CLI --gpu-chain flag in main.c. The library API must never do this.
 *
 *   T1a: Alignment count parity — mapping reads with the sr preset against the
 *        existing small test dataset must produce > 0 alignments. This catches
 *        the class of regression where valid reads silently produce no output.
 */

#include "test_map_common.h"
#include <climits>

// ============================================================================
// T1b: Option-parsing guard
//
// Verifies that mm_set_opt("sr", ...) + mm_mapopt_update() do NOT set
// max_chain_skip = INT32_MAX. This guards the AIOSS-4486 regression: the
// override must only be applied by the CLI when --gpu-chain is active, not
// baked into the library preset.
// ============================================================================

TEST(SrPresetRegression, MaxChainSkipNotForcedForCpuSr)
{
	mm_idxopt_t iopt;
	mm_mapopt_t opt;
	mm_idxopt_init(&iopt);
	mm_mapopt_init(&opt);
	mm_set_opt("sr", &iopt, &opt);

	// MM_F_GPU_CHAIN must not be set by the sr preset alone
	EXPECT_EQ(opt.flag & MM_F_GPU_CHAIN, static_cast<long long>(0))
	    << "AIOSS-4486 regression: sr preset must not set MM_F_GPU_CHAIN";

	// max_chain_skip must not be INT32_MAX without --gpu-chain
	EXPECT_NE(opt.max_chain_skip, INT32_MAX)
	    << "AIOSS-4486 regression: max_chain_skip must not be INT32_MAX for "
	       "cpu-only sr runs; this override is only valid when --gpu-chain is "
	       "active (MM_F_GPU_CHAIN set)";
}

// ============================================================================
// T1a: Alignment count parity
//
// Maps the small test dataset using the sr preset and verifies that at least
// one alignment is produced. This catches the symptom of AIOSS-4486 where
// ~60% of reads silently produced no alignment record.
//
// Uses the existing "small" test data (original dataset with known mappable
// reads). The sr preset uses shorter k-mer (k=21) vs long-read presets,
// so it can still find anchors in the long-read reference — enough to verify
// the chaining logic produces output rather than discarding all chains.
// ============================================================================

class SrAlignmentCountTest : public ::testing::Test
{
      protected:
	mm_idx_t* mi = nullptr;
	mm_mapopt_t opt;
	mm_idxopt_t iopt;
	mm_tbuf_t* tbuf = nullptr;

	std::string ref_path;
	std::string query_path;

	void SetUp() override
	{
		// Use the existing original test data (MT-human reference, MT-orang query)
		std::string suite_dir = TEST_SUITE_DIR;
		ref_path = suite_dir + "/original/MT-human.fa";
		query_path = suite_dir + "/original/MT-orang.fa";

		mm_idxopt_init(&iopt);
		mm_mapopt_init(&opt);
		mm_set_opt("sr", &iopt, &opt);

		mm_idx_reader_t* rdr = mm_idx_reader_open(ref_path.c_str(), &iopt, nullptr);
		if (rdr) {
			mi = mm_idx_reader_read(rdr, 1);
			mm_idx_reader_close(rdr);
		}
		if (mi) {
			mm_mapopt_update(&opt, mi);
			mm_idx_index_name(mi);
		}
		tbuf = mm_tbuf_init();
	}

	void TearDown() override
	{
		if (mi) mm_idx_destroy(mi);
		if (tbuf) mm_tbuf_destroy(tbuf);
	}
};

TEST_F(SrAlignmentCountTest, ProducesAlignmentsForSrPreset)
{
	ASSERT_NE(mi, nullptr) << "Failed to build index from " << ref_path;
	ASSERT_NE(tbuf, nullptr);

	// Load query sequences
	mm_bseq_file_t* fp = mm_bseq_open(query_path.c_str());
	ASSERT_NE(fp, nullptr) << "Failed to open query file: " << query_path;

	int n_seqs = 0;
	mm_bseq1_t* seqs = mm_bseq_read(fp, INT64_MAX, 0, &n_seqs);
	mm_bseq_close(fp);

	ASSERT_GT(n_seqs, 0) << "No query sequences loaded from " << query_path;

	int total_hits = 0;
	for (int i = 0; i < n_seqs; i++) {
		int n_regs = 0;
		mm_reg1_t* regs =
		    mm_map(mi, seqs[i].l_seq, seqs[i].seq, &n_regs, tbuf, &opt, seqs[i].name);
		total_hits += n_regs;
		free(regs);
	}

	for (int i = 0; i < n_seqs; i++) {
		free(seqs[i].seq);
		free(seqs[i].qual);
		free(seqs[i].name);
		free(seqs[i].comment);
	}
	free(seqs);

	// With the AIOSS-4486 regression, total_hits would be 0 because max_chain_skip=INT32_MAX
	// causes all short-read chains to be discarded. At least 1 alignment must be produced.
	EXPECT_GT(total_hits, 0)
	    << "AIOSS-4486 regression: sr preset produced no alignments across " << n_seqs
	    << " queries. This indicates max_chain_skip may be incorrectly set to INT32_MAX.";
}
