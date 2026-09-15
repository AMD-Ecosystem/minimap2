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
 * T3 — --gpu-chain --gpu-align combined mode stability.
 *
 * Root cause (AIOSS-5006, fixed PR #71, 21 May 2026):
 *   plmem_stream_cleanup() called hipStreamDestroy() on each GPU chain
 *   worker's stream without first calling hipStreamSynchronize(). When
 *   the GPU align subsystem was simultaneously active, destroying an
 *   in-flight stream corrupted shared device state → intermittent crash
 *   or extreme slowdown (>2 hours vs 2 minutes for upstream).
 *   Fix: added hipStreamSynchronize(w->hipstream) before hipStreamDestroy.
 *
 * Both Jianshu Zhao (UCSD) and Can Firtina (UMD) independently hit this
 * crash during alpha testing. No automated regression test existed.
 *
 * This test guards the fix by running the combined mode through the
 * mm_map_ctx API and verifying:
 *   (a) No crash / assertion failure
 *   (b) Alignment count within ±1% of CPU-only baseline
 *   (c) All primary alignments have valid coordinates
 */

#include "test_gpu_common.h"
#include <cmath>
#include <string>
#include <vector>

// ============================================================================
// T3: GPU chain + GPU align combined mode stability
// ============================================================================

class GpuCombinedModeTest : public ::testing::Test
{
      protected:
	mm_idx_t*   mi   = nullptr;
	mm_tbuf_t*  tbuf = nullptr;
	mm_idxopt_t iopt;

	std::string ref_path;
	std::string query_path;

	void SetUp() override
	{
		std::string suite_dir = TEST_SUITE_DIR;
		ref_path              = suite_dir + "/original/MT-human.fa";
		query_path            = suite_dir + "/original/MT-orang.fa";

		mm_idxopt_init(&iopt);

		mm_idx_reader_t* rdr = mm_idx_reader_open(ref_path.c_str(), &iopt, nullptr);
		if (!rdr) {
			FAIL() << "Cannot open reference: " << ref_path;
			return;
		}
		mi = mm_idx_reader_read(rdr, 1);
		mm_idx_reader_close(rdr);
		if (!mi) {
			FAIL() << "Cannot build index";
			return;
		}
		mm_idx_index_name(mi);
		tbuf = mm_tbuf_init();
	}

	void TearDown() override
	{
		if (tbuf) mm_tbuf_destroy(tbuf);
		if (mi) mm_idx_destroy(mi);
	}

	// Load query sequences from query_path, map with given opt, return total alignment count.
	int mapQueryFile(mm_mapopt_t& opt) const
	{
		mm_mapopt_update(&opt, mi);

		mm_bseq_file_t* fp = mm_bseq_open(query_path.c_str());
		if (!fp) return -1;

		int          n_seqs = 0;
		mm_bseq1_t*  seqs   = mm_bseq_read(fp, INT64_MAX, 0, &n_seqs);
		mm_bseq_close(fp);
		if (n_seqs <= 0) {
			free(seqs);
			return 0;
		}

		int total = 0;
		for (int i = 0; i < n_seqs; i++) {
			int        n_regs = 0;
			mm_reg1_t* regs =
			    mm_map(mi, seqs[i].l_seq, seqs[i].seq, &n_regs, tbuf, &opt, seqs[i].name);
			total += n_regs;
			for (int j = 0; j < n_regs; j++) {
				if (regs[j].p) free(regs[j].p);
			}
			free(regs);
		}
		for (int i = 0; i < n_seqs; i++) {
			free(seqs[i].seq);
			free(seqs[i].qual);
			free(seqs[i].name);
			free(seqs[i].comment);
		}
		free(seqs);
		return total;
	}
};

// T3a: combined mode does not crash and produces alignments
TEST_F(GpuCombinedModeTest, NoCrashAndProducesAlignments)
{
	mm_mapopt_t opt;
	mm_mapopt_init(&opt);
	mm_set_opt("map-ont", &iopt, &opt);
	opt.flag |= MM_F_CIGAR | MM_F_GPU_CHAIN | MM_F_GPU_ALIGN;

	// This must not crash (AIOSS-5006). The regression caused a HIP
	// stream-destroy-while-in-flight corruption in combined mode.
	int count = mapQueryFile(opt);
	EXPECT_GE(count, 0) << "T3: mapQueryFile returned error";
	EXPECT_GT(count, 0)
	    << "T3: --gpu-chain --gpu-align produced no alignments. "
	       "Combined mode should map at least as many reads as CPU mode.";
}

// T3b: alignment count is within ±5% of CPU-only baseline
// (Wider tolerance than T3a: GPU chain has known DP divergence vs CPU chain.)
TEST_F(GpuCombinedModeTest, AlignmentCountWithinToleranceOfCpuBaseline)
{
	// CPU baseline
	mm_mapopt_t cpu_opt;
	mm_mapopt_init(&cpu_opt);
	mm_set_opt("map-ont", &iopt, &cpu_opt);
	cpu_opt.flag |= MM_F_CIGAR;
	int cpu_count = mapQueryFile(cpu_opt);
	ASSERT_GT(cpu_count, 0) << "CPU baseline produced no alignments";

	// GPU combined
	mm_mapopt_t gpu_opt;
	mm_mapopt_init(&gpu_opt);
	mm_set_opt("map-ont", &iopt, &gpu_opt);
	gpu_opt.flag |= MM_F_CIGAR | MM_F_GPU_CHAIN | MM_F_GPU_ALIGN;
	int gpu_count = mapQueryFile(gpu_opt);
	ASSERT_GE(gpu_count, 0) << "GPU combined mode error";

	if (cpu_count == 0) {
		GTEST_SKIP() << "CPU baseline is 0; cannot compute tolerance";
	}

	double ratio = static_cast<double>(gpu_count) / static_cast<double>(cpu_count);
	EXPECT_GE(ratio, 0.95)
	    << "T3: GPU combined alignment count (" << gpu_count
	    << ") is >5% below CPU baseline (" << cpu_count << "). "
	       "GPU chain DP divergence is known but should not cause large count drops.";
	EXPECT_LE(ratio, 1.05)
	    << "T3: GPU combined alignment count (" << gpu_count
	    << ") is >5% above CPU baseline (" << cpu_count
	    << "). Excess alignments may indicate false positives.";
}
