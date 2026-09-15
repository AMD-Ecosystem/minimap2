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
 * Test fixtures for mm_map_* tests.
 *
 * This header provides:
 * - MapTestFixture: Base fixture that creates index from test data reference sequence
 * - MapParamTestFixture: Parameterized fixture for testing against all test datas
 * - Helper methods for creating chain_read_t structures
 *
 * Test data (inputs and expected values) is defined in JSON files in test_suite/<category>/expected/ referencing FASTA files.
 * Utility functions are in test_helpers.h.
 */
#ifndef TEST_FIXTURES_H
#define TEST_FIXTURES_H

#include <gtest/gtest.h>
#include <cassert>
#include "map_priv.h"
#include "bseq.h"
#include "kalloc.h"
#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "test_data.h"
#include "test_helpers.h"

// ============================================================================
// Test name generator for parameterized tests
// ============================================================================

inline std::string testDataNameGenerator(
    const ::testing::TestParamInfo<const MappingTestData*>& info)
{
	// Handle null or empty name
	if (!info.param || !info.param->name || info.param->name[0] == '\0') {
		return "UnnamedTest_" + std::to_string(info.index);
	}

	std::string name = info.param->name;

	// Replace all non-alphanumeric characters with underscores
	// GoogleTest only accepts [A-Za-z0-9_] in test names
	for (char& c : name) {
		if (!std::isalnum(static_cast<unsigned char>(c))) {
			c = '_';
		}
	}

	// Ensure name doesn't start with a digit (prepend underscore if so)
	if (!name.empty() && std::isdigit(static_cast<unsigned char>(name[0]))) {
		name = "_" + name;
	}

	// Handle empty result after sanitization
	if (name.empty()) {
		return "Test_" + std::to_string(info.index);
	}

	return name;
}

// ============================================================================
// Base test fixture for mapping tests
// ============================================================================

/**
 * Common test fixture for mapping tests.
 * Creates index from test data reference file and provides helpers.
 */
class MapTestFixture : public ::testing::Test
{
      protected:
	mm_idx_t* mi = nullptr;
	mm_mapopt_t opt;
	mm_idxopt_t iopt;
	mm_tbuf_t* tbuf = nullptr;

	// Cached query sequence (loaded from file)
	mm_bseq1_t* query_seqs = nullptr;
	int n_query_seqs = 0;

	// Get test data - must be implemented by derived classes
	virtual const MappingTestData& getTestData() const = 0;

	void SetUp() override
	{
		const MappingTestData& tc = getTestData();

		// Initialize index options
		mm_idxopt_init(&iopt);
		mm_mapopt_init(&opt);

		// Apply preset if specified (e.g. "map-ont", "map-pb", "sr")
		// Must be applied BEFORE building index, as presets may set k/w
		if (tc.preset && tc.preset[0] != '\0') {
			mm_set_opt(tc.preset, &iopt, &opt);
		}

		// Apply explicit k/w overrides from test data (takes precedence over preset)
		if (tc.k > 0) iopt.k = tc.k;
		if (tc.w > 0) iopt.w = tc.w;

		// Build index from reference FASTA file (supports large sequences)
		mm_idx_reader_t* idx_rdr = mm_idx_reader_open(tc.ref_path, &iopt, nullptr);
		if (idx_rdr) {
			mi = mm_idx_reader_read(idx_rdr, 1);
			mm_idx_reader_close(idx_rdr);
		}

		// Update mapopt based on index
		if (mi) {
			mm_mapopt_update(&opt, mi);
			mm_idx_index_name(mi);
		}

		// Load query sequences from file (use large chunk size to read ALL sequences)
		mm_bseq_file_t* fp = mm_bseq_open(tc.query_path);
		if (fp) {
			query_seqs = mm_bseq_read(fp, INT64_MAX, 0, &n_query_seqs);
			mm_bseq_close(fp);
		}

		// Create thread buffer
		tbuf = mm_tbuf_init();
	}

	void TearDown() override
	{
		// Free query sequences
		if (query_seqs) {
			for (int i = 0; i < n_query_seqs; i++) {
				free(query_seqs[i].seq);
				free(query_seqs[i].qual);
				free(query_seqs[i].name);
				free(query_seqs[i].comment);
			}
			free(query_seqs);
		}
		if (mi) mm_idx_destroy(mi);
		if (tbuf) mm_tbuf_destroy(tbuf);
	}

	// ========================================================================
	// Query Sequence Accessors
	// ========================================================================

	/**
     * Get the primary query sequence for this test case.
     * Uses query_index from test data to select the correct query
     * from multi-query FASTA files.
     */
	const mm_bseq1_t& getPrimaryQuery() const
	{
		int idx = getTestData().query_index;
		assert(idx >= 0 && idx < n_query_seqs);
		return query_seqs[idx];
	}

	/**
     * Get primary query sequence pointer (convenience accessor).
     */
	const char* getQuerySeq() const { return getPrimaryQuery().seq; }

	/**
     * Get primary query sequence length (convenience accessor).
     */
	int getQueryLen() const { return getPrimaryQuery().l_seq; }

	/**
     * Get primary query sequence name (convenience accessor).
     */
	const char* getQueryName() const { return getPrimaryQuery().name; }

	// ========================================================================
	// chain_read_t Creation Helpers
	// ========================================================================

	/**
     * Create chain_read_t from sequence string.
     */
	chain_read_t createRead(const char* name, const char* seq, void* km, int qlen = -1)
	{
		chain_read_t read;
		memset(&read, 0, sizeof(chain_read_t));

		read.n_seg = 1;
		read.qlens = (int*)kmalloc(km, sizeof(int));
		read.qlens[0] = (qlen >= 0) ? qlen : strlen(seq);

		read.qseqs = (const char**)kmalloc(km, sizeof(char*));
		read.qseqs[0] = seq;

		if (name) {
			strncpy(read.seq.name, name, 199);
			read.seq.name[199] = '\0';
		}
		read.seq.qlen_sum = read.qlens[0];

		return read;
	}

	/**
     * Create chain_read_t from loaded sequence (mm_bseq1_t).
     * Common pattern for chaining and alignment tests.
     */
	chain_read_t createReadFromSeq(void* km, const mm_bseq1_t& seq)
	{
		chain_read_t read;
		memset(&read, 0, sizeof(read));
		read.n_seg = 1;
		read.qlens = (int*)kmalloc(km, sizeof(int));
		read.qseqs = (const char**)kmalloc(km, sizeof(char*));
		read.qlens[0] = seq.l_seq;
		read.qseqs[0] = seq.seq;
		read.seq.qlen_sum = seq.l_seq;
		strncpy(read.seq.name, seq.name, 199);
		read.seq.name[199] = '\0';
		return read;
	}

	/**
     * Create minimal chain_read_t for chaining tests (no sequence data).
     */
	chain_read_t createMinimalRead(const char* name, int qlen, void* km)
	{
		chain_read_t read;
		memset(&read, 0, sizeof(chain_read_t));
		read.n_seg = 1;
		read.qlens = (int*)kmalloc(km, sizeof(int));
		read.qlens[0] = qlen;
		read.seq.qlen_sum = qlen;
		strncpy(read.seq.name, name, 199);
		read.seq.name[199] = '\0';
		return read;
	}

	// ========================================================================
	// Anchor/Chain Population Helpers
	// ========================================================================

	/**
     * Populate chain_read_t with constant anchor data from test data.
     * Allows skipping mm_map_seed and testing mm_map_chain in isolation.
     */
	void populateAnchorsFromConstants(chain_read_t& read, void* km,
	    const MappingTestData& tc, uint32_t q_span = 0)
	{
		if (q_span == 0) q_span = tc.getKmerSize();

		read.n = tc.n_expected_anchors;
		read.a = (mm128_t*)kmalloc(km, read.n * sizeof(mm128_t));

		for (size_t i = 0; i < tc.n_expected_anchors; i++) {
			read.a[i].x = tc.expected_anchors[i].x;
			read.a[i].y = ((uint64_t)q_span << 32) | tc.expected_anchors[i].y;
		}

		read.n_mini_pos = tc.n_expected_anchors;
		read.rep_len = 0;
		read.frag_gap = 0; // Will be set by mm_map_chain
	}

	/**
     * Create chain_read_t with constant chain data for testing mm_map_align.
     * Skips both seeding and chaining, using pre-computed chain anchors.
     */
	void createReadWithConstantChain(chain_read_t& read, void* km, const MappingTestData& tc,
	    const mm_bseq1_t& seq)
	{
		const ExpectedChain* chain = tc.getPrimaryChain();

		memset(&read, 0, sizeof(chain_read_t));

		// Set up sequence info from loaded sequence
		read.n_seg = 1;
		read.qlens = (int*)kmalloc(km, sizeof(int));
		read.qlens[0] = seq.l_seq;
		read.seq.qlen_sum = seq.l_seq;
		strncpy(read.seq.name, seq.name, 199);

		// Set up sequence pointers (needed for DP alignment)
		read.qseqs = (const char**)kmalloc(km, sizeof(char*));
		read.qseqs[0] = seq.seq;

		// Populate anchors from expected chain
		const uint32_t q_span = tc.getKmerSize();
		const int chain_len = chain ? chain->length : 0;

		// Find the starting anchor index by matching chain's first_qpos
		// Some anchors at the beginning may be outside the main chain range
		int chain_start = 0;
		if (chain) {
			for (int i = 0; i < tc.n_expected_anchors; i++) {
				const ExpectedAnchor* anchor = tc.getAnchor(i);
				if (anchor && anchor->y == chain->first_qpos) {
					chain_start = i;
					break;
				}
			}
		}

		// Determine how many anchors are actually available from chain_start
		int n_available_anchors = tc.n_expected_anchors - chain_start;
		int actual_chain_len = std::min(chain_len, n_available_anchors);

		read.n = actual_chain_len;
		read.a = (mm128_t*)kmalloc(km, read.n * sizeof(mm128_t));
		for (int i = 0; i < actual_chain_len; i++) {
			const ExpectedAnchor* anchor = tc.getAnchor(chain_start + i);
			if (anchor) {
				read.a[i].x = anchor->x;
				read.a[i].y = ((uint64_t)q_span << 32) | anchor->y;
			}
		}

		// Set chain output (what mm_map_chain would produce)
		read.n_u = 1;
		read.u = (uint64_t*)kmalloc(km, sizeof(uint64_t));
		int32_t chain_score = chain ? chain->score : 0;
		read.u[0] = ((uint64_t)chain_score << 32) | actual_chain_len;

		// mini_pos is needed but can be empty for alignment
		read.mini_pos = (uint64_t*)kmalloc(km, sizeof(uint64_t));
		read.mini_pos[0] = 0;

		read.n_mini_pos = actual_chain_len;
		read.rep_len = 0;
		read.frag_gap = 5000;
	}

	// ========================================================================
	// Pipeline Helpers
	// ========================================================================

	/**
     * Run the full seed -> chain -> align pipeline.
     * Requires MM_F_CIGAR to be set in opt for CIGAR generation.
     */
	void runAlignPipeline(chain_read_t& read, mm_reg1_t** regs, int* n_regs, void* km)
	{
		mm_map_seed(mi, &opt, &read, &tbuf->timers.seed, km, NULL);
		mm_map_chain(mi, &opt, &read, &tbuf->timers.chain, km, NULL);
		mm_map_align(mi, &opt, &read, regs, n_regs, &tbuf->timers.align, km, NULL, NULL, 0);
	}

	// ========================================================================
	// Sequence Access Helpers
	// ========================================================================

	/**
     * Get the number of loaded query sequences.
     */
	int getNumQuerySeqs() const { return n_query_seqs; }

	/**
     * Get query sequence by index.
     */
	const mm_bseq1_t& getQuerySeq(int idx) const { return query_seqs[idx]; }

	/**
     * Get all MappingTestData entries that share the same query file as the
     * current test case, filtered to those with a precomputed alignment
     * baseline (n_expected_alignments > 0).
     *
     * The registry holds one MappingTestData per query in the JSON "queries"
     * array, each with its own query_index and n_expected_alignments. This
     * method returns all such entries so callers can assert per-read GPU
     * alignment counts against their individual precomputed CPU baselines,
     * rather than comparing against a single representative read's baseline.
     *
     * Empty for datasets whose JSON has only one query entry without
     * alignment baselines (e.g. the small/sr test data).
     */
	std::vector<const MappingTestData*> getAllQueryData() const
	{
		const MappingTestData& current = getTestData();
		std::vector<const MappingTestData*> result;
		for (auto* td : getAllTestData()) {
			if (td &&
			    td->query_path == current.query_path &&
			    td->n_expected_alignments > 0) {
				result.push_back(td);
			}
		}
		return result;
	}
};

// ============================================================================
// Parameterized test fixture - runs tests against all test data from JSON
// ============================================================================

/**
 * Parameterized test fixture for mapping tests.
 * Runs each test against all test data loaded from JSON files.
 */
class MapParamTestFixture : public MapTestFixture,
			    public ::testing::WithParamInterface<const MappingTestData*>
{
      protected:
	const MappingTestData& getTestData() const override
	{
		return *GetParam();
	}
};

// ============================================================================
// Behavioral test fixture — single representative dataset
// ============================================================================

/**
 * Fixture for behavioral/robustness tests that don't check exact expected
 * values from JSON. Uses a single representative dataset (the first one with
 * anchors, chains, and alignments) so each test runs once instead of 14×.
 *
 * Use this for: determinism, flag effects, edge cases, mode variants,
 * parameter sensitivity, error handling.
 */
class BehavioralTestFixture : public MapTestFixture
{
      protected:
	const MappingTestData& getTestData() const override
	{
		static const MappingTestData* rep = findRepresentativeDataset();
		return *rep;
	}

      private:
	static const MappingTestData* findRepresentativeDataset()
	{
		const auto& all = getAllTestData();
		// Prefer a dataset with anchors, chains, and alignments
		for (auto* td : all) {
			if (td->n_expected_anchors > 0 &&
			    td->n_expected_chains > 0 &&
			    td->n_expected_alignments > 0) {
				return td;
			}
		}
		return all[0];
	}
};

#endif // TEST_FIXTURES_H
