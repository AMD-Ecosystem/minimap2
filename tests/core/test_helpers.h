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
 * Test helper utilities for mm_map_* tests.
 * 
 * This header provides:
 * - RAII helpers for memory management (ScopedMemPool, ScopedAlignments)
 * - Utility functions for chain/anchor data extraction
 * - CIGAR validation helpers (validateCigar, CigarStats)
 * - Alignment coordinate validation (validateAlignmentCoordinates)
 * - Chain colinearity validation (validateChainColinearity)
 * - Anchor bounds validation (validateAnchorBounds)
 * - File I/O helpers
 * - String utilities (reverseComplement)
 * 
 * These are standalone utilities that don't depend on test fixtures.
 */
#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <gtest/gtest.h>
#include "map_priv.h"
#include "bseq.h"
#include "kalloc.h"
#include "test_data_types.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>

// ============================================================================
// RAII Helpers for Memory Management
// ============================================================================

/**
 * RAII wrapper for kalloc memory pool.
 * Automatically destroys the pool when going out of scope.
 */
class ScopedMemPool
{
      public:
	ScopedMemPool() : km_(km_init()) {}
	~ScopedMemPool()
	{
		if (km_) km_destroy(km_);
	}

	// Non-copyable
	ScopedMemPool(const ScopedMemPool&) = delete;
	ScopedMemPool& operator=(const ScopedMemPool&) = delete;

	// Move semantics
	ScopedMemPool(ScopedMemPool&& other) noexcept : km_(other.km_) { other.km_ = nullptr; }
	ScopedMemPool& operator=(ScopedMemPool&& other) noexcept
	{
		if (this != &other) {
			if (km_) km_destroy(km_);
			km_ = other.km_;
			other.km_ = nullptr;
		}
		return *this;
	}

	void* get() const { return km_; }
	operator void*() const { return km_; }

	// Helper to allocate within this pool
	template <typename T>
	T* alloc(size_t count = 1) { return (T*)kmalloc(km_, count * sizeof(T)); }

	template <typename T>
	void free(T* ptr) { kfree(km_, ptr); }

      private:
	void* km_;
};

/**
 * RAII wrapper for alignment results.
 * Automatically frees results when going out of scope.
 */
class ScopedAlignments
{
      public:
	ScopedAlignments() : regs_(nullptr), n_regs_(0) {}
	~ScopedAlignments() { free_results(); }

	// Non-copyable
	ScopedAlignments(const ScopedAlignments&) = delete;
	ScopedAlignments& operator=(const ScopedAlignments&) = delete;

	mm_reg1_t*& regs() { return regs_; }
	int& count() { return n_regs_; }

	mm_reg1_t* operator->() { return regs_; }
	mm_reg1_t& operator[](int i) { return regs_[i]; }
	const mm_reg1_t& operator[](int i) const { return regs_[i]; }

	bool empty() const { return n_regs_ == 0 || regs_ == nullptr; }
	int size() const { return n_regs_; }

      private:
	void free_results()
	{
		if (regs_) {
			for (int i = 0; i < n_regs_; i++) {
				if (regs_[i].p) free(regs_[i].p);
			}
			free(regs_);
			regs_ = nullptr;
			n_regs_ = 0;
		}
	}

	mm_reg1_t* regs_;
	int n_regs_;
};

// ============================================================================
// Chain/Anchor Data Extraction Utilities
// ============================================================================

// Extract chain score from u array element (upper 32 bits)
inline int32_t getChainScore(uint64_t u)
{
	return (int32_t)(u >> 32);
}

// Extract chain length from u array element (lower 32 bits)
inline int32_t getChainLen(uint64_t u)
{
	return (int32_t)(u & 0xFFFFFFFF);
}

// Extract query position from anchor y field (lower 32 bits)
inline uint32_t getAnchorQPos(uint64_t y)
{
	return (uint32_t)(y & 0xFFFFFFFF);
}

// Extract q_span from anchor y field (upper 32 bits)
inline uint32_t getAnchorQSpan(uint64_t y)
{
	return (uint32_t)(y >> 32);
}

// ============================================================================
// CIGAR Validation Utilities
// ============================================================================

// CIGAR statistics structure
struct CigarStats {
	uint32_t qlen = 0; // Query length from CIGAR
	uint32_t rlen = 0; // Reference length from CIGAR
	uint32_t n_match = 0; // Match operations
	uint32_t n_ins = 0; // Insertions
	uint32_t n_del = 0; // Deletions
	bool valid = true; // Whether all operations are valid
};

// Calculate CIGAR query and reference consumption
// Query-consuming ops: M(0), I(1), =(7), X(8)
// Ref-consuming ops: M(0), D(2), N(3), =(7), X(8)
inline std::pair<int, int> calculateCigarConsumption(const mm_reg1_t* r)
{
	if (!r || !r->p) return {0, 0};
	int query_consumed = 0, ref_consumed = 0;
	for (uint32_t k = 0; k < r->p->n_cigar; k++) {
		int op = r->p->cigar[k] & 0xf;
		int len = r->p->cigar[k] >> 4;
		if (op == 0 || op == 1 || op == 7 || op == 8) query_consumed += len; // M, I, =, X
		if (op == 0 || op == 2 || op == 3 || op == 7 || op == 8) ref_consumed += len; // M, D, N, =, X
	}
	return {query_consumed, ref_consumed};
}

// Validate CIGAR operations and return statistics
inline CigarStats validateCigar(const mm_reg1_t* r)
{
	CigarStats stats;
	if (!r || !r->p || r->p->n_cigar == 0) {
		stats.valid = false;
		return stats;
	}

	for (uint32_t i = 0; i < r->p->n_cigar; i++) {
		uint32_t op = r->p->cigar[i] & 0xf;
		uint32_t len = r->p->cigar[i] >> 4;

		if (len == 0 || op > 8) {
			stats.valid = false;
			continue;
		}

		switch (op) {
		case 0:
		case 7: // M, =
			stats.qlen += len;
			stats.rlen += len;
			stats.n_match += len;
			break;
		case 1: // I
			stats.qlen += len;
			stats.n_ins += len;
			break;
		case 2: // D
			stats.rlen += len;
			stats.n_del += len;
			break;
		case 3: // N (intron)
			stats.rlen += len;
			break;
		case 4: // S (soft clip)
			stats.qlen += len;
			break;
		case 8: // X (mismatch)
			stats.qlen += len;
			stats.rlen += len;
			break;
		}
	}
	return stats;
}

// Validate mapping quality range (uses gtest EXPECT macros)
inline void validateMappingQuality(const mm_reg1_t* regs, int n_regs, const char* context = "")
{
	for (int i = 0; i < n_regs; i++) {
		EXPECT_GE(regs[i].mapq, 0) << "MAPQ should be >= 0 for alignment " << i << context;
		EXPECT_LE(regs[i].mapq, 60) << "MAPQ should be <= 60 for alignment " << i << context;
	}
}

// ============================================================================
// Memory Cleanup Utilities
// ============================================================================

// Free alignment results (including CIGAR)
inline void freeAlignmentResults(mm_reg1_t* regs, int n_regs)
{
	if (!regs) return;
	for (int i = 0; i < n_regs; i++) {
		if (regs[i].p) free(regs[i].p);
	}
	free(regs);
}

// Free mm_bseq1_t array
inline void freeSeqs(mm_bseq1_t* seqs, int n_seq)
{
	if (!seqs) return;
	for (int i = 0; i < n_seq; i++) {
		free(seqs[i].name);
		free(seqs[i].seq);
		if (seqs[i].qual) free(seqs[i].qual);
		if (seqs[i].comment) free(seqs[i].comment);
	}
	free(seqs);
}

// ============================================================================
// File I/O Utilities
// ============================================================================

// Write sequence to FASTA file with 70bp line wrapping
inline bool writeFasta(const char* path, const char* name, const char* seq)
{
	std::ofstream ofs(path);
	if (!ofs) return false;
	ofs << '>' << name << '\n';
	size_t n = strlen(seq);
	for (size_t i = 0; i < n; i += 70) {
		ofs.write(seq + i, std::min<size_t>(70, n - i));
		ofs << '\n';
	}
	return ofs.good();
}

// ============================================================================
// String Utilities
// ============================================================================

// Reverse complement a DNA sequence
inline std::string reverseComplement(const char* seq)
{
	std::string result;
	size_t len = strlen(seq);
	result.reserve(len);
	for (size_t i = len; i > 0; --i) {
		char c = seq[i - 1];
		switch (c) {
		case 'A':
		case 'a':
			result += 'T';
			break;
		case 'T':
		case 't':
			result += 'A';
			break;
		case 'G':
		case 'g':
			result += 'C';
			break;
		case 'C':
		case 'c':
			result += 'G';
			break;
		default:
			result += 'N';
			break;
		}
	}
	return result;
}

// ============================================================================
// Alignment Validation Utilities
// ============================================================================

// Validate alignment coordinates are consistent and within bounds
inline void validateAlignmentCoordinates(const mm_reg1_t* r, int qlen, const mm_idx_t* mi,
    const char* context = "")
{
	EXPECT_GE(r->qs, 0) << context << " Query start should be >= 0";
	EXPECT_GT(r->qe, r->qs) << context << " Query end should be > start";
	EXPECT_LE(r->qe, qlen) << context << " Query end should be <= query length";

	EXPECT_GE(r->rs, 0) << context << " Reference start should be >= 0";
	EXPECT_GT(r->re, r->rs) << context << " Reference end should be > start";

	EXPECT_GE(r->rid, 0) << context << " Reference ID should be >= 0";
	if (mi && r->rid >= 0 && r->rid < mi->n_seq) {
		EXPECT_LE(r->re, (int32_t)mi->seq[r->rid].len)
		    << context << " Reference end should be <= reference length";
	}
}

// Validate chain colinearity (anchors should be sorted by query position)
inline bool validateChainColinearity(const mm128_t* anchors, int chain_len)
{
	for (int i = 1; i < chain_len; i++) {
		uint32_t prev_qpos = getAnchorQPos(anchors[i - 1].y);
		uint32_t curr_qpos = getAnchorQPos(anchors[i].y);
		if (prev_qpos > curr_qpos) {
			return false;
		}
	}
	return true;
}

/**
 * Validate anchor array bounds.
 * 
 * Checks that:
 * - q_pos is within query length (q_pos < qlen)
 * - q_span is positive (> 0)
 * 
 * Note: We do NOT check q_pos + q_span <= qlen because minimizers near the
 * end of a query can legitimately have their k-mer span extend past the query
 * boundary. For example, a minimizer at position qlen-k+1 would have
 * q_pos + q_span = qlen + 1, which is valid behavior.
 */
inline void validateAnchorBounds(const mm128_t* anchors, int64_t n_anchors, int qlen,
    const char* context = "")
{
	for (int64_t i = 0; i < n_anchors; i++) {
		uint32_t q_pos = getAnchorQPos(anchors[i].y);
		uint32_t q_span = getAnchorQSpan(anchors[i].y);

		EXPECT_LT(q_pos, (uint32_t)qlen)
		    << context << " Anchor " << i << " query position within bounds";
		EXPECT_GT(q_span, 0u)
		    << context << " Anchor " << i << " q_span should be positive";
		// Note: q_pos + q_span can exceed qlen for minimizers near query end
	}
}

// ============================================================================
// Alignment Result Comparison Utilities
// ============================================================================

/**
 * Structure for comparing two alignment results.
 * Used for determinism and regression testing.
 */
struct AlignmentDifference {
	bool score_diff = false;
	bool coord_diff = false; // qs, qe, rs, re
	bool strand_diff = false;
	bool mapq_diff = false;
	bool cigar_len_diff = false;
	bool mlen_diff = false;
	bool blen_diff = false;
	int score_delta = 0;
	int mapq_delta = 0;
	std::string details;
};

/**
 * Compare two alignment results in detail.
 * Returns a difference structure describing all variations.
 */
inline AlignmentDifference compareAlignments(const mm_reg1_t* r1, const mm_reg1_t* r2)
{
	AlignmentDifference diff;
	if (!r1 || !r2) return diff;

	if (r1->score != r2->score) {
		diff.score_diff = true;
		diff.score_delta = r1->score - r2->score;
	}

	if (r1->qs != r2->qs || r1->qe != r2->qe ||
	    r1->rs != r2->rs || r1->re != r2->re) {
		diff.coord_diff = true;
	}

	if (r1->rev != r2->rev) {
		diff.strand_diff = true;
	}

	if (r1->mapq != r2->mapq) {
		diff.mapq_diff = true;
		diff.mapq_delta = r1->mapq - r2->mapq;
	}

	if ((r1->p && r2->p) && (r1->p->n_cigar != r2->p->n_cigar)) {
		diff.cigar_len_diff = true;
	}

	if (r1->mlen != r2->mlen) {
		diff.mlen_diff = true;
	}

	if (r1->blen != r2->blen) {
		diff.blen_diff = true;
	}

	return diff;
}

/**
 * Validate alignment score is reasonable given its CIGAR stats.
 * Score should correlate with sequence identity and alignment length.
 */
inline void validateAlignmentScoring(const mm_reg1_t* r, const char* context = "")
{
	if (!r) return;

	// Score should generally increase with alignment length
	int align_span = std::max(r->qe - r->qs, r->re - r->rs);
	EXPECT_GT(r->score, 0) << context << " Score should be positive";
	EXPECT_LT(r->score, align_span * 2)
	    << context << " Score should be reasonable for span " << align_span;

	// For good alignments, score should be significant relative to length
	if (r->mlen > 0) {
		double score_per_match = (double)r->score / r->mlen;
		EXPECT_GT(score_per_match, 0.5)
		    << context << " Score per matched base should be reasonable";
		EXPECT_LT(score_per_match, 10.0)
		    << context << " Score per matched base shouldn't be too high";
	}
}

/**
 * Verify that an alignment's anchors are properly covered by its coordinates.
 * Used for pipeline consistency: chain → align mapping.
 */
inline bool validateChainCoverageInAlignment(const mm128_t* anchors, int n_anchors,
    const mm_reg1_t* align)
{
	if (!anchors || n_anchors == 0 || !align) return true;

	// Anchors should fit within alignment bounds
	for (int i = 0; i < n_anchors; i++) {
		uint32_t q_pos = getAnchorQPos(anchors[i].y);
		uint32_t ref_pos = (uint32_t)anchors[i].x;

		// Query position should be within or near alignment bounds
		// (slightly outside is OK due to clipping)
		if (q_pos < align->qs - 100 || q_pos > align->qe + 100) {
			return false;
		}
	}
	return true;
}

/**
 * Validate MAPQ distribution across multiple alignments.
 * Primary should typically have higher MAPQ than secondaries.
 */
inline void validateMAPQOrdering(const mm_reg1_t* regs, int n_regs,
    const char* context = "")
{
	if (!regs || n_regs <= 1) return;

	for (int i = 1; i < n_regs; i++) {
		// Primary (i=0) typically has MAPQ >= secondary alignments
		// (unless primary has low confidence)
		if (regs[0].mapq > 0) {
			EXPECT_GE(regs[0].mapq, regs[i].mapq / 2)
			    << context << " Primary MAPQ should be competitive with secondary " << i;
		}
	}
}

// ============================================================================
// Alignment Verification Helpers
// ============================================================================

/**
 * Verify an array of mm_reg1_t alignments against expected values.
 * Reports up to 10 mismatches per field and emits a summary on failure.
 *
 * @param regs      Pointer to actual alignment array
 * @param n_actual  Number of actual alignments
 * @param tc        Test data containing expected alignments
 * @param label     Context label for failure messages (e.g. test name)
 */
inline void verifyAlignments(const mm_reg1_t* regs, int n_actual,
    const MappingTestData& tc,
    const char* label = "")
{
	EXPECT_EQ(n_actual, tc.n_expected_alignments)
	    << label << " alignment count mismatch: got " << n_actual
	    << ", expected " << tc.n_expected_alignments;

	int n_compare = std::min(n_actual, tc.n_expected_alignments);
	int mm_score = 0, mm_qs = 0, mm_qe = 0, mm_rs = 0, mm_re = 0;
	int mm_mapq = 0, mm_rev = 0, mm_dp_score = 0, mm_blen = 0, mm_mlen = 0;
	int mm_dp_max = 0, mm_n_ambi = 0, mm_n_cigar = 0;
	constexpr int kMaxReport = 10;

	for (int i = 0; i < n_compare; i++) {
		const ExpectedAlignment* ea = tc.getAlignment(i);
		ASSERT_NE(ea, nullptr) << "Expected alignment " << i << " should exist";
		const mm_reg1_t& r = regs[i];

#define MM2_CHECK_FIELD(actual, expected, counter, name) \
	if ((actual) != (expected)) { \
		(counter)++; \
		if ((counter) <= kMaxReport) \
			EXPECT_EQ(actual, expected) << "Alignment " << i << " " << (name); \
	}

		MM2_CHECK_FIELD(r.score, ea->score, mm_score, "score");
		MM2_CHECK_FIELD(r.qs, ea->qs, mm_qs, "qs");
		MM2_CHECK_FIELD(r.qe, ea->qe, mm_qe, "qe");
		MM2_CHECK_FIELD(r.rs, ea->rs, mm_rs, "rs");
		MM2_CHECK_FIELD(r.re, ea->re, mm_re, "re");
		MM2_CHECK_FIELD(r.mapq, ea->mapq, mm_mapq, "mapq");
		MM2_CHECK_FIELD(r.rev, ea->rev, mm_rev, "rev");
		MM2_CHECK_FIELD((int32_t)r.blen, ea->blen, mm_blen, "blen");
		MM2_CHECK_FIELD((int32_t)r.mlen, ea->mlen, mm_mlen, "mlen");

		if (r.p != nullptr) {
			MM2_CHECK_FIELD(r.p->dp_score, ea->dp_score, mm_dp_score, "dp_score");
			MM2_CHECK_FIELD(r.p->dp_max, ea->dp_max, mm_dp_max, "dp_max");
			MM2_CHECK_FIELD((int32_t)r.p->n_ambi, ea->n_ambi, mm_n_ambi, "n_ambi");
			MM2_CHECK_FIELD((int32_t)r.p->n_cigar, ea->n_cigar_ops, mm_n_cigar, "n_cigar_ops");
		}
#undef MM2_CHECK_FIELD
	}

	if (mm_score || mm_qs || mm_qe || mm_rs || mm_re || mm_mapq || mm_rev ||
	    mm_dp_score || mm_blen || mm_mlen || mm_dp_max || mm_n_ambi || mm_n_cigar) {
		ADD_FAILURE()
		    << label << " verification summary (" << tc.name << "): "
		    << n_compare << " alignments compared, mismatches: "
		    << "score=" << mm_score
		    << " qs=" << mm_qs << " qe=" << mm_qe
		    << " rs=" << mm_rs << " re=" << mm_re
		    << " mapq=" << mm_mapq << " rev=" << mm_rev
		    << " dp_score=" << mm_dp_score
		    << " blen=" << mm_blen << " mlen=" << mm_mlen
		    << " dp_max=" << mm_dp_max << " n_ambi=" << mm_n_ambi
		    << " n_cigar=" << mm_n_cigar;
	}
}

// ============================================================================
// Chain Verification Helpers
// ============================================================================

/**
 * Verify chain_read_t chain results against expected values.
 * Reports up to 10 mismatches per field and emits a summary on failure.
 *
 * Checks: chain count, per-chain score/length, and anchor positions
 * (first_qpos, first_tpos, last_qpos, last_tpos) for each chain.
 *
 * @param read     chain_read_t with chaining results (u[] and a[] arrays)
 * @param tc       Test data containing expected chains
 * @param label    Context label for failure messages
 */
inline void verifyChains(const chain_read_t& read, const MappingTestData& tc,
    const char* label = "")
{
	EXPECT_EQ(read.n_u, tc.n_expected_chains)
	    << label << " chain count mismatch: got " << read.n_u
	    << ", expected " << tc.n_expected_chains;

	int n_compare = std::min(read.n_u, tc.n_expected_chains);
	int mm_score = 0, mm_length = 0;
	int mm_first_qpos = 0, mm_first_tpos = 0, mm_last_qpos = 0, mm_last_tpos = 0;
	constexpr int kMaxReport = 10;

	int64_t anchor_offset = 0;
	for (int i = 0; i < n_compare; i++) {
		const ExpectedChain* ec = tc.getChain(i);
		ASSERT_NE(ec, nullptr) << "Expected chain " << i << " should exist";

		int32_t score = getChainScore(read.u[i]);
		int32_t length = getChainLen(read.u[i]);

#define MM2_CHECK_CHAIN(actual, expected, counter, name) \
	if ((actual) != (expected)) { \
		(counter)++; \
		if ((counter) <= kMaxReport) \
			EXPECT_EQ(actual, expected) << label << " Chain " << i << " " << (name); \
	}

		MM2_CHECK_CHAIN(score, ec->score, mm_score, "score");
		MM2_CHECK_CHAIN(length, ec->length, mm_length, "length");

		// Check anchor positions for this chain
		if (read.a && length > 0 && anchor_offset + length <= read.n) {
			uint32_t first_qpos = getAnchorQPos(read.a[anchor_offset].y);
			uint32_t first_tpos = (uint32_t)(read.a[anchor_offset].x);
			uint32_t last_qpos = getAnchorQPos(read.a[anchor_offset + length - 1].y);
			uint32_t last_tpos = (uint32_t)(read.a[anchor_offset + length - 1].x);

			MM2_CHECK_CHAIN(first_qpos, ec->first_qpos, mm_first_qpos, "first_qpos");
			MM2_CHECK_CHAIN(first_tpos, ec->first_tpos, mm_first_tpos, "first_tpos");
			MM2_CHECK_CHAIN(last_qpos, ec->last_qpos, mm_last_qpos, "last_qpos");
			MM2_CHECK_CHAIN(last_tpos, ec->last_tpos, mm_last_tpos, "last_tpos");
		}

#undef MM2_CHECK_CHAIN

		anchor_offset += length;
	}

	if (mm_score || mm_length || mm_first_qpos || mm_first_tpos ||
	    mm_last_qpos || mm_last_tpos) {
		ADD_FAILURE()
		    << label << " chain verification summary (" << tc.name << "): "
		    << n_compare << " chains compared, mismatches: "
		    << "score=" << mm_score << " length=" << mm_length
		    << " first_qpos=" << mm_first_qpos << " first_tpos=" << mm_first_tpos
		    << " last_qpos=" << mm_last_qpos << " last_tpos=" << mm_last_tpos;
	}
}

#endif // TEST_HELPERS_H
