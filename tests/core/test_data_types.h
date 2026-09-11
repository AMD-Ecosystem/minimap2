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
 * Test data type definitions for mapping pipeline tests.
 * 
 * This file contains only the structure definitions for test data.
 * Actual test data is defined in separate test_data_*.h files.
 */
#ifndef TEST_DATA_TYPES_H
#define TEST_DATA_TYPES_H

#include <cstdint>
#include <cstddef>

// ============================================================================
// Test Data Structures
// ============================================================================

/**
 * Expected anchor values (output of mm_map_seed)
 */
struct ExpectedAnchor {
	uint64_t x; // Encoded reference position and strand
	uint32_t y; // Query position
};

/**
 * Expected values for a single chain (output of mm_map_chain)
 */
struct ExpectedChain {
	int32_t score; // Chain score
	int32_t length; // Number of anchors in chain
	uint32_t first_qpos; // First anchor query position
	uint32_t first_tpos; // First anchor target position
	uint32_t last_qpos; // Last anchor query position
	uint32_t last_tpos; // Last anchor target position
};

/**
 * Expected values for a single alignment (output of mm_map_align)
 */
struct ExpectedAlignment {
	int32_t score; // Chain/alignment score
	int32_t dp_score; // DP alignment score
	int32_t qs; // Query start
	int32_t qe; // Query end
	int32_t rs; // Reference start
	int32_t re; // Reference end
	int32_t mapq; // Mapping quality
	int32_t rev; // Strand (0 = forward, 1 = reverse)
	int32_t n_cigar_ops; // Exact number of CIGAR operations
	int32_t blen; // Block length (aligned region length)
	int32_t mlen; // Number of matching bases
	int32_t dp_max; // Maximum DP score during alignment
	int32_t n_ambi; // Number of ambiguous bases
};

/**
 * Complete test data for the mapping pipeline.
 * Contains both input data and expected outputs for each stage.
 * 
 * For large sequences (GB-scale), only file paths are stored.
 * For small test sequences, the sequence data may be cached in memory.
 */
class MappingTestData
{
      public:
	// Test data identification
	const char* name; // Human-readable test data name
	const char* description; // Description of what this test data covers

	// Input: Reference sequence
	const char* ref_name; // Reference sequence name
	const char* ref_path; // Path to reference FASTA file (always available)

	// Input: Query sequence
	const char* query_name; // Query sequence name
	const char* query_path; // Path to query FASTA file (always available)

	// Expected: Seeding output (mm_map_seed)
	const ExpectedAnchor* expected_anchors; // Array of expected anchors
	size_t n_expected_anchors; // Number of expected anchors

	// Expected: Chaining output (mm_map_chain)
	int n_expected_chains; // Number of expected chains
	const ExpectedChain* expected_chains; // Array of expected chain values

	// Expected: Alignment output (mm_map_align)
	int n_expected_alignments; // Number of expected alignments
	const ExpectedAlignment* expected_alignments; // Array of expected alignments

	// Index parameters (0 = use defaults)
	int k; // K-mer size (0 = default 15)
	int w; // Minimizer window (0 = default 10)

	// Preset name (nullptr = no preset, use defaults)
	const char* preset; // e.g. "map-ont", "map-pb", "sr"

	// Query index within multi-query FASTA files (0-based)
	int query_index; // Which query sequence to use from the FASTA file

	// ========================================================================
	// Helper methods
	// ========================================================================

	/**
     * Get the primary (first) expected chain.
     * Returns nullptr if no chains are expected.
     */
	const ExpectedChain* getPrimaryChain() const
	{
		return (n_expected_chains > 0) ? &expected_chains[0] : nullptr;
	}

	/**
     * Get the primary (first) expected alignment.
     * Returns nullptr if no alignments are expected.
     */
	const ExpectedAlignment* getPrimaryAlignment() const
	{
		return (n_expected_alignments > 0) ? &expected_alignments[0] : nullptr;
	}

	/**
     * Get k-mer size (returns default 15 if not specified).
     */
	int getKmerSize() const
	{
		return (k > 0) ? k : 15;
	}

	/**
     * Get minimizer window (returns default 10 if not specified).
     */
	int getMinimizerWindow() const
	{
		return (w > 0) ? w : 10;
	}

	/**
     * Get expected chain by index.
     * Returns nullptr if index is out of bounds.
     */
	const ExpectedChain* getChain(int index) const
	{
		return (index >= 0 && index < n_expected_chains) ? &expected_chains[index] : nullptr;
	}

	/**
     * Get expected alignment by index.
     * Returns nullptr if index is out of bounds.
     */
	const ExpectedAlignment* getAlignment(int index) const
	{
		return (index >= 0 && index < n_expected_alignments) ? &expected_alignments[index] : nullptr;
	}

	/**
     * Get expected anchor by index.
     * Returns nullptr if index is out of bounds.
     */
	const ExpectedAnchor* getAnchor(size_t index) const
	{
		return (index < n_expected_anchors) ? &expected_anchors[index] : nullptr;
	}

	/**
     * Check if this test data has expected anchors.
     */
	bool hasAnchors() const { return n_expected_anchors > 0; }

	/**
     * Check if this test data has expected chains.
     */
	bool hasChains() const { return n_expected_chains > 0; }

	/**
     * Check if this test data has expected alignments.
     */
	bool hasAlignments() const { return n_expected_alignments > 0; }
};

#endif // TEST_DATA_TYPES_H
