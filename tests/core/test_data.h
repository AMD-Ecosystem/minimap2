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
 * Test data aggregation for mapping pipeline tests.
 *
 * Test data is loaded from JSON files in test_suite/<category>/expected/ referencing FASTA files in test_suite/original/.
 * No C++ header editing is needed to add new test data.
 */
#ifndef TEST_DATA_H
#define TEST_DATA_H

#include "test_data_types.h"
#include "test_data_loader.h"
#include <vector>

// ============================================================================
// Runtime test data access
// ============================================================================

/**
 * Get all test data loaded from JSON files.
 * Returns pointers to MappingTestData structures.
 */
inline std::vector<const MappingTestData*> getAllTestData()
{
	TestDataRegistry& registry = TestDataRegistry::getInstance();
	registry.loadAll();
	return registry.getMappingTestDataPointers();
}

// ============================================================================
// Filtered test data access
//
// Parameterized tests that validate against expected outputs should only be
// instantiated for datasets that actually have those outputs, otherwise they
// become noisy "Skipped" entries in the test report. The helpers below filter
// the registry at parameterization time so skipped cases are never registered.
// ============================================================================

/**
 * Get test data entries that have expected anchors.
 */
inline std::vector<const MappingTestData*> getTestDataWithAnchors()
{
	std::vector<const MappingTestData*> result;
	for (auto* td : getAllTestData()) {
		if (td && td->hasAnchors()) result.push_back(td);
	}
	return result;
}

/**
 * Get test data entries that have both expected chains and anchors.
 * Chain-stage parameterized tests require both: anchors drive the pipeline and
 * chains are compared against the expected baseline.
 */
inline std::vector<const MappingTestData*> getTestDataWithChains()
{
	std::vector<const MappingTestData*> result;
	for (auto* td : getAllTestData()) {
		if (td && td->hasChains() && td->hasAnchors()) result.push_back(td);
	}
	return result;
}

/**
 * Get test data entries that have expected alignments.
 */
inline std::vector<const MappingTestData*> getTestDataWithAlignments()
{
	std::vector<const MappingTestData*> result;
	for (auto* td : getAllTestData()) {
		if (td && td->hasAlignments()) result.push_back(td);
	}
	return result;
}

/**
 * Get test data entries that have both expected alignments and anchors.
 * Used by tests that exercise a GPU/CPU pipeline starting from seeding and
 * compare the final alignment against the baseline.
 */
inline std::vector<const MappingTestData*> getTestDataWithAlignmentsAndAnchors()
{
	std::vector<const MappingTestData*> result;
	for (auto* td : getAllTestData()) {
		if (td && td->hasAlignments() && td->hasAnchors()) result.push_back(td);
	}
	return result;
}

#endif // TEST_DATA_H
