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

// Unit tests for kalloc memory allocator
#include <gtest/gtest.h>

extern "C" {
#include "kalloc.h"
#include <string.h>
}

// Test kalloc basic functionality
class KallocTest : public ::testing::Test
{
      protected:
	void* km;

	void SetUp() override
	{
		km = km_init();
	}

	void TearDown() override
	{
		km_destroy(km);
	}
};

TEST_F(KallocTest, InitAndDestroy)
{
	EXPECT_NE(km, nullptr);
}

TEST_F(KallocTest, SimpleAllocation)
{
	void* ptr = kmalloc(km, 100);
	ASSERT_NE(ptr, nullptr);

	// Write to the allocated memory to ensure it's valid
	memset(ptr, 0xAB, 100);

	unsigned char* bytes = (unsigned char*)ptr;
	for (int i = 0; i < 100; ++i) {
		EXPECT_EQ(bytes[i], 0xAB);
	}
}

TEST_F(KallocTest, MultipleAllocations)
{
	void* ptr1 = kmalloc(km, 50);
	void* ptr2 = kmalloc(km, 100);
	void* ptr3 = kmalloc(km, 150);

	ASSERT_NE(ptr1, nullptr);
	ASSERT_NE(ptr2, nullptr);
	ASSERT_NE(ptr3, nullptr);

	// Pointers should be different
	EXPECT_NE(ptr1, ptr2);
	EXPECT_NE(ptr2, ptr3);
	EXPECT_NE(ptr1, ptr3);
}

TEST_F(KallocTest, CallocZeroInitialization)
{
	size_t count = 50;
	void* ptr = kcalloc(km, count, sizeof(int));
	ASSERT_NE(ptr, nullptr);

	// Check that memory is zero-initialized
	int* ints = (int*)ptr;
	for (size_t i = 0; i < count; ++i) {
		EXPECT_EQ(ints[i], 0);
	}
}

TEST_F(KallocTest, LargeAllocation)
{
	// Allocate 1MB
	size_t size = 1024 * 1024;
	void* ptr = kmalloc(km, size);
	ASSERT_NE(ptr, nullptr);

	// Write pattern to verify memory is usable
	unsigned char* bytes = (unsigned char*)ptr;
	for (size_t i = 0; i < 1000; i += 100) {
		bytes[i] = (unsigned char)(i & 0xFF);
	}

	// Verify pattern
	for (size_t i = 0; i < 1000; i += 100) {
		EXPECT_EQ(bytes[i], (unsigned char)(i & 0xFF));
	}
}

TEST_F(KallocTest, AlignedAllocation)
{
	// Allocate with specific alignment
	void* ptr = kmalloc(km, 256);
	ASSERT_NE(ptr, nullptr);

	// Check alignment (should be at least pointer-sized)
	uintptr_t addr = (uintptr_t)ptr;
	EXPECT_EQ(addr % sizeof(void*), 0);
}

// Test without memory pool (using NULL)
TEST(KallocNullTest, DirectMalloc)
{
	void* ptr = kmalloc(NULL, 100);
	ASSERT_NE(ptr, nullptr);

	memset(ptr, 0xCD, 100);
	kfree(NULL, ptr);
}

TEST(KallocNullTest, DirectCalloc)
{
	void* ptr = kcalloc(NULL, 10, sizeof(int));
	ASSERT_NE(ptr, nullptr);

	int* ints = (int*)ptr;
	for (int i = 0; i < 10; ++i) {
		EXPECT_EQ(ints[i], 0);
	}

	kfree(NULL, ptr);
}

TEST_F(KallocTest, Realloc)
{
	// Initial allocation
	void* ptr = kmalloc(km, 50);
	ASSERT_NE(ptr, nullptr);

	// Write pattern
	unsigned char* bytes = (unsigned char*)ptr;
	for (int i = 0; i < 50; ++i) {
		bytes[i] = (unsigned char)(i & 0xFF);
	}

	// Realloc to larger size
	ptr = krealloc(km, ptr, 100);
	ASSERT_NE(ptr, nullptr);

	// Original data should be preserved
	bytes = (unsigned char*)ptr;
	for (int i = 0; i < 50; ++i) {
		EXPECT_EQ(bytes[i], (unsigned char)(i & 0xFF)) << "Data should be preserved at index " << i;
	}
}

TEST_F(KallocTest, ReallocShrink)
{
	// Initial allocation
	void* ptr = kmalloc(km, 100);
	ASSERT_NE(ptr, nullptr);

	// Write pattern
	unsigned char* bytes = (unsigned char*)ptr;
	for (int i = 0; i < 100; ++i) {
		bytes[i] = (unsigned char)(i & 0xFF);
	}

	// Realloc to smaller size
	ptr = krealloc(km, ptr, 50);
	ASSERT_NE(ptr, nullptr);

	// Data within new size should be preserved
	bytes = (unsigned char*)ptr;
	for (int i = 0; i < 50; ++i) {
		EXPECT_EQ(bytes[i], (unsigned char)(i & 0xFF)) << "Data should be preserved at index " << i;
	}
}

TEST_F(KallocTest, ReallocNull)
{
	// Realloc with NULL should behave like malloc
	void* ptr = krealloc(km, nullptr, 100);
	ASSERT_NE(ptr, nullptr);

	// Should be usable
	memset(ptr, 0xAB, 100);
}
