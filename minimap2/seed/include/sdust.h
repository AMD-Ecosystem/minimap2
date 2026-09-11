/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2018-2022 Heng Li; Dana-Farber Cancer Institute; Broad Institute
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

#ifndef SDUST_H
#define SDUST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct sdust_buf_s;
typedef struct sdust_buf_s sdust_buf_t;

/**
 * Wrapper for sdust_core that allocates and destroys its own buffer
 * Identifies low-complexity regions in a DNA sequence using the SDUST algorithm
 * @param km     Thread-local memory pool; using NULL falls back to malloc()
 * @param seq    DNA sequence as byte array
 * @param l_seq  Length of sequence; if negative, strlen() is used
 * @param T      Entropy threshold (default 20; higher = more aggressive masking)
 * @param W      Window size (default 64)
 * @param n      Output: number of masked regions found
 * @return       Heap-allocated array of n masked regions encoded as (start<<32)|finish; caller must free
 */
uint64_t* sdust(void* km, const uint8_t* seq, int l_seq, int T, int W, int* n);

// the following interface dramatically reduce heap allocations when sdust is frequently called.

/**
 * Initialize an SDUST buffer with empty structures for tracking masked regions
 * @param km  Thread-local memory pool; using NULL falls back to malloc()
 * @return    Pointer to initialized sdust_buf_t structure
 */
sdust_buf_t* sdust_buf_init(void* km);

/**
 * Free all memory associated with an SDUST buffer
 * @param buf  Pointer to sdust_buf_t structure to destroy
 */
void sdust_buf_destroy(sdust_buf_t* buf);

/**
 * Core SDUST algorithm to identify low-complexity regions in a DNA sequence
 * Based on Morgulis et al. (2006) symmetric DUST algorithm
 * @param seq    DNA sequence as byte array (uses seq_nt4_table for conversion)
 * @param l_seq  Length of sequence; if negative, strlen() is used
 * @param T      Entropy threshold (default 20; higher = more aggressive masking)
 * @param W      Window size (default 64)
 * @param n      Output: number of masked regions found
 * @param buf    Reusable buffer for internal state (avoids repeated allocations)
 * @return       Array of n masked regions encoded as (start<<32)|finish
 */
const uint64_t* sdust_core(const uint8_t* seq, int l_seq, int T, int W, int* n, sdust_buf_t* buf);

#ifdef __cplusplus
}
#endif

#endif
