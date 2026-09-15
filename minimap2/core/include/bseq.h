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

#ifndef MM_BSEQ_H
#define MM_BSEQ_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

struct mm_bseq_file_s;
typedef struct mm_bseq_file_s mm_bseq_file_t;

typedef struct {
	int l_seq, rid;
	char *name, *seq, *qual, *comment;
} mm_bseq1_t;

mm_bseq_file_t* mm_bseq_open(const char* fn);
void mm_bseq_close(mm_bseq_file_t* fp);
mm_bseq1_t* mm_bseq_read3(mm_bseq_file_t* fp, int64_t chunk_size, int with_qual, int with_comment, int frag_mode, int* n_);
mm_bseq1_t* mm_bseq_read2(mm_bseq_file_t* fp, int64_t chunk_size, int with_qual, int frag_mode, int* n_);
mm_bseq1_t* mm_bseq_read(mm_bseq_file_t* fp, int64_t chunk_size, int with_qual, int* n_);
mm_bseq1_t* mm_bseq_read_frag2(int n_fp, mm_bseq_file_t** fp, int64_t chunk_size, int with_qual, int with_comment, int* n_);
mm_bseq1_t* mm_bseq_read_frag(int n_fp, mm_bseq_file_t** fp, int64_t chunk_size, int with_qual, int* n_);
int mm_bseq_eof(mm_bseq_file_t* fp);

extern unsigned char seq_nt4_table[256];
extern unsigned char seq_comp_table[256];

static inline int mm_qname_len(const char* s)
{
	int l;
	l = strlen(s);
	return l >= 3 && s[l - 1] >= '0' && s[l - 1] <= '9' && s[l - 2] == '/' ? l - 2 : l;
}

static inline int mm_qname_same(const char* s1, const char* s2)
{
	int l1, l2;
	l1 = mm_qname_len(s1);
	l2 = mm_qname_len(s2);
	return (l1 == l2 && strncmp(s1, s2, l1) == 0);
}

/**
 * Compute the reverse complement of a biological sequence in-place
 * @s: pointer to sequence structure to reverse complement
 *
 * Performs two operations:
 * 1. Reverses the nucleotide sequence order
 * 2. Complements each base (A↔T, C↔G) using seq_comp_table
 * Also reverses quality scores if present to maintain alignment with sequence.
 * This is used to convert between forward and reverse DNA strands.
 */
static inline void mm_revcomp_bseq(mm_bseq1_t* s)
{
	int i, t, l = s->l_seq;
	for (i = 0; i < l >> 1; ++i) {
		t = s->seq[l - i - 1];
		s->seq[l - i - 1] = seq_comp_table[(uint8_t)s->seq[i]];
		s->seq[i] = seq_comp_table[t];
	}
	if (l & 1) s->seq[l >> 1] = seq_comp_table[(uint8_t)s->seq[l >> 1]];
	if (s->qual)
		for (i = 0; i < l >> 1; ++i)
			t = s->qual[l - i - 1], s->qual[l - i - 1] = s->qual[i], s->qual[i] = t;
}

#ifdef __cplusplus
}
#endif

#endif
