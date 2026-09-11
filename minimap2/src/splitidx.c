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

#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include "minimap.h"
#include "misc.h"
#include "mm_log.h"

FILE* mm_split_init(const char* prefix, const mm_idx_t* mi)
{
	char* fn;
	FILE* fp;
	uint32_t i, k = mi->k;
	fn = (char*)calloc(strlen(prefix) + 10, 1);
	sprintf(fn, "%s.%.4d.tmp", prefix, mi->index);
	if ((fp = fopen(fn, "wb")) == NULL) {
		if (mm_verbose >= 1)
			mm_log("[ERROR]\033[1;31m failed to write to temporary file '%s'\033[0m: %s", fn, strerror(errno));
		exit(1);
	}
	mm_err_fwrite(&k, 4, 1, fp);
	mm_err_fwrite(&mi->n_seq, 4, 1, fp);
	for (i = 0; i < mi->n_seq; ++i) {
		uint32_t l;
		l = strlen(mi->seq[i].name);
		mm_err_fwrite(&l, 1, 4, fp);
		mm_err_fwrite(mi->seq[i].name, 1, l, fp);
		mm_err_fwrite(&mi->seq[i].len, 4, 1, fp);
	}
	free(fn);
	return fp;
}

mm_idx_t* mm_split_merge_prep(const char* prefix, int n_splits, FILE** fp, uint32_t* n_seq_part)
{
	mm_idx_t* mi = 0;
	char* fn;
	int i, j;

	if (n_splits < 1) return 0;
	fn = CALLOC(char, strlen(prefix) + 10);
	for (i = 0; i < n_splits; ++i) {
		sprintf(fn, "%s.%.4d.tmp", prefix, i);
		if ((fp[i] = fopen(fn, "rb")) == 0) {
			if (mm_verbose >= 1)
				mm_log("ERROR: failed to open temporary file '%s': %s", fn, strerror(errno));
			for (j = 0; j < i; ++j)
				fclose(fp[j]);
			free(fn);
			return 0;
		}
	}
	free(fn);

	mi = CALLOC(mm_idx_t, 1);
	for (i = 0; i < n_splits; ++i) {
		mm_err_fread(&mi->k, 4, 1, fp[i]); // TODO: check if k is all the same
		mm_err_fread(&n_seq_part[i], 4, 1, fp[i]);
		mi->n_seq += n_seq_part[i];
	}
	mi->seq = CALLOC(mm_idx_seq_t, mi->n_seq);
	for (i = j = 0; i < n_splits; ++i) {
		uint32_t k;
		for (k = 0; k < n_seq_part[i]; ++k, ++j) {
			uint32_t l;
			mm_err_fread(&l, 1, 4, fp[i]);
			mi->seq[j].name = (char*)calloc(l + 1, 1);
			mm_err_fread(mi->seq[j].name, 1, l, fp[i]);
			mm_err_fread(&mi->seq[j].len, 4, 1, fp[i]);
		}
	}
	return mi;
}

void mm_split_rm_tmp(const char* prefix, int n_splits)
{
	int i;
	char* fn;
	fn = CALLOC(char, strlen(prefix) + 10);
	for (i = 0; i < n_splits; ++i) {
		sprintf(fn, "%s.%.4d.tmp", prefix, i);
		remove(fn);
	}
	free(fn);
}
