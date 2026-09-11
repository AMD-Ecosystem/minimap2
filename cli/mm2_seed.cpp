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
 * @file mm2_seed.cpp
 * @brief Minimap2 Seeding Stage CLI
 * 
 * Performs only the seeding stage of minimap2's mapping pipeline.
 * Takes reference and query files as input, outputs JSON with seed matches.
 * 
 * Usage:
 *   mm2_seed [options] <reference.fa> <query.fa> -o <output.json>
 */

#include <getopt.h>
#include <cstring>
#include "cli_common.h"

extern "C" {
#include "mm2_seed.h"
#include "chain_read.h"
#include "index.h"
}

/* Default seeding parameters from mm_mapopt_init() in options.c */
#define DEFAULT_SDUST_THRES 0 /* opt->sdust_thres = 0 (no SDUST masking) */
#define DEFAULT_MAX_MAX_OCC 4095 /* opt->max_max_occ = 4095 */
#define DEFAULT_OCC_DIST 500 /* opt->occ_dist = 500 */
#define DEFAULT_Q_OCC_FRAC 0.01f /* opt->q_occ_frac = 0.01f */

// Apply preset configurations for seeding
static int apply_preset(const char* preset, int* k, int* w)
{
	if (strcmp(preset, "map-ont") == 0) {
		*k = 15;
		*w = 10; // default
	} else if (strcmp(preset, "map-pb") == 0 || strcmp(preset, "map10k") == 0) {
		*k = 19;
		*w = 10; // HPC indexing
	} else if (strcmp(preset, "map-hifi") == 0 || strcmp(preset, "map-ccs") == 0) {
		*k = 19;
		*w = 19;
	} else if (strncmp(preset, "asm", 3) == 0) {
		*k = 19;
		*w = (strcmp(preset, "asm20") == 0) ? 10 : 19;
	} else if (strcmp(preset, "sr") == 0 || strcmp(preset, "short") == 0) {
		*k = 21;
		*w = 11;
	} else if (strncmp(preset, "splice", 6) == 0 || strcmp(preset, "cdna") == 0) {
		*k = 15;
		*w = 5;
	} else if (strcmp(preset, "ava-ont") == 0) {
		*k = 15;
		*w = 5;
	} else if (strcmp(preset, "ava-pb") == 0) {
		*k = 19;
		*w = 5;
	} else {
		mm_log_error("Unknown preset '{}'", preset);
		return -1;
	}
	return 0;
}

static void print_usage(const char* prog)
{
	mm_log("Usage: %s [options] <reference.fa> <query.fa> -o <output.json>\n", prog);
	mm_log("Options:");
	mm_log("  -x STR    Preset configuration [default]");
	mm_log("            map-ont: Oxford Nanopore reads");
	mm_log("            map-pb: PacBio CLR reads");
	mm_log("            map-hifi: PacBio HiFi/CCS reads");
	mm_log("            asm5/asm10/asm20: assembly alignment");
	mm_log("            sr: short reads");
	mm_log("            splice: spliced alignment");
	mm_log("  -k INT    K-mer size (default: 15)");
	mm_log("  -w INT    Minimizer window size (default: 10)");
	mm_log("  -d INT    SDUST threshold for masking low-complexity (default: 0, disabled)");
	mm_log("  -m INT    Max occurrence threshold (default: auto-computed from index)");
	mm_log("  -M INT    Hard max occurrence limit (default: 4095)");
	mm_log("  -D INT    Occurrence distance for selective filtering (default: 500)");
	mm_log("  -o FILE   Output JSON file (required)");
	mm_log("  -V, --version  Show version");
	mm_log("  -h              Show help");
}

int main(int argc, char* argv[])
{
	mm_logger_t* logger = mm_logger_create(0, NULL);
	mm_set_tl_logger(logger);

	int k = 15, w = 10;
	int sdust_thres = DEFAULT_SDUST_THRES;
	int max_occ_override = -1; // -1 means auto-compute from index
	int max_max_occ = DEFAULT_MAX_MAX_OCC;
	int occ_dist = DEFAULT_OCC_DIST;
	const char* output_path = NULL;
	const char* preset = NULL;

	static struct option long_options[] = {
	    {"version", no_argument, 0, 'V'},
	    {0, 0, 0, 0}};

	int c;
	int option_index = 0;
	while ((c = getopt_long(argc, argv, "x:k:w:d:m:M:D:o:Vh", long_options, &option_index)) != -1) {
		switch (c) {
		case 'x':
			preset = optarg;
			break;
		case 'k':
			k = atoi(optarg);
			break;
		case 'w':
			w = atoi(optarg);
			break;
		case 'd':
			sdust_thres = atoi(optarg);
			break;
		case 'm':
			max_occ_override = atoi(optarg);
			break;
		case 'M':
			max_max_occ = atoi(optarg);
			break;
		case 'D':
			occ_dist = atoi(optarg);
			break;
		case 'o':
			output_path = optarg;
			break;
		case 'V': {
			const mm_version_t* v = mm_version_info();
			mm_print("%s", v->full);
			return 0;
		}
		case 'h':
			print_usage(argv[0]);
			return 0;
		default:
			print_usage(argv[0]);
			return 1;
		}
	}

	if (optind + 2 > argc || !output_path) {
		mm_log_error("Missing required arguments");
		print_usage(argv[0]);
		return 1;
	}

	// Apply preset if specified
	if (preset) {
		if (apply_preset(preset, &k, &w) < 0) {
			return 1;
		}
	}

	// Validate parameters
	if (k <= 0 || k > 28) {
		mm_log_error("k must be between 1 and 28");
		return 1;
	}
	if (w <= 0) {
		mm_log_error("w must be positive");
		return 1;
	}

	const char* ref_path = argv[optind];
	const char* query_path = argv[optind + 1];

	// Load index
	mm_idxopt_t iopt;
	mm_idxopt_init(&iopt);
	iopt.k = k;
	iopt.w = w;

	mm_idx_t* mi = load_index(ref_path, &iopt);
	if (!mi) return 1;

	// Initialize mapping options
	mm_mapopt_t opt;
	mm_mapopt_init(&opt);
	opt.sdust_thres = sdust_thres;
	opt.max_max_occ = max_max_occ;
	opt.occ_dist = occ_dist;
	opt.q_occ_frac = DEFAULT_Q_OCC_FRAC;

	// Compute mid_occ dynamically like mm_mapopt_update() does, unless overridden
	if (max_occ_override > 0) {
		opt.mid_occ = max_occ_override;
	} else {
		opt.mid_occ = mm_idx_cal_max_occ(mi, 2e-4f);
		if (opt.mid_occ < 10) opt.mid_occ = 10;
		if (opt.mid_occ > 1000000) opt.mid_occ = 1000000;
	}

	// Load queries
	int n_seqs = 0;
	mm_bseq1_t* seqs = load_queries(query_path, &n_seqs);
	if (!seqs) {
		mm_log_error("Failed to load queries from {}", query_path);
		mm_idx_destroy(mi);
		return 1;
	}

	if (n_seqs == 0) {
		mm_log_warn("No sequences found in query file");
	}

	// Create output JSON using helper
	cJSON* output_json = create_output_json("seed_output", "Minimap2 seeding stage output",
	    ref_path, query_path, k, w);
	if (!output_json) {
		mm_log_error("Failed to create output JSON");
		free_queries(seqs, n_seqs);
		mm_idx_destroy(mi);
		return 1;
	}

	// Add additional seeding params
	cJSON* params = cJSON_GetObjectItem(output_json, "index_params");
	cJSON_AddNumberToObject(params, "sdust_thres", opt.sdust_thres);
	cJSON_AddNumberToObject(params, "mid_occ", opt.mid_occ);
	cJSON_AddNumberToObject(params, "max_max_occ", opt.max_max_occ);
	cJSON_AddNumberToObject(params, "occ_dist", opt.occ_dist);

	cJSON* queries_arr = cJSON_CreateArray();

	// Process each query using seed library
	for (int i = 0; i < n_seqs; i++) {
		KallocGuard km;

		// Setup chain_read_t structure
		chain_read_t read;
		memset(&read, 0, sizeof(chain_read_t));

		int qlens[1] = {seqs[i].l_seq};
		const char* qseqs[1] = {seqs[i].seq};

		read.n_seg = 1;
		read.qlens = qlens;
		read.qseqs = qseqs;
		strncpy(read.seq.name, seqs[i].name, sizeof(read.seq.name) - 1);
		read.seq.name[sizeof(read.seq.name) - 1] = '\0';

		// Call mm_map_seed
		mm_stage_timer_t t_seed = {0};
		mm_map_seed(mi, &opt, &read, &t_seed, km.get(), NULL);

		// Create query JSON
		cJSON* query_item = cJSON_CreateObject();
		cJSON_AddStringToObject(query_item, "name", seqs[i].name);
		cJSON_AddNumberToObject(query_item, "length", seqs[i].l_seq);

		// Add seeds array
		cJSON* seeds_arr = cJSON_CreateArray();
		for (int64_t j = 0; j < read.n; j++) {
			cJSON_AddItemToArray(seeds_arr, json_anchor(read.a[j].x, read.a[j].y));
		}
		cJSON_AddItemToObject(query_item, "seeds", seeds_arr);
		cJSON_AddNumberToObject(query_item, "n_seeds", (int)read.n);

		cJSON_AddItemToArray(queries_arr, query_item);

		// Memory freed when km goes out of scope
	}

	cJSON_AddItemToObject(output_json, "queries", queries_arr);

	if (json_write_file(output_path, output_json) != 0) {
		mm_log_error("Failed to write output file");
		cJSON_Delete(output_json);
		free_queries(seqs, n_seqs);
		mm_idx_destroy(mi);
		return 1;
	}

	mm_log_info("Seeding complete. Processed {} queries.", n_seqs);

	// Cleanup
	cJSON_Delete(output_json);
	free_queries(seqs, n_seqs);
	mm_idx_destroy(mi);

	mm_logger_destroy(logger);
	return 0;
}
