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
 * @file mm2_align.cpp
 * @brief Minimap2 Alignment Stage CLI
 * 
 * Performs the alignment stage of minimap2's mapping pipeline.
 * Takes reference, query files, and chains JSON as input.
 * Outputs JSON with seeds, chains, and alignments.
 * 
 * Usage:
 *   mm2_align [options] <reference.fa> <query.fa> -c <chains.json> -o <output.json>
 */

#include <getopt.h>
#include <vector>
#include "cli_common.h"

#include "chain_read.h"
#include "mm2_align.h"

// Apply preset configurations for alignment
static int apply_preset(const char* preset, int* a, int* b, int* q, int* e,
    int* q2, int* e2, int* zdrop, int* zdrop_inv, int* min_dp_max)
{
	// Set defaults first
	*a = 2;
	*b = 4;
	*q = 4;
	*e = 2;
	*q2 = 24;
	*e2 = 1;
	*zdrop = 400;
	*zdrop_inv = 200;
	*min_dp_max = 80; // default: min_chain_score (40) * a (2)

	if (strcmp(preset, "map-ont") == 0 || strcmp(preset, "map-pb") == 0 || strcmp(preset, "map10k") == 0) {
		// Use defaults
	} else if (strcmp(preset, "map-hifi") == 0 || strcmp(preset, "map-ccs") == 0) {
		*a = 1;
		*b = 4;
		*q = 6;
		*q2 = 26;
		*e = 2;
		*e2 = 1;
		*min_dp_max = 200;
	} else if (strcmp(preset, "asm5") == 0) {
		*a = 1;
		*b = 19;
		*q = 39;
		*q2 = 81;
		*e = 3;
		*e2 = 1;
		*zdrop = 200;
		*zdrop_inv = 200;
		*min_dp_max = 200;
	} else if (strcmp(preset, "asm10") == 0) {
		*a = 1;
		*b = 9;
		*q = 16;
		*q2 = 41;
		*e = 2;
		*e2 = 1;
		*zdrop = 200;
		*zdrop_inv = 200;
		*min_dp_max = 200;
	} else if (strcmp(preset, "asm20") == 0) {
		*a = 1;
		*b = 4;
		*q = 6;
		*q2 = 26;
		*e = 2;
		*e2 = 1;
		*zdrop = 200;
		*zdrop_inv = 200;
		*min_dp_max = 200;
	} else if (strcmp(preset, "sr") == 0 || strcmp(preset, "short") == 0) {
		*a = 2;
		*b = 8;
		*q = 12;
		*q2 = 24;
		*e = 2;
		*e2 = 1;
		*zdrop = 100;
		*zdrop_inv = 100;
		*min_dp_max = 40;
	} else if (strncmp(preset, "splice", 6) == 0 || strcmp(preset, "cdna") == 0) {
		*a = 1;
		*b = 2;
		*q = 2;
		*e = 1;
		*q2 = 32;
		*e2 = 0;
		*zdrop = 200;
		*zdrop_inv = 100;
		if (strcmp(preset, "splice:hq") == 0) {
			*b = 4;
			*q = 6;
			*q2 = 24;
		}
	} else if (strcmp(preset, "ava-ont") == 0 || strcmp(preset, "ava-pb") == 0) {
		// Use defaults
	} else {
		mm_log_error("Unknown preset '{}'", preset);
		return -1;
	}
	return 0;
}

// ============================================================================
// chain_read_t helpers (init_chain_read, free_chain_read in cli_common.h)
// ============================================================================

// ============================================================================
// mm_reg1_t helpers (from align_priv.h)
// ============================================================================

/** Create JSON object for a single alignment (mm_reg1_t). */
static cJSON* alignment_to_json(const mm_reg1_t* r)
{
	cJSON* align = cJSON_CreateObject();
	if (!r) return align;

	cJSON_AddNumberToObject(align, "rid", r->rid);
	cJSON_AddNumberToObject(align, "score", r->score);
	cJSON_AddNumberToObject(align, "dp_score", r->p ? r->p->dp_score : 0);
	cJSON_AddNumberToObject(align, "qs", r->qs);
	cJSON_AddNumberToObject(align, "qe", r->qe);
	cJSON_AddNumberToObject(align, "rs", r->rs);
	cJSON_AddNumberToObject(align, "re", r->re);
	cJSON_AddNumberToObject(align, "mapq", r->mapq);
	cJSON_AddNumberToObject(align, "rev", r->rev);
	cJSON_AddNumberToObject(align, "n_cigar_ops", r->p ? r->p->n_cigar : 0);

	if (r->p && r->p->n_cigar > 0) {
		size_t cigar_size = r->p->n_cigar * 16;
		char* cigar_str = (char*)malloc(cigar_size);
		if (!cigar_str) {
			cJSON_AddStringToObject(align, "cigar", "(allocation failed)");
		} else {
			cigar_str[0] = '\0';

			static const char ops[] = "MIDNSHP=X";
			size_t pos = 0;
			for (uint32_t i = 0; i < r->p->n_cigar; i++) {
				int op = r->p->cigar[i] & 0xf;
				int len = r->p->cigar[i] >> 4;
				pos += snprintf(cigar_str + pos, cigar_size - pos, "%d%c", len, ops[op]);
			}
			cJSON_AddStringToObject(align, "cigar", cigar_str);
			free(cigar_str);
		}
	}
	return align;
}

// ============================================================================
// Main
// ============================================================================

static void print_usage(const char* prog)
{
	mm_log("Usage: %s [options] <reference.fa> <query.fa> -c <chains.json> -o <output.json>\n", prog);
	mm_log("Options:");
	mm_log("  -x STR              Preset configuration");
	mm_log("                      map-ont/map-pb/map-hifi/asm5/asm10/asm20/sr/splice");
	mm_log("  -k INT              K-mer size (default: 15)");
	mm_log("  -w INT              Minimizer window size (default: 10)");
	mm_log("  -c FILE             Input chains JSON file (required)");
	mm_log("  -o FILE             Output JSON file (required)");
	mm_log("\nAlignment scoring:");
	mm_log("  -A INT              Matching score [2]");
	mm_log("  -B INT              Mismatch penalty [4]");
	mm_log("  -O INT[,INT]        Gap open penalty [4,24]");
	mm_log("  -E INT[,INT]        Gap extension penalty [2,1]");
	mm_log("  -z INT[,INT]        Z-drop score [400,200]");
	mm_log("  -s INT              Minimum DP alignment score [40]");
#ifdef MM_ENABLE_HIP
	mm_log("\nGPU options:");
	mm_log("  --use-gpu           Enable GPU acceleration for alignment");
	mm_log("  --gpu-flush-threshold INT  accumulated tasks before GPU flush [8192]");
	mm_log("  --gpu-batch-max-align INT  max alignments per GPU batch [16384]");
	mm_log("  --gpu-batch-max-mem NUM    GPU memory limit for batching [4G]");
	mm_log("  --gpu-accum-pool-size INT  shared accumulator pool size [0=auto]");
#endif
	mm_log("\nOther options:");
	mm_log("  -V                  Show version");
	mm_log("  -h                  Show help");
	mm_log("\nExample:");
	mm_log("  %s ref.fa query.fa -c chains.json -o output.json", prog);
	mm_log("  %s -A 1 -B 2 -O 2,10 -E 1,1 ref.fa query.fa -c chains.json -o output.json", prog);
}

int main(int argc, char* argv[])
{
	mm_logger_t* logger = mm_logger_create(0, NULL);
	mm_set_tl_logger(logger);

	int k = 15, w = 10;
	const char* chains_path = NULL;
	const char* output_path = NULL;
	const char* preset = nullptr;
#ifdef MM_ENABLE_HIP
	bool use_gpu = false;
	int gpu_flush_threshold = 0;
	int gpu_batch_max_align = 0;
	int64_t gpu_batch_max_mem = 0;
	int gpu_accum_pool_size = 0;
#endif

	// Parse long options
	static struct option long_options[] = {
#ifdef MM_ENABLE_HIP
	    {"use-gpu", no_argument, 0, 0},
	    {"gpu-flush-threshold", required_argument, 0, 1},
	    {"gpu-batch-max-align", required_argument, 0, 2},
	    {"gpu-batch-max-mem", required_argument, 0, 3},
	    {"gpu-accum-pool-size", required_argument, 0, 4},
#endif
	    {"version", no_argument, 0, 'V'},
	    {0, 0, 0, 0}};

	int c;
	int option_index = 0;
	char* s;

	// Temporary storage for alignment parameters
	int opt_a = -1, opt_b = -1;
	int opt_q = -1, opt_q2 = -1;
	int opt_e = -1, opt_e2 = -1;
	int opt_zdrop = -1, opt_zdrop_inv = -1;
	int opt_min_dp_max = -1;

	while ((c = getopt_long(argc, argv, "x:k:w:c:o:A:B:O:E:z:s:Vh", long_options, &option_index)) != -1) {
#ifdef MM_ENABLE_HIP
		if (c == 0) {
			use_gpu = true;
			continue;
		}
		if (c == 1) {
			gpu_flush_threshold = atoi(optarg);
			continue;
		}
		if (c == 2) {
			gpu_batch_max_align = atoi(optarg);
			continue;
		}
		if (c == 3) {
			double x = strtod(optarg, &s);
			if (*s == 'G' || *s == 'g')
				x *= 1e9;
			else if (*s == 'M' || *s == 'm')
				x *= 1e6;
			else if (*s == 'K' || *s == 'k')
				x *= 1e3;
			gpu_batch_max_mem = (int64_t)(x + .499);
			continue;
		}
		if (c == 4) {
			gpu_accum_pool_size = atoi(optarg);
			continue;
		}
#endif

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
		case 'c':
			chains_path = optarg;
			break;
		case 'o':
			output_path = optarg;
			break;
		case 'A':
			opt_a = atoi(optarg);
			break;
		case 'B':
			opt_b = atoi(optarg);
			break;
		case 'O':
			opt_q = opt_q2 = strtol(optarg, &s, 10);
			if (*s == ',') opt_q2 = strtol(s + 1, &s, 10);
			break;
		case 'E':
			opt_e = opt_e2 = strtol(optarg, &s, 10);
			if (*s == ',') opt_e2 = strtol(s + 1, &s, 10);
			break;
		case 'z':
			opt_zdrop = opt_zdrop_inv = strtol(optarg, &s, 10);
			if (*s == ',') opt_zdrop_inv = strtol(s + 1, &s, 10);
			break;
		case 's':
			opt_min_dp_max = atoi(optarg);
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

	// Apply preset if specified (before validating individual params)
	if (preset) {
		int preset_a, preset_b, preset_q, preset_e, preset_q2, preset_e2;
		int preset_zdrop, preset_zdrop_inv, preset_min_dp_max;

		if (apply_preset(preset, &preset_a, &preset_b, &preset_q, &preset_e,
			&preset_q2, &preset_e2, &preset_zdrop, &preset_zdrop_inv,
			&preset_min_dp_max) < 0) {
			return 1;
		}

		// Apply preset values only if not overridden by command line
		if (opt_a < 0) opt_a = preset_a;
		if (opt_b < 0) opt_b = preset_b;
		if (opt_q < 0) opt_q = preset_q;
		if (opt_e < 0) opt_e = preset_e;
		if (opt_q2 < 0) opt_q2 = preset_q2;
		if (opt_e2 < 0) opt_e2 = preset_e2;
		if (opt_zdrop < 0) opt_zdrop = preset_zdrop;
		if (opt_zdrop_inv < 0) opt_zdrop_inv = preset_zdrop_inv;
		if (opt_min_dp_max < 0) opt_min_dp_max = preset_min_dp_max;
	}

	if (optind + 2 > argc || !chains_path || !output_path) {
		mm_log_error("Missing required arguments");
		print_usage(argv[0]);
		return 1;
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

	// Validate alignment scoring parameters
	if (opt_a >= 0 && opt_a > 255) {
		mm_log_error("Matching score (-A) must be <= 255");
		return 1;
	}
	if (opt_b >= 0 && opt_b > 255) {
		mm_log_error("Mismatch penalty (-B) must be <= 255");
		return 1;
	}
	if (opt_q >= 0 && opt_q > 255) {
		mm_log_error("Gap open penalty (-O) must be <= 255");
		return 1;
	}
	if (opt_e >= 0 && opt_e > 255) {
		mm_log_error("Gap extension penalty (-E) must be <= 255");
		return 1;
	}

	const char* ref_path = argv[optind];
	const char* query_path = argv[optind + 1];

	// Initialize context
	mm2_context ctx;

	// Load chains JSON first to get params
	if (!ctx.load_input(chains_path)) {
		mm_log_error("Failed to read input JSON from {}", chains_path);
		return 1;
	}
	get_params_from_json(ctx.input_json, &k, &w);

	if (!ctx.init(ref_path, query_path, k, w)) return 1;
	ctx.opt.flag |= MM_F_CIGAR; // Enable CIGAR output

	// Apply alignment scoring parameters if provided
	if (opt_a >= 0) ctx.opt.a = opt_a;
	if (opt_b >= 0) ctx.opt.b = opt_b;
	if (opt_q >= 0) ctx.opt.q = opt_q;
	if (opt_q2 >= 0) ctx.opt.q2 = opt_q2;
	if (opt_e >= 0) ctx.opt.e = opt_e;
	if (opt_e2 >= 0) ctx.opt.e2 = opt_e2;
	if (opt_zdrop >= 0) ctx.opt.zdrop = opt_zdrop;
	if (opt_zdrop_inv >= 0) ctx.opt.zdrop_inv = opt_zdrop_inv;
	if (opt_min_dp_max >= 0) ctx.opt.min_dp_max = opt_min_dp_max;

#ifdef MM_ENABLE_HIP
	// Apply GPU settings if enabled
	if (use_gpu) {
		ctx.opt.flag |= MM_F_GPU_ALIGN;
		if (gpu_flush_threshold > 0) ctx.opt.gpu_flush_threshold = gpu_flush_threshold;
		if (gpu_batch_max_align > 0) ctx.opt.gpu_batch_max_align = gpu_batch_max_align;
		if (gpu_batch_max_mem > 0) ctx.opt.gpu_batch_max_mem = gpu_batch_max_mem;
		if (gpu_accum_pool_size > 0) ctx.opt.gpu_accum_pool_size = gpu_accum_pool_size;
		mm_log_info("GPU alignment enabled");
	}
#endif

	// Create output JSON
	ctx.create_output("align_output", "Minimap2 alignment stage output",
	    ref_path, query_path, k, w);

	cJSON* queries_arr = cJSON_CreateArray();
	cJSON* input_queries = ctx.get_input_queries();

#ifdef MM_ENABLE_HIP
	/* Build the GPU device list before any ctx creation.
	 * Auto-discover all visible GPUs (no --gpu-devices flag for this tool). */
	gpu_device_list_t device_list;
	if (use_gpu) {
		device_list = mm_gpu_create_device_list(NULL, 0);
		if (device_list.empty()) {
			mm_log_error("No GPU devices found for accelerated alignment");
			return 1;
		}

		mm_align_ctx_t* align_ctx = mm_align_ctx_create(&ctx.opt, 1, std::move(device_list));
		gpu_align_accum_t* accum = mm_gpu_align_ctx_acquire_accum(
		    align_ctx->gpu_align, &ctx.opt, ctx.mi);

		struct read_result {
			chain_read_t read;
			mm_reg1_t* regs;
			int n_regs;
			cJSON* query_item;
		};

		std::vector<read_result> results(ctx.n_seqs);

		for (int i = 0; i < ctx.n_seqs; i++) {
			read_result& rr = results[i];
			rr.regs = NULL;
			rr.n_regs = 0;

			init_chain_read(&rr.read, &ctx.seqs[i], NULL);

			cJSON* chain_query = json_find_query(input_queries, ctx.seqs[i].name);
			if (chain_query) {
				rr.read.n = load_seeds_from_json(chain_query, &rr.read.a, NULL);
				rr.read.n_u = load_chains_from_json(chain_query, &rr.read.u, NULL);

				cJSON* frag_gap_item = cJSON_GetObjectItem(chain_query, "frag_gap");
				rr.read.frag_gap = frag_gap_item ? frag_gap_item->valueint : 1000;

				cJSON* rep_len_item = cJSON_GetObjectItem(chain_query, "rep_len");
				rr.read.rep_len = rep_len_item ? rep_len_item->valueint : 0;
			}

			rr.query_item = cJSON_CreateObject();
			cJSON_AddStringToObject(rr.query_item, "name", ctx.seqs[i].name);
			cJSON_AddNumberToObject(rr.query_item, "length", ctx.seqs[i].l_seq);
			add_seeds_to_query(rr.query_item, rr.read.a, rr.read.n);
			add_chains_to_query_json(rr.query_item, &rr.read);

			if (rr.read.n > 0 && rr.read.n_u > 0) {
				mm_gpu_accum_add_read(accum, &rr.read,
				    &rr.regs, &rr.n_regs, NULL, NULL, NULL);
				if (mm_gpu_accum_should_flush(accum))
					mm_gpu_accum_flush(accum, NULL, NULL);
			}
		}

		mm_gpu_accum_flush(accum, NULL, NULL);

		for (int i = 0; i < ctx.n_seqs; i++) {
			read_result& rr = results[i];
			cJSON* alignments_arr = cJSON_CreateArray();
			for (int j = 0; j < rr.n_regs; j++) {
				cJSON_AddItemToArray(alignments_arr, alignment_to_json(&rr.regs[j]));
				if (rr.regs[j].p) free(rr.regs[j].p);
			}
			free(rr.regs);
			cJSON_AddItemToObject(rr.query_item, "alignments", alignments_arr);
			cJSON_AddItemToArray(queries_arr, rr.query_item);
			free_chain_read(&rr.read);
		}

		mm_gpu_align_ctx_release_accum(align_ctx->gpu_align, accum);
		mm_align_ctx_destroy(align_ctx);
	} else
#endif
	{
		// CPU path
		for (int i = 0; i < ctx.n_seqs; i++) {
			KallocGuard km;

			chain_read_t read;
			init_chain_read(&read, &ctx.seqs[i], km);

			cJSON* chain_query = json_find_query(input_queries, ctx.seqs[i].name);
			if (chain_query) {
				read.n = load_seeds_from_json(chain_query, &read.a, km);
				read.n_u = load_chains_from_json(chain_query, &read.u, km);

				cJSON* frag_gap_item = cJSON_GetObjectItem(chain_query, "frag_gap");
				read.frag_gap = frag_gap_item ? frag_gap_item->valueint : 1000;

				cJSON* rep_len_item = cJSON_GetObjectItem(chain_query, "rep_len");
				read.rep_len = rep_len_item ? rep_len_item->valueint : 0;
			}

			cJSON* query_item = cJSON_CreateObject();
			cJSON_AddStringToObject(query_item, "name", ctx.seqs[i].name);
			cJSON_AddNumberToObject(query_item, "length", ctx.seqs[i].l_seq);
			add_seeds_to_query(query_item, read.a, read.n);
			add_chains_to_query_json(query_item, &read);

			cJSON* alignments_arr = cJSON_CreateArray();

			if (read.n > 0 && read.n_u > 0) {
				mm_reg1_t* regs = NULL;
				int n_regs = 0;

				mm_map_align(ctx.mi, &ctx.opt, &read, &regs, &n_regs, NULL, km, NULL, NULL, 0);

				for (int j = 0; j < n_regs; j++) {
					cJSON_AddItemToArray(alignments_arr, alignment_to_json(&regs[j]));
					if (regs[j].p) free(regs[j].p);
				}
				free(regs);
			}

			cJSON_AddItemToObject(query_item, "alignments", alignments_arr);
			cJSON_AddItemToArray(queries_arr, query_item);
			free_chain_read(&read);
		}
	}

	cJSON_AddItemToObject(ctx.output_json, "queries", queries_arr);

	if (ctx.write_output(output_path) != 0) {
		mm_log_error("Failed to write output file");
		return 1;
	}

	mm_log_info("Alignment complete. Processed {} queries.", ctx.n_seqs);
	mm_logger_destroy(logger);
	return 0;
}
