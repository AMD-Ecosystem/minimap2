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
 * @file mm2_chain.cpp
 * @brief Minimap2 Chaining Stage CLI
 *
 * Performs the chaining stage of minimap2's mapping pipeline.
 * Uses mm_map_chain from the full pipeline for consistency.
 *
 * Input:  JSON file from mm2_seed (or compatible format) + reference index
 * Output: JSON file with chains for use by mm2_align
 *
 * Usage:
 *   mm2_chain [options] <reference.fa> -s <seeds.json> -o <output.json>
 *
 * Example:
 *   mm2_seed -k 15 -w 10 ref.fa query.fa -o seeds.json
 *   mm2_chain ref.fa -s seeds.json -o chains.json
 */

#include <getopt.h>
#include "cli_common.h"

#include "chain_read.h"
#include "minimap.h" /* Include before mm2_chain.h for typedefs */
#include "mm2_chain.h"
#include "chain_priv.h"

// Apply preset configurations for chaining
static int apply_preset(const char* preset, mm_mapopt_t* opt)
{
	if (strcmp(preset, "map-ont") == 0 || strcmp(preset, "map-pb") == 0 ||
	    strcmp(preset, "map10k") == 0 || strcmp(preset, "map-hifi") == 0 ||
	    strcmp(preset, "map-ccs") == 0) {
		// Use defaults already set by mm_mapopt_init()
	} else if (strcmp(preset, "ava-ont") == 0 || strcmp(preset, "ava-pb") == 0) {
		opt->max_chain_skip = 25;
		opt->bw = opt->bw_long = 2000;
	} else if (strncmp(preset, "asm", 3) == 0) {
		opt->bw = 1000;
		opt->bw_long = 100000;
		opt->max_gap = 10000;
	} else if (strcmp(preset, "sr") == 0 || strcmp(preset, "short") == 0) {
		opt->max_gap = 100;
		opt->bw = opt->bw_long = 100;
		opt->min_cnt = 2;
		opt->min_chain_score = 25;
	} else if (strncmp(preset, "splice", 6) == 0 || strcmp(preset, "cdna") == 0) {
		opt->max_gap = 2000;
		opt->max_gap_ref = opt->bw = opt->bw_long = 200000;
	} else {
		mm_log_error("Unknown preset '{}'", preset);
		return -1;
	}
	return 0;
}

// ============================================================================
// Output JSON Creation
// ============================================================================

static cJSON* create_chain_output_json(cJSON* input_json, const mm_mapopt_t* opt, const mm_idx_t* mi)
{
	cJSON* root = cJSON_CreateObject();
	cJSON_AddStringToObject(root, "name", "chain_output");
	cJSON_AddStringToObject(root, "description", "Minimap2 chaining stage output");

	/* Copy reference and query info from input */
	cJSON* ref = cJSON_GetObjectItem(input_json, "reference");
	if (ref) cJSON_AddItemToObject(root, "reference", cJSON_Duplicate(ref, 1));

	cJSON* query = cJSON_GetObjectItem(input_json, "query");
	if (query) cJSON_AddItemToObject(root, "query", cJSON_Duplicate(query, 1));

	/* Add chaining parameters */
	cJSON* params = cJSON_CreateObject();
	cJSON_AddNumberToObject(params, "k", mi->k);
	cJSON_AddNumberToObject(params, "max_gap", opt->max_gap);
	cJSON_AddNumberToObject(params, "bw", opt->bw);
	cJSON_AddNumberToObject(params, "min_cnt", opt->min_cnt);
	cJSON_AddNumberToObject(params, "min_chain_score", opt->min_chain_score);
	cJSON_AddStringToObject(params, "algorithm",
	    (opt->flag & MM_F_RMQ) ? "rmq" : "dp");
	cJSON_AddItemToObject(root, "chain_params", params);

	/* Copy index_params if present */
	cJSON* idx_params = cJSON_GetObjectItem(input_json, "index_params");
	if (idx_params) cJSON_AddItemToObject(root, "index_params", cJSON_Duplicate(idx_params, 1));

	return root;
}

/** Add chains to query item from chain_read_t. */
static void add_chains_to_query_from_read(cJSON* query_item, const chain_read_t* read)
{
	add_chains_to_query_json(query_item, read);
	cJSON_AddNumberToObject(query_item, "frag_gap", read->frag_gap);
	cJSON_AddNumberToObject(query_item, "rep_len", read->rep_len);
}

// ============================================================================
// Main
// ============================================================================

static void print_usage(const char* prog)
{
	mm_log("\nUsage: %s [options] <reference.fa> -s <seeds.json> -o <output.json>", prog);
	mm_log("\nChains seed anchors from mm2_seed output using mm_map_chain.");
	mm_log("\nRequired arguments:");
	mm_log("  <reference.fa>  Reference FASTA file (for index parameters)");
	mm_log("  -s FILE         Input seeds JSON file (from mm2_seed)");
	mm_log("  -o FILE         Output chains JSON file");
	mm_log("\nPreset:");
	mm_log("  -x STR    Preset configuration");
	mm_log("            map-ont/map-pb/map-hifi/asm5/asm10/asm20/sr/splice");
	mm_log("\nChaining options:");
	mm_log("  -k INT    K-mer size [from input or 15]");
	mm_log("  -w INT    Minimizer window size [from input or 10]");
	mm_log("  -g INT    Maximum gap in reference/query [5000]");
	mm_log("  -G INT    Maximum gap on reference (overrides -g) [0=use -g]");
	mm_log("  -b INT    Bandwidth for DP chaining [500]");
	mm_log("  -B INT    Long-join bandwidth (for ultra-long reads) [0=disabled]");
	mm_log("  -n INT    Minimum anchor count per chain [3]");
	mm_log("  -c INT    Minimum chain score [40]");
	mm_log("  -r        Use RMQ algorithm (faster for ultra-long reads)");
	mm_log("\nAdvanced chaining options:");
	mm_log("  --max-chain-skip INT     Max anchors to skip in chaining [25]");
	mm_log("  --max-chain-iter INT     Max chaining DP iterations [5000]");
	mm_log("  --chain-gap-scale FLOAT  Gap cost scaling factor [1.0]");
	mm_log("  --chain-skip-scale FLOAT Skip cost scaling factor [1.0]");
#ifdef MM_ENABLE_HIP
	mm_log("\nGPU options:");
	mm_log("  --use-gpu           Enable GPU acceleration for chaining");
	mm_log("  --gpu-config FILE   GPU configuration file (optional; overrides auto-detected defaults)");
#endif
	mm_log("\nOther options:");
	mm_log("  -V        Show version");
	mm_log("  -h        Show this help message");
	mm_log("\nExample:");
	mm_log("  mm2_seed -k 15 -w 10 ref.fa query.fa -o seeds.json");
	mm_log("  %s ref.fa -s seeds.json -o chains.json", prog);
#ifdef MM_ENABLE_HIP
	mm_log("  %s --use-gpu --gpu-config gpu_config.json ref.fa -s seeds.json -o chains.json", prog);
#endif
	mm_log("");
}

int main(int argc, char* argv[])
{
	mm_logger_t* logger = mm_logger_create(0, NULL);
	mm_set_tl_logger(logger);

	mm_mapopt_t opt;
	mm_mapopt_init(&opt);

	const char* seeds_path = NULL;
	const char* output_path = NULL;
#ifdef MM_ENABLE_HIP
	const char* gpu_config_path = NULL;
#endif
	const char* preset = NULL;
	int k_override = -1;
	int w_override = -1;
#ifdef MM_ENABLE_HIP
	bool use_gpu = false;
#endif

	// Parse long options
	static struct option long_options[] = {
#ifdef MM_ENABLE_HIP
	    {"use-gpu", no_argument, 0, 0},
	    {"gpu-config", required_argument, 0, 1},
#endif
	    {"max-chain-skip", required_argument, 0, 2},
	    {"max-chain-iter", required_argument, 0, 3},
	    {"chain-gap-scale", required_argument, 0, 4},
	    {"chain-skip-scale", required_argument, 0, 5},
	    {"version", no_argument, 0, 'V'},
	    {0, 0, 0, 0}};

	int c;
	int option_index = 0;
	while ((c = getopt_long(argc, argv, "x:k:w:g:G:b:B:n:c:rs:o:Vh", long_options, &option_index)) != -1) {
		if (c == 'x') {
			preset = optarg;
			continue;
		}
#ifdef MM_ENABLE_HIP
		else if (c == 0) {
			// --use-gpu
			use_gpu = true;
			continue;
		} else if (c == 1) {
			// --gpu-config
			gpu_config_path = optarg;
			continue;
		}
#endif
		else if (c == 2) {
			// --max-chain-skip
			opt.max_chain_skip = (strcmp(optarg, "infinity") == 0 || strcmp(optarg, "inf") == 0) ? INT32_MAX : atoi(optarg);
			continue;
		} else if (c == 3) {
			// --max-chain-iter
			opt.max_chain_iter = atoi(optarg);
			continue;
		} else if (c == 4) {
			// --chain-gap-scale
			opt.chain_gap_scale = atof(optarg);
			continue;
		} else if (c == 5) {
			// --chain-skip-scale
			opt.chain_skip_scale = atof(optarg);
			continue;
		}

		switch (c) {
		case 'k':
			k_override = atoi(optarg);
			if (k_override <= 0 || k_override > 28) {
				mm_log_error("k must be between 1 and 28");
				return 1;
			}
			break;
		case 'w':
			w_override = atoi(optarg);
			if (w_override <= 0) {
				mm_log_error("w must be positive");
				return 1;
			}
			break;
		case 'g':
			opt.max_gap = atoi(optarg);
			if (opt.max_gap <= 0) {
				mm_log_error("max_gap must be positive");
				return 1;
			}
			break;
		case 'G':
			opt.max_gap_ref = atoi(optarg);
			break;
		case 'b':
			opt.bw = atoi(optarg);
			if (opt.bw <= 0) {
				mm_log_error("bandwidth must be positive");
				return 1;
			}
			break;
		case 'B':
			opt.bw_long = atoi(optarg);
			break;
		case 'n':
			opt.min_cnt = atoi(optarg);
			if (opt.min_cnt < 1) {
				mm_log_error("min_cnt must be at least 1");
				return 1;
			}
			break;
		case 'c':
			opt.min_chain_score = atoi(optarg);
			break;
		case 'r':
			opt.flag |= MM_F_RMQ;
			break;
		case 's':
			seeds_path = optarg;
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
		case '?':
			mm_log_error("Unknown option");
			print_usage(argv[0]);
			return 1;
		default:
			print_usage(argv[0]);
			return 1;
		}
	}

	// Apply preset if specified
	if (preset) {
		if (apply_preset(preset, &opt) < 0) {
			return 1;
		}
	}

#ifdef MM_ENABLE_HIP
	// Apply GPU settings
	if (use_gpu) {
		if (opt.flag & MM_F_RMQ) {
			mm_log_warn("GPU chaining is not compatible with RMQ algorithm. GPU will be disabled.");
		} else {
			opt.flag |= MM_F_GPU_CHAIN;
			opt.gpu_config_file = gpu_config_path; // NULL means auto-detect
			if (gpu_config_path)
				mm_log_info("GPU chaining enabled with config: {}", opt.gpu_config_file);
			else
				mm_log_info("GPU chaining enabled with auto-detected config");
		}
	}
#endif

	if (optind >= argc) {
		mm_log_error("Missing reference file");
		print_usage(argv[0]);
		return 1;
	}

	if (!seeds_path || !output_path) {
		mm_log_error("Missing required arguments (-s and -o)");
		print_usage(argv[0]);
		return 1;
	}

	const char* ref_path = argv[optind];

	/* Load input JSON first to get index params */
	JsonGuard input_json(json_read_file(seeds_path));
	if (!input_json) {
		mm_log_error("Failed to read input JSON from {}", seeds_path);
		return 1;
	}

	/* Get k, w from input JSON if not overridden */
	int k = 15, w = 10;
	cJSON* idx_params = cJSON_GetObjectItem(input_json, "index_params");
	if (idx_params) {
		cJSON* k_item = cJSON_GetObjectItem(idx_params, "k");
		cJSON* w_item = cJSON_GetObjectItem(idx_params, "w");
		if (k_item && cJSON_IsNumber(k_item)) k = k_item->valueint;
		if (w_item && cJSON_IsNumber(w_item)) w = w_item->valueint;
	}
	if (k_override > 0) k = k_override;
	if (w_override > 0) w = w_override;

	/* Load index */
	mm_idxopt_t iopt;
	mm_idxopt_init(&iopt);
	iopt.k = k;
	iopt.w = w;

	mm_idx_t* mi = load_index(ref_path, &iopt);
	if (!mi) {
		mm_log_error("Failed to load index from {}", ref_path);
		return 1;
	}

	/* Build the GPU device list before any ctx creation.
	 * Auto-discover all visible GPUs (no --gpu-devices flag for this tool). */
	gpu_device_list_t device_list;
#ifdef MM_ENABLE_HIP
	if (opt.flag & MM_F_GPU_CHAIN) {
		device_list = mm_gpu_create_device_list(NULL, 0);
		if (device_list.empty()) {
			mm_log_error("No GPU devices found for GPU chaining");
			mm_idx_destroy(mi);
			return 1;
		}
	}
#endif
	mm_chain_ctx_t* chain_ctx = mm_chain_ctx_create(&opt, /*n_threads=*/1,
	    std::move(device_list)
	);
	if (chain_ctx && chain_ctx->gpu_chain) {
		mm_log_info("GPU initialized for chaining");
	} else if (opt.flag & MM_F_GPU_CHAIN) {
		/* User asked for --gpu-chain but ctx creation returned NULL. Likely
		 * causes: CPU-only build (no MM_ENABLE_HIP) or GPU init failure
		 * inside the chain library. Warn loudly so the user knows they're
		 * silently on the CPU path. */
		mm_log_warn("GPU chaining requested (--gpu-chain) but ctx init returned NULL; "
			    "falling back to CPU. Check that this is a GPU-enabled build.");
	}

	/* Create output JSON */
	JsonGuard output_json(create_chain_output_json(input_json, &opt, mi));
	cJSON* queries_arr = cJSON_CreateArray();

	cJSON* input_queries = cJSON_GetObjectItem(input_json, "queries");
	if (!input_queries) {
		mm_log_error("No queries found in input JSON");
		mm_chain_ctx_destroy(chain_ctx);
		mm_idx_destroy(mi);
		return 1;
	}

	int n_queries = 0;
	int n_chains_total = 0;

	cJSON* query_in;
	cJSON_ArrayForEach(query_in, input_queries)
	{
		KallocGuard km;

		/* Get query info */
		cJSON* name_item = cJSON_GetObjectItem(query_in, "name");
		cJSON* length_item = cJSON_GetObjectItem(query_in, "length");
		const char* qname = name_item ? name_item->valuestring : "unknown";
		int qlen = length_item ? length_item->valueint : 0;

		/* Initialize chain_read_t */
		chain_read_t read;
		memset(&read, 0, sizeof(read));
		read.n_seg = 1;
		read.seq.qlen_sum = qlen;
		strncpy(read.seq.name, qname, sizeof(read.seq.name) - 1);
		read.seq.name[sizeof(read.seq.name) - 1] = '\0';

		/* Load seeds into chain_read_t */
		read.n = load_seeds_from_json(query_in, &read.a, km.get());

		/* Save original anchors before chaining (chaining may modify/free them) */
		mm128_t* original_anchors = read.a;
		int64_t original_n = read.n;

		/* Create query output item */
		cJSON* query_out = cJSON_CreateObject();
		cJSON_AddStringToObject(query_out, "name", qname);
		cJSON_AddNumberToObject(query_out, "length", qlen);

		/* Run chaining using mm_map_chain */
		if (read.n > 0) {
			mm_stage_timer_t t_chain = {0};
			mm_map_chain(mi, &opt, &read, &t_chain, km.get(), NULL);
			n_chains_total += read.n_u;
		}

		/* Add seeds to output - use read.a if still valid (chains found), 
         * otherwise use original anchors (no chains found) */
		if (read.a != NULL && read.n > 0) {
			/* Chaining reordered anchors - use the reordered ones */
			add_seeds_to_query(query_out, read.a, read.n);
		} else if (original_anchors != NULL && original_n > 0) {
			/* No chains found, use original anchors */
			add_seeds_to_query(query_out, original_anchors, original_n);
		}

		/* Add chains to output */
		add_chains_to_query_from_read(query_out, &read);

		cJSON_AddItemToArray(queries_arr, query_out);
		n_queries++;

		/* Memory freed when km goes out of scope */
	}

	cJSON_AddItemToObject(output_json, "queries", queries_arr);

	/* Write output */
	if (json_write_file(output_path, output_json) != 0) {
		mm_log_error("Failed to write output file");
		mm_chain_ctx_destroy(chain_ctx);
		mm_idx_destroy(mi);
		return 1;
	}

	mm_log_info("Chaining complete. Processed {} queries, found {} chains.",
	    n_queries, n_chains_total);

	mm_chain_ctx_destroy(chain_ctx);
	mm_idx_destroy(mi);
	mm_logger_destroy(logger);
	return 0;
}
