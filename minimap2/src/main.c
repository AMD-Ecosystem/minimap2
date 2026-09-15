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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <signal.h>
#include <stdint.h>
#include <unistd.h>
#include "minimap.h"
#include "map_priv.h"
#include "ketopt.h"
#include "mm_log.h"
#include "mm_dump.h"
#include "core_ctx.h"

#if defined(MM_ENABLE_HIP)
extern int mm_gpu_check_available(void);

static int mm_parse_gpu_device_ids_arg(mm_mapopt_t* opt, const char* arg, const char* opt_name)
{
	if (opt->gpu_device_ids) {
		free(opt->gpu_device_ids);
		opt->gpu_device_ids = NULL;
		opt->gpu_num_devices = 0;
	}

	/* NULL/empty means auto-discover all visible GPUs. */
	if (arg == NULL || *arg == '\0')
		return 0;

	char* arg_copy = strdup(arg);
	if (!arg_copy) {
		mm_log("[ERROR]\033[1;31m memory allocation failed for --%s\033[0m", opt_name);
		return -1;
	}

	/* Count commas to estimate device count. */
	int count = 1;
	for (const char* p = arg_copy; *p; ++p)
		if (*p == ',') ++count;

	opt->gpu_device_ids = (int*)malloc((size_t)count * sizeof(int));
	if (!opt->gpu_device_ids) {
		mm_log("[ERROR]\033[1;31m memory allocation failed for device id array\033[0m");
		free(arg_copy);
		return -1;
	}

	int idx = 0;
	char* saveptr = NULL;
	char* token = strtok_r(arg_copy, ",", &saveptr);
	while (token && idx < count) {
		while (isspace((unsigned char)*token)) ++token;
		char* endtrim = token + strlen(token);
		while (endtrim > token && isspace((unsigned char)*(endtrim - 1))) --endtrim;
		*endtrim = '\0';

		char* endptr = NULL;
		long val = strtol(token, &endptr, 10);
		if (token[0] == '\0' || endptr == token || *endptr != '\0' || val < 0 || val > 255) {
			mm_log("[ERROR]\033[1;31m invalid GPU device id in --%s: '%s'\033[0m", opt_name, token);
			free(opt->gpu_device_ids);
			opt->gpu_device_ids = NULL;
			opt->gpu_num_devices = 0;
			free(arg_copy);
			return -1;
		}
		opt->gpu_device_ids[idx++] = (int)val;
		token = strtok_r(NULL, ",", &saveptr);
	}

	opt->gpu_num_devices = idx;
	free(arg_copy);
	return 0;
}
#endif

static volatile sig_atomic_t s_signal_received = 0;
static mm_logger_t* s_signal_logger;

static void signal_handler(int sig)
{
	s_signal_received = sig;
	const char msg[] = "\n[minimap2] caught signal, shutting down...\n";
	(void)!write(STDERR_FILENO, msg, sizeof(msg) - 1);
	// Re-raise the signal with default handler to get proper exit behavior
	signal(sig, SIG_DFL);
	raise(sig);
}

static void atexit_flush_logger(void)
{
	if (s_signal_logger) s_signal_logger->flush(s_signal_logger);
}

static void setup_signal_handlers(void)
{
	atexit(atexit_flush_logger);
	signal(SIGTERM, signal_handler);
	signal(SIGINT, signal_handler);
	signal(SIGABRT, signal_handler);
}

#ifdef __linux__
#include <sys/resource.h>
#include <sys/time.h>
void liftrlimit()
{
	struct rlimit r;
	getrlimit(RLIMIT_AS, &r);
	r.rlim_cur = r.rlim_max;
	setrlimit(RLIMIT_AS, &r);
}
#else
void liftrlimit()
{
}
#endif

static ko_longopt_t long_options[] = {
    {"bucket-bits", ko_required_argument, 300},
    {"mb-size", ko_required_argument, 'K'},
    {"seed", ko_required_argument, 302},
    {"no-kalloc", ko_no_argument, 303},
    {"print-qname", ko_no_argument, 304},
    {"no-self", ko_no_argument, 'D'},
    {"print-seeds", ko_no_argument, 306},
    {"max-chain-skip", ko_required_argument, 307},
    {"min-dp-len", ko_required_argument, 308},
    {"print-aln-seq", ko_no_argument, 309},
    {"splice", ko_no_argument, 310},
    {"cost-non-gt-ag", ko_required_argument, 'C'},
    {"no-long-join", ko_no_argument, 312},
    {"sr", ko_optional_argument, 313},
    {"frag", ko_required_argument, 314},
    {"secondary", ko_required_argument, 315},
    {"cs", ko_optional_argument, 316},
    {"end-bonus", ko_required_argument, 317},
    {"no-pairing", ko_no_argument, 318}, // deprecated but reserved for backward compatibility
    {"splice-flank", ko_required_argument, 319},
    {"idx-no-seq", ko_no_argument, 320},
    {"end-seed-pen", ko_required_argument, 321},
    {"for-only", ko_no_argument, 322},
    {"rev-only", ko_no_argument, 323},
    {"heap-sort", ko_required_argument, 324},
    {"all-chain", ko_no_argument, 'P'},
    {"dual", ko_required_argument, 326},
    {"max-clip-ratio", ko_required_argument, 327},
    {"min-occ-floor", ko_required_argument, 328},
    {"MD", ko_no_argument, 329},
    {"lj-min-ratio", ko_required_argument, 330},
    {"score-N", ko_required_argument, 331},
    {"eqx", ko_no_argument, 332},
    {"paf-no-hit", ko_no_argument, 333},
    {"split-prefix", ko_required_argument, 334},
    {"no-end-flt", ko_no_argument, 335},
    {"hard-mask-level", ko_no_argument, 336},
    {"cap-sw-mem", ko_required_argument, 337},
    {"max-qlen", ko_required_argument, 338},
    {"max-chain-iter", ko_required_argument, 339},
    {"junc-bed", ko_required_argument, 340},
    {"junc-bonus", ko_required_argument, 341},
    {"sam-hit-only", ko_no_argument, 342},
    {"chain-gap-scale", ko_required_argument, 343},
    {"alt", ko_required_argument, 344},
    {"alt-drop", ko_required_argument, 345},
    {"mask-len", ko_required_argument, 346},
    {"rmq", ko_optional_argument, 347},
    {"qstrand", ko_no_argument, 348},
    {"cap-kalloc", ko_required_argument, 349},
    {"cap-kalloc-largest", ko_required_argument, 365},
    {"q-occ-frac", ko_required_argument, 350},
    {"chain-skip-scale", ko_required_argument, 351},
    {"print-chains", ko_no_argument, 352},
    {"no-hash-name", ko_no_argument, 353},
    {"secondary-seq", ko_no_argument, 354},
    {"ds", ko_no_argument, 355},
    {"rmq-inner", ko_required_argument, 356},
    {"spsc", ko_required_argument, 357},
    {"junc-pen", ko_required_argument, 358},
    {"pairing", ko_required_argument, 359},
    {"jump-min-match", ko_required_argument, 360},
    {"write-junc", ko_no_argument, 361},
    {"pass1", ko_required_argument, 362},
    {"spsc-scale", ko_required_argument, 363},
    {"spsc0", ko_required_argument, 364},
    {"dbg-seed-occ", ko_no_argument, 501},
    {"debug-log", ko_required_argument, 502},
    {"help", ko_no_argument, 'h'},
    {"max-intron-len", ko_required_argument, 'G'},
    {"version", ko_no_argument, 'V'},
    {"min-count", ko_required_argument, 'n'},
    {"min-chain-score", ko_required_argument, 'm'},
    {"mask-level", ko_required_argument, 'M'},
    {"min-dp-score", ko_required_argument, 's'},
    {"sam", ko_no_argument, 'a'},
#if defined(MM_ENABLE_HIP)
    {"gpu-chain", ko_no_argument, 600}, // use gpu for chaining
    {"gpu-cfg", ko_required_argument, 601},
    {"gpu-align", ko_no_argument, 602}, // use gpu for alignment
    {"gpu-flush-threshold", ko_required_argument, 605},
    {"gpu-batch-max-align", ko_required_argument, 606},
    {"gpu-batch-max-mem", ko_required_argument, 607},
    {"gpu-batch-max-mem-cap", ko_required_argument, 611},
    {"gpu-chain-workers", ko_required_argument, 609},
    {"gpu-accum-pool-size", ko_required_argument, 610},
    {"multi-gpu", ko_optional_argument, 612}, // optional explicit ids; absent => auto-discover all GPUs
#endif
    {"extra-out-dir", ko_required_argument, 700}, // extra output directory for timer stats and intermediates
    {"dump-intermediates", ko_optional_argument, 701}, // dump intermediate data: FLATBUF (default) or JSON
    {"dump-chained-seeds", ko_no_argument, 702}, // include chained seed indices in FLATBUF dump
    {0, 0, 0}};

static inline int64_t mm_parse_num2(const char* str, char** q)
{
	double x;
	char* p;
	x = strtod(str, &p);
	if (*p == 'G' || *p == 'g')
		x *= 1e9, ++p;
	else if (*p == 'M' || *p == 'm')
		x *= 1e6, ++p;
	else if (*p == 'K' || *p == 'k')
		x *= 1e3, ++p;
	if (q) *q = p;
	return (int64_t)(x + .499);
}

static inline int64_t mm_parse_num(const char* str)
{
	return mm_parse_num2(str, 0);
}

static inline void yes_or_no(mm_mapopt_t* opt, int64_t flag, int long_idx, const char* arg, int yes_to_set)
{
	if (yes_to_set) {
		if (strcmp(arg, "yes") == 0 || strcmp(arg, "y") == 0)
			opt->flag |= flag;
		else if (strcmp(arg, "no") == 0 || strcmp(arg, "n") == 0)
			opt->flag &= ~flag;
		else
			mm_log("[WARNING]\033[1;31m option '--%s' only accepts 'yes' or 'no'.\033[0m", long_options[long_idx].name);
	} else {
		if (strcmp(arg, "yes") == 0 || strcmp(arg, "y") == 0)
			opt->flag &= ~flag;
		else if (strcmp(arg, "no") == 0 || strcmp(arg, "n") == 0)
			opt->flag |= flag;
		else
			mm_log("[WARNING]\033[1;31m option '--%s' only accepts 'yes' or 'no'.\033[0m", long_options[long_idx].name);
	}
}

int main(int argc, char* argv[])
{
	const char* opt_str = "2aSDw:k:K:t:r:f:Vv:g:G:I:d:XT:s:x:Hcp:M:n:z:A:B:b:O:E:m:N:Qu:R:hF:LC:yYPo:e:U:J:j:";
	ketopt_t o = KETOPT_INIT;
	mm_mapopt_t opt;
	mm_idxopt_t ipt;
	int i, c, n_threads = 3, n_parts, old_best_n = -1;
	float spsc_scale = 0.7f;
	int max_chain_skip_user_set = 0;
	int gpu_device_selection_set = 0;
	char *fnw = 0, *rg = 0, *fn_bed_junc = 0, *fn_bed_jump = 0, *fn_bed_pass1 = 0, *fn_spsc = 0, *s, *alt_list = 0;
	char* debug_log_file = 0;
	int help_to_stdout = 0;
	const char* extra_out_dir = NULL;
	mm_dump_format_t dump_format = MM_DUMP_NONE;
	int dump_chained_seeds = 0;
	mm_idx_reader_t* idx_rdr;
	mm_idx_t* mi;

	mm_verbose = 3;
	liftrlimit();
	mm_set_opt(0, &ipt, &opt);

	// Initialize async logging system (will be re-initialized if --debug-log is provided)
	mm_logger_t* logger = mm_logger_create(1, NULL); // 1 = async mode, NULL = stderr for debug
	mm_set_tl_logger(logger);
	s_signal_logger = logger;
	setup_signal_handlers(); // Flush logs on SIGTERM, SIGINT, SIGABRT

	while ((c = ketopt(&o, argc, argv, 1, opt_str, long_options)) >= 0) { // test command line options and apply option -x/preset first
		if (c == 'x') {
			if (mm_set_opt(o.arg, &ipt, &opt) < 0) {
				mm_log("[ERROR] unknown preset '%s'", o.arg);
				return 1;
			}
		} else if (c == 354) {
			// --debug-log: Skip in first pass, will be handled in second pass
			continue;
		} else if (c == ':') {
			mm_log("[ERROR] missing option argument");
			return 1;
		} else if (c == '?') {
			mm_log("[ERROR] unknown option in \"%s\"", argv[o.i - 1]);
			return 1;
		}
	}
	o = KETOPT_INIT;

	while ((c = ketopt(&o, argc, argv, 1, opt_str, long_options)) >= 0) {
		if (c == 'w')
			ipt.w = atoi(o.arg);
		else if (c == 'k')
			ipt.k = atoi(o.arg);
		else if (c == 'H')
			ipt.flag |= MM_I_HPC;
		else if (c == 'd')
			fnw = o.arg; // the above are indexing related options, except -I
		else if (c == 't')
			n_threads = atoi(o.arg);
		else if (c == 'v')
			mm_verbose = atoi(o.arg);
		else if (c == 'g')
			opt.max_gap = (int)mm_parse_num(o.arg);
		else if (c == 'G')
			mm_mapopt_max_intron_len(&opt, (int)mm_parse_num(o.arg));
		else if (c == 'F')
			opt.max_frag_len = (int)mm_parse_num(o.arg);
		else if (c == 'N')
			old_best_n = opt.best_n, opt.best_n = atoi(o.arg);
		else if (c == 'p')
			opt.pri_ratio = atof(o.arg);
		else if (c == 'M')
			opt.mask_level = atof(o.arg);
		else if (c == 'c')
			opt.flag |= MM_F_OUT_CG | MM_F_CIGAR;
		else if (c == 'D')
			opt.flag |= MM_F_NO_DIAG;
		else if (c == 'P')
			opt.flag |= MM_F_ALL_CHAINS;
		else if (c == 'X')
			opt.flag |= MM_F_ALL_CHAINS | MM_F_NO_DIAG | MM_F_NO_DUAL | MM_F_NO_LJOIN; // -D -P --no-long-join --dual=no
		else if (c == 'a')
			opt.flag |= MM_F_OUT_SAM | MM_F_CIGAR;
		else if (c == 'Q')
			opt.flag |= MM_F_NO_QUAL;
		else if (c == 'Y')
			opt.flag |= MM_F_SOFTCLIP;
		else if (c == 'L')
			opt.flag |= MM_F_LONG_CIGAR;
		else if (c == 'y')
			opt.flag |= MM_F_COPY_COMMENT;
		else if (c == 'T')
			opt.sdust_thres = atoi(o.arg);
		else if (c == 'n')
			opt.min_cnt = atoi(o.arg);
		else if (c == 'm')
			opt.min_chain_score = atoi(o.arg);
		else if (c == 'A')
			opt.a = atoi(o.arg);
		else if (c == 'B')
			opt.b = atoi(o.arg);
		else if (c == 'b')
			opt.transition = atoi(o.arg);
		else if (c == 's')
			opt.min_dp_max = atoi(o.arg);
		else if (c == 'C')
			opt.noncan = atoi(o.arg);
		else if (c == 'I')
			ipt.batch_size = mm_parse_num(o.arg);
		else if (c == 'K')
			opt.mini_batch_size = mm_parse_num(o.arg);
		else if (c == 'e')
			opt.occ_dist = mm_parse_num(o.arg);
		else if (c == 'R')
			rg = o.arg;
		else if (c == 'h')
			help_to_stdout = 1;
		else if (c == '2')
			opt.flag |= MM_F_2_IO_THREADS;
		else if (c == 'j')
			fn_bed_jump = o.arg;
		else if (c == 'J') {
			int t;
			t = atoi(o.arg);
			if (t == 0)
				opt.flag |= MM_F_SPLICE_OLD;
			else if (t == 1)
				opt.flag &= ~MM_F_SPLICE_OLD;
		} else if (c == 'o') {
			if (strcmp(o.arg, "-") != 0) {
				if (freopen(o.arg, "wb", stdout) == NULL) {
					mm_log("[ERROR]\033[1;31m failed to write the output to file '%s'\033[0m: %s", o.arg, strerror(errno));
					exit(1);
				}
			}
		} else if (c == 300)
			ipt.bucket_bits = atoi(o.arg); // --bucket-bits
		else if (c == 302)
			opt.seed = atoi(o.arg); // --seed
		else if (c == 303)
			mm_dbg_flag |= MM_DBG_NO_KALLOC; // --no-kalloc
		else if (c == 304)
			mm_dbg_flag |= MM_DBG_PRINT_QNAME; // --print-qname
		else if (c == 306)
			mm_dbg_flag |= MM_DBG_PRINT_QNAME | MM_DBG_PRINT_SEED, n_threads = 1; // --print-seed
		else if (c == 307) // --max-chain-skip
			max_chain_skip_user_set = 1,
			opt.max_chain_skip = (strcmp(o.arg, "infinity") == 0 || strcmp(o.arg, "inf") == 0) ? INT32_MAX : atoi(o.arg);
		else if (c == 339)
			opt.max_chain_iter = atoi(o.arg); // --max-chain-iter
		else if (c == 308)
			opt.min_ksw_len = atoi(o.arg); // --min-dp-len
		else if (c == 309)
			mm_dbg_flag |= MM_DBG_PRINT_QNAME | MM_DBG_PRINT_ALN_SEQ, n_threads = 1; // --print-aln-seq
		else if (c == 310)
			opt.flag |= MM_F_SPLICE; // --splice
		else if (c == 312)
			opt.flag |= MM_F_NO_LJOIN; // --no-long-join
		else if (c == 317)
			opt.end_bonus = atoi(o.arg); // --end-bonus
		else if (c == 318)
			opt.flag |= MM_F_INDEPEND_SEG; // --no-pairing (deprecated)
		else if (c == 320)
			ipt.flag |= MM_I_NO_SEQ; // --idx-no-seq
		else if (c == 321)
			opt.anchor_ext_shift = atoi(o.arg); // --end-seed-pen
		else if (c == 322)
			opt.flag |= MM_F_FOR_ONLY; // --for-only
		else if (c == 323)
			opt.flag |= MM_F_REV_ONLY; // --rev-only
		else if (c == 327)
			opt.max_clip_ratio = atof(o.arg); // --max-clip-ratio
		else if (c == 328)
			opt.min_mid_occ = atoi(o.arg); // --min-occ-floor
		else if (c == 329)
			opt.flag |= MM_F_OUT_MD; // --MD
		else if (c == 331)
			opt.sc_ambi = atoi(o.arg); // --score-N
		else if (c == 332)
			opt.flag |= MM_F_EQX; // --eqx
		else if (c == 333)
			opt.flag |= MM_F_PAF_NO_HIT; // --paf-no-hit
		else if (c == 334)
			opt.split_prefix = o.arg; // --split-prefix
		else if (c == 335)
			opt.flag |= MM_F_NO_END_FLT; // --no-end-flt
		else if (c == 336)
			opt.flag |= MM_F_HARD_MLEVEL; // --hard-mask-level
		else if (c == 337)
			opt.max_sw_mat = mm_parse_num(o.arg); // --cap-sw-mat
		else if (c == 338)
			opt.max_qlen = mm_parse_num(o.arg); // --max-qlen
		else if (c == 340)
			fn_bed_junc = o.arg; // --junc-bed
		else if (c == 341)
			opt.junc_bonus = atoi(o.arg); // --junc-bonus
		else if (c == 342)
			opt.flag |= MM_F_SAM_HIT_ONLY; // --sam-hit-only
		else if (c == 343)
			opt.chain_gap_scale = atof(o.arg); // --chain-gap-scale
		else if (c == 351)
			opt.chain_skip_scale = atof(o.arg); // --chain-skip-scale
		else if (c == 344)
			alt_list = o.arg; // --alt
		else if (c == 345)
			opt.alt_drop = atof(o.arg); // --alt-drop
		else if (c == 346)
			opt.mask_len = mm_parse_num(o.arg); // --mask-len
		else if (c == 348)
			opt.flag |= MM_F_QSTRAND | MM_F_NO_INV; // --qstrand
		else if (c == 349)
			opt.cap_kalloc = mm_parse_num(o.arg); // --cap-kalloc
		else if (c == 365)
			opt.cap_kalloc_largest = mm_parse_num(o.arg); // --cap-kalloc-largest
		else if (c == 350)
			opt.q_occ_frac = atof(o.arg); // --q-occ-frac
		else if (c == 352)
			mm_dbg_flag |= MM_DBG_PRINT_CHAIN; // --print-chains
		else if (c == 353)
			opt.flag |= MM_F_NO_HASH_NAME; // --no-hash-name
		else if (c == 354)
			opt.flag |= MM_F_SECONDARY_SEQ; // --secondary-seq
		else if (c == 355)
			opt.flag |= MM_F_OUT_DS; // --ds
		else if (c == 356)
			opt.rmq_inner_dist = mm_parse_num(o.arg); // --rmq-inner
		else if (c == 357)
			fn_spsc = o.arg; // --spsc
		else if (c == 360)
			opt.jump_min_match = mm_parse_num(o.arg); // --jump-min-match
		else if (c == 361)
			opt.flag |= MM_F_OUT_JUNC | MM_F_CIGAR; // --write-junc
		else if (c == 362)
			fn_bed_pass1 = o.arg; // --jump-pass1
		else if (c == 501)
			mm_dbg_flag |= MM_DBG_SEED_FREQ; // --dbg-seed-occ
		else if (c == 363)
			spsc_scale = atof(o.arg); // --spsc-scale
		else if (c == 358 || c == 364)
			opt.junc_pen = atoi(o.arg); // --junc-pen or --spsc0
		else if (c == 502)
			debug_log_file = o.arg; // --debug-log
		else if (c == 330) {
			mm_log("[WARNING] \033[1;31m --lj-min-ratio has been deprecated.\033[0m");
		} else if (c == 313) { // --sr
			if (o.arg == 0 || strcmp(o.arg, "dna") == 0) {
				opt.flag |= MM_F_SR;
			} else if (strcmp(o.arg, "rna") == 0) {
				opt.flag |= MM_F_SR_RNA;
			} else if (strcmp(o.arg, "no") == 0) {
				opt.flag &= ~(uint64_t)(MM_F_SR | MM_F_SR_RNA);
			} else if (mm_verbose >= 2) {
				opt.flag |= MM_F_SR;
				fprintf(stderr, "[WARNING]\033[1;31m --sr only takes 'dna' or 'rna'. Invalid values are assumed to be 'dna'.\033[0m\n");
			}
		} else if (c == 314) { // --frag
			yes_or_no(&opt, MM_F_FRAG_MODE, o.longidx, o.arg, 1);
		} else if (c == 315) { // --secondary
			yes_or_no(&opt, MM_F_NO_PRINT_2ND, o.longidx, o.arg, 0);
		} else if (c == 316) { // --cs
			opt.flag |= MM_F_OUT_CS | MM_F_CIGAR;
			if (o.arg == 0 || strcmp(o.arg, "short") == 0) {
				opt.flag &= ~MM_F_OUT_CS_LONG;
			} else if (strcmp(o.arg, "long") == 0) {
				opt.flag |= MM_F_OUT_CS_LONG;
			} else if (strcmp(o.arg, "none") == 0) {
				opt.flag &= ~MM_F_OUT_CS;
			} else if (mm_verbose >= 2) {
				mm_log("[WARNING]\033[1;31m --cs only takes 'short' or 'long'. Invalid values are assumed to be 'short'.\033[0m");
			}
		} else if (c == 319) { // --splice-flank
			yes_or_no(&opt, MM_F_SPLICE_FLANK, o.longidx, o.arg, 1);
		} else if (c == 324) { // --heap-sort
			yes_or_no(&opt, MM_F_HEAP_SORT, o.longidx, o.arg, 1);
		} else if (c == 326) { // --dual
			yes_or_no(&opt, MM_F_NO_DUAL, o.longidx, o.arg, 0);
		} else if (c == 347) { // --rmq
			if (o.arg)
				yes_or_no(&opt, MM_F_RMQ, o.longidx, o.arg, 1);
			else
				opt.flag |= MM_F_RMQ;
		} else if (c == 359) { // --pairing
			if (strcmp(o.arg, "no") == 0)
				opt.flag |= MM_F_INDEPEND_SEG;
			else if (strcmp(o.arg, "weak") == 0)
				opt.flag |= MM_F_WEAK_PAIRING, opt.flag &= ~(uint64_t)MM_F_INDEPEND_SEG;
			else {
				if (strcmp(o.arg, "strong") != 0 && mm_verbose >= 2)
					fprintf(stderr, "[WARNING]\033[1;31m unrecognized argument for --pairing; assuming 'strong'.\033[0m\n");
				opt.flag &= ~(uint64_t)(MM_F_INDEPEND_SEG | MM_F_WEAK_PAIRING);
			}
		} else if (c == 'S') {
			opt.flag |= MM_F_OUT_CS | MM_F_CIGAR | MM_F_OUT_CS_LONG;
			if (mm_verbose >= 2)
				mm_log("[WARNING]\033[1;31m option -S is deprecated and may be removed in future. Please use --cs=long instead.\033[0m");
		} else if (c == 'V') {
			const mm_version_t* v = mm_version_info();
			mm_print("%s", v->full);
			return 0;
		} else if (c == 'r') {
			opt.bw = (int)mm_parse_num2(o.arg, &s);
			if (*s == ',') opt.bw_long = (int)mm_parse_num2(s + 1, &s);
		} else if (c == 'U') {
			opt.min_mid_occ = strtol(o.arg, &s, 10);
			if (*s == ',') opt.max_mid_occ = strtol(s + 1, &s, 10);
		} else if (c == 'f') {
			double x;
			char* p;
			x = strtod(o.arg, &p);
			if (x < 1.0)
				opt.mid_occ_frac = x, opt.mid_occ = 0;
			else
				opt.mid_occ = (int)(x + .499);
			if (*p == ',') opt.max_occ = (int)(strtod(p + 1, &p) + .499);
		} else if (c == 'u') {
			if (*o.arg == 'b')
				opt.flag |= MM_F_SPLICE_FOR | MM_F_SPLICE_REV; // both strands
			else if (*o.arg == 'f')
				opt.flag |= MM_F_SPLICE_FOR, opt.flag &= ~MM_F_SPLICE_REV; // match GT-AG
			else if (*o.arg == 'r')
				opt.flag |= MM_F_SPLICE_REV, opt.flag &= ~MM_F_SPLICE_FOR; // match CT-AC (reverse complement of GT-AG)
			else if (*o.arg == 'n')
				opt.flag &= ~(MM_F_SPLICE_FOR | MM_F_SPLICE_REV); // don't try to match the GT-AG signal
			else {
				mm_log("[ERROR]\033[1;31m unrecognized cDNA direction\033[0m");
				return 1;
			}
		} else if (c == 'z') {
			opt.zdrop = opt.zdrop_inv = strtol(o.arg, &s, 10);
			if (*s == ',') opt.zdrop_inv = strtol(s + 1, &s, 10);
		} else if (c == 'O') {
			opt.q = opt.q2 = strtol(o.arg, &s, 10);
			if (*s == ',') opt.q2 = strtol(s + 1, &s, 10);
		} else if (c == 'E') {
			opt.e = opt.e2 = strtol(o.arg, &s, 10);
			if (*s == ',') opt.e2 = strtol(s + 1, &s, 10);
		}
#if defined(MM_ENABLE_HIP)
		else if (c == 600) {
			opt.flag |= MM_F_GPU_CHAIN; // use gpu for chaining
		} else if (c == 601) {
			/* --gpu-cfg=<path>. Store the argv pointer directly: argv[]
			 * lives until process exit, and opt.gpu_config_file is now a
			 * non-owning pointer. Replaces a previous strcpy of o.arg into
			 * a fixed 1024-byte buffer, which was a stack-buffer overflow
			 * for any --gpu-cfg path longer than 1023 bytes. */
			opt.gpu_config_file = o.arg;
		} else if (c == 602) {
			opt.flag |= MM_F_GPU_ALIGN; // use gpu for alignment
		} else if (c == 605) {
			opt.gpu_flush_threshold = atoi(o.arg);
		} else if (c == 606) {
			opt.gpu_batch_max_align = atoi(o.arg);
		} else if (c == 607) {
			opt.gpu_batch_max_mem = mm_parse_num(o.arg);
		} else if (c == 611) {
			opt.gpu_batch_max_mem_cap = mm_parse_num(o.arg);
		} else if (c == 609) {
			opt.gpu_chain_workers = atoi(o.arg);
		} else if (c == 610) {
			opt.gpu_accum_pool_size = atoi(o.arg);
		} else if (c == 612) {
			/* --multi-gpu or --multi-gpu=<id1,id2,...>
			 * no arg: auto-discover all visible GPUs
			 * arg: explicit subset */
			if (mm_parse_gpu_device_ids_arg(&opt, o.arg, "multi-gpu") != 0)
				return 1;
			gpu_device_selection_set = 1;
		}
#endif
		else if (c == 700) {
			extra_out_dir = o.arg;
		} else if (c == 701) {
			if (o.arg == 0 || strcasecmp(o.arg, "FLATBUF") == 0 || strcasecmp(o.arg, "FLATBUFFERS") == 0) {
				dump_format = MM_DUMP_FLATBUF;
			} else if (strcasecmp(o.arg, "JSON") == 0) {
				dump_format = MM_DUMP_JSON;
			} else {
				mm_log("[ERROR] --dump-intermediates accepts 'FLATBUF' (default) or 'JSON'");
				return 1;
			}
		} else if (c == 702) {
			dump_chained_seeds = 1;
		}
	}
	// Dump system is initialized after mapctx creation (see below)

	// GPU operations are thread-safe: per-thread trbufs + a worker pool sized by
	// the GPU's XCC topology (2 workers/XCC, min 8). --gpu-chain-workers overrides
	// the auto count; use -t N normally alongside --gpu-chain / --gpu-align.
#if defined(MM_ENABLE_HIP)
	if ((opt.flag & MM_F_GPU_CHAIN) && (opt.flag & MM_F_RMQ)) {
		// RMQ chaining (asm5/asm10/asm20 presets, or --rmq) has no GPU kernel;
		// the per-read chain dispatch already falls back to CPU mg_lchain_rmq.
		// Do NOT force max_chain_skip = INT32_MAX here (that is a GPU-DP-only
		// tweak); leaving it at the preset default keeps the CPU RMQ fallback
		// bit-identical to plain CPU mode / upstream.
		mm_log_warn("GPU chaining is not supported with RMQ chaining (asm5/asm10/asm20 or --rmq); chaining runs on the CPU. Output matches CPU mode; --gpu-chain has no effect here.");
	} else if ((opt.flag & MM_F_GPU_CHAIN) && !max_chain_skip_user_set) {
		// Keep legacy GPU-chaining behavior while preserving CPU parity when GPU is not requested.
		opt.max_chain_skip = INT32_MAX;
		mm_log_debug("GPU chaining: defaulting --max-chain-skip to infinity (no early-stop); pass --max-chain-skip N for a finite value.");
	}
	if ((opt.flag & (MM_F_GPU_CHAIN | MM_F_GPU_ALIGN)) && !gpu_device_selection_set) {
		/* CLI default: single-GPU mode on device 0 unless user requests multi-GPU or explicit ids. */
		opt.gpu_device_ids = (int*)malloc(sizeof(int));
		if (!opt.gpu_device_ids) {
			mm_log("[ERROR]\033[1;31m memory allocation failed for default GPU device selection\033[0m");
			return 1;
		}
		opt.gpu_device_ids[0] = 0;
		opt.gpu_num_devices = 1;
	}
#endif
	if (!fnw && !(opt.flag & MM_F_CIGAR))
		ipt.flag |= MM_I_NO_SEQ;
	if (mm_check_opt(&ipt, &opt) < 0)
		return 1;
	if (opt.best_n == 0) {
		mm_log("[WARNING]\033[1;31m changed '-N 0' to '-N %d --secondary=no'.\033[0m", old_best_n);
		opt.best_n = old_best_n, opt.flag |= MM_F_NO_PRINT_2ND;
	}

	// Re-initialize logging if --debug-log was provided
	if (debug_log_file) {
		mm_set_tl_logger(NULL);
		mm_logger_destroy(logger);
		logger = mm_logger_create(1, debug_log_file);
		mm_set_tl_logger(logger);
		s_signal_logger = logger;
	}

	if (argc == o.ind || help_to_stdout) {
#define help_out(fmt, ...) \
	do { \
		mm_logger_t* _lg = mm_tl_logger(); \
		const char* _msg = mm_log_format(fmt, ##__VA_ARGS__); \
		if (help_to_stdout) \
			_lg->print(_lg, _msg); \
		else \
			_lg->log(_lg, _msg); \
	} while (0)
		help_out("Usage: minimap2 [options] <target.fa>|<target.idx> [query.fa] [...]");
		help_out("Options:");
		help_out("  Indexing:");
		help_out("    -H           use homopolymer-compressed k-mer (preferrable for PacBio)");
		help_out("    -k INT       k-mer size (no larger than 28) [%d]", ipt.k);
		help_out("    -w INT       minimizer window size [%d]", ipt.w);
		help_out("    -I NUM       split index for every ~NUM input bases [8G]");
		help_out("    -d FILE      dump index to FILE []");
		help_out("  Mapping:");
		help_out("    -f FLOAT     filter out top FLOAT fraction of repetitive minimizers [%g]", opt.mid_occ_frac);
		help_out("    -g NUM       stop chain enlongation if there are no minimizers in INT-bp [%d]", opt.max_gap);
		help_out("    -G NUM       max intron length (effective with -xsplice; changing -r) [200k]");
		help_out("    -F NUM       max fragment length (effective with -xsr or in the fragment mode) [800]");
		help_out("    -r NUM[,NUM] chaining/alignment bandwidth and long-join bandwidth [%d,%d]", opt.bw, opt.bw_long);
		help_out("    -n INT       minimal number of minimizers on a chain [%d]", opt.min_cnt);
		help_out("    -m INT       minimal chaining score (matching bases minus log gap penalty) [%d]", opt.min_chain_score);
		//		help_out("    -T INT       SDUST threshold; 0 to disable SDUST [%d]", opt.sdust_thres); // TODO: this option is never used; might be buggy
		help_out("    -X           skip self and dual mappings (for the all-vs-all mode)");
		help_out("    -p FLOAT     min secondary-to-primary score ratio [%g]", opt.pri_ratio);
		help_out("    -N INT       retain at most INT secondary alignments [%d]", opt.best_n);
		help_out("  Alignment:");
		help_out("    -A INT       matching score [%d]", opt.a);
		help_out("    -B INT       mismatch penalty (larger value for lower divergence) [%d]", opt.b);
		help_out("    -O INT[,INT] gap open penalty [%d,%d]", opt.q, opt.q2);
		help_out("    -E INT[,INT] gap extension penalty; a k-long gap costs min{O1+k*E1,O2+k*E2} [%d,%d]", opt.e, opt.e2);
		help_out("    -z INT[,INT] Z-drop score and inversion Z-drop score [%d,%d]", opt.zdrop, opt.zdrop_inv);
		help_out("    -s INT       minimal peak DP alignment score [%d]", opt.min_dp_max);
		help_out("    -u CHAR      how to find GT-AG. f:transcript strand, b:both strands, n:don't match GT-AG [n]");
		help_out("    -J INT       splice mode. 0: original minimap2 model; 1: miniprot model [1]");
		help_out("    -j FILE      junctions in BED12 to extend *short* RNA-seq alignment []");
		help_out("  Input/Output:");
		help_out("    -a           output in the SAM format (PAF by default)");
		help_out("    -o FILE      output alignments to FILE [stdout]");
		help_out("    -L           write CIGAR with >65535 ops at the CG tag");
		help_out("    -R STR       SAM read group line in a format like '@RG\\tID:foo\\tSM:bar' []");
		help_out("    -c           output CIGAR in PAF");
		help_out("    --cs[=STR]   output the cs tag; STR is 'short' (if absent) or 'long' [none]");
		help_out("    --ds         output the ds tag, which is an extension to cs");
		help_out("    --MD         output the MD tag");
		help_out("    --eqx        write =/X CIGAR operators");
		help_out("    -Y           use soft clipping for supplementary alignments");
		help_out("    -t INT       number of threads [%d]", n_threads);
		help_out("    -K NUM       minibatch size for mapping [500M]");
		help_out("    -v INT       verbose level [%d]", mm_verbose);
		help_out("    -y           copy FASTA/Q comments to output SAM");
		help_out("    --debug-log FILE  redirect debug output to FILE [stderr]");
		help_out("    --extra-out-dir DIR  output directory for timer stats and intermediate data");
		help_out("    --dump-intermediates[=STR]  dump seeds/chains/alignments; STR is 'FLATBUF' (default) or 'JSON'");
		help_out("    --dump-chained-seeds        include per-chain seed indices in FLATBUF dump (increases size)");
		help_out("    --version    show version number");
#if defined(MM_ENABLE_HIP)
		help_out("  GPU Acceleration:");
		help_out("    --gpu-chain        use GPU for chaining (compatible with -t N; default workers auto-sized by GPU)");
		help_out("    --gpu-align        use GPU for alignment");
		help_out("    --gpu-cfg FILE     GPU configuration JSON file (optional; overrides auto-detected defaults) []");
		help_out("    --gpu-flush-threshold INT  accumulated tasks before GPU flush; 0 = auto-detect [0]");
		help_out("    --gpu-batch-max-align INT  max alignments per GPU batch; 0 = auto-detect [0]");
		help_out("    --gpu-chain-workers INT    number of GPU chain workers; 0 = auto-detect (2/XCC, min 8) [0]");
		help_out("    --gpu-batch-max-mem NUM    GPU memory limit per accumulator for alignment batching; 0 = auto-detect [0]");
		help_out("    --gpu-batch-max-mem-cap NUM  hard cap on auto-derived per-accumulator GPU budget; 0 = compile-time default (4G) [0]");
		help_out("    --gpu-accum-pool-size INT  shared accumulator pool size; 0 = auto-detect [0]");
		help_out("    --multi-gpu[=STR]  use multi-GPU mode; STR(optional)=comma-separated device ids (e.g., 0,1); absent=auto-all []");
#endif
		help_out("  Preset:");
		help_out("    -x STR       preset (always applied before other options; see minimap2.1 for details) []");
		help_out("                 - lr:hq - accurate long reads (error rate <1%%) against a reference genome");
		help_out("                 - splice/splice:hq - spliced alignment for long reads/accurate long reads");
		help_out("                 - splice:sr - spliced alignment for short RNA-seq reads");
		help_out("                 - asm5/asm10/asm20 - asm-to-ref mapping, for ~0.1/1/5%% sequence divergence");
		help_out("                 - sr - short reads against a reference");
		help_out("                 - map-pb/map-hifi/map-ont/map-iclr - CLR/HiFi/Nanopore/ICLR vs reference mapping");
		help_out("                 - ava-pb/ava-ont - PacBio CLR/Nanopore read overlap");
		help_out("\nSee `man minimap2' for detailed description of these and other advanced command-line options.");
#undef help_out
		return help_to_stdout ? 0 : 1;
	}

	if ((opt.flag & MM_F_SR) && argc - o.ind > 3) {
		mm_log("[ERROR] incorrect input: in the sr mode, please specify no more than two query files.");
		return 1;
	}
	idx_rdr = mm_idx_reader_open(argv[o.ind], &ipt, fnw);
	if (idx_rdr == 0) {
		mm_log("[ERROR] failed to open file '%s': %s", argv[o.ind], strerror(errno));
		return 1;
	}
	if (!idx_rdr->is_idx && fnw == 0 && argc - o.ind < 2) {
		mm_log("[ERROR] missing input: please specify a query file to map or option -d to keep the index");
		mm_idx_reader_close(idx_rdr);
		return 1;
	}
	if (opt.best_n == 0 && (opt.flag & MM_F_CIGAR) && mm_verbose >= 2)
		mm_log("[WARNING]\033[1;31m `-N 0' reduces alignment accuracy. Please use --secondary=no to suppress secondary alignments.\033[0m");

		/* Hoist the mapping context to process scope: the GPU chaining context
	 * and core context are created once up front.  mm_mapctx_create
	 * internally calls mm_core_ctx_create and mm_gpu_chain_ctx_create.
	 * On CPU-only paths (or if MM_F_GPU_CHAIN is unset) the gpu_chain
	 * member is NULL and all calls run the CPU chain unchanged. */
#if defined(MM_ENABLE_HIP)
	if ((opt.flag & (MM_F_GPU_CHAIN | MM_F_GPU_ALIGN)) && !mm_gpu_check_available()) {
		mm_log("[ERROR]\033[1;31m --gpu-chain/--gpu-align requested but no AMD GPU detected. "
		       "Check that ROCm is installed and a GPU is visible (HIP_VISIBLE_DEVICES).\033[0m");
		mm_idx_reader_close(idx_rdr);
		return 1;
	}
#endif
	mm_mapctx_t* mapctx = mm_mapctx_create(logger, &opt, n_threads);

	{
		const char* env_dir = getenv("MM2_EXTRA_OUT_DIR");
		if (!extra_out_dir && env_dir && env_dir[0] != '\0')
			extra_out_dir = env_dir;
		if (dump_format != MM_DUMP_NONE && !extra_out_dir) {
			mm_log("[ERROR] --dump-intermediates requires --extra-out-dir");
			mm_mapctx_destroy(mapctx);
			return 1;
		}
		if (extra_out_dir && dump_format != MM_DUMP_NONE) {
			mm_core_ctx_set_dump(mapctx->core_ctx, dump_format, extra_out_dir);
			mapctx->core_ctx->dump_cfg.dump_chained_seeds = dump_chained_seeds;
			mm_dump_init(mapctx->core_ctx);
		}
	}

	while ((mi = mm_idx_reader_read(idx_rdr, n_threads)) != 0) {
		int ret;
		if ((opt.flag & MM_F_CIGAR) && (mi->flag & MM_I_NO_SEQ)) {
			mm_log("[ERROR] the prebuilt index doesn't contain sequences.");
			mm_idx_destroy(mi);
			mm_idx_reader_close(idx_rdr);
			return 1;
		}
		if ((opt.flag & MM_F_OUT_SAM) && idx_rdr->n_parts == 1) {
			if (mm_idx_reader_eof(idx_rdr)) {
				if (opt.split_prefix == 0)
					ret = mm_write_sam_hdr(mi, rg, mm_version(), argc, argv);
				else
					ret = mm_write_sam_hdr(0, rg, mm_version(), argc, argv);
			} else {
				ret = mm_write_sam_hdr(0, rg, mm_version(), argc, argv);
				if (opt.split_prefix == 0 && mm_verbose >= 2)
					mm_log("[WARNING]\033[1;31m For a multi-part index, no @SQ lines will be outputted. Please use --split-prefix.\033[0m");
			}
			if (ret != 0) {
				mm_idx_destroy(mi);
				mm_idx_reader_close(idx_rdr);
				return 1;
			}
		}
		if (mm_verbose >= 3)
			mm_log("[M::%s::%.3f*%.2f] loaded/built the index for %d target sequence(s)",
			    __func__, realtime() - mm_realtime0, cputime() / (realtime() - mm_realtime0), mi->n_seq);
		if (argc != o.ind + 1) mm_mapopt_update(&opt, mi);
		if (mm_verbose >= 3) mm_idx_stat(mi);
		if (fn_bed_junc) {
			mm_idx_bed_read(mi, fn_bed_junc, 1);
			if (mi->I == 0 && mm_verbose >= 2)
				fprintf(stderr, "[WARNING] failed to load the junction BED file\n");
		}
		if (fn_bed_jump) {
			mm_idx_jjump_read(mi, fn_bed_jump, MM_JUNC_ANNO, -1);
			if (mi->J == 0 && mm_verbose >= 2)
				fprintf(stderr, "[WARNING] failed to load the jump BED file\n");
		}
		if (fn_bed_pass1) {
			mm_idx_jjump_read(mi, fn_bed_pass1, MM_JUNC_MISC, 5);
			if (mi->J == 0 && mm_verbose >= 2)
				fprintf(stderr, "[WARNING] failed to load the pass-1 jump BED file\n");
		}
		if (fn_spsc) {
			mm_idx_spsc_read2(mi, fn_spsc, mm_max_spsc_bonus(&opt), spsc_scale);
			if (mi->spsc == 0 && mm_verbose >= 2)
				fprintf(stderr, "[WARNING] failed to load the splice score file\n");
		}
		if (alt_list) mm_idx_alt_read(mi, alt_list);
		if (argc - (o.ind + 1) == 0) {
			mm_idx_destroy(mi);
			continue; // no query files
		}
		/* The hoisted mapctx is reused for every index part now;
		 * no per-part HIP init/teardown. */
		ret = 0;
		if (!(opt.flag & MM_F_FRAG_MODE)) {
			for (i = o.ind + 1; i < argc; ++i) {
				ret = mm_map_file_ctx(mi, argv[i], &opt, n_threads, mapctx);
				if (ret < 0) break;
			}
		} else {
			ret = mm_map_file_frag_ctx(mi, argc - (o.ind + 1), (const char**)&argv[o.ind + 1], &opt, n_threads, mapctx);
		}
		mm_idx_destroy(mi);
		if (ret < 0) {
			mm_log("ERROR: failed to map the query file");
			exit(EXIT_FAILURE);
		}
	}
	n_parts = idx_rdr->n_parts;
	mm_idx_reader_close(idx_rdr);

	if (opt.split_prefix)
		mm_split_merge_ctx(argc - (o.ind + 1), (const char**)&argv[o.ind + 1], &opt, n_parts, mapctx);

	mm_timer_stats_t* ts = mapctx->timer_stats;

	if (fflush(stdout) == EOF) {
		mm_log("[ERROR] failed to write the results: %s", strerror(errno));
		exit(EXIT_FAILURE);
	}

	{
		double wall_time = realtime() - mm_realtime0;
		double cpu_t = cputime();
		double peak_rss = peakrss() / 1024.0 / 1024.0 / 1024.0;
		if (extra_out_dir && ts)
			mm_dump_write_timer_json(extra_out_dir, ts, wall_time, cpu_t, peak_rss, n_threads);
	}

	if (mm_verbose >= 3) {
		mm_log("[M::%s] Version: %s", __func__, mm_version());
		// Log CMD with all arguments
		char cmd_buf[4096];
		int pos = 0;
		for (i = 0; i < argc && pos < (int)sizeof(cmd_buf) - 1; ++i)
			pos += snprintf(cmd_buf + pos, sizeof(cmd_buf) - pos, "%s%s", i ? " " : "", argv[i]);
		mm_log("[M::%s] CMD: %s", __func__, cmd_buf);
		const double wall_time = realtime() - mm_realtime0;
		mm_log("[M::%s] Real time: %.3f sec; CPU: %.3f sec; Peak RSS: %.3f GB", __func__, wall_time, cputime(), peakrss() / 1024.0 / 1024.0 / 1024.0);
		if (ts) {
			const double total_sum = ts->seed_sum + ts->chain_sum + ts->align_sum;
			const double inv = 1.0 / (double)n_threads;
			const double stage_per_thread_equiv = total_sum * inv;
			const double overhead = wall_time - stage_per_thread_equiv;
			const double per_batch_thread = (ts->batch_count > 0) ? (total_sum / ((double)n_threads * (double)ts->batch_count)) : 0.0;
			const double per_query = (ts->query_count > 0) ? (total_sum / (double)ts->query_count) : 0.0;
			mm_log_debug("----------------------------------------------------");
			mm_log_debug("          Sum (sec)  Avg/thread (sec)");
			mm_log_debug("----------------------------------------------------");
			mm_log_debug("Seed    = %11.3f %11.3f", ts->seed_sum, ts->seed_sum * inv);
			mm_log_debug("Chain   = %11.3f %11.3f", ts->chain_sum, ts->chain_sum * inv);
			mm_log_debug("Align   = %11.3f %11.3f", ts->align_sum, ts->align_sum * inv);
			mm_log_debug("----------------------------------------------------");
			mm_log_debug("Per-call (per-read) statistics:");
			mm_log_debug("          Min (sec)    Max (sec)    Avg (sec)    Count");
			mm_log_debug("Seed    = %11.6f %11.6f %11.6f %llu", ts->seed_call_min, ts->seed_call_max, ts->seed_call_avg, (unsigned long long)ts->seed_call_count);
			mm_log_debug("Chain   = %11.6f %11.6f %11.6f %llu", ts->chain_call_min, ts->chain_call_max, ts->chain_call_avg, (unsigned long long)ts->chain_call_count);
			mm_log_debug("Align   = %11.6f %11.6f %11.6f %llu", ts->align_call_min, ts->align_call_max, ts->align_call_avg, (unsigned long long)ts->align_call_count);
			mm_log_debug("----------------------------------------------------");
			mm_log_debug("Batches processed = %llu", (unsigned long long)ts->batch_count);
			mm_log_debug("Queries processed = %llu", (unsigned long long)ts->query_count);
			mm_log_debug("Pool resets = %llu", (unsigned long long)ts->pool_reset_count);
			mm_log_debug("Avg total per thread per batch = %.3f secs", per_batch_thread);
			mm_log_debug("Avg total per query = %.6f secs", per_query);
			mm_log_debug("Approx non-stage overhead (wall - stage/thread) = %.3f secs", overhead);
			mm_log_debug("Total (seed + chain + align) for %d thread(s) = %.3f secs", n_threads, total_sum);
		}
	}

	// Shutdown logging system and flush buffers
	mm_dump_cleanup(mapctx->core_ctx);
	mm_mapctx_destroy(mapctx);
	s_signal_logger = NULL;
	mm_logger_destroy(logger);

	return 0;
}
