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

#include <cstdlib>
#include <cstring>
#include <string>
#include <new>

#include "mm_json_dump.h"
#include "mm_dump_util.h"
#include "chain_read.h"
#include "mm_log.h"
#include "cJSON.h"

struct mm_json_state_s {
	std::string base_dir;
	bool initialized;
};

namespace
{

std::string sanitize_filename(const char* src)
{
	std::string result;
	result.reserve(256);

	for (const char* p = src; *p && result.size() < 255; p++) {
		char c = *p;
		if (c == '/' || c == '\\' || c == ':' || c == '*' ||
		    c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
			result += '_';
		} else {
			result += c;
		}
	}
	return result;
}

cJSON* json_anchor(uint64_t x, uint64_t y)
{
	cJSON* anchor = cJSON_CreateObject();
	mm_dump_json_add_uint64(anchor, "x", x);
	mm_dump_json_add_uint64(anchor, "y", y);
	return anchor;
}

cJSON* json_chain(int32_t score, int32_t length,
    uint32_t first_qpos, uint32_t first_tpos,
    uint32_t last_qpos, uint32_t last_tpos)
{
	cJSON* chain = cJSON_CreateObject();
	cJSON_AddNumberToObject(chain, "score", score);
	cJSON_AddNumberToObject(chain, "length", length);
	cJSON_AddNumberToObject(chain, "first_qpos", first_qpos);
	cJSON_AddNumberToObject(chain, "first_tpos", first_tpos);
	cJSON_AddNumberToObject(chain, "last_qpos", last_qpos);
	cJSON_AddNumberToObject(chain, "last_tpos", last_tpos);
	return chain;
}

} // anonymous namespace

extern "C" {

mm_json_state_t* mm_json_state_create(const char* dir)
{
	mm_json_state_t* st = new (std::nothrow) mm_json_state_t();
	if (!st) return NULL;
	st->base_dir = dir ? dir : "./mm2_json_dump";
	st->initialized = false;
	return st;
}

void mm_json_state_destroy(mm_json_state_t* st)
{
	if (!st) return;
	mm_json_dump_cleanup(st);
	delete st;
}

int mm_json_dump_init(mm_json_state_t* st)
{
	if (!st) return -1;

	const std::string& base_dir = st->base_dir;

	if (mm_dump_mkdir_p(base_dir) != 0) {
		mm_log_warn("Failed to create JSON dump directory: {}", base_dir.c_str());
		return -1;
	}

	if (mm_dump_mkdir_p(base_dir + "/seeds") != 0 ||
	    mm_dump_mkdir_p(base_dir + "/chains") != 0 ||
	    mm_dump_mkdir_p(base_dir + "/alignments") != 0 ||
	    mm_dump_mkdir_p(base_dir + "/stats") != 0) {
		mm_log_warn("Failed to create JSON dump subdirectories in: {}", base_dir.c_str());
		return -1;
	}

	mm_log_debug("JSON dump enabled, output dir: {}", base_dir.c_str());
	st->initialized = true;
	return 0;
}

void mm_json_dump_cleanup(mm_json_state_t* st)
{
	if (st) st->initialized = false;
}

int mm_json_dump_seeds(mm_json_state_t* st, const char* qname,
    const chain_read_t* read, const mm_idx_t* mi)
{
	if (!st || !st->initialized || !qname || !read) return 0;

	cJSON* root = cJSON_CreateObject();
	if (!root) return -1;

	cJSON_AddStringToObject(root, "name", qname);
	cJSON_AddNumberToObject(root, "length", read->seq.qlen_sum);
	cJSON_AddNumberToObject(root, "n_seg", read->n_seg);
	cJSON_AddNumberToObject(root, "n_seeds", static_cast<double>(read->n));
	cJSON_AddNumberToObject(root, "n_mini_pos", read->n_mini_pos);
	cJSON_AddNumberToObject(root, "rep_len", read->rep_len);
	cJSON_AddNumberToObject(root, "n_minimizers", read->n_minimizers);

	if (read->qlens && read->n_seg > 0) {
		cJSON* qlens_arr = cJSON_CreateArray();
		for (int i = 0; i < read->n_seg; i++) {
			cJSON_AddItemToArray(qlens_arr, cJSON_CreateNumber(read->qlens[i]));
		}
		cJSON_AddItemToObject(root, "qlens", qlens_arr);
	}

	if (read->qseqs && read->n_seg > 0) {
		cJSON* seqs_arr = cJSON_CreateArray();
		for (int i = 0; i < read->n_seg && i < 3; i++) {
			if (read->qseqs[i]) {
				char prefix[51];
				int len = read->qlens ? read->qlens[i] : 0;
				if (len < 0) len = 0;
				if (len > 50) len = 50;
				memcpy(prefix, read->qseqs[i], len);
				prefix[len] = '\0';
				cJSON_AddItemToArray(seqs_arr, cJSON_CreateString(prefix));
			}
		}
		cJSON_AddItemToObject(root, "seq_prefixes", seqs_arr);
	}

	if (mi) {
		cJSON* idx_params = cJSON_CreateObject();
		cJSON_AddNumberToObject(idx_params, "k", mi->k);
		cJSON_AddNumberToObject(idx_params, "w", mi->w);
		cJSON_AddItemToObject(root, "index_params", idx_params);
	}

	cJSON* seeds_arr = cJSON_CreateArray();
	if (read->a && read->n > 0) {
		for (int64_t i = 0; i < read->n; i++) {
			cJSON_AddItemToArray(seeds_arr, json_anchor(read->a[i].x, read->a[i].y));
		}
	}
	cJSON_AddItemToObject(root, "seeds", seeds_arr);

	std::string safe_name = sanitize_filename(qname);
	std::string filepath = st->base_dir + "/seeds/" + safe_name + ".json";

	int ret = mm_dump_write_json_file(filepath, root);
	cJSON_Delete(root);
	return ret;
}

int mm_json_dump_chains(mm_json_state_t* st, const char* qname,
    const chain_read_t* read, const mm_idx_t* mi)
{
	if (!st || !st->initialized || !qname || !read) return 0;

	cJSON* root = cJSON_CreateObject();
	if (!root) return -1;

	cJSON_AddStringToObject(root, "name", qname);
	cJSON_AddNumberToObject(root, "length", read->seq.qlen_sum);

	int64_t actual_n_anchors = 0;
	if (read->u && read->n_u > 0) {
		for (int i = 0; i < read->n_u; i++) {
			actual_n_anchors += static_cast<int32_t>(read->u[i] & 0xFFFFFFFF);
		}
	}

	cJSON_AddNumberToObject(root, "n_seeds", static_cast<double>(actual_n_anchors));
	cJSON_AddNumberToObject(root, "n_chains", read->n_u);
	cJSON_AddNumberToObject(root, "n_seg", read->n_seg);
	cJSON_AddNumberToObject(root, "frag_gap", read->frag_gap);
	cJSON_AddNumberToObject(root, "rep_len", read->rep_len);
	cJSON_AddNumberToObject(root, "n_mini_pos", read->n_mini_pos);

	if (mi) {
		cJSON* idx_params = cJSON_CreateObject();
		cJSON_AddNumberToObject(idx_params, "k", mi->k);
		cJSON_AddNumberToObject(idx_params, "w", mi->w);
		cJSON_AddItemToObject(root, "index_params", idx_params);
	}

	cJSON* seeds_arr = cJSON_CreateArray();
	if (read->a && actual_n_anchors > 0) {
		for (int64_t i = 0; i < actual_n_anchors; i++) {
			cJSON_AddItemToArray(seeds_arr, json_anchor(read->a[i].x, read->a[i].y));
		}
	}
	cJSON_AddItemToObject(root, "seeds", seeds_arr);

	cJSON* chains_arr = cJSON_CreateArray();
	int64_t anchor_offset = 0;

	if (read->u && read->n_u > 0 && read->a) {
		for (int i = 0; i < read->n_u; i++) {
			int32_t chain_len = static_cast<int32_t>(read->u[i] & 0xFFFFFFFF);
			int32_t chain_score = static_cast<int32_t>(read->u[i] >> 32);

			if (chain_len <= 0 || anchor_offset < 0 || anchor_offset >= actual_n_anchors) {
				anchor_offset += chain_len;
				continue;
			}
			if (anchor_offset + chain_len > actual_n_anchors) {
				chain_len = static_cast<int32_t>(actual_n_anchors - anchor_offset);
			}
			if (chain_len <= 0) {
				continue;
			}

			uint32_t first_qpos = static_cast<uint32_t>(read->a[anchor_offset].y & 0xFFFFFFFF);
			uint32_t first_tpos = static_cast<uint32_t>(read->a[anchor_offset].x);
			uint32_t last_qpos = static_cast<uint32_t>(read->a[anchor_offset + chain_len - 1].y & 0xFFFFFFFF);
			uint32_t last_tpos = static_cast<uint32_t>(read->a[anchor_offset + chain_len - 1].x);

			cJSON* chain = json_chain(chain_score, chain_len,
			    first_qpos, first_tpos, last_qpos, last_tpos);

			int rev = read->a[anchor_offset].x >> 63;
			uint32_t rid = (read->a[anchor_offset].x << 1) >> 33;
			cJSON_AddNumberToObject(chain, "rev", rev);
			cJSON_AddNumberToObject(chain, "rid", rid);
			if (mi && rid < mi->n_seq) {
				cJSON_AddStringToObject(chain, "rname", mi->seq[rid].name);
			}

			cJSON_AddItemToArray(chains_arr, chain);
			anchor_offset += chain_len;
		}
	}
	cJSON_AddItemToObject(root, "chains", chains_arr);

	std::string safe_name = sanitize_filename(qname);
	std::string filepath = st->base_dir + "/chains/" + safe_name + ".json";

	int ret = mm_dump_write_json_file(filepath, root);
	cJSON_Delete(root);
	return ret;
}

int mm_json_dump_alignments(mm_json_state_t* st, const char* qname,
    int n_segs, int qlen_sum, const int* n_regs, mm_reg1_t** regs,
    const mm_idx_t* mi)
{
	if (!st || !st->initialized || !qname || !n_regs || !regs) return 0;

	cJSON* root = cJSON_CreateObject();
	if (!root) return -1;

	cJSON_AddStringToObject(root, "name", qname);
	cJSON_AddNumberToObject(root, "length", qlen_sum);

	int total_alignments = 0;
	for (int s = 0; s < n_segs; s++) {
		if (regs[s]) total_alignments += n_regs[s];
	}
	cJSON_AddNumberToObject(root, "n_alignments", total_alignments);
	cJSON_AddNumberToObject(root, "n_segs", n_segs);

	if (mi) {
		cJSON* idx_params = cJSON_CreateObject();
		cJSON_AddNumberToObject(idx_params, "k", mi->k);
		cJSON_AddNumberToObject(idx_params, "w", mi->w);
		cJSON_AddItemToObject(root, "index_params", idx_params);
	}

	cJSON* alignments_arr = cJSON_CreateArray();

	for (int s = 0; s < n_segs; s++) {
		if (regs[s]) {
			for (int a = 0; a < n_regs[s]; a++) {
				mm_reg1_t* r = &regs[s][a];
				cJSON* aln = cJSON_CreateObject();

				cJSON_AddNumberToObject(aln, "rid", r->rid);
				cJSON_AddNumberToObject(aln, "score", r->score);
				cJSON_AddNumberToObject(aln, "dp_score", r->p ? r->p->dp_score : 0);
				cJSON_AddNumberToObject(aln, "qs", r->qs);
				cJSON_AddNumberToObject(aln, "qe", r->qe);
				cJSON_AddNumberToObject(aln, "rs", r->rs);
				cJSON_AddNumberToObject(aln, "re", r->re);
				cJSON_AddNumberToObject(aln, "mapq", r->mapq);
				cJSON_AddNumberToObject(aln, "rev", r->rev);
				cJSON_AddNumberToObject(aln, "n_cigar_ops", r->p ? r->p->n_cigar : 0);

				if (r->p && r->p->n_cigar > 0) {
					std::string cigar_str = mm_dump_build_cigar(r->p->cigar, r->p->n_cigar);
					cJSON_AddStringToObject(aln, "cigar", cigar_str.c_str());
				}

				if (mi && r->rid >= 0 && static_cast<uint32_t>(r->rid) < mi->n_seq) {
					cJSON_AddStringToObject(aln, "rname", mi->seq[r->rid].name);
				}

				cJSON_AddNumberToObject(aln, "seg_id", s);
				cJSON_AddNumberToObject(aln, "score0", r->score0);
				cJSON_AddNumberToObject(aln, "inv", r->inv);
				cJSON_AddNumberToObject(aln, "parent", r->parent);
				cJSON_AddNumberToObject(aln, "subsc", r->subsc);
				cJSON_AddNumberToObject(aln, "mlen", r->mlen);
				cJSON_AddNumberToObject(aln, "blen", r->blen);
				cJSON_AddNumberToObject(aln, "n_sub", r->n_sub);
				cJSON_AddNumberToObject(aln, "sam_pri", r->sam_pri);

				if (r->p) {
					cJSON_AddNumberToObject(aln, "dp_max", r->p->dp_max);
					cJSON_AddNumberToObject(aln, "dp_max2", r->p->dp_max2);
					cJSON_AddNumberToObject(aln, "n_ambi", r->p->n_ambi);
				}

				cJSON_AddItemToArray(alignments_arr, aln);
			}
		}
	}
	cJSON_AddItemToObject(root, "alignments", alignments_arr);

	std::string safe_name = sanitize_filename(qname);
	std::string filepath = st->base_dir + "/alignments/" + safe_name + ".json";

	int ret = mm_dump_write_json_file(filepath, root);
	cJSON_Delete(root);
	return ret;
}

} // extern "C"
