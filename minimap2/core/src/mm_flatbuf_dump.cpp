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

#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <new>

#include "mm_flatbuf_dump.h"
#include "mm_dump_util.h"
#include "chain_read.h"
#include "mm_log.h"

#include "flatbuffers/flatbuffers.h"
#include "generated/query_dump_generated.h"

struct mm_flatbuf_state_s {
	std::string base_dir;
	bool initialized;
	bool dump_chained_seeds;
};

namespace
{

struct thread_buf_t {
	std::vector<mm2::dump::Anchor> seeds;
	std::vector<uint32_t> chained_seed_indices;

	struct chain_info_t {
		int32_t score;
		uint32_t seed_offset;
		int32_t length;
		int32_t first_qpos;
		int32_t first_tpos;
		int32_t last_qpos;
		int32_t last_tpos;
		int32_t rev;
		int32_t rid;
		std::string rname;
	};
	std::vector<chain_info_t> chains;

	int32_t qlen_sum;
	int32_t n_seg;
	int32_t rep_len;
	int32_t n_mini_pos;
	int32_t frag_gap;
	int32_t k;
	int32_t w;

	void clear()
	{
		seeds.clear();
		chained_seed_indices.clear();
		chains.clear();
		qlen_sum = n_seg = rep_len = n_mini_pos = frag_gap = k = w = 0;
	}
};

// Per-query buffer map: in the batched pipeline, chaining and alignment
// happen in separate loops, so a single thread-local buffer gets
// overwritten. This map preserves each query's chain data until its
// alignment dump writes the file and erases the entry.
thread_local std::unordered_map<std::string, thread_buf_t> tl_qbufs;

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

} // anonymous namespace

extern "C" {

mm_flatbuf_state_t* mm_flatbuf_state_create(const char* dir)
{
	mm_flatbuf_state_t* st = new (std::nothrow) mm_flatbuf_state_t();
	if (!st) return NULL;
	st->base_dir = dir ? dir : "./mm2_flatbuf_dump";
	st->initialized = false;
	st->dump_chained_seeds = false;
	return st;
}

void mm_flatbuf_state_destroy(mm_flatbuf_state_t* st)
{
	if (!st) return;
	mm_flatbuf_dump_cleanup(st);
	delete st;
}

void mm_flatbuf_state_set_dump_chained_seeds(mm_flatbuf_state_t* st, int enable)
{
	if (st) st->dump_chained_seeds = (enable != 0);
}

int mm_flatbuf_dump_init(mm_flatbuf_state_t* st)
{
	if (!st) return -1;

	if (mm_dump_mkdir_p(st->base_dir) != 0) {
		mm_log_warn("Failed to create FlatBuffers dump directory: {}", st->base_dir.c_str());
		return -1;
	}

	mm_log_debug("FlatBuffers dump enabled, output dir: {}", st->base_dir.c_str());
	st->initialized = true;
	return 0;
}

void mm_flatbuf_dump_cleanup(mm_flatbuf_state_t* st)
{
	if (st) st->initialized = false;
}

int mm_flatbuf_dump_seeds(mm_flatbuf_state_t* st, const char* qname,
    const chain_read_t* read, const mm_idx_t* mi)
{
	if (!st || !st->initialized || !qname || !read) return 0;

	std::string key(qname);
	thread_buf_t& buf = tl_qbufs[key];
	buf.clear();

	buf.qlen_sum = read->seq.qlen_sum;
	buf.n_seg = read->n_seg;
	buf.rep_len = read->rep_len;
	buf.n_mini_pos = read->n_mini_pos;
	buf.frag_gap = read->frag_gap;
	buf.k = mi ? mi->k : 0;
	buf.w = mi ? mi->w : 0;

	if (read->a && read->n > 0) {
		buf.seeds.reserve(static_cast<size_t>(read->n));
		for (int64_t i = 0; i < read->n; i++) {
			buf.seeds.emplace_back(read->a[i].x, read->a[i].y);
		}
	}

	return 0;
}

int mm_flatbuf_dump_chains(mm_flatbuf_state_t* st, const char* qname,
    const chain_read_t* read, const mm_idx_t* mi)
{
	if (!st || !st->initialized || !qname || !read) return 0;

	std::string key(qname);
	thread_buf_t& buf = tl_qbufs[key];
	buf.chained_seed_indices.clear();
	buf.chains.clear();

	buf.qlen_sum = read->seq.qlen_sum;
	buf.n_seg = read->n_seg;
	buf.rep_len = read->rep_len;
	buf.n_mini_pos = read->n_mini_pos;
	buf.frag_gap = read->frag_gap;
	buf.k = mi ? mi->k : 0;
	buf.w = mi ? mi->w : 0;

	int64_t actual_n_anchors = 0;
	if (read->u && read->n_u > 0) {
		for (int i = 0; i < read->n_u; i++) {
			actual_n_anchors += static_cast<int32_t>(read->u[i] & 0xFFFFFFFF);
		}
	}

	// Build index lookup only if chained seed tracking is enabled
	if (st->dump_chained_seeds && read->a && actual_n_anchors > 0 && !buf.seeds.empty()) {
		struct PairHash {
			size_t operator()(const std::pair<uint64_t, uint64_t>& p) const
			{
				return std::hash<uint64_t>()(p.first) ^ (std::hash<uint64_t>()(p.second) << 1);
			}
		};
		std::unordered_map<std::pair<uint64_t, uint64_t>, uint32_t, PairHash> seed_idx_map;
		seed_idx_map.reserve(buf.seeds.size());
		for (size_t i = 0; i < buf.seeds.size(); i++) {
			auto key_pair = std::make_pair(buf.seeds[i].x(), buf.seeds[i].y());
			seed_idx_map.emplace(key_pair, static_cast<uint32_t>(i));
		}

		buf.chained_seed_indices.reserve(static_cast<size_t>(actual_n_anchors));
		for (int64_t i = 0; i < actual_n_anchors; i++) {
			auto it = seed_idx_map.find(std::make_pair(read->a[i].x, read->a[i].y));
			uint32_t idx = (it != seed_idx_map.end()) ? it->second : UINT32_MAX;
			buf.chained_seed_indices.push_back(idx);
		}
	}

	int64_t anchor_offset = 0;
	if (read->u && read->n_u > 0 && read->a) {
		buf.chains.reserve(static_cast<size_t>(read->n_u));
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
			if (chain_len <= 0) continue;

			thread_buf_t::chain_info_t ci;
			ci.score = chain_score;
			ci.seed_offset = static_cast<uint32_t>(anchor_offset);
			ci.length = chain_len;
			ci.first_qpos = static_cast<int32_t>(read->a[anchor_offset].y & 0xFFFFFFFF);
			ci.first_tpos = static_cast<int32_t>(read->a[anchor_offset].x);
			ci.last_qpos = static_cast<int32_t>(read->a[anchor_offset + chain_len - 1].y & 0xFFFFFFFF);
			ci.last_tpos = static_cast<int32_t>(read->a[anchor_offset + chain_len - 1].x);
			ci.rev = static_cast<int32_t>(read->a[anchor_offset].x >> 63);
			ci.rid = static_cast<int32_t>((read->a[anchor_offset].x << 1) >> 33);
			if (mi && static_cast<uint32_t>(ci.rid) < mi->n_seq)
				ci.rname = mi->seq[ci.rid].name;

			buf.chains.push_back(std::move(ci));
			anchor_offset += chain_len;
		}
	}

	return 0;
}

int mm_flatbuf_dump_alignments(mm_flatbuf_state_t* st, const char* qname,
    int n_segs, int qlen_sum, const int* n_regs, mm_reg1_t** regs,
    const mm_idx_t* mi, int dump_chained_seeds)
{
	(void)dump_chained_seeds; // use st->dump_chained_seeds instead
	if (!st || !st->initialized || !qname || !n_regs || !regs) return 0;

	std::string key(qname);
	auto it = tl_qbufs.find(key);
	thread_buf_t empty_buf;
	thread_buf_t& buf = (it != tl_qbufs.end()) ? it->second : empty_buf;
	flatbuffers::FlatBufferBuilder fbb(1024 * 1024);

	auto name = fbb.CreateString(qname);

	auto seeds_vec = fbb.CreateVectorOfStructs(buf.seeds.data(), buf.seeds.size());

	flatbuffers::Offset<flatbuffers::Vector<uint32_t>> chained_vec = 0;
	if (st->dump_chained_seeds && !buf.chained_seed_indices.empty()) {
		chained_vec = fbb.CreateVector(buf.chained_seed_indices);
	}

	std::vector<flatbuffers::Offset<mm2::dump::Chain>> chain_offsets;
	chain_offsets.reserve(buf.chains.size());
	for (const auto& ci : buf.chains) {
		auto rname_off = ci.rname.empty()
		    ? flatbuffers::Offset<flatbuffers::String>(0)
		    : fbb.CreateString(ci.rname);
		chain_offsets.push_back(mm2::dump::CreateChain(fbb,
		    ci.score, ci.seed_offset, ci.length,
		    ci.first_qpos, ci.first_tpos,
		    ci.last_qpos, ci.last_tpos,
		    ci.rev, ci.rid, rname_off));
	}
	auto chains_vec = fbb.CreateVector(chain_offsets);

	std::vector<flatbuffers::Offset<mm2::dump::Alignment>> aln_offsets;
	for (int s = 0; s < n_segs; s++) {
		if (!regs[s]) continue;
		for (int a = 0; a < n_regs[s]; a++) {
			mm_reg1_t* r = &regs[s][a];

			int32_t chain_idx = -1;
			// Match alignment to chain using score0 (initial chain score), rid, and strand.
			// We cannot use r->as because mm_squeeze_a() modifies it during alignment.
			for (size_t ci = 0; ci < buf.chains.size(); ci++) {
				const auto& ch = buf.chains[ci];
				if (r->score0 == ch.score &&
				    r->rid == ch.rid &&
				    static_cast<int32_t>(r->rev) == ch.rev) {
					chain_idx = static_cast<int32_t>(ci);
					break;
				}
			}

			flatbuffers::Offset<flatbuffers::String> rname_off = 0;
			if (mi && r->rid >= 0 && static_cast<uint32_t>(r->rid) < mi->n_seq)
				rname_off = fbb.CreateString(mi->seq[r->rid].name);

			flatbuffers::Offset<flatbuffers::String> cigar_off = 0;
			if (r->p && r->p->n_cigar > 0) {
				std::string cigar_str = mm_dump_build_cigar(r->p->cigar, r->p->n_cigar);
				cigar_off = fbb.CreateString(cigar_str);
			}

			aln_offsets.push_back(mm2::dump::CreateAlignment(fbb,
			    static_cast<int32_t>(s),
			    r->rid,
			    r->score,
			    r->score0,
			    r->p ? r->p->dp_score : 0,
			    r->qs, r->qe,
			    r->rs, r->re,
			    static_cast<int32_t>(r->mapq),
			    static_cast<int32_t>(r->rev),
			    r->mlen, r->blen,
			    r->n_sub,
			    static_cast<int32_t>(r->sam_pri),
			    r->p ? static_cast<int32_t>(r->p->n_cigar) : 0,
			    r->p ? r->p->dp_max : 0,
			    r->p ? r->p->dp_max2 : 0,
			    r->p ? static_cast<int32_t>(r->p->n_ambi) : 0,
			    chain_idx,
			    static_cast<int32_t>(r->split),
			    r->parent,
			    static_cast<int32_t>(r->inv),
			    r->subsc,
			    r->as,
			    rname_off,
			    cigar_off));
		}
	}
	auto aligns_vec = fbb.CreateVector(aln_offsets);

	mm2::dump::QueryMeta meta(buf.qlen_sum, buf.n_seg, buf.rep_len,
	    buf.n_mini_pos, buf.frag_gap, buf.k, buf.w);

	auto dump = mm2::dump::CreateQueryDump(fbb, name, &meta,
	    seeds_vec, chained_vec, chains_vec, aligns_vec);
	mm2::dump::FinishQueryDumpBuffer(fbb, dump);

	std::string safe_name = sanitize_filename(qname);
	std::string filepath = st->base_dir + "/" + safe_name + ".bin";

	FILE* f = fopen(filepath.c_str(), "wb");
	if (!f) {
		mm_log_warn("Failed to open FlatBuffers dump file: {}", filepath.c_str());
		tl_qbufs.erase(key);
		return -1;
	}

	size_t written = fwrite(fbb.GetBufferPointer(), 1, fbb.GetSize(), f);
	fclose(f);
	tl_qbufs.erase(key);

	if (written != fbb.GetSize()) {
		mm_log_warn("Incomplete write to FlatBuffers dump file: {}", filepath.c_str());
		return -1;
	}

	return 0;
}

} // extern "C"
