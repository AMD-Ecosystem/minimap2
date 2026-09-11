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

#ifndef MM_CORE_CTX_H
#define MM_CORE_CTX_H

#include <limits.h>
#include "mm_log.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum mm_dump_format_e {
	MM_DUMP_NONE = 0,
	MM_DUMP_JSON = 1,
	MM_DUMP_FLATBUF = 2
} mm_dump_format_t;

typedef struct mm_dump_cfg_s {
	mm_dump_format_t format;
	int enabled;
	int dump_chained_seeds;
	char dir[PATH_MAX];
	void* state;
} mm_dump_cfg_t;

// Per-context state for a single mapping run.  verbose and dbg_flag
// are initialised from the application-level globals (mm_verbose,
// mm_dbg_flag) at creation time, but thereafter belong to this context
// and may diverge from the globals.  Pipeline code should read these
// fields instead of the globals.
typedef struct mm_core_ctx_s {
	mm_logger_t* log;
	int verbose; // context-level verbosity (seeded from mm_verbose)
	int dbg_flag; // context-level debug flags (seeded from mm_dbg_flag)
	mm_dump_cfg_t dump_cfg;
} mm_core_ctx_t;

// Read verbose/dbg_flag from context when available, falling back to the
// application-level globals when ctx is NULL (legacy callers).
static inline int mm_core_ctx_verbose(const mm_core_ctx_t* ctx) { return ctx ? ctx->verbose : mm_verbose; }
static inline int mm_core_ctx_dbg_flag(const mm_core_ctx_t* ctx) { return ctx ? ctx->dbg_flag : mm_dbg_flag; }

mm_core_ctx_t* mm_core_ctx_create(mm_logger_t* log);
void mm_core_ctx_destroy(mm_core_ctx_t* ctx);
void mm_core_ctx_reset(mm_core_ctx_t* ctx);

void mm_core_ctx_set_verbose(mm_core_ctx_t* ctx, int verbose);
void mm_core_ctx_set_dbg_flag(mm_core_ctx_t* ctx, int dbg_flag);
void mm_core_ctx_set_log(mm_core_ctx_t* ctx, mm_logger_t* log);

void mm_core_ctx_set_dump(mm_core_ctx_t* ctx, mm_dump_format_t format, const char* dir);
const mm_dump_cfg_t* mm_core_ctx_dump_cfg(const mm_core_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif // MM_CORE_CTX_H
