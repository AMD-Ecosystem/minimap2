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

#include "core_ctx.h"
#include "minimap.h"

#include <stdlib.h>
#include <string.h>

extern "C" {

mm_core_ctx_t* mm_core_ctx_create(mm_logger_t* log)
{
	if (!log) return NULL;
	mm_core_ctx_t* ctx = static_cast<mm_core_ctx_t*>(calloc(1, sizeof(mm_core_ctx_t)));
	if (!ctx) return NULL;
	ctx->log = log;
	ctx->verbose = mm_verbose;
	ctx->dbg_flag = mm_dbg_flag;
	return ctx;
}

void mm_core_ctx_destroy(mm_core_ctx_t* ctx)
{
	if (!ctx) return;
	free(ctx);
}

void mm_core_ctx_reset(mm_core_ctx_t* ctx)
{
	if (!ctx) return;
	ctx->verbose = mm_verbose;
	ctx->dbg_flag = mm_dbg_flag;
}

void mm_core_ctx_set_verbose(mm_core_ctx_t* ctx, int verbose)
{
	if (ctx) ctx->verbose = verbose;
}

void mm_core_ctx_set_dbg_flag(mm_core_ctx_t* ctx, int dbg_flag)
{
	if (ctx) ctx->dbg_flag = dbg_flag;
}

void mm_core_ctx_set_log(mm_core_ctx_t* ctx, mm_logger_t* log)
{
	if (ctx && log) ctx->log = log;
}

void mm_core_ctx_set_dump(mm_core_ctx_t* ctx, mm_dump_format_t format, const char* dir)
{
	if (!ctx) return;
	ctx->dump_cfg.format = format;
	ctx->dump_cfg.enabled = (format != MM_DUMP_NONE && dir && dir[0] != '\0') ? 1 : 0;
	ctx->dump_cfg.state = NULL;
	if (dir) {
		strncpy(ctx->dump_cfg.dir, dir, PATH_MAX - 1);
		ctx->dump_cfg.dir[PATH_MAX - 1] = '\0';
	} else {
		ctx->dump_cfg.dir[0] = '\0';
	}
}

const mm_dump_cfg_t* mm_core_ctx_dump_cfg(const mm_core_ctx_t* ctx)
{
	return ctx ? &ctx->dump_cfg : NULL;
}

} // extern "C"
