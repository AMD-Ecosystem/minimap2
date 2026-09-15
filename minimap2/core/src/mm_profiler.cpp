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
 * @file mm_profiler.cpp
 * @brief Profiler backend implementation for ITT/NVTX/ROCTX
 * 
 * This file provides the C-callable profiler backend functions.
 * Colors are defined once here and used by all backends.
 */

#include "mm_profiler.h"

#ifdef MM_USE_NVTX
#include <pthread.h> /* for pthread_self() in thread naming */
#endif

#ifdef MM_USE_ROCTX
#include <roctracer/roctx.h>
#endif

#ifdef MM_USE_NVTX
#include <nvToolsExt.h>
#endif

#ifdef MM_USE_ITT
#include <ittnotify.h>

// Thread-safe lazy initialization using static local variable (C++11 guarantees thread safety)
static __itt_domain* get_itt_domain()
{
	static __itt_domain* domain = __itt_domain_create("minimap2");
	return domain;
}
#endif

/* Single color table used by all backends (ARGB format) */
static const uint32_t PROFILE_COLORS[] = {
    0xFFFF0000, /* RED */
    0xFF00FF00, /* GREEN */
    0xFF0000FF, /* BLUE */
    0xFFFFFF00, /* YELLOW */
    0xFF00FFFF, /* CYAN */
    0xFFFF00FF, /* MAGENTA */
    0xFFFF8000, /* ORANGE */
    0xFFFFFFFF, /* WHITE */
    0xFF808080 /* DEFAULT (gray) */
};

static inline uint32_t get_color(int color_idx)
{
	if (color_idx < 0 || color_idx >= MM_PROFILE_COLOR_COUNT) {
		return PROFILE_COLORS[MM_PROFILE_COLOR_DEFAULT];
	}
	return PROFILE_COLORS[color_idx];
}

/* ============================================================================
 * Profiler Backend Implementation
 * ============================================================================ */

extern "C" void mm_profiler_push(const char* name)
{
#ifdef MM_USE_ROCTX
	roctxRangePushA(name);
#endif
#ifdef MM_USE_NVTX
	nvtxRangePushA(name);
#endif
#ifdef MM_USE_ITT
	__itt_string_handle* handle = __itt_string_handle_create(name);
	__itt_task_begin(get_itt_domain(), __itt_null, __itt_null, handle);
#endif
	(void)name;
}

extern "C" void mm_profiler_push_color(const char* name, int color)
{
#ifdef MM_USE_ROCTX
	/* ROCTX: roctxRangePushA only supports message; color requires extended API (roctxRangePushEx) */
	roctxRangePushA(name);
#endif
#ifdef MM_USE_NVTX
	nvtxEventAttributes_t attr = {0};
	attr.version = NVTX_VERSION;
	attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
	attr.message.ascii = name;
	attr.colorType = NVTX_COLOR_ARGB;
	attr.color = get_color(color);
	nvtxRangePushEx(&attr);
#endif
#ifdef MM_USE_ITT
	/* ITT doesn't support colors */
	__itt_string_handle* handle = __itt_string_handle_create(name);
	__itt_task_begin(get_itt_domain(), __itt_null, __itt_null, handle);
#endif
	(void)name;
	(void)color;
}

extern "C" void mm_profiler_pop(void)
{
#ifdef MM_USE_ROCTX
	roctxRangePop();
#endif
#ifdef MM_USE_NVTX
	nvtxRangePop();
#endif
#ifdef MM_USE_ITT
	__itt_task_end(get_itt_domain());
#endif
}

extern "C" void mm_profiler_mark(const char* name)
{
#ifdef MM_USE_ROCTX
	roctxMarkA(name);
#endif
#ifdef MM_USE_NVTX
	nvtxMarkA(name);
#endif
#ifdef MM_USE_ITT
	__itt_string_handle* handle = __itt_string_handle_create(name);
	__itt_marker(get_itt_domain(), __itt_null, handle, __itt_scope_track);
#endif
	(void)name;
}

extern "C" void mm_profiler_thread_name(const char* name)
{
#ifdef MM_USE_ROCTX
	/* ROCTX doesn't have thread naming API */
#endif
#ifdef MM_USE_NVTX
	nvtxNameOsThread(pthread_self(), name);
#endif
#ifdef MM_USE_ITT
	__itt_thread_set_name(name);
#endif
	(void)name;
}
