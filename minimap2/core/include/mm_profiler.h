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
 * @file mm_profiler.h
 * @brief Unified profiling interface for ITT (Intel VTune), NVTX (NVIDIA), and ROCTX (AMD)
 *
 * This header provides a unified API for emitting profiling events across different
 * profiler backends. It supports:
 * - ITT (Intel Threading Tools) for Intel VTune Profiler
 * - NVTX (NVIDIA Tools Extension) for NVIDIA Nsight Systems/Compute
 * - ROCTX (ROCm TX) for AMD rocprof/rocprofv3
 * - Trace logging with high-resolution timing
 *
 * Usage:
 *   // C-style macros (works in both C and C++)
 *   MM_PROFILE_RANGE_PUSH("MyFunction");
 *   // ... code ...
 *   MM_PROFILE_RANGE_POP();
 *
 *   // C++ RAII style (automatically pops on scope exit)
 *   {
 *       MM_PROFILE_SCOPE("MyFunction");
 *       // ... code ...
 *   } // automatically pops here
 *
 *   // Colored ranges (NVTX/ROCTX)
 *   MM_PROFILE_RANGE_PUSH_COLOR("Alignment", MM_PROFILE_COLOR_GREEN);
 *
 *   // Markers (instant events)
 *   MM_PROFILE_MARK("checkpoint");
 *
 * Build flags:
 *   -DMM_USE_ITT    - Enable Intel ITT instrumentation
 *   -DMM_USE_NVTX   - Enable NVIDIA NVTX instrumentation
 *   -DMM_USE_ROCTX  - Enable AMD ROCTX instrumentation
 *   -DMM_PROFILE    - Enable all available profiling (auto-detect)
 *   -DMM_TRACE      - Enable trace logging with timing
 *
 * If none are defined, all macros become no-ops with zero overhead.
 */

#ifndef MM_PROFILER_H
#define MM_PROFILER_H

#include <stdint.h>
#include "mm_log.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Auto-detection of profiling backends
 * ============================================================================ */

#ifdef MM_PROFILE
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#ifndef MM_USE_ITT
#define MM_USE_ITT 1
#endif
#endif

#if defined(__CUDACC__) || defined(USECUDA)
#ifndef MM_USE_NVTX
#define MM_USE_NVTX 1
#endif
#endif

#if defined(__HIP__) || defined(__HIPCC__) || defined(USEHIP)
#ifndef MM_USE_ROCTX
#define MM_USE_ROCTX 1
#endif
#endif
#endif

/* ============================================================================
 * Color definitions (single source of truth)
 * ============================================================================ */

typedef enum {
	MM_PROFILE_COLOR_RED = 0,
	MM_PROFILE_COLOR_GREEN = 1,
	MM_PROFILE_COLOR_BLUE = 2,
	MM_PROFILE_COLOR_YELLOW = 3,
	MM_PROFILE_COLOR_CYAN = 4,
	MM_PROFILE_COLOR_MAGENTA = 5,
	MM_PROFILE_COLOR_ORANGE = 6,
	MM_PROFILE_COLOR_WHITE = 7,
	MM_PROFILE_COLOR_DEFAULT = 8,
	MM_PROFILE_COLOR_COUNT = 9
} mm_profile_color_t;

/* ============================================================================
 * Profiling enabled check
 * ============================================================================ */

#if defined(MM_USE_ITT) || defined(MM_USE_NVTX) || defined(MM_USE_ROCTX)
#define MM_PROFILING_ENABLED 1
#else
#define MM_PROFILING_ENABLED 0
#endif

/* ============================================================================
 * Backend function declarations (implemented in mm_profiler.cpp)
 * ============================================================================ */

#if MM_PROFILING_ENABLED

/**
 * Push a profiler range (implemented in mm_profiler.cpp)
 */
void mm_profiler_push(const char* name);

/**
 * Push a profiler range with color
 */
void mm_profiler_push_color(const char* name, int color);

/**
 * Pop a profiler range
 */
void mm_profiler_pop(void);

/**
 * Mark an instant profiler event
 */
void mm_profiler_mark(const char* name);

/**
 * Set thread name
 */
void mm_profiler_thread_name(const char* name);

/* Simplified inline wrappers - just call backend functions */
static inline void mm_profile_range_push(const char* name)
{
	mm_profiler_push(name);
}

static inline void mm_profile_range_push_color(const char* name, mm_profile_color_t color)
{
	mm_profiler_push_color(name, (int)color);
}

static inline void mm_profile_range_pop(void)
{
	mm_profiler_pop();
}

static inline void mm_profile_mark(const char* name)
{
	mm_profiler_mark(name);
}

static inline void mm_profile_thread_name(const char* name)
{
	mm_profiler_thread_name(name);
}

#else /* !MM_PROFILING_ENABLED */

/* No-op implementations when profiling is disabled */
static inline void mm_profile_range_push(const char* name) { (void)name; }
static inline void mm_profile_range_push_color(const char* name, mm_profile_color_t color)
{
	(void)name;
	(void)color;
}
static inline void mm_profile_range_pop(void) {}
static inline void mm_profile_mark(const char* name) { (void)name; }
static inline void mm_profile_thread_name(const char* name) { (void)name; }

#endif /* MM_PROFILING_ENABLED */

/* ============================================================================
 * Trace Logging with High-Resolution Timing
 * ============================================================================ */

#ifdef MM_TRACE

/* Thread-local storage for trace timing stack */
#ifndef __cplusplus
#define MM_TRACE_TLS _Thread_local
#else
#define MM_TRACE_TLS thread_local
#endif

#define MM_TRACE_MAX_DEPTH 64

typedef struct {
	uint64_t start_times[MM_TRACE_MAX_DEPTH];
	const char* names[MM_TRACE_MAX_DEPTH];
} mm_trace_stack_t;

static inline mm_trace_stack_t* mm_trace_get_stack(void)
{
	static MM_TRACE_TLS mm_trace_stack_t stack = {0};
	return &stack;
}

static inline void mm_trace_begin_impl(const char* name, const char* file, int line)
{
	mm_trace_stack_t* stack = mm_trace_get_stack();
	int depth = mm_trace_get_depth(); /* Use single source of truth for depth */
	uint64_t start_time = mm_trace_enter(name, file, line); /* This increments depth */

	if (depth < MM_TRACE_MAX_DEPTH) {
		stack->start_times[depth] = start_time;
		stack->names[depth] = name;
	}
	mm_profile_range_push(name);
}

static inline void mm_trace_begin_impl_color(const char* name, mm_profile_color_t color, const char* file, int line)
{
	mm_trace_stack_t* stack = mm_trace_get_stack();
	int depth = mm_trace_get_depth(); /* Use single source of truth for depth */
	uint64_t start_time = mm_trace_enter(name, file, line); /* This increments depth */

	if (depth < MM_TRACE_MAX_DEPTH) {
		stack->start_times[depth] = start_time;
		stack->names[depth] = name;
	}
	mm_profile_range_push_color(name, color);
}

static inline void mm_trace_end_impl(const char* name, const char* file, int line)
{
	mm_trace_stack_t* stack = mm_trace_get_stack();
	int depth = mm_trace_get_depth(); /* Get depth before decrement */
	uint64_t start_time = 0;

	if (depth > 0 && depth <= MM_TRACE_MAX_DEPTH) {
		start_time = stack->start_times[depth - 1];
	}
	mm_trace_exit(name, file, line, start_time); /* This decrements depth */
	mm_profile_range_pop();
}

static inline void mm_trace_mark_impl(const char* name, const char* file, int line)
{
	mm_trace_mark_event(name, file, line);
	mm_profile_mark(name);
}

#define MM_TRACE_BEGIN(name) mm_trace_begin_impl(name, __FILE__, __LINE__)
#define MM_TRACE_END(name) mm_trace_end_impl(name, __FILE__, __LINE__)
#define MM_TRACE_MARK(name) mm_trace_mark_impl(name, __FILE__, __LINE__)
#define MM_TRACE_LOG(fmt, ...) mm_trace_log_msg(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define MM_TRACE_FUNCTION() MM_TRACE_BEGIN(__func__)
#define MM_TRACE_FUNCTION_END() MM_TRACE_END(__func__)

#else /* !MM_TRACE */

#define MM_TRACE_BEGIN(name) ((void)0)
#define MM_TRACE_END(name) ((void)0)
#define MM_TRACE_MARK(name) ((void)0)
#define MM_TRACE_LOG(fmt, ...) ((void)0)
#define MM_TRACE_FUNCTION() ((void)0)
#define MM_TRACE_FUNCTION_END() ((void)0)

#endif /* MM_TRACE */

/* ============================================================================
 * Convenience Macros for C code
 * ============================================================================ */

#ifdef MM_TRACE
#define MM_PROFILE_RANGE_PUSH(name) mm_trace_begin_impl(name, __FILE__, __LINE__)
#define MM_PROFILE_RANGE_PUSH_COLOR(name, color) mm_trace_begin_impl_color(name, color, __FILE__, __LINE__)
#define MM_PROFILE_RANGE_POP() mm_trace_end_impl("", __FILE__, __LINE__)
#define MM_PROFILE_MARK(name) mm_trace_mark_impl(name, __FILE__, __LINE__)
#else
#define MM_PROFILE_RANGE_PUSH(name) mm_profile_range_push(name)
#define MM_PROFILE_RANGE_PUSH_COLOR(name, color) mm_profile_range_push_color(name, color)
#define MM_PROFILE_RANGE_POP() mm_profile_range_pop()
#define MM_PROFILE_MARK(name) mm_profile_mark(name)
#endif

#define MM_PROFILE_THREAD_NAME(name) mm_profile_thread_name(name)
#define MM_PROFILE_FUNCTION() MM_PROFILE_RANGE_PUSH(__func__)
#define MM_PROFILE_FUNCTION_END() MM_PROFILE_RANGE_POP()

/* Domain-specific colored ranges */
#define MM_PROFILE_SEED(name) MM_PROFILE_RANGE_PUSH_COLOR(name, MM_PROFILE_COLOR_GREEN)
#define MM_PROFILE_CHAIN(name) MM_PROFILE_RANGE_PUSH_COLOR(name, MM_PROFILE_COLOR_BLUE)
#define MM_PROFILE_ALIGN(name) MM_PROFILE_RANGE_PUSH_COLOR(name, MM_PROFILE_COLOR_RED)
#define MM_PROFILE_IO(name) MM_PROFILE_RANGE_PUSH_COLOR(name, MM_PROFILE_COLOR_YELLOW)
#define MM_PROFILE_GPU(name) MM_PROFILE_RANGE_PUSH_COLOR(name, MM_PROFILE_COLOR_CYAN)

#ifdef __cplusplus
}
#endif

/* ============================================================================
 * C++ RAII wrappers
 * ============================================================================ */

#ifdef __cplusplus

namespace mm_profile
{

class ScopedRange
{
      public:
	explicit ScopedRange(const char* name) { mm_profile_range_push(name); }
	ScopedRange(const char* name, mm_profile_color_t color) { mm_profile_range_push_color(name, color); }
	~ScopedRange() { mm_profile_range_pop(); }

	ScopedRange(const ScopedRange&) = delete;
	ScopedRange& operator=(const ScopedRange&) = delete;
};

} /* namespace mm_profile */

#define MM_PROFILE_SCOPE(name) \
	mm_profile::ScopedRange _mm_profile_scope_##__LINE__(name)

#define MM_PROFILE_SCOPE_COLOR(name, color) \
	mm_profile::ScopedRange _mm_profile_scope_##__LINE__(name, color)

#define MM_PROFILE_SCOPE_FUNCTION() \
	mm_profile::ScopedRange _mm_profile_scope_func(__func__)

#define MM_PROFILE_SCOPE_SEED(name) MM_PROFILE_SCOPE_COLOR(name, MM_PROFILE_COLOR_GREEN)
#define MM_PROFILE_SCOPE_CHAIN(name) MM_PROFILE_SCOPE_COLOR(name, MM_PROFILE_COLOR_BLUE)
#define MM_PROFILE_SCOPE_ALIGN(name) MM_PROFILE_SCOPE_COLOR(name, MM_PROFILE_COLOR_RED)
#define MM_PROFILE_SCOPE_IO(name) MM_PROFILE_SCOPE_COLOR(name, MM_PROFILE_COLOR_YELLOW)
#define MM_PROFILE_SCOPE_GPU(name) MM_PROFILE_SCOPE_COLOR(name, MM_PROFILE_COLOR_CYAN)

#ifdef MM_TRACE

namespace mm_trace
{

class ScopedTrace
{
      public:
	ScopedTrace(const char* name, const char* file, int line)
	    : name_(name), file_(file), line_(line)
	{
		mm_trace_begin_impl(name, file, line);
	}
	~ScopedTrace() { mm_trace_end_impl(name_, file_, line_); }

	ScopedTrace(const ScopedTrace&) = delete;
	ScopedTrace& operator=(const ScopedTrace&) = delete;

      private:
	const char* name_;
	const char* file_;
	int line_;
};

} /* namespace mm_trace */

#define MM_TRACE_SCOPE(name) \
	mm_trace::ScopedTrace _mm_trace_scope_##__LINE__(name, __FILE__, __LINE__)

#define MM_TRACE_SCOPE_FUNCTION() \
	mm_trace::ScopedTrace _mm_trace_scope_func(__func__, __FILE__, __LINE__)

#else /* !MM_TRACE */

#define MM_TRACE_SCOPE(name) ((void)0)
#define MM_TRACE_SCOPE_FUNCTION() ((void)0)

#endif /* MM_TRACE */

#endif /* __cplusplus */

#endif /* MM_PROFILER_H */
