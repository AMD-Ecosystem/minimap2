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

#ifndef MM_LOG_H
#define MM_LOG_H

#include <stdint.h>
#include "minimap.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Logger helpers — thread-local plumbing and logging macros
 *
 * The mm_logger_t struct is defined in minimap.h so that library users can
 * provide custom implementations.  This header adds the internal
 * thread-local binding, the format helper, and the convenience macros
 * used throughout the minimap2 codebase.
 *
 * Thread-local override:
 *   mm_set_tl_logger(lg) — bind `lg` to the calling thread.  A diagnostic
 *                           message is printed to stderr if the thread
 *                           already had a different logger bound (indicates
 *                           an unexpected double-set).
 *   mm_tl_logger()       — return the thread-local logger if set,
 *                           otherwise fall back to the active logger.
 *                           If neither exists, a sync fallback logger is
 *                           created automatically.
 *                           This function never returns NULL.
 *
 * All logging macros (mm_log, mm_print, mm_log_debug, …) call
 * mm_tl_logger() internally, so callers never pass a logger explicitly.
 * Entry points (main(), worker callbacks, CLI main()) should call
 * mm_set_tl_logger() once after creating or obtaining their logger so
 * that every function on that thread automatically uses it.
 * ============================================================================ */

mm_logger_t* mm_get_default_logger(void);

void mm_set_tl_logger(mm_logger_t* logger);
mm_logger_t* mm_tl_logger(void);

/* ---- printf-style format helper (returns thread-local buffer) ---- */
const char* mm_log_format(const char* fmt, ...);

#ifdef __cplusplus
} /* extern "C" */

/* ---- C++ path: fmt::format through logger struct ---- */
#include <spdlog/fmt/fmt.h>

#define mm_log_debug(...) \
	do { \
		mm_logger_t* _lg = mm_tl_logger(); \
		_lg->log_debug(_lg, __FILE__, __LINE__, \
		    static_cast<const char*>(__FUNCTION__), \
		    fmt::format(__VA_ARGS__).c_str()); \
	} while (0)

#define mm_log_trace(...) \
	do { \
		mm_logger_t* _lg = mm_tl_logger(); \
		_lg->log_trace(_lg, __FILE__, __LINE__, \
		    static_cast<const char*>(__FUNCTION__), \
		    fmt::format(__VA_ARGS__).c_str()); \
	} while (0)

#define mm_log_info(...) \
	do { \
		mm_logger_t* _lg = mm_tl_logger(); \
		_lg->log_info(_lg, __FILE__, __LINE__, \
		    static_cast<const char*>(__FUNCTION__), \
		    fmt::format(__VA_ARGS__).c_str()); \
	} while (0)

#define mm_log_warn(...) \
	do { \
		mm_logger_t* _lg = mm_tl_logger(); \
		_lg->log_warn(_lg, __FILE__, __LINE__, \
		    static_cast<const char*>(__FUNCTION__), \
		    fmt::format(__VA_ARGS__).c_str()); \
	} while (0)

#define mm_log_error(...) \
	do { \
		mm_logger_t* _lg = mm_tl_logger(); \
		_lg->log_error(_lg, __FILE__, __LINE__, \
		    static_cast<const char*>(__FUNCTION__), \
		    fmt::format(__VA_ARGS__).c_str()); \
	} while (0)

#define mm_gpu_log_info(...) \
	do { \
		mm_logger_t* _lg = mm_tl_logger(); \
		_lg->log_debug(_lg, __FILE__, __LINE__, \
		    static_cast<const char*>(__FUNCTION__), \
		    fmt::format(__VA_ARGS__).c_str()); \
	} while (0)

#else
/* ---- C path: printf-style format + dispatch through struct ---- */

#define mm_log_debug(fmt, ...) \
	do { \
		mm_logger_t* _lg = mm_tl_logger(); \
		_lg->log_debug(_lg, __FILE__, __LINE__, __FUNCTION__, \
		    mm_log_format(fmt, ##__VA_ARGS__)); \
	} while (0)
#define mm_log_trace(fmt, ...) \
	do { \
		mm_logger_t* _lg = mm_tl_logger(); \
		_lg->log_trace(_lg, __FILE__, __LINE__, __FUNCTION__, \
		    mm_log_format(fmt, ##__VA_ARGS__)); \
	} while (0)
#define mm_log_info(fmt, ...) \
	do { \
		mm_logger_t* _lg = mm_tl_logger(); \
		_lg->log_info(_lg, __FILE__, __LINE__, __FUNCTION__, \
		    mm_log_format(fmt, ##__VA_ARGS__)); \
	} while (0)
#define mm_log_warn(fmt, ...) \
	do { \
		mm_logger_t* _lg = mm_tl_logger(); \
		_lg->log_warn(_lg, __FILE__, __LINE__, __FUNCTION__, \
		    mm_log_format(fmt, ##__VA_ARGS__)); \
	} while (0)
#define mm_log_error(fmt, ...) \
	do { \
		mm_logger_t* _lg = mm_tl_logger(); \
		_lg->log_error(_lg, __FILE__, __LINE__, __FUNCTION__, \
		    mm_log_format(fmt, ##__VA_ARGS__)); \
	} while (0)
#define mm_gpu_log_info(fmt, ...) \
	do { \
		mm_logger_t* _lg = mm_tl_logger(); \
		_lg->log_debug(_lg, __FILE__, __LINE__, __FUNCTION__, \
		    mm_log_format(fmt, ##__VA_ARGS__)); \
	} while (0)

#endif /* __cplusplus */

/* ---- mm_print / mm_log: shared by C and C++ ---- */
#define mm_print(fmt, ...) \
	do { \
		mm_logger_t* _lg = mm_tl_logger(); \
		_lg->print(_lg, mm_log_format(fmt, ##__VA_ARGS__)); \
	} while (0)
#define mm_print_str(s) \
	do { \
		mm_logger_t* _lg = mm_tl_logger(); \
		_lg->print(_lg, (s)); \
	} while (0)
#define mm_log(fmt, ...) \
	do { \
		mm_logger_t* _lg = mm_tl_logger(); \
		_lg->log(_lg, mm_log_format(fmt, ##__VA_ARGS__)); \
	} while (0)

/* ============================================================================
 * Profiler Trace Logging
 * ============================================================================ */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Trace enter - log function/region entry with timing
 * @param name Name of the function/region
 * @param file Source file
 * @param line Line number
 * @return Timestamp (nanoseconds) for duration calculation
 */
#ifdef MM_TRACE
uint64_t mm_trace_enter(const char* name, const char* file, int line);
#else
static inline uint64_t mm_trace_enter(const char* name, const char* file, int line)
{
	(void)name;
	(void)file;
	(void)line;
	return 0;
}
#endif

/**
 * Trace exit - log function/region exit with duration
 * @param name Name of the function/region
 * @param file Source file
 * @param line Line number
 * @param start_time Timestamp from mm_trace_enter
 */
#ifdef MM_TRACE
void mm_trace_exit(const char* name, const char* file, int line, uint64_t start_time);
#else
static inline void mm_trace_exit(const char* name, const char* file, int line, uint64_t start_time)
{
	(void)name;
	(void)file;
	(void)line;
	(void)start_time;
}
#endif

/**
 * Trace mark - log an instant event
 * @param name Name of the marker
 * @param file Source file
 * @param line Line number
 */
#ifdef MM_TRACE
void mm_trace_mark_event(const char* name, const char* file, int line);
#else
static inline void mm_trace_mark_event(const char* name, const char* file, int line)
{
	(void)name;
	(void)file;
	(void)line;
}
#endif

/**
 * Trace log - log a message at current trace depth
 * @param file Source file
 * @param line Line number
 * @param fmt Format string
 */
#ifdef MM_TRACE
void mm_trace_log_msg(const char* file, int line, const char* fmt, ...);
#else
#define mm_trace_log_msg(file, line, fmt, ...) ((void)0)
#endif

/**
 * Get current trace depth (for indentation)
 */
#ifdef MM_TRACE
int mm_trace_get_depth(void);
#else
static inline int mm_trace_get_depth(void) { return 0; }
#endif

#ifdef __cplusplus
}
#endif

#endif // MM_LOG_H
