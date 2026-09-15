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

#include "mm_log.h"
#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <memory>
#include <cstdarg>
#include <string>
#include <type_traits>
#include <atomic>
#include <thread>
#include <mutex>
#include <sstream>

// Format a printf-style va_list into a std::string without truncation.
static std::string vformat(const char* fmt, va_list args)
{
	char stack_buf[512];
	va_list args_copy;
	va_copy(args_copy, args);
	int n = vsnprintf(stack_buf, sizeof(stack_buf), fmt, args_copy);
	va_end(args_copy);
	if (n < 0) return fmt;
	if (static_cast<size_t>(n) < sizeof(stack_buf))
		return std::string(stack_buf, static_cast<size_t>(n));
	std::string result(static_cast<size_t>(n) + 1, '\0');
	vsnprintf(&result[0], static_cast<size_t>(n) + 1, fmt, args);
	result.resize(static_cast<size_t>(n));
	return result;
}

/* ============================================================================
 * mm_default_logger — spdlog-backed implementation of mm_logger_t
 * ============================================================================ */

class mm_default_logger : public mm_logger_t
{
	std::shared_ptr<spdlog::logger> print_logger_;
	std::shared_ptr<spdlog::logger> log_logger_;
	std::shared_ptr<spdlog::logger> debug_logger_;
	std::shared_ptr<spdlog::logger> info_logger_;

	static void print_impl(mm_logger_t* self, const char* msg)
	{
		auto* dl = static_cast<mm_default_logger*>(self);
		if (dl->print_logger_)
			dl->print_logger_->log(spdlog::level::info, msg);
		else
			fprintf(stdout, "%s\n", msg);
	}

	static void log_impl(mm_logger_t* self, const char* msg)
	{
		auto* dl = static_cast<mm_default_logger*>(self);
		if (dl->log_logger_) {
			if (!dl->log_logger_->should_log(spdlog::level::info)) return;
			dl->log_logger_->log(spdlog::level::info, msg);
		} else {
			fprintf(stderr, "%s\n", msg);
		}
	}

	static void log_trace_impl(mm_logger_t* self, const char* file,
	    int line, const char* func, const char* msg)
	{
		auto* dl = static_cast<mm_default_logger*>(self);
		if (dl->debug_logger_) {
			if (!dl->debug_logger_->should_log(spdlog::level::trace)) return;
			dl->debug_logger_->log({file, line, func}, spdlog::level::trace, msg);
		} else {
			fprintf(stderr, "%s\n", msg);
		}
	}

	static void log_debug_impl(mm_logger_t* self, const char* file,
	    int line, const char* func, const char* msg)
	{
		auto* dl = static_cast<mm_default_logger*>(self);
		if (dl->debug_logger_) {
			if (!dl->debug_logger_->should_log(spdlog::level::debug)) return;
			dl->debug_logger_->log({file, line, func}, spdlog::level::debug, msg);
		} else {
			fprintf(stderr, "%s\n", msg);
		}
	}

	static void log_info_impl(mm_logger_t* self, const char* file,
	    int line, const char* func, const char* msg)
	{
		auto* dl = static_cast<mm_default_logger*>(self);
		if (dl->info_logger_) {
			if (!dl->info_logger_->should_log(spdlog::level::info)) return;
			dl->info_logger_->log({file, line, func}, spdlog::level::info, msg);
		} else {
			fprintf(stdout, "%s\n", msg);
		}
	}

	static void log_warn_impl(mm_logger_t* self, const char* file,
	    int line, const char* func, const char* msg)
	{
		auto* dl = static_cast<mm_default_logger*>(self);
		if (dl->debug_logger_) {
			if (!dl->debug_logger_->should_log(spdlog::level::warn)) return;
			dl->debug_logger_->log({file, line, func}, spdlog::level::warn, msg);
		} else {
			fprintf(stderr, "%s\n", msg);
		}
	}

	static void log_error_impl(mm_logger_t* self, const char* file,
	    int line, const char* func, const char* msg)
	{
		auto* dl = static_cast<mm_default_logger*>(self);
		if (dl->debug_logger_) {
			if (!dl->debug_logger_->should_log(spdlog::level::err)) return;
			dl->debug_logger_->log({file, line, func}, spdlog::level::err, msg);
		} else {
			fprintf(stderr, "%s\n", msg);
		}
	}

	static void flush_impl(mm_logger_t* self)
	{
		auto* dl = static_cast<mm_default_logger*>(self);
		if (dl->print_logger_) dl->print_logger_->flush();
		if (dl->log_logger_) dl->log_logger_->flush();
		if (dl->debug_logger_) dl->debug_logger_->flush();
		if (dl->info_logger_) dl->info_logger_->flush();
	}

      public:
	mm_default_logger()
	{
		print = print_impl;
		log = log_impl;
		log_trace = log_trace_impl;
		log_debug = log_debug_impl;
		log_info = log_info_impl;
		log_warn = log_warn_impl;
		log_error = log_error_impl;
		flush = flush_impl;
	}

	mm_default_logger(const mm_default_logger&) = delete;
	mm_default_logger& operator=(const mm_default_logger&) = delete;

	void init(int async_mode, const char* debug_file)
	{
		try {
			auto print_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

			std::shared_ptr<spdlog::sinks::sink> log_sink;
			std::shared_ptr<spdlog::sinks::sink> debug_sink;

			if (debug_file) {
				log_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(debug_file, true);
				debug_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(debug_file, false);
			} else {
				log_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
				debug_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
			}

			auto info_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

			if (async_mode)
				spdlog::init_thread_pool(8192, 1);

			static std::atomic<uint64_t> id_gen{0};
			uint64_t id = id_gen.fetch_add(1, std::memory_order_relaxed);
			auto tid = std::this_thread::get_id();
			auto uname = [&](const char* base) {
				std::ostringstream os;
				os << base << "_" << tid << "_" << id;
				return os.str();
			};

			auto make_logger = [&](const std::string& name,
					       const std::shared_ptr<spdlog::sinks::sink>& sink)
			    -> std::shared_ptr<spdlog::logger> {
				if (async_mode)
					return std::make_shared<spdlog::async_logger>(
					    name, sink, spdlog::thread_pool(),
					    spdlog::async_overflow_policy::block);
				return std::make_shared<spdlog::logger>(name, sink);
			};

			print_logger_ = make_logger(uname("print"), print_sink);
			log_logger_ = make_logger(uname("log"), log_sink);
			debug_logger_ = make_logger(uname("debug"), debug_sink);
			info_logger_ = make_logger(uname("info"), info_sink);

			print_logger_->set_pattern("%v");
			print_logger_->set_level(spdlog::level::info);
			log_logger_->set_pattern("%v");
			log_logger_->set_level(spdlog::level::info);

			static constexpr const char* ts_pattern =
			    "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] [%s:%#] %v";
			debug_logger_->set_pattern(ts_pattern);
#ifdef MM_TRACE
			debug_logger_->set_level(spdlog::level::trace);
#else
			debug_logger_->set_level(spdlog::level::debug);
#endif
			info_logger_->set_pattern(ts_pattern);
			info_logger_->set_level(spdlog::level::info);

			if (debug_file) {
				log_logger_->flush_on(spdlog::level::trace);
				debug_logger_->flush_on(spdlog::level::trace);
			}

			spdlog::register_logger(print_logger_);
			spdlog::register_logger(log_logger_);
			spdlog::register_logger(debug_logger_);
			spdlog::register_logger(info_logger_);

		} catch (const spdlog::spdlog_ex& ex) {
			fprintf(stderr, "Log initialization failed: %s\n", ex.what());
		}
	}

	void shutdown()
	{
		flush_impl(this);
		if (print_logger_) spdlog::drop(print_logger_->name());
		if (log_logger_) spdlog::drop(log_logger_->name());
		if (debug_logger_) spdlog::drop(debug_logger_->name());
		if (info_logger_) spdlog::drop(info_logger_->name());
		print_logger_ = nullptr;
		log_logger_ = nullptr;
		debug_logger_ = nullptr;
		info_logger_ = nullptr;
	}

	std::shared_ptr<spdlog::logger>& get_debug_logger() { return debug_logger_; }
};

static_assert(std::is_standard_layout_v<mm_logger_t>, "mm_logger_t must be standard-layout");

// Process-owned default logger. It is the fallback that every mm_log_* call
// resolves to when the calling thread has not bound its own logger via
// mm_set_tl_logger(). It is created lazily on first use and never destroyed:
// the logging macros dereference the resolved logger without a lock, so the
// fallback must outlive every thread that might log through it.
//
// Crucially, user loggers (mm_logger_create/destroy) are NEVER installed here.
// Tying the global fallback to a user logger's lifetime would let one context's
// teardown free the object other threads are still logging through — a
// use-after-free (jump through a freed vtable). See AIOSS-4548.
//
// The default logger lives in a function-local static inside
// mm_get_or_create_default_logger(). C++11 guarantees its initialization runs
// exactly once and is thread-safe (the compiler inserts a one-time guard), so no
// explicit mutex or atomic is needed. After that one-time init the pointer is
// never mutated, so every later call is just a cheap guard-byte check plus a
// load — no per-call lock and no data race.
static mm_default_logger* mm_get_or_create_default_logger()
{
	static mm_default_logger* s_default_logger = []() -> mm_default_logger* {
		auto* dl = new (std::nothrow) mm_default_logger();
		if (dl) dl->init(0, nullptr);
		return dl;
	}();
	return s_default_logger;
}

extern "C" mm_logger_t* mm_logger_create(int async_mode, const char* debug_file)
{
	auto* logger = new (std::nothrow) mm_default_logger();
	if (!logger) return nullptr;
	logger->init(async_mode, debug_file);
	return static_cast<mm_logger_t*>(logger);
}

extern "C" void mm_logger_destroy(mm_logger_t* logger)
{
	if (!logger) return;
	auto* dl = static_cast<mm_default_logger*>(logger);
	dl->shutdown();
	delete dl;
}

extern "C" mm_logger_t* mm_get_default_logger(void)
{
	return static_cast<mm_logger_t*>(mm_get_or_create_default_logger());
}

static thread_local mm_logger_t* tl_logger = nullptr;

extern "C" void mm_set_tl_logger(mm_logger_t* logger)
{
	// Multiple mapping contexts may legitimately run on the same thread over
	// time (e.g. ThreadPoolExecutor reusing workers across Aligners), so
	// rebinding the thread-local logger is expected and not diagnostic-worthy.
	tl_logger = logger;
}

extern "C" mm_logger_t* mm_tl_logger(void)
{
	if (tl_logger) return tl_logger;
	// No thread-local logger bound: fall back to the process default. The
	// getter's function-local static makes this race-free and lock-free after the
	// one-time init, so there is no per-call mutex on the hot logging path.
	return static_cast<mm_logger_t*>(mm_get_or_create_default_logger());
}

/* ---- mm_log_format: printf-style → thread-local buffer ---- */

static thread_local std::string g_format_buf;

extern "C" const char* mm_log_format(const char* fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	g_format_buf = vformat(fmt, args);
	va_end(args);
	return g_format_buf.c_str();
}

/* ============================================================================
 * Profiler Trace Logging Implementation
 * ============================================================================ */

#ifdef MM_TRACE

#include <chrono>
#include <spdlog/fmt/fmt.h>
#include <string>

// Thread-local trace depth for indentation
static thread_local int g_trace_depth = 0;

static inline uint64_t get_time_ns()
{
	auto now = std::chrono::steady_clock::now();
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	    now.time_since_epoch())
	    .count();
}

static std::string format_duration(uint64_t ns)
{
	if (ns >= 1000000000ULL)
		return fmt::format("{:.3f}s", static_cast<double>(ns) / 1e9);
	if (ns >= 1000000ULL)
		return fmt::format("{:.3f}ms", static_cast<double>(ns) / 1e6);
	if (ns >= 1000ULL)
		return fmt::format("{:.3f}us", static_cast<double>(ns) / 1e3);
	return fmt::format("{}ns", ns);
}

// Emit a trace-level message with indentation via g_debug_logger (or stderr).
template <typename... Args>
static void trace_log(spdlog::source_loc loc, fmt::format_string<Args...> fstr, Args&&... args)
{
	auto msg = fmt::format(fstr, std::forward<Args>(args)...);
	auto* dl = static_cast<mm_default_logger*>(mm_tl_logger());
	if (dl) {
		auto& dbg = dl->get_debug_logger();
		if (__builtin_expect(dbg != nullptr, 1)) {
			dbg->log(loc, spdlog::level::trace, msg);
			return;
		}
	}
	fmt::print(stderr, "{}\n", msg);
}

extern "C" uint64_t mm_trace_enter(const char* name, const char* file, int line)
{
	std::string indent(static_cast<size_t>(g_trace_depth) * 2, ' ');
	trace_log({file, line, ""}, "{}>> {}", indent, name);

	if (g_trace_depth >= 64) {
		static bool warned = false;
		if (!warned) {
			warned = true;
			auto* dl = static_cast<mm_default_logger*>(mm_tl_logger());
			if (dl) {
				auto& dbg = dl->get_debug_logger();
				if (dbg)
					dbg->log(spdlog::level::warn,
					    "Warning: trace depth exceeded 64, timing data may be incorrect");
			}
		}
	}

	g_trace_depth++;
	return get_time_ns();
}

extern "C" void mm_trace_exit(const char* name, const char* file, int line, uint64_t start_time)
{
	if (g_trace_depth > 0) g_trace_depth--;
	std::string indent(static_cast<size_t>(g_trace_depth) * 2, ' ');

	if (start_time > 0)
		trace_log({file, line, ""}, "{}<< {} [{}]", indent, name,
		    format_duration(get_time_ns() - start_time));
	else
		trace_log({file, line, ""}, "{}<<", indent);
}

extern "C" void mm_trace_mark_event(const char* name, const char* file, int line)
{
	std::string indent(static_cast<size_t>(g_trace_depth) * 2, ' ');
	trace_log({file, line, ""}, "{}-- {}", indent, name);
}

extern "C" void mm_trace_log_msg(const char* file, int line, const char* fmt_str, ...)
{
	va_list args;
	va_start(args, fmt_str);
	std::string user_msg = vformat(fmt_str, args);
	va_end(args);
	std::string indent(static_cast<size_t>(g_trace_depth) * 2, ' ');
	trace_log({file, line, ""}, "{}   {}", indent, user_msg);
}

extern "C" int mm_trace_get_depth(void)
{
	return g_trace_depth;
}

#endif /* MM_TRACE */