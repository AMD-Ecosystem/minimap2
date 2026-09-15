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

#ifndef HIP_RAII_H_
#define HIP_RAII_H_

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdio>
#include <utility>

#include "mm_log.h"

static inline bool hip_raii_check(hipError_t err, const char* file, int line, const char* func)
{
	if (err == hipSuccess) return true;
	mm_log_error("[hip_raii] HIP error {} ({}) at {}:{}:{}", (int)err,
	    hipGetErrorString(err), file, line, func);
	return false;
}
#define HIP_RAII_OK(x) hip_raii_check((x), __FILE__, __LINE__, __FUNCTION__)

template <typename T>
class hip_device_buf
{
      public:
	hip_device_buf() = default;

	hip_device_buf(const hip_device_buf&) = delete;
	hip_device_buf& operator=(const hip_device_buf&) = delete;

	hip_device_buf(hip_device_buf&& o) noexcept
	    : ptr_(o.ptr_), cap_(o.cap_), stream_(o.stream_), pool_(o.pool_)
	{
		o.ptr_ = nullptr;
		o.cap_ = 0;
	}

	hip_device_buf& operator=(hip_device_buf&& o) noexcept
	{
		if (this != &o) {
			free();
			ptr_ = o.ptr_;
			cap_ = o.cap_;
			stream_ = o.stream_;
			pool_ = o.pool_;
			o.ptr_ = nullptr;
			o.cap_ = 0;
		}
		return *this;
	}

	~hip_device_buf() { free(); }

	void bind(hipStream_t s, hipMemPool_t p)
	{
		stream_ = s;
		pool_ = p;
	}

	hipError_t grow(size_t count)
	{
		if (count <= cap_) return hipSuccess;
		free();
		hipError_t err = hipMallocFromPoolAsync((void**)&ptr_, count * sizeof(T), pool_, stream_);
		if (err == hipSuccess)
			cap_ = count;
		return err;
	}

	void free()
	{
		if (ptr_) {
			HIP_RAII_OK(hipFreeAsync(ptr_, stream_));
			ptr_ = nullptr;
			cap_ = 0;
		}
	}

	T* get() const { return ptr_; }
	T* operator*() const { return ptr_; }
	size_t capacity() const { return cap_; }
	explicit operator bool() const { return ptr_ != nullptr; }

      private:
	T* ptr_ = nullptr;
	size_t cap_ = 0;
	hipStream_t stream_ = nullptr;
	hipMemPool_t pool_ = nullptr;
};

template <typename T>
class hip_pinned_buf
{
      public:
	hip_pinned_buf() = default;

	hip_pinned_buf(const hip_pinned_buf&) = delete;
	hip_pinned_buf& operator=(const hip_pinned_buf&) = delete;

	hip_pinned_buf(hip_pinned_buf&& o) noexcept : ptr_(o.ptr_), cap_(o.cap_)
	{
		o.ptr_ = nullptr;
		o.cap_ = 0;
	}

	hip_pinned_buf& operator=(hip_pinned_buf&& o) noexcept
	{
		if (this != &o) {
			free();
			ptr_ = o.ptr_;
			cap_ = o.cap_;
			o.ptr_ = nullptr;
			o.cap_ = 0;
		}
		return *this;
	}

	~hip_pinned_buf() { free(); }

	hipError_t alloc(size_t count)
	{
		free();
		hipError_t err = hipHostMalloc((void**)&ptr_, count * sizeof(T), hipHostMallocDefault);
		if (err == hipSuccess)
			cap_ = count;
		return err;
	}

	hipError_t grow(size_t count)
	{
		if (count <= cap_) return hipSuccess;
		return alloc(count);
	}

	void free()
	{
		if (ptr_) {
			HIP_RAII_OK(hipHostFree(ptr_));
			ptr_ = nullptr;
			cap_ = 0;
		}
	}

	T* get() const { return ptr_; }
	T* operator*() const { return ptr_; }
	size_t capacity() const { return cap_; }
	explicit operator bool() const { return ptr_ != nullptr; }

      private:
	T* ptr_ = nullptr;
	size_t cap_ = 0;
};

class hip_stream
{
      public:
	hip_stream() = default;

	hip_stream(const hip_stream&) = delete;
	hip_stream& operator=(const hip_stream&) = delete;

	hip_stream(hip_stream&& o) noexcept : stream_(o.stream_), owned_(o.owned_)
	{
		o.stream_ = nullptr;
		o.owned_ = false;
	}

	hip_stream& operator=(hip_stream&& o) noexcept
	{
		if (this != &o) {
			destroy();
			stream_ = o.stream_;
			owned_ = o.owned_;
			o.stream_ = nullptr;
			o.owned_ = false;
		}
		return *this;
	}

	~hip_stream() { destroy(); }

	hipError_t create()
	{
		destroy();
		hipError_t err = hipStreamCreate(&stream_);
		if (err == hipSuccess) owned_ = true;
		return err;
	}

	void adopt(hipStream_t s)
	{
		destroy();
		stream_ = s;
		owned_ = false;
	}

	void destroy()
	{
		if (stream_ && owned_) {
			HIP_RAII_OK(hipStreamSynchronize(stream_));
			HIP_RAII_OK(hipStreamDestroy(stream_));
		}
		stream_ = nullptr;
		owned_ = false;
	}

	hipStream_t get() const { return stream_; }
	operator hipStream_t() const { return stream_; }
	explicit operator bool() const { return stream_ != nullptr; }

      private:
	hipStream_t stream_ = nullptr;
	bool owned_ = false;
};

#endif /* HIP_RAII_H_ */
