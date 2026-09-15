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

#include "minimap.h"

// ---- Compile-time macros passed via -D from CMake (cmake/version.cmake) ----
// Provide safe defaults so the file compiles even without CMake.

#ifndef MM_VERSION_MAJOR
#define MM_VERSION_MAJOR 0
#endif
#ifndef MM_VERSION_MINOR
#define MM_VERSION_MINOR 0
#endif
#ifndef MM_VERSION_PATCH
#define MM_VERSION_PATCH 0
#endif

// String macros — CMake passes these as -DMM_VERSION_FULL=\"2.24.0+g34cc1a3\"
#ifndef MM_VERSION_FULL
#define MM_VERSION_FULL "0.0.0-unknown"
#endif
#ifndef MM_VERSION_PRERELEASE
#define MM_VERSION_PRERELEASE ""
#endif
#ifndef MM_GIT_HASH
#define MM_GIT_HASH "unknown"
#endif
#ifndef MM_GIT_HASH_SHORT
#define MM_GIT_HASH_SHORT "unknown"
#endif
#ifndef MM_GIT_DIRTY
#define MM_GIT_DIRTY 0
#endif

// ---- Static version info struct, populated at compile time ----

static const mm_version_t _mm_version_info = {
    .major = MM_VERSION_MAJOR,
    .minor = MM_VERSION_MINOR,
    .patch = MM_VERSION_PATCH,
    .full = MM_VERSION_FULL,
    .prerelease = MM_VERSION_PRERELEASE,
    .git_hash = MM_GIT_HASH,
    .git_short = MM_GIT_HASH_SHORT,
    .git_dirty = MM_GIT_DIRTY,
};

const char* mm_version(void)
{
	return MM_VERSION_FULL;
}

const mm_version_t* mm_version_info(void)
{
	return &_mm_version_info;
}
