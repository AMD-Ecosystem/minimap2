# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# =============================================================================
# Unified Version Management
# =============================================================================
# Primary source: git tags (e.g. v2.24, v2.24.0, v2.24.0-alpha.1, v2.24.0a1).
# Fallback:       VERSION file at the project root (for tarballs, sdists,
#                 shallow clones without tags).
#
# During configure, the VERSION file is (re-)generated so that consumers
# outside of CMake (pyproject.toml / scikit-build-core) can read it.
#
# Consumers:
#   - CMake project()          -> PROJECT_VERSION
#   - C code (mm_version.c)    -> MM_VERSION_FULL, MM_VERSION_MAJOR, etc.
#   - Python (pyproject.toml)  -> package metadata version (reads VERSION file)
#   - Cython (mappy.pyx)       -> __version__
#   - CPack (.deb)             -> CPACK_PACKAGE_VERSION
#
# Version format examples:
#   Release:     2.24.0
#   Pre-release: 2.24.0-alpha.1  (SemVer tag: v2.24.0-alpha.1)
#   Pre-release: 2.24.0-alpha.1  (PEP 440 tag: v2.24.0a1)
#   Dev build:   2.24.0-alpha.1+g34cc1a3
#   Dirty:       2.24.0-alpha.1+g34cc1a3.dirty
# =============================================================================

# ---- Derive version from git describe (primary) or VERSION file (fallback) --

set(_got_version_from_git FALSE)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    # Try git describe --tags --match "v*"
    # Output examples: "v2.24-267-g34cc1a3", "v2.24.0-0-gabcdef0", "v2.24.0-alpha.1-5-g1234abc"
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --match "v*" --long
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE _git_describe
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _git_result
    )
    if(_git_result EQUAL 0 AND _git_describe)
        # Parse:  v<TAG_BODY>-<DISTANCE>-g<HASH>
        # TAG_BODY may be:  2.24  |  2.24.0  |  2.24.0-alpha.1
        # We split from the RIGHT: last two dash-separated tokens are distance and hash.
        string(REGEX MATCH "^v(.+)-([0-9]+)-g([0-9a-f]+)$" _match "${_git_describe}")
        if(_match)
            set(_tag_body   "${CMAKE_MATCH_1}")   # e.g. "2.24" or "2.24.0-alpha.1"
            set(MM_GIT_DISTANCE "${CMAKE_MATCH_2}")
            set(MM_GIT_HASH_SHORT "${CMAKE_MATCH_3}")

            # Parse tag body: MAJOR.MINOR[.PATCH][-PRERELEASE]
            # Supports both SemVer (2.24.0-alpha.1) and PEP 440 (2.24.0a1) tags.
            string(REGEX MATCH "^([0-9]+)\\.([0-9]+)(\\.([0-9]+))?(-.+)?$" _tv "${_tag_body}")
            if(_tv)
                set(MM_VERSION_MAJOR "${CMAKE_MATCH_1}")
                set(MM_VERSION_MINOR "${CMAKE_MATCH_2}")
                if(CMAKE_MATCH_4)
                    set(MM_VERSION_PATCH "${CMAKE_MATCH_4}")
                else()
                    set(MM_VERSION_PATCH "0")
                endif()
                set(MM_VERSION_PRERELEASE "${CMAKE_MATCH_5}")  # includes leading '-' or ""
                set(_got_version_from_git TRUE)
            else()
                # Try PEP 440 style: MAJOR.MINOR.PATCH{a|b|rc}N (e.g. 2.24.0a1)
                string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(a|b|rc)([0-9]+)$" _pv "${_tag_body}")
                if(_pv)
                    set(MM_VERSION_MAJOR "${CMAKE_MATCH_1}")
                    set(MM_VERSION_MINOR "${CMAKE_MATCH_2}")
                    set(MM_VERSION_PATCH "${CMAKE_MATCH_3}")
                    set(_pep_tag "${CMAKE_MATCH_4}")
                    set(_pep_num "${CMAKE_MATCH_5}")
                    # Convert PEP 440 prerelease to SemVer: a1->-alpha.1, b2->-beta.2, rc1->-rc.1
                    if(_pep_tag STREQUAL "a")
                        set(MM_VERSION_PRERELEASE "-alpha.${_pep_num}")
                    elseif(_pep_tag STREQUAL "b")
                        set(MM_VERSION_PRERELEASE "-beta.${_pep_num}")
                    elseif(_pep_tag STREQUAL "rc")
                        set(MM_VERSION_PRERELEASE "-rc.${_pep_num}")
                    endif()
                    set(_got_version_from_git TRUE)
                endif()
            endif()
        endif()
    endif()

    # Full commit hash
    if(NOT MM_GIT_HASH_SHORT STREQUAL "")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            OUTPUT_VARIABLE MM_GIT_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()

    # Dirty check
    execute_process(
        COMMAND ${GIT_EXECUTABLE} diff --quiet HEAD
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RESULT_VARIABLE _git_dirty_result
        ERROR_QUIET
    )
    if(NOT _git_dirty_result EQUAL 0)
        set(MM_GIT_DIRTY TRUE)
    else()
        set(MM_GIT_DIRTY FALSE)
    endif()
endif()

# ---- Fallback: read VERSION file (tarballs, sdists, shallow clones) ----
if(NOT _got_version_from_git)
    if(EXISTS "${CMAKE_SOURCE_DIR}/VERSION")
        file(READ "${CMAKE_SOURCE_DIR}/VERSION" _version_raw)
        string(STRIP "${_version_raw}" _version_str)
        # VERSION file is PEP 440: MAJOR.MINOR.PATCH[{a|b|rc}N]
        string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)((a|b|rc)([0-9]*))?$" _match "${_version_str}")
        if(_match)
            set(MM_VERSION_MAJOR "${CMAKE_MATCH_1}")
            set(MM_VERSION_MINOR "${CMAKE_MATCH_2}")
            set(MM_VERSION_PATCH "${CMAKE_MATCH_3}")
            # Convert PEP 440 prerelease back to SemVer: a1->-alpha.1, b2->-beta.2, rc1->-rc.1
            set(_pep_tag "${CMAKE_MATCH_5}")
            set(_pep_num "${CMAKE_MATCH_6}")
            if(_pep_tag)
                if(NOT _pep_num)
                    set(_pep_num "0")
                endif()
                if(_pep_tag STREQUAL "a")
                    set(MM_VERSION_PRERELEASE "-alpha.${_pep_num}")
                elseif(_pep_tag STREQUAL "b")
                    set(MM_VERSION_PRERELEASE "-beta.${_pep_num}")
                elseif(_pep_tag STREQUAL "rc")
                    set(MM_VERSION_PRERELEASE "-rc.${_pep_num}")
                endif()
            else()
                set(MM_VERSION_PRERELEASE "")
            endif()
        else()
            message(FATAL_ERROR "VERSION file does not match PEP 440 (MAJOR.MINOR.PATCH[{a|b|rc}N]): '${_version_str}'")
        endif()
    else()
        message(FATAL_ERROR "Cannot determine version: no git tags and no VERSION file found.")
    endif()
endif()

# Defaults for git metadata when unavailable
if(NOT DEFINED MM_GIT_HASH)
    set(MM_GIT_HASH "unknown")
endif()
if(NOT DEFINED MM_GIT_HASH_SHORT OR MM_GIT_HASH_SHORT STREQUAL "")
    set(MM_GIT_HASH_SHORT "unknown")
endif()
if(NOT DEFINED MM_GIT_DISTANCE)
    set(MM_GIT_DISTANCE "0")
endif()
if(NOT DEFINED MM_GIT_DIRTY)
    set(MM_GIT_DIRTY FALSE)
endif()

# Strip leading dash from prerelease for convenience
if(MM_VERSION_PRERELEASE)
    string(SUBSTRING "${MM_VERSION_PRERELEASE}" 1 -1 MM_VERSION_PRERELEASE_CLEAN)
else()
    set(MM_VERSION_PRERELEASE_CLEAN "")
endif()

# Base version string (no build metadata)
set(MM_BASE_VERSION "${MM_VERSION_MAJOR}.${MM_VERSION_MINOR}.${MM_VERSION_PATCH}")
if(MM_VERSION_PRERELEASE_CLEAN)
    string(APPEND MM_BASE_VERSION "-${MM_VERSION_PRERELEASE_CLEAN}")
endif()

# ---- Build version strings for each target format ----

# SemVer string for C/C++ (MM_VERSION in mm_version.h)
# Format: MAJOR.MINOR.PATCH[-PRERELEASE][+gHASH[.dirty]]
set(MM_VERSION_STRING "${MM_VERSION_MAJOR}.${MM_VERSION_MINOR}.${MM_VERSION_PATCH}")
if(MM_VERSION_PRERELEASE_CLEAN)
    string(APPEND MM_VERSION_STRING "-${MM_VERSION_PRERELEASE_CLEAN}")
endif()

# Build metadata suffix (after '+')
set(_build_meta "")
if(NOT MM_GIT_HASH_SHORT STREQUAL "unknown")
    set(_build_meta "g${MM_GIT_HASH_SHORT}")
endif()
if(MM_GIT_DIRTY)
    if(_build_meta)
        string(APPEND _build_meta ".dirty")
    else()
        set(_build_meta "dirty")
    endif()
endif()
if(_build_meta)
    string(APPEND MM_VERSION_STRING "+${_build_meta}")
endif()

# Full version for C (with build suffix identifying this fork)
set(MM_VERSION_FULL "${MM_VERSION_STRING}")

# CMake project version (MAJOR.MINOR.PATCH only — no prerelease)
set(MM_CMAKE_VERSION "${MM_VERSION_MAJOR}.${MM_VERSION_MINOR}.${MM_VERSION_PATCH}")

# PEP 440 version for Python
# Mapping: alpha.N -> aN, beta.N -> bN, rc.N -> rcN
set(MM_PYTHON_VERSION "${MM_VERSION_MAJOR}.${MM_VERSION_MINOR}.${MM_VERSION_PATCH}")
if(MM_VERSION_PRERELEASE_CLEAN)
    # Parse prerelease: alpha.1, beta.2, rc.1
    string(REGEX MATCH "^(alpha|beta|rc)\\.?([0-9]*)" _pre_match "${MM_VERSION_PRERELEASE_CLEAN}")
    if(_pre_match)
        set(_pre_tag "${CMAKE_MATCH_1}")
        set(_pre_num "${CMAKE_MATCH_2}")
        if(NOT _pre_num)
            set(_pre_num "0")
        endif()
        if(_pre_tag STREQUAL "alpha")
            string(APPEND MM_PYTHON_VERSION "a${_pre_num}")
        elseif(_pre_tag STREQUAL "beta")
            string(APPEND MM_PYTHON_VERSION "b${_pre_num}")
        elseif(_pre_tag STREQUAL "rc")
            string(APPEND MM_PYTHON_VERSION "rc${_pre_num}")
        endif()
    else()
        # Unknown prerelease format, use .dev0
        string(APPEND MM_PYTHON_VERSION ".dev0")
    endif()
endif()

# Debian version: ~ sorts before release, so 2.24.0~alpha1 < 2.24.0
set(MM_DEB_VERSION "${MM_VERSION_MAJOR}.${MM_VERSION_MINOR}.${MM_VERSION_PATCH}")
if(MM_VERSION_PRERELEASE_CLEAN)
    string(REGEX MATCH "^(alpha|beta|rc)\\.?([0-9]*)" _pre_match "${MM_VERSION_PRERELEASE_CLEAN}")
    if(_pre_match)
        set(_pre_tag "${CMAKE_MATCH_1}")
        set(_pre_num "${CMAKE_MATCH_2}")
        if(NOT _pre_num)
            set(_pre_num "0")
        endif()
        string(APPEND MM_DEB_VERSION "~${_pre_tag}${_pre_num}")
    endif()
endif()

# RPM version: MAJOR.MINOR.PATCH with pre-release in Release field
# RPM doesn't allow hyphens in Version; pre-release goes into Release tag.
# e.g. Version: 2.24.0  Release: 0.1.alpha1  (pre-release)
#      Version: 2.24.0  Release: 1            (stable)
set(MM_RPM_VERSION "${MM_VERSION_MAJOR}.${MM_VERSION_MINOR}.${MM_VERSION_PATCH}")
if(MM_VERSION_PRERELEASE_CLEAN)
    string(REGEX MATCH "^(alpha|beta|rc)\\.?([0-9]*)" _pre_match "${MM_VERSION_PRERELEASE_CLEAN}")
    if(_pre_match)
        set(_pre_tag "${CMAKE_MATCH_1}")
        set(_pre_num "${CMAKE_MATCH_2}")
        if(NOT _pre_num)
            set(_pre_num "0")
        endif()
        set(MM_RPM_RELEASE "0.1.${_pre_tag}${_pre_num}")
    else()
        message(FATAL_ERROR
            "Unsupported prerelease format '${MM_VERSION_PRERELEASE_CLEAN}' for RPM packaging. "
            "Only alpha, beta, and rc are supported (e.g. v2.24.0-alpha.1). "
            "Please use a supported prerelease tag or a stable version.")
    endif()
else()
    set(MM_RPM_RELEASE "1")
endif()

# ---- Generate VERSION file (PEP 440 — so pyproject.toml / sdists can read it) ----
file(WRITE "${CMAKE_SOURCE_DIR}/VERSION" "${MM_PYTHON_VERSION}\n")

# ---- Compile definitions for mm_version.c ----
# These are passed as -D flags to the compiler for the mm2_core target.
# String values are escaped with \" so they become C string literals.
if(MM_GIT_DIRTY)
    set(_dirty_int 1)
else()
    set(_dirty_int 0)
endif()

set(MM_VERSION_DEFINITIONS
    MM_VERSION_MAJOR=${MM_VERSION_MAJOR}
    MM_VERSION_MINOR=${MM_VERSION_MINOR}
    MM_VERSION_PATCH=${MM_VERSION_PATCH}
    MM_VERSION_FULL="${MM_VERSION_FULL}"
    MM_VERSION_PRERELEASE="${MM_VERSION_PRERELEASE_CLEAN}"
    MM_GIT_HASH="${MM_GIT_HASH}"
    MM_GIT_HASH_SHORT="${MM_GIT_HASH_SHORT}"
    MM_GIT_DIRTY=${_dirty_int}
)

# ---- Report ----
message(STATUS "=== Version Info ===")
if(_got_version_from_git)
    set(_version_source "git tag")
else()
    set(_version_source "VERSION file (fallback)")
endif()
message(STATUS "  Source:          ${_version_source}")
message(STATUS "  Base version:    ${MM_BASE_VERSION}")
message(STATUS "  SemVer (C/C++):  ${MM_VERSION_FULL}")
message(STATUS "  CMake project:   ${MM_CMAKE_VERSION}")
message(STATUS "  Python (PEP440): ${MM_PYTHON_VERSION}")
if(BUILD_DEB_PACKAGE)
    message(STATUS "  Debian:          ${MM_DEB_VERSION}")
endif()
if(BUILD_RPM_PACKAGE OR BUILD_SLES_RPM_PACKAGE)
    message(STATUS "  RPM:             ${MM_RPM_VERSION}-${MM_RPM_RELEASE}")
endif()
message(STATUS "  Git hash:        ${MM_GIT_HASH_SHORT}")
message(STATUS "  Git distance:    ${MM_GIT_DISTANCE}")
message(STATUS "  Git dirty:       ${MM_GIT_DIRTY}")
message(STATUS "====================")

