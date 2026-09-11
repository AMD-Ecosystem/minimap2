#!/usr/bin/env python3
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

"""
Shared pipeline stage definitions for minimap2 profiling scripts.

Used by:
  - analyze_stages.py   (classifies functions into stages)
  - generate_profiling_report.py  (compares stages between CPU and GPU runs)
"""

# Pipeline stage names in display order
PIPELINE_STAGES = ["seeding", "io", "chaining", "alignment", "gpu_other", "other"]

# C macro name -> stage mapping (used when scanning minimap2 source)
PROFILE_MACRO_TO_STAGE = {
    "MM_PROFILE_SEED": "seeding",
    "MM_PROFILE_CHAIN": "chaining",
    "MM_PROFILE_ALIGN": "alignment",
    "MM_PROFILE_IO": "io",
    "MM_PROFILE_GPU": "gpu_other",
}


def reclassify_gpu_chain(name: str, current_stage: str) -> str:
    """Reclassify gpu_chain_* functions from gpu_other to chaining."""
    if "gpu_chain" in name.lower() and current_stage == "gpu_other":
        return "chaining"
    return current_stage


def shorten_kernel_name(name: str) -> str:
    """Shorten a GPU kernel name by removing template parameters."""
    return name.split("(")[0] if "(" in name else name
