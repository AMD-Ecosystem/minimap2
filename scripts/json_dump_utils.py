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
Shared utilities for parsing minimap2 --extra-out-dir --dump-intermediates JSON output.

Used by:
  - generate_test_data.py  (combines per-query JSON into test data)
  - compare_json_intermediates.py  (compares per-query JSON between runs)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Subdirectory names under the --extra-out-dir --dump-intermediates JSON output
STAGE_DIRS = ("seeds", "chains", "alignments")

# Field names in per-query JSON files
SEED_FIELDS = ("x", "y")
CHAIN_FIELDS = ("score", "length", "first_qpos", "first_tpos",
                "last_qpos", "last_tpos", "rev", "rid", "rname")
ALIGNMENT_FIELDS = ("rid", "score", "dp_score", "qs", "qe", "rs", "re",
                    "mapq", "rev", "rname", "cigar",
                    "n_cigar_ops", "seg_id", "score0", "inv", "parent",
                    "subsc", "mlen", "blen", "n_sub", "sam_pri",
                    "dp_max", "dp_max2", "n_ambi")

# Top-level metadata fields compared for each stage (beyond array data)
SEED_META_FIELDS = ("name", "length", "n_seeds", "n_seg", "rep_len",
                    "n_mini_pos", "n_minimizers")
CHAIN_META_FIELDS = ("name", "length", "n_seeds", "n_chains", "n_seg",
                     "frag_gap", "rep_len", "n_mini_pos")
ALIGNMENT_META_FIELDS = ("name", "length", "n_alignments", "n_segs")

# Index parameter fields (nested under "index_params" in all stages)
INDEX_PARAM_FIELDS = ("k", "w")

# All known top-level keys per stage (for detecting unknown fields)
SEED_ALL_KEYS = set(SEED_META_FIELDS) | {"seeds", "index_params", "qlens", "seq_prefixes"}
CHAIN_ALL_KEYS = set(CHAIN_META_FIELDS) | {"seeds", "chains", "index_params"}
ALIGNMENT_ALL_KEYS = set(ALIGNMENT_META_FIELDS) | {"alignments", "index_params"}

STAGE_ALL_KEYS = {
    "seeds": SEED_ALL_KEYS,
    "chains": CHAIN_ALL_KEYS,
    "alignments": ALIGNMENT_ALL_KEYS,
}


def load_json(filepath) -> Optional[Dict]:
    """Load a JSON file. Return None if not found or invalid."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def discover_query_names(dump_dir, stages=None) -> Set[str]:
    """Discover all query names present across stage subdirectories.

    Returns the set of query names (filename stems without .json).
    """
    stages = stages or STAGE_DIRS
    names: Set[str] = set()
    for stage in stages:
        stage_path = Path(dump_dir) / stage
        if stage_path.is_dir():
            for entry in stage_path.iterdir():
                if entry.suffix == ".json":
                    names.add(entry.stem)
    return names


def load_stage_json(dump_dir, stage: str, query_name: str) -> Optional[Dict]:
    """Load a single per-query JSON file for a given stage."""
    path = Path(dump_dir) / stage / f"{query_name}.json"
    return load_json(path)
