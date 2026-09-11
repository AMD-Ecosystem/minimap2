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
compare_benchmark_data.py — Compare 1 baseline against N test directories.

Each directory (baseline and test) is expected to contain either:
  *.bin                   FlatBuffers binary files (from --dump-intermediates FLATBUF)
or:
  seeds/          per-query JSON files
  chains/         per-query JSON files
  alignments/     per-query JSON files

Plus optionally:
  stats/timer_summary.json          (optional, from direct minimap2 run)
  timer_summary_avg.json            (optional, from generate_benchmark_data.py)

The script compares every test directory against the single baseline:
  1. Intermediate correctness — per-query comparison (FLATBUF or JSON)
  2. Timing comparison        — wall time, per-stage breakdown, speedup

Usage:
  python3 scripts/compare_benchmark_data.py <baseline> <test1> [test2 ...] [OPTIONS]

Examples:
  python3 scripts/compare_benchmark_data.py baseline/ t1/ t2/ --timing-only
"""

import argparse
import json
import multiprocessing
import os
import sys
import threading
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import zstandard as zstd
from tqdm import tqdm

from json_dump_utils import STAGE_DIRS
from compare_json_intermediates import (
    compare_query_data as _json_compare_query_data,
    SummaryStats as JsonSummaryStats,
)
from compare_binary_intermediates import (
    _compare_one_query as _bin_compare_one_query,
    parse_query as parse_bin_query,
    SummaryStats as BinSummaryStats,
)

# Union type for either backend's SummaryStats
AnySummaryStats = Union[JsonSummaryStats, BinSummaryStats]


# ─────────────────────────────────────────────────────────────────────────────
# Colors
# ─────────────────────────────────────────────────────────────────────────────

class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    BOLD = '\033[1m'
    NC = '\033[0m'

    @classmethod
    def disable(cls):
        cls.GREEN = cls.RED = cls.YELLOW = cls.BLUE = cls.BOLD = cls.NC = ''


def header(msg: str):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 72}{Colors.NC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{msg}{Colors.NC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 72}{Colors.NC}\n")


def subheader(msg: str):
    print(f"\n{Colors.YELLOW}--- {msg} ---{Colors.NC}")


# ─────────────────────────────────────────────────────────────────────────────
# Source discovery
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceInfo:
    """Lightweight descriptor for an intermediate data source."""
    backend: str    # 'bin' or 'json'
    kind: str       # 'tar', 'zip', or 'dir'
    path: Path


def find_intermediates(data_dir: Path) -> Optional[SourceInfo]:
    """Detect intermediate source type and location.

    Returns SourceInfo or None.  Prefers bin > json, zip > dir.
    """
    for candidate in [data_dir, data_dir / "intermediates"]:
        # Zip archive (preferred — supports fast random access)
        zip_path = candidate / "intermediates.zip"
        if zip_path.is_file():
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
            if any(n.endswith((".bin", ".bin.zst")) for n in names):
                return SourceInfo("bin", "zip", zip_path)
            if any(n.startswith(("seeds/", "chains/", "alignments/")) for n in names):
                return SourceInfo("json", "zip", zip_path)

        # Flat binary directory
        if candidate.is_dir() and any(candidate.glob("*.bin")):
            return SourceInfo("bin", "dir", candidate)

        # JSON stage directories
        if any((candidate / s).is_dir() for s in STAGE_DIRS):
            return SourceInfo("json", "dir", candidate)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Source reader — unified access to tar / zip / dir
# ─────────────────────────────────────────────────────────────────────────────

class _SourceReader:
    """Context manager that indexes and reads raw bytes from a data source.

    Keys are normalised:
      bin backend  →  query name  (str)
      json backend →  (stage, query_name)  tuple
    """

    def __init__(self, info: SourceInfo):
        self.info = info
        self._handle = None
        self._index: Dict[Any, Any] = {}     # key → access token

    # ── context manager ────────────────────────────────────────────────────

    def __enter__(self):
        if self.info.kind == "zip":
            self._handle = zipfile.ZipFile(self.info.path, "r")
            for name in self._handle.namelist():
                key = self._normalise(name)
                if key is not None:
                    self._index[key] = name
        else:  # dir
            if self.info.backend == "bin":
                for f in self.info.path.glob("*.bin"):
                    self._index[f.stem] = f
            else:
                for stage in STAGE_DIRS:
                    sd = self.info.path / stage
                    if sd.is_dir():
                        for f in sd.glob("*.json"):
                            self._index[(stage, f.stem)] = f
        return self

    def __exit__(self, *args):
        if self._handle is not None:
            self._handle.close()

    # ── public API ─────────────────────────────────────────────────────────

    def keys(self):
        return set(self._index.keys())

    def read(self, key) -> Optional[bytes]:
        """Read raw bytes for *key*.  May be zstd-compressed."""
        token = self._index.get(key)
        if token is None:
            return None
        if self.info.kind == "zip":
            return self._handle.read(token)
        return token.read_bytes()           # dir: token is a Path

    # ── internal ───────────────────────────────────────────────────────────

    def _normalise(self, name: str):
        """Map an archive member name to a normalised key, or None to skip."""
        clean = name.replace(".zst", "")
        if self.info.backend == "bin":
            if clean.endswith(".bin"):
                return Path(clean).stem
        else:
            if clean.endswith(".json"):
                parts = Path(clean).parts
                if len(parts) >= 2 and parts[-2] in STAGE_DIRS:
                    return (parts[-2], Path(parts[-1]).stem)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Worker functions — run in child processes via ProcessPoolExecutor
# ─────────────────────────────────────────────────────────────────────────────

_ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'

# Shared counter for per-entry progress (set via pool initializer)
_progress_counter = None


def _init_worker(counter):
    """Pool initializer — gives each worker a reference to the shared counter."""
    global _progress_counter
    _progress_counter = counter


def _decompress(raw: bytes) -> bytes:
    """Decompress zstd if needed, otherwise return as-is."""
    if len(raw) >= 4 and raw[:4] == _ZSTD_MAGIC:
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(raw)
    return raw


def _process_chunk(args):
    """Process a chunk of keys in a worker process.

    Each process opens its own file handles, reads/decompresses/compares,
    and returns aggregated counts plus any output lines.

    Returns (partial_stats, output_lines) where:
      partial_stats = {label: {stage: [matches, mismatches, missing1, missing2]}}
      output_lines  = list of formatted strings for verbose/mismatch output
    """
    global _progress_counter
    chunk, baseline_info, test_infos, stages, backend, verbose = args

    partial = {}
    for label, _ in test_infos:
        partial[label] = {s: [0, 0, 0, 0] for s in stages}
    output_lines = []

    with _SourceReader(baseline_info) as src_baseline:
        test_readers = []
        try:
            for label, ti in test_infos:
                r = _SourceReader(ti)
                r.__enter__()
                test_readers.append((label, r))

            for key in chunk:
                raw_base = src_baseline.read(key)

                if backend == "bin":
                    d1 = parse_bin_query(_decompress(raw_base)) if raw_base else None
                    for label, r in test_readers:
                        raw_t = r.read(key)
                        d2 = parse_bin_query(_decompress(raw_t)) if raw_t else None
                        for stage, result in _bin_compare_one_query(key, d1, d2, stages):
                            counts = partial[label][stage]
                            if result.match:
                                counts[0] += 1
                            else:
                                counts[1] += 1
                                if result.missing_in_dir1:
                                    counts[2] += 1
                                elif result.missing_in_dir2:
                                    counts[3] += 1
                            if verbose or not result.match:
                                output_lines.append(f"  [{label}] {result}")
                                if verbose and result.differences:
                                    for diff in result.differences:
                                        output_lines.append(f"    {diff}")
                else:
                    stage, qname = key
                    d1 = json.loads(_decompress(raw_base)) if raw_base else None
                    for label, r in test_readers:
                        raw_t = r.read(key)
                        d2 = json.loads(_decompress(raw_t)) if raw_t else None
                        result = _json_compare_query_data(qname, d1, d2, stage)
                        counts = partial[label][stage]
                        if result.match:
                            counts[0] += 1
                        else:
                            counts[1] += 1
                            if result.missing_in_dir1:
                                counts[2] += 1
                            elif result.missing_in_dir2:
                                counts[3] += 1
                        if verbose or not result.match:
                            output_lines.append(f"  [{label}] {result}")
                            if verbose and result.differences:
                                for diff in result.differences:
                                    output_lines.append(f"    {diff}")

                if _progress_counter is not None:
                    with _progress_counter.get_lock():
                        _progress_counter.value += 1
        finally:
            for _, r in reversed(test_readers):
                r.__exit__(None, None, None)

    return partial, output_lines


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

_TIMER_CANDIDATES = [
    "timer_summary_avg.json",
    "stats/timer_summary.json",
]


def find_timing(data_dir: Path) -> Optional[Path]:
    """Return the path to the first timer JSON found."""
    for candidate in _TIMER_CANDIDATES:
        p = data_dir / candidate
        if p.is_file():
            return p
    return None


def load_timing(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Intermediate comparison — 4-step pipeline
# ─────────────────────────────────────────────────────────────────────────────

def compare_intermediates(
    baseline_info: SourceInfo,
    test_infos: List[Tuple[str, SourceInfo]],
    stages: List[str],
    read_filter: Optional[str],
    verbose: bool,
) -> Dict[str, Dict[str, AnySummaryStats]]:
    """Compare intermediates between one baseline and multiple tests.

    *test_infos* is a list of ``(label, SourceInfo)`` pairs.

    Returns ``{label: {stage: stats}}`` for each test.

    Pipeline:
      1. Enumerate keys from baseline + all test sources (fast central-dir reads)
      2. Split key list into chunks
      3. Dispatch chunks to ProcessPoolExecutor — each process opens its own
         file handles, reads/decompresses/compares, returns aggregated counts
      4. Merge partial counts in main process
    """
    backend = baseline_info.backend

    # Validate backends match
    valid_tests: List[Tuple[str, SourceInfo]] = []
    for label, ti in test_infos:
        if ti.backend != backend:
            print(f"  {Colors.RED}[{label}] Backend mismatch: baseline={backend}, "
                  f"test={ti.backend}. Skipping.{Colors.NC}")
        else:
            valid_tests.append((label, ti))
    if not valid_tests:
        return {}

    print(f"  Baseline: {baseline_info.path}  ({baseline_info.kind})")
    for label, ti in valid_tests:
        print(f"  Test [{label}]: {ti.path}  ({ti.kind})")

    # ── Step 1: enumerate keys (temporary readers, closed immediately) ─────
    with _SourceReader(baseline_info) as src:
        keys_base = src.keys()
    if read_filter:
        if backend == "bin":
            keys_base = {k for k in keys_base if read_filter in k}
        else:
            keys_base = {k for k in keys_base if read_filter in k[1]}
    if backend == "json":
        stage_set = set(stages)
        keys_base = {k for k in keys_base if k[0] in stage_set}

    all_keys = set(keys_base)
    for label, ti in valid_tests:
        with _SourceReader(ti) as r:
            ks = r.keys()
        if read_filter:
            if backend == "bin":
                ks = {k for k in ks if read_filter in k}
            else:
                ks = {k for k in ks if read_filter in k[1]}
        if backend == "json":
            ks = {k for k in ks if k[0] in stage_set}
        all_keys |= ks
    all_keys = sorted(all_keys)

    if not all_keys:
        print(f"  {Colors.YELLOW}No entries to compare{Colors.NC}")
        return {}

    # ── Step 2: split into chunks for ProcessPoolExecutor ──────────────────
    n_workers = min(os.cpu_count() or 4, max(1, len(all_keys)))
    chunk_size = -(-len(all_keys) // n_workers)
    chunks = [all_keys[i:i + chunk_size] for i in range(0, len(all_keys), chunk_size)]

    print(f"  Comparing {len(all_keys)} entries × {len(valid_tests)} tests "
          f"across {len(chunks)} processes ...")

    # ── Step 3: dispatch to worker processes ───────────────────────────────
    _Stats = BinSummaryStats if backend == "bin" else JsonSummaryStats
    per_test: Dict[str, Dict[str, AnySummaryStats]] = {}
    for label, _ in valid_tests:
        per_test[label] = {s: _Stats() for s in stages}

    progress_counter = multiprocessing.Value('i', 0)
    pbar = tqdm(total=len(all_keys), desc="  Comparing", unit="entry",
                bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    # Monitor thread: polls shared counter and refreshes tqdm
    stop_monitor = threading.Event()
    def _monitor():
        while not stop_monitor.wait(0.25):
            pbar.n = progress_counter.value
            pbar.refresh()
    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()

    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_worker,
                             initargs=(progress_counter,)) as pool:
        futures = []
        for chunk in chunks:
            args = (chunk, baseline_info, valid_tests, stages, backend, verbose)
            futures.append(pool.submit(_process_chunk, args))

        # ── Step 4: merge results ──────────────────────────────────────────
        for future in as_completed(futures):
            partial, output_lines = future.result()
            for line in output_lines:
                tqdm.write(line)
            for label in partial:
                for stage in partial[label]:
                    m, mm, md1, md2 = partial[label][stage]
                    stats = per_test[label][stage]
                    stats.matches += m
                    stats.mismatches += mm
                    stats.total += m + mm
                    stats.missing_dir1 += md1
                    stats.missing_dir2 += md2

    stop_monitor.set()
    monitor.join()
    pbar.n = progress_counter.value
    pbar.refresh()
    pbar.close()

    # ── Per-test totals ───────────────────────────────────────────────────
    for label, stats_d in per_test.items():
        total_m = sum(s.matches for s in stats_d.values())
        total_mm = sum(s.mismatches for s in stats_d.values())
        total = total_m + total_mm
        if total > 0:
            pct = total_m / total * 100
            color = Colors.GREEN if total_mm == 0 else Colors.RED
            print(f"\n  {color}[{label}] {total_m}/{total} queries match ({pct:.1f}%){Colors.NC}")
        else:
            print(f"  {Colors.YELLOW}[{label}] No queries to compare{Colors.NC}")

    return per_test


def print_intermediates_summary(
    per_test: Dict[str, Dict[str, AnySummaryStats]],
    test_labels: List[str],
) -> bool:
    """Print summary table. Returns True if all passed."""
    header("INTERMEDIATE COMPARISON SUMMARY")

    stages = list(STAGE_DIRS)
    col_w = 16

    hdr = f"  {'Test':<28}"
    for s in stages:
        hdr += f" | {s:^{col_w}}"
    hdr += f" | {'TOTAL':^{col_w}}"
    sep = f"  {'─' * 28}" + f"-+-{'─' * col_w}" * (len(stages) + 1)
    print(hdr)
    print(sep)

    all_pass = True
    for label in test_labels:
        stats = per_test.get(label)
        if not stats:
            continue
        row = f"  {label:<28}"
        t_total = t_match = 0
        for s in stages:
            st = stats.get(s)
            if st is None:
                st_total, st_matches, st_mismatches = 0, 0, 0
            else:
                st_total, st_matches, st_mismatches = st.total, st.matches, st.mismatches
            t_total += st_total
            t_match += st_matches
            if st_total == 0:
                row += f" | {'—':^{col_w}}"
            else:
                pct = st_matches / st_total * 100
                color = Colors.GREEN if st_mismatches == 0 else Colors.RED
                cell = f"{st_matches}/{st_total} ({pct:.0f}%)"
                row += f" | {color}{cell:^{col_w}}{Colors.NC}"
                if st_mismatches > 0:
                    all_pass = False

        if t_total > 0:
            pct = t_match / t_total * 100
            color = Colors.GREEN if t_match == t_total else Colors.RED
            cell = f"{t_match}/{t_total} ({pct:.0f}%)"
            row += f" | {color}{cell:^{col_w}}{Colors.NC}"
        else:
            row += f" | {'—':^{col_w}}"
        print(row)

    print()
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ratio_color(ratio: float) -> str:
    if ratio <= 1.0:
        return Colors.GREEN
    if ratio <= 1.1:
        return Colors.YELLOW
    return Colors.RED


def _timing_row(label: str, rv, tv, col_w: int = 14,
                *, pct_base_r: float = 0, pct_base_t: float = 0):
    """Print one row of the timing detail table."""
    try:
        rv_f, tv_f = float(rv), float(tv)
    except (TypeError, ValueError):
        print(f"  {label:<40} {str(rv):>{col_w}} {str(tv):>{col_w}}")
        return

    diff = tv_f - rv_f
    ratio = tv_f / rv_f if rv_f != 0 else float('inf')
    ratio_str = f"{ratio:.3f}x" if ratio != float('inf') else "N/A"
    diff_str = f"{diff:+.4f}"
    color = _ratio_color(ratio) if ratio != float('inf') else Colors.NC
    pct_r = f" ({rv_f / pct_base_r * 100:5.1f}%)" if pct_base_r > 0 else ""
    pct_t = f" ({tv_f / pct_base_t * 100:5.1f}%)" if pct_base_t > 0 else ""

    print(f"  {label:<40} {rv_f:>{col_w}.4f}{pct_r:>8} {tv_f:>{col_w}.4f}{pct_t:>8} "
          f"{diff_str:>{col_w}} {color}{ratio_str:>{col_w}}{Colors.NC}")


# ─────────────────────────────────────────────────────────────────────────────
# Timing comparison
# ─────────────────────────────────────────────────────────────────────────────

def print_timing_detail(
    baseline_data: dict,
    test_data: dict,
    label: str,
):
    """Print full timing comparison for one test vs baseline."""
    overview_fields = ["n_threads", "batch_count", "query_count"]
    time_fields = ["wall_time_sec", "cpu_time_sec", "peak_rss_gb"]
    aggregate_fields = [
        "stage_total_sum_sec", "stage_per_thread_equiv_sec",
        "avg_total_per_thread_per_batch_sec", "avg_total_per_query_sec",
        "approx_non_stage_overhead_sec",
    ]
    stage_metrics = [
        "sum_sec", "total_per_thread_sec",
        "min_thread_batch_sec", "max_thread_batch_sec", "avg_thread_batch_sec",
    ]
    col_w = 14

    subheader(label)
    short = label[:col_w]
    hdr = (f"  {'Field':<40} {'Baseline':>{col_w}} {'%wall':>8} "
           f"{short:>{col_w}} {'%wall':>8} "
           f"{'Diff':>{col_w}} {'Ratio':>{col_w}}")
    sep = "  " + "─" * (40 + col_w + 8 + 1 + col_w + 8 + 1 + col_w + 1 + col_w)
    print(hdr)
    print(sep)

    r_wall = baseline_data.get("wall_time_sec", 0)
    t_wall = test_data.get("wall_time_sec", 0)

    # Overview
    print(f"  {Colors.BOLD}Overview{Colors.NC}")
    for fld in overview_fields:
        _timing_row(f"  {fld}", baseline_data.get(fld, "—"), test_data.get(fld, "—"), col_w)

    # Time & Resources
    print(f"  {Colors.BOLD}Time & Resources{Colors.NC}")
    for fld in time_fields:
        _timing_row(f"  {fld}", baseline_data.get(fld, 0), test_data.get(fld, 0), col_w)

    # Per-stage breakdown
    ref_stages = baseline_data.get("stages", {})
    test_stages = test_data.get("stages", {})
    all_stages = sorted(set(ref_stages) | set(test_stages))

    if all_stages:
        print(f"  {Colors.BOLD}Per-Stage Breakdown{Colors.NC}")
        for sn in all_stages:
            rs = ref_stages.get(sn, {})
            ts = test_stages.get(sn, {})
            for metric in stage_metrics:
                fld = f"  stage/{sn}/{metric}"
                if metric == "sum_sec":
                    _timing_row(fld, rs.get(metric, 0), ts.get(metric, 0), col_w,
                                pct_base_r=r_wall, pct_base_t=t_wall)
                else:
                    _timing_row(fld, rs.get(metric, 0), ts.get(metric, 0), col_w)

    # Aggregates
    print(f"  {Colors.BOLD}Aggregates{Colors.NC}")
    for fld in aggregate_fields:
        pct_r = r_wall if fld in ("stage_total_sum_sec", "approx_non_stage_overhead_sec") else 0
        pct_t = t_wall if fld in ("stage_total_sum_sec", "approx_non_stage_overhead_sec") else 0
        _timing_row(f"  {fld}",
                     baseline_data.get(fld, 0), test_data.get(fld, 0), col_w,
                     pct_base_r=pct_r, pct_base_t=pct_t)
    print()


def print_stage_breakdown(
    baseline_data: dict,
    test_data_list: List[dict],
    test_labels: List[str],
):
    """Print per-stage time breakdown: baseline vs all tests side-by-side."""
    header("PER-STAGE TIME BREAKDOWN")

    col_w = 12
    stage_names = ["seed", "chain", "align"]
    run_labels = ["Baseline"] + test_labels

    # Header
    hdr = f"  {'Metric':<25}"
    for rl in run_labels:
        hdr += f" {rl[:col_w]:>{col_w}} {'%wall':>6}"
    sep = f"  {'─' * 25}" + ("─" + "─" * col_w + "─" + "─" * 6) * len(run_labels)
    print(hdr)
    print(sep)

    all_data = [baseline_data] + test_data_list

    row_specs = [("wall_time_sec", lambda d: d.get("wall_time_sec", 0))]
    for sn in stage_names:
        row_specs.append(
            (f"  {sn}", lambda d, _sn=sn: d.get("stages", {}).get(_sn, {}).get("sum_sec", 0))
        )
    row_specs.append(("  stage_total", lambda d: d.get("stage_total_sum_sec", 0)))
    row_specs.append(("  overhead", lambda d: d.get("approx_non_stage_overhead_sec", 0)))

    for row_name, extractor in row_specs:
        row = f"  {row_name:<25}"
        for rd in all_data:
            if rd is None:
                row += f" {'N/A':>{col_w}} {'':>6}"
                continue
            val = extractor(rd)
            wall = rd.get("wall_time_sec", 0)
            pct = val / wall * 100 if wall > 0 else 0
            row += f" {val:>{col_w}.4f}"
            row += f" {'':>6}" if row_name == "wall_time_sec" else f" {pct:>5.1f}%"
        print(row)

    # Speedup row
    rw = baseline_data.get("wall_time_sec", 0) if baseline_data else 0
    if rw > 0:
        row = f"  {'speedup (wall)':25}"
        row += f" {'1.000x':>{col_w}} {'':>6}"
        for td in test_data_list:
            tw = td.get("wall_time_sec", 0) if td else 0
            if tw > 0:
                sp = rw / tw
                color = Colors.GREEN if sp >= 1.0 else Colors.RED
                row += f" {color}{sp:>{col_w}.3f}x{Colors.NC} {'':>6}"
            else:
                row += f" {'N/A':>{col_w}} {'':>6}"
        print(row)

    print()


def print_wall_time_summary(
    baseline_data: dict,
    test_data_list: List[dict],
    test_labels: List[str],
):
    """Print compact wall-time summary row."""
    header("WALL TIME SUMMARY")

    col_w = 14
    hdr = f"  {'Baseline':>{col_w}}"
    for label in test_labels:
        hdr += f" {label[:col_w]:>{col_w}} {'speedup':>{col_w}}"
    sep = f"  {'─' * col_w}" + f"─{'─' * col_w}─{'─' * col_w}" * len(test_labels)
    print(hdr)
    print(sep)

    rw = baseline_data.get("wall_time_sec", 0) if baseline_data else 0
    row = f"  {rw:>{col_w}.4f}" if rw > 0 else f"  {'N/A':>{col_w}}"

    for td in test_data_list:
        tw = td.get("wall_time_sec", 0) if td else 0
        if tw > 0:
            row += f" {tw:>{col_w}.4f}"
            if rw > 0:
                sp = rw / tw
                color = Colors.GREEN if sp >= 1.0 else Colors.RED
                row += f" {color}{sp:>{col_w}.3f}x{Colors.NC}"
            else:
                row += f" {'N/A':>{col_w}}"
        else:
            row += f" {'N/A':>{col_w}} {'':>{col_w}}"

    print(row)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_results(
    per_test_stats: Dict[str, Dict[str, AnySummaryStats]],
    baseline_dir: Path,
    baseline_timing: Optional[dict],
    test_dirs: List[Path],
    test_labels: List[str],
    test_timing_list: List[Optional[dict]],
    output_path: Path,
):
    """Export results to JSON."""
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "baseline_dir": str(baseline_dir),
            "test_dirs": {label: str(td) for label, td in zip(test_labels, test_dirs)},
        },
        "intermediates": {},
        "timing": {},
    }

    for label, stats in per_test_stats.items():
        data["intermediates"][label] = {}
        for stage, s in stats.items():
            data["intermediates"][label][stage] = {
                "total": s.total,
                "matches": s.matches,
                "mismatches": s.mismatches,
                "missing_baseline": getattr(s, "missing_dir1", 0),
                "missing_test": getattr(s, "missing_dir2", 0),
            }

    rw = baseline_timing.get("wall_time_sec", 0) if baseline_timing else 0
    for label, td in zip(test_labels, test_timing_list):
        if not baseline_timing or not td:
            continue
        tw = td.get("wall_time_sec", 0)
        data["timing"][label] = {
            "baseline": baseline_timing,
            "test": td,
            "speedup": rw / tw if tw > 0 else None,
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results exported to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare 1 baseline against N test directories: "
                    "intermediate correctness + timing comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python3 scripts/compare_benchmark_data.py baseline/ test/
  python3 scripts/compare_benchmark_data.py baseline/ run1/ run2/ --labels v1 v2
  python3 scripts/compare_benchmark_data.py baseline/ test/ -s chains -v
  python3 scripts/compare_benchmark_data.py baseline/ t1/ t2/ --timing-only
  python3 scripts/compare_benchmark_data.py baseline/ test/ -o results.json
""",
    )

    parser.add_argument("baseline", type=Path, help="Baseline data directory")
    parser.add_argument("test_dirs", type=Path, nargs="+",
                        help="One or more test data directories to compare")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Custom labels for each test directory")
    parser.add_argument("-s", "--stage", choices=["seeds", "chains", "alignments"],
                        action="append", dest="stages",
                        help="Stage(s) to compare (default: all)")
    parser.add_argument("-r", "--read", dest="read_filter",
                        help="Filter to reads matching substring")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show per-query match details")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only show summary tables")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")
    parser.add_argument("-o", "--output", type=Path,
                        help="Export results to JSON file")
    parser.add_argument("--timing-only", action="store_true",
                        help="Skip intermediate comparison")
    parser.add_argument("--intermediates-only", action="store_true",
                        help="Skip timing comparison")

    args = parser.parse_args()
    if args.no_color:
        Colors.disable()

    # Validate directories
    if not args.baseline.exists():
        print(f"Error: baseline not found: {args.baseline}", file=sys.stderr)
        sys.exit(1)
    for td in args.test_dirs:
        if not td.exists():
            print(f"Error: test directory not found: {td}", file=sys.stderr)
            sys.exit(1)

    # Build labels
    if args.labels:
        if len(args.labels) != len(args.test_dirs):
            print(f"Error: --labels count ({len(args.labels)}) != "
                  f"test_dirs count ({len(args.test_dirs)})", file=sys.stderr)
            sys.exit(1)
        test_labels = args.labels
    else:
        test_labels = []
        seen: Dict[str, int] = {}
        for td in args.test_dirs:
            name = td.name
            if name in seen:
                seen[name] += 1
                test_labels.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                test_labels.append(name)

    stages = args.stages if args.stages else list(STAGE_DIRS)

    # ── Discover paths ─────────────────────────────────────────────────────
    baseline_info = find_intermediates(args.baseline)
    baseline_timer_path = find_timing(args.baseline)
    baseline_timing = load_timing(baseline_timer_path)

    header("Minimap2 Benchmark Data Comparison")
    print(f"  Baseline:       {args.baseline}")
    if baseline_info:
        print(f"  Intermediates:  {baseline_info.path}  ({baseline_info.backend}, {baseline_info.kind})")
    else:
        print(f"  Intermediates:  not found")
    print(f"  Timing:         {baseline_timer_path or 'not found'}")
    for label, td in zip(test_labels, args.test_dirs):
        ti = find_intermediates(td)
        tt = find_timing(td)
        if ti:
            print(f"  Test [{label}]:  {td}  (inter={ti.backend}/{ti.kind}, timing={tt is not None})")
        else:
            print(f"  Test [{label}]:  {td}  (inter=none, timing={tt is not None})")
    print(f"  Stages:         {', '.join(stages)}")
    if args.read_filter:
        print(f"  Filter:         {args.read_filter}")

    # ── Intermediate comparison ────────────────────────────────────────────
    per_test_stats: Dict[str, Dict[str, AnySummaryStats]] = {}
    intermediates_pass = True

    if not args.timing_only and baseline_info:
        header("INTERMEDIATE COMPARISON")

        # Discover test intermediates and filter to compatible backends
        test_infos: List[Tuple[str, SourceInfo]] = []
        for label, td in zip(test_labels, args.test_dirs):
            test_info = find_intermediates(td)
            if not test_info:
                print(f"  {Colors.YELLOW}[{label}] No intermediates found, skipping{Colors.NC}")
                continue
            test_infos.append((label, test_info))

        if test_infos:
            show_verbose = args.verbose and not args.summary_only
            per_test_stats = compare_intermediates(
                baseline_info, test_infos,
                stages, args.read_filter, show_verbose,
            )

        if per_test_stats:
            intermediates_pass = print_intermediates_summary(per_test_stats, test_labels)

    # ── Timing comparison ──────────────────────────────────────────────────
    test_timing_list: List[Optional[dict]] = []
    if not args.intermediates_only and baseline_timing:
        for label, td in zip(test_labels, args.test_dirs):
            test_timing_list.append(load_timing(find_timing(td)))

        has_timing = any(t is not None for t in test_timing_list)
        if has_timing:
            # Detailed per-test comparison
            if not args.summary_only:
                header("TIMING COMPARISON (DETAIL)")
                for label, td_timing in zip(test_labels, test_timing_list):
                    if td_timing:
                        print_timing_detail(baseline_timing, td_timing, label)

            # Per-stage breakdown (all tests side-by-side)
            valid_data = [t for t in test_timing_list if t is not None]
            valid_labels = [l for l, t in zip(test_labels, test_timing_list) if t is not None]
            print_stage_breakdown(baseline_timing, valid_data, valid_labels)
            print_wall_time_summary(baseline_timing, valid_data, valid_labels)

    # ── Export ─────────────────────────────────────────────────────────────
    if args.output:
        export_results(
            per_test_stats, args.baseline, baseline_timing,
            args.test_dirs, test_labels, test_timing_list,
            args.output,
        )

    # ── Result ─────────────────────────────────────────────────────────────
    header("RESULT")
    if intermediates_pass:
        print(f"  {Colors.GREEN}All intermediate comparisons passed.{Colors.NC}")
    else:
        print(f"  {Colors.RED}Some intermediate comparisons have differences.{Colors.NC}")

    rw = baseline_timing.get("wall_time_sec", 0) if baseline_timing else 0
    for label, td_timing in zip(test_labels, test_timing_list):
        if not td_timing:
            continue
        tw = td_timing.get("wall_time_sec", 0)
        if rw > 0 and tw > 0:
            ratio = tw / rw
            if ratio <= 1.0:
                print(f"  {Colors.GREEN}[{label}] {1/ratio:.2f}x faster than baseline{Colors.NC}")
            else:
                print(f"  {Colors.YELLOW}[{label}] {ratio:.2f}x slower than baseline{Colors.NC}")

    print()
    return 0 if intermediates_pass else 1


if __name__ == "__main__":
    sys.exit(main())
