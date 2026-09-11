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
Compare FlatBuffers intermediate dump directories between two minimap2 runs.

Compares per-query .bin files for seeds, chains, and alignments to
identify mismatches between CPU and GPU runs, or between different
configurations. Each directory is expected to contain one .bin file
per query sequence (produced by minimap2 --dump-intermediates FLATBUF).

Arguments:
  dir1                  First directory with FlatBuffers .bin files
  dir2                  Second directory with FlatBuffers .bin files
  -s, --stage STAGE     Stage(s) to compare: seeds, chains, alignments (default: all)
  -r, --read SUBSTR     Filter to reads whose filename contains SUBSTR
  -v, --verbose         Show all comparisons including matches
  -o, --output FILE     Export mismatches to JSON file
  --summary-only        Only show summary, skip per-file output

Expected directory layout:
  dir1/
    query1.bin, query2.bin, ...
  dir2/
    query1.bin, query2.bin, ...

Output:
  stdout                Per-query MATCH / mismatch report + summary table
  -o mismatches.json    Structured export of all mismatched fields

Key behaviors:
  - Compares files by name across dir1/dir2, reports missing files
  - Shows first 5 mismatches per query, then a count of remaining
  - Seed comparison: (x, y) coordinate pairs
  - Chain comparison: score, length, first/last positions, strand, rid, rname
  - Alignment comparison: score, dp_score, coordinates, mapq, cigar
  - CIGAR strings compared by content; length-only diff if > 10 chars apart
  - Query metadata (qlen_sum, n_seg, k, w) compared when present
  - Exit code 0 always (mismatches reported, not treated as errors)

Usage:
    python compare_binary_intermediates.py <dir1> <dir2> [options]

Examples:
    # Compare two FlatBuffers dump directories
    python compare_binary_intermediates.py /tmp/cpu_dump/ /tmp/gpu_dump/

    # Compare only chains with verbose output
    python compare_binary_intermediates.py dir1/ dir2/ --stage chains -v

    # Compare specific read
    python compare_binary_intermediates.py dir1/ dir2/ --read "m64136_200706_123635/1008/ccs"

    # Export mismatches for further analysis
    python compare_binary_intermediates.py dir1/ dir2/ -o mismatches.json
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "generated"))

from mm2.dump.QueryDump import QueryDump

STAGES = ("seeds", "chains", "alignments")

CHAIN_FIELDS = ("score", "length", "first_qpos", "first_tpos",
                "last_qpos", "last_tpos", "rev", "rid", "rname")

ALIGNMENT_FIELDS = ("seg_id", "rid", "rname", "score", "dp_score",
                    "qs", "qe", "rs", "re", "mapq", "rev",
                    "mlen", "blen", "n_sub", "sam_pri", "cigar",
                    "n_cigar_ops", "dp_max", "n_ambi")

META_FIELDS = ("qlen_sum", "n_seg", "k", "w")


@dataclass
class ComparisonResult:
    name: str
    stage: str
    match: bool = True
    missing_in_dir1: bool = False
    missing_in_dir2: bool = False
    differences: List[str] = field(default_factory=list)

    def __str__(self):
        if self.match:
            return f"  MATCH   {self.stage}/{self.name}"
        status = []
        if self.missing_in_dir1:
            status.append("missing in dir1")
        elif self.missing_in_dir2:
            status.append("missing in dir2")
        else:
            status.append(f"{len(self.differences)} differences")
        return f"  DIFF    {self.stage}/{self.name}: {', '.join(status)}"


@dataclass
class SummaryStats:
    total: int = 0
    matches: int = 0
    mismatches: int = 0
    missing_dir1: int = 0
    missing_dir2: int = 0

    def add(self, result: ComparisonResult):
        self.total += 1
        if result.match:
            self.matches += 1
        else:
            self.mismatches += 1
            if result.missing_in_dir1:
                self.missing_dir1 += 1
            elif result.missing_in_dir2:
                self.missing_dir2 += 1


def list_queries(dump_dir: Path) -> List[str]:
    return sorted(f.stem for f in dump_dir.iterdir() if f.suffix == ".bin")


def load_query(filepath: Path) -> Dict:
    with open(filepath, "rb") as f:
        buf = f.read()
    return parse_query(buf)


def parse_query(buf: bytes) -> Dict:
    """Parse a FlatBuffers query dump from raw bytes."""
    q = QueryDump.GetRootAs(bytearray(buf))

    meta = q.Meta()
    meta_dict = {}
    if meta:
        meta_dict = {
            "qlen_sum": meta.QlenSum(),
            "n_seg": meta.NSeg(),
            "rep_len": meta.RepLen(),
            "n_mini_pos": meta.NMiniPos(),
            "frag_gap": meta.FragGap(),
            "k": meta.K(),
            "w": meta.W(),
        }

    seeds = []
    for i in range(q.SeedsLength()):
        s = q.Seeds(i)
        seeds.append((s.X(), s.Y()))

    chained_seed_indices = []
    for i in range(q.ChainedSeedIndicesLength()):
        chained_seed_indices.append(q.ChainedSeedIndices(i))

    chains = []
    for i in range(q.ChainsLength()):
        c = q.Chains(i)
        chains.append({
            "score": c.Score(),
            "seed_offset": c.SeedOffset(),
            "length": c.Length(),
            "first_qpos": c.FirstQpos(),
            "first_tpos": c.FirstTpos(),
            "last_qpos": c.LastQpos(),
            "last_tpos": c.LastTpos(),
            "rev": c.Rev(),
            "rid": c.Rid(),
            "rname": c.Rname().decode() if c.Rname() else "",
        })

    alignments = []
    for i in range(q.AlignmentsLength()):
        a = q.Alignments(i)
        alignments.append({
            "seg_id": a.SegId(),
            "rid": a.Rid(),
            "score": a.Score(),
            "score0": a.Score0(),
            "dp_score": a.DpScore(),
            "qs": a.Qs(),
            "qe": a.Qe(),
            "rs": a.Rs(),
            "re": a.Re(),
            "mapq": a.Mapq(),
            "rev": a.Rev(),
            "mlen": a.Mlen(),
            "blen": a.Blen(),
            "n_sub": a.NSub(),
            "sam_pri": a.SamPri(),
            "n_cigar_ops": a.NCigarOps(),
            "dp_max": a.DpMax(),
            "dp_max2": a.DpMax2(),
            "n_ambi": a.NAmbi(),
            "chain_idx": a.ChainIdx(),
            "rname": a.Rname().decode() if a.Rname() else "",
            "cigar": a.Cigar().decode() if a.Cigar() else "",
        })

    return {
        "name": q.Name().decode() if q.Name() else "",
        "metadata": meta_dict,
        "seeds": seeds,
        "chained_seed_indices": chained_seed_indices,
        "chains": chains,
        "alignments": alignments,
    }


def compare_metadata(d1: Dict, d2: Dict) -> List[str]:
    diffs = []
    m1 = d1.get("metadata", {})
    m2 = d2.get("metadata", {})
    for key in META_FIELDS:
        v1 = m1.get(key)
        v2 = m2.get(key)
        if v1 != v2:
            diffs.append(f"meta.{key}: {v1} vs {v2}")
    return diffs


def compare_seeds(d1: Dict, d2: Dict) -> List[str]:
    s1 = d1["seeds"]
    s2 = d2["seeds"]
    diffs = []
    if len(s1) != len(s2):
        diffs.append(f"seed count: {len(s1)} vs {len(s2)}")

    mismatched = 0
    for i, (a, b) in enumerate(zip(s1, s2)):
        if a != b:
            mismatched += 1
            if mismatched <= 5:
                diffs.append(f"seed[{i}]: {a} vs {b}")
    if mismatched > 5:
        diffs.append(f"... and {mismatched - 5} more seed mismatches")
    return diffs


def compare_chains(d1: Dict, d2: Dict) -> List[str]:
    c1 = d1["chains"]
    c2 = d2["chains"]
    diffs = []
    if len(c1) != len(c2):
        diffs.append(f"chain count: {len(c1)} vs {len(c2)}")

    mismatched = 0
    for i, (a, b) in enumerate(zip(c1, c2)):
        chain_diffs = []
        for col in CHAIN_FIELDS:
            if a.get(col) != b.get(col):
                chain_diffs.append(f"{col}: {a.get(col)} vs {b.get(col)}")
        if chain_diffs:
            mismatched += 1
            if mismatched <= 5:
                diffs.append(f"chain[{i}]: {'; '.join(chain_diffs)}")
    if mismatched > 5:
        diffs.append(f"... and {mismatched - 5} more chain mismatches")
    return diffs


def compare_alignments(d1: Dict, d2: Dict) -> List[str]:
    a1 = d1["alignments"]
    a2 = d2["alignments"]
    diffs = []
    if len(a1) != len(a2):
        diffs.append(f"alignment count: {len(a1)} vs {len(a2)}")

    mismatched = 0
    for i, (x, y) in enumerate(zip(a1, a2)):
        aln_diffs = []
        for col in ALIGNMENT_FIELDS:
            if col == "cigar":
                continue
            if x.get(col) != y.get(col):
                aln_diffs.append(f"{col}: {x.get(col)} vs {y.get(col)}")

        cigar1 = x.get("cigar", "")
        cigar2 = y.get("cigar", "")
        if cigar1 != cigar2:
            len_diff = abs(len(cigar1) - len(cigar2))
            if len_diff > 10:
                aln_diffs.append(f"cigar length: {len(cigar1)} vs {len(cigar2)}")
            else:
                aln_diffs.append("cigar differs")

        if aln_diffs:
            mismatched += 1
            if mismatched <= 5:
                diffs.append(f"alignment[{i}]: {'; '.join(aln_diffs)}")
    if mismatched > 5:
        diffs.append(f"... and {mismatched - 5} more alignment mismatches")
    return diffs


STAGE_COMPARATORS = {
    "seeds": compare_seeds,
    "chains": compare_chains,
    "alignments": compare_alignments,
}


def _compare_one_query(qname, d1, d2, stages):
    """Compare a single query across all stages. Returns list of (stage, ComparisonResult)."""
    results = []
    for stage in stages:
        result = ComparisonResult(name=qname, stage=stage)
        if d1 is None and d2 is None:
            result.match = False
            result.missing_in_dir1 = True
            result.missing_in_dir2 = True
        elif d1 is None:
            result.match = False
            result.missing_in_dir1 = True
        elif d2 is None:
            result.match = False
            result.missing_in_dir2 = True
        else:
            diffs = compare_metadata(d1, d2)
            diffs.extend(STAGE_COMPARATORS[stage](d1, d2))
            result.differences = diffs
            result.match = len(diffs) == 0
        results.append((stage, result))
    return results


def _compare_loaded_queries(all_names, map1, map2, stages, verbose):
    """Shared comparison logic operating on pre-loaded query dicts."""
    stats = {stage: SummaryStats() for stage in stages}
    results = {stage: [] for stage in stages}

    with ThreadPoolExecutor() as pool:
        futures = {}
        for qname in all_names:
            d1 = map1.get(qname)
            d2 = map2.get(qname)
            future = pool.submit(_compare_one_query, qname, d1, d2, stages)
            futures[future] = qname

        # Collect results keyed by query name to preserve sorted order
        result_map = {}
        for future in as_completed(futures):
            result_map[futures[future]] = future.result()

    for qname in all_names:
        for stage, result in result_map[qname]:
            stats[stage].add(result)
            results[stage].append(result)
            if verbose or not result.match:
                print(str(result))
                if verbose and result.differences:
                    for diff in result.differences:
                        print(f"    {diff}")

    return stats, results


def compare_directories(dir1: Path, dir2: Path,
                        stages: List[str],
                        read_filter: Optional[str] = None,
                        verbose: bool = False) -> Tuple[Dict[str, SummaryStats],
                                                         Dict[str, List[ComparisonResult]]]:
    queries1 = set(list_queries(dir1))
    queries2 = set(list_queries(dir2))

    if read_filter:
        queries1 = {q for q in queries1 if read_filter in q}
        queries2 = {q for q in queries2 if read_filter in q}

    all_names = sorted(queries1 | queries2)

    map1 = {q: load_query(dir1 / f"{q}.bin") for q in queries1 if q in all_names}
    map2 = {q: load_query(dir2 / f"{q}.bin") for q in queries2 if q in all_names}

    return _compare_loaded_queries(all_names, map1, map2, stages, verbose)


def print_summary(stats: Dict[str, SummaryStats]):
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    total_matches = 0
    total_mismatches = 0

    for stage, s in stats.items():
        if s.total == 0:
            continue
        match_pct = (s.matches / s.total * 100) if s.total > 0 else 0
        print(f"\n{stage.upper()}:")
        print(f"  Total queries:  {s.total}")
        print(f"  Matches:        {s.matches} ({match_pct:.1f}%)")
        print(f"  Mismatches:     {s.mismatches}")
        if s.missing_dir1 > 0:
            print(f"  Missing dir1:   {s.missing_dir1}")
        if s.missing_dir2 > 0:
            print(f"  Missing dir2:   {s.missing_dir2}")

        total_matches += s.matches
        total_mismatches += s.mismatches

    print("\n" + "-" * 60)
    total = total_matches + total_mismatches
    if total > 0:
        overall_pct = total_matches / total * 100
        print(f"OVERALL: {total_matches}/{total} queries match ({overall_pct:.1f}%)")
        if total_mismatches == 0:
            print("All queries match.")
        else:
            print(f"{total_mismatches} queries have differences")


def export_mismatches(results: Dict[str, List[ComparisonResult]],
                      output_file: Path):
    mismatches = {}
    for stage, stage_results in results.items():
        stage_mismatches = []
        for r in stage_results:
            if not r.match:
                stage_mismatches.append({
                    "name": r.name,
                    "missing_in_dir1": r.missing_in_dir1,
                    "missing_in_dir2": r.missing_in_dir2,
                    "differences": r.differences,
                })
        if stage_mismatches:
            mismatches[stage] = stage_mismatches

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(mismatches, f, indent=2)
    print(f"\nMismatches exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare FlatBuffers intermediate dump directories between two minimap2 runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dir1", type=Path,
                        help="First dump directory")
    parser.add_argument("dir2", type=Path,
                        help="Second dump directory")
    parser.add_argument("-s", "--stage", choices=list(STAGES),
                        action="append", dest="stages",
                        help="Stage(s) to compare (default: all)")
    parser.add_argument("-r", "--read", dest="read_filter",
                        help="Filter to specific read name (substring match)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show all comparisons including matches")
    parser.add_argument("-o", "--output", type=Path,
                        help="Export mismatches to JSON file")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only show summary, no per-query output")

    args = parser.parse_args()

    if not args.dir1.is_dir():
        print(f"Error: {args.dir1} is not a directory")
        sys.exit(1)
    if not args.dir2.is_dir():
        print(f"Error: {args.dir2} is not a directory")
        sys.exit(1)

    stages = args.stages if args.stages else list(STAGES)

    print(f"Comparing FlatBuffers intermediates:")
    print(f"  Dir 1: {args.dir1}")
    print(f"  Dir 2: {args.dir2}")
    print(f"  Stages: {', '.join(stages)}")
    if args.read_filter:
        print(f"  Filter: {args.read_filter}")
    print()

    verbose = args.verbose and not args.summary_only
    stats, results = compare_directories(
        args.dir1, args.dir2, stages,
        read_filter=args.read_filter,
        verbose=verbose,
    )

    print_summary(stats)

    if args.output:
        export_mismatches(results, args.output)


if __name__ == "__main__":
    main()
