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
Compare JSON intermediate files between two minimap2 runs.

Compares per-query JSON files for seeds, chains, and alignments to
identify mismatches between CPU and GPU runs, or between different
configurations. Each stage directory is expected to contain one JSON
file per query sequence (produced by minimap2 --extra-out-dir DIR --dump-intermediates JSON).

Arguments:
  dir1                  First directory with JSON intermediates
  dir2                  Second directory with JSON intermediates
  -s, --stage STAGE     Stage(s) to compare: seeds, chains, alignments (default: all)
  -r, --read SUBSTR     Filter to reads whose filename contains SUBSTR
  -v, --verbose         Show all comparisons including matches
  -o, --output FILE     Export mismatches to JSON file
  --summary-only        Only show summary, skip per-file output

Expected directory layout:
  dir1/
    seeds/       query1.json, query2.json, ...
    chains/      query1.json, query2.json, ...
    alignments/  query1.json, query2.json, ...

Output:
  stdout                Per-file MATCH / mismatch report + summary table
  -o mismatches.json    Structured export of all mismatched fields

Key behaviors:
  - Compares files by name across dir1/dir2, reports missing files
  - Shows first 5 mismatches per file, then a count of remaining
  - Seed comparison: (x, y) coordinate pairs
  - Chain comparison: score, length, first/last positions, strand, rid
  - Alignment comparison: score, dp_score, coordinates, mapq, cigar
  - CIGAR strings compared by content; length-only diff if > 10 chars apart
  - Exit code 0 always (mismatches reported, not treated as errors)

Usage:
    python compare_json_intermediates.py <dir1> <dir2> [options]

Examples:
    # Compare two full intermediate directories
    python compare_json_intermediates.py /path/to/cpu_run /path/to/gpu_run

    # Compare only chains with verbose output
    python compare_json_intermediates.py dir1 dir2 --stage chains -v

    # Compare specific read
    python compare_json_intermediates.py dir1 dir2 --read "m64136_200706_123635/1008/ccs"

    # Export mismatches for further analysis
    python compare_json_intermediates.py dir1 dir2 -o mismatches.json
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from json_dump_utils import (
    load_json, CHAIN_FIELDS, SEED_FIELDS, ALIGNMENT_FIELDS, STAGE_DIRS,
    SEED_META_FIELDS, CHAIN_META_FIELDS, ALIGNMENT_META_FIELDS,
    INDEX_PARAM_FIELDS, STAGE_ALL_KEYS,
)


@dataclass
class ComparisonResult:
    """Result of comparing a single JSON file."""
    name: str
    stage: str
    match: bool = True
    missing_in_dir1: bool = False
    missing_in_dir2: bool = False
    differences: List[str] = field(default_factory=list)
    
    def __str__(self):
        if self.match:
            return f"✓ {self.stage}/{self.name}: MATCH"
        status = []
        if self.missing_in_dir1:
            status.append("missing in dir1")
        elif self.missing_in_dir2:
            status.append("missing in dir2")
        else:
            status.append(f"{len(self.differences)} differences")
        return f"✗ {self.stage}/{self.name}: {', '.join(status)}"


@dataclass 
class SummaryStats:
    """Summary statistics for comparison."""
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


def compare_meta_fields(data1: Dict, data2: Dict, fields: tuple,
                        stage: Optional[str] = None) -> List[str]:
    """Compare top-level metadata fields between two JSON objects.
    
    Raises errors for:
    - Fields in `fields` that are missing from BOTH objects
    - Fields missing from one side but present in the other
    - Unknown fields not in the stage schema (schema drift detection)
    """
    differences = []
    for fld in fields:
        in1 = fld in data1
        in2 = fld in data2
        if not in1 and not in2:
            differences.append(f"MISSING {fld}: not found in either file")
            continue
        if not in1:
            differences.append(f"MISSING {fld}: not in dir1 (dir2={data2[fld]})")
            continue
        if not in2:
            differences.append(f"MISSING {fld}: not in dir2 (dir1={data1[fld]})")
            continue
        if data1[fld] != data2[fld]:
            differences.append(f"{fld}: {data1[fld]} vs {data2[fld]}")
    
    # Compare qlens array if present in either
    qlens1 = data1.get('qlens', [])
    qlens2 = data2.get('qlens', [])
    if qlens1 != qlens2:
        differences.append(f"qlens: {qlens1} vs {qlens2}")
    
    # Compare seq_prefixes array if present in either
    sp1 = data1.get('seq_prefixes', [])
    sp2 = data2.get('seq_prefixes', [])
    if sp1 != sp2:
        differences.append(f"seq_prefixes: {sp1} vs {sp2}")
    
    # Compare index_params if present
    ip1 = data1.get('index_params', {})
    ip2 = data2.get('index_params', {})
    if ip1 or ip2:
        for fld in INDEX_PARAM_FIELDS:
            v1 = ip1.get(fld) if ip1 else None
            v2 = ip2.get(fld) if ip2 else None
            in1 = ip1 and fld in ip1
            in2 = ip2 and fld in ip2
            if not in1 and not in2:
                continue  # index_params may be absent in older builds
            if not in1:
                differences.append(f"MISSING index_params.{fld}: not in dir1 (dir2={v2})")
            elif not in2:
                differences.append(f"MISSING index_params.{fld}: not in dir2 (dir1={v1})")
            elif v1 != v2:
                differences.append(f"index_params.{fld}: {v1} vs {v2}")
    
    # Detect unknown top-level keys (schema drift)
    if stage and stage in STAGE_ALL_KEYS:
        known = STAGE_ALL_KEYS[stage]
        for data, label in [(data1, "dir1"), (data2, "dir2")]:
            unknown = set(data.keys()) - known
            if unknown:
                differences.append(f"UNKNOWN keys in {label}: {sorted(unknown)}")
    
    return differences


def compare_seeds(data1: Dict, data2: Dict) -> List[str]:
    """Compare seed data between two JSON objects."""
    differences = compare_meta_fields(data1, data2, SEED_META_FIELDS, stage="seeds")
    
    # Compare seeds array
    seeds1 = data1.get('seeds', [])
    seeds2 = data2.get('seeds', [])
    
    if len(seeds1) != len(seeds2):
        differences.append(f"seeds array length: {len(seeds1)} vs {len(seeds2)}")
    else:
        mismatched_seeds = 0
        for i, (s1, s2) in enumerate(zip(seeds1, seeds2)):
            for fld in SEED_FIELDS:
                if fld not in s1:
                    differences.append(f"MISSING seed[{i}].{fld}: not in dir1")
                if fld not in s2:
                    differences.append(f"MISSING seed[{i}].{fld}: not in dir2")
            x1, y1 = s1.get('x'), s1.get('y')
            x2, y2 = s2.get('x'), s2.get('y')
            if x1 != x2 or y1 != y2:
                mismatched_seeds += 1
                if mismatched_seeds <= 5:  # Show first 5 mismatches
                    differences.append(f"seed[{i}]: ({x1}, {y1}) vs ({x2}, {y2})")
        if mismatched_seeds > 5:
            differences.append(f"... and {mismatched_seeds - 5} more seed mismatches")
    
    return differences


def compare_chains(data1: Dict, data2: Dict) -> List[str]:
    """Compare chain data between two JSON objects."""
    differences = compare_meta_fields(data1, data2, CHAIN_META_FIELDS, stage="chains")
    
    # Compare chains array
    chains1 = data1.get('chains', [])
    chains2 = data2.get('chains', [])
    
    if len(chains1) != len(chains2):
        differences.append(f"chains array length: {len(chains1)} vs {len(chains2)}")
    
    # Compare seeds array within chain JSON (post-chain anchors)
    seeds1 = data1.get('seeds', [])
    seeds2 = data2.get('seeds', [])
    if len(seeds1) != len(seeds2):
        differences.append(f"chain seeds array length: {len(seeds1)} vs {len(seeds2)}")
    else:
        mismatched_seeds = 0
        for i, (s1, s2) in enumerate(zip(seeds1, seeds2)):
            if s1.get('x') != s2.get('x') or s1.get('y') != s2.get('y'):
                mismatched_seeds += 1
                if mismatched_seeds <= 3:
                    differences.append(f"chain seed[{i}]: ({s1.get('x')}, {s1.get('y')}) vs ({s2.get('x')}, {s2.get('y')})")
        if mismatched_seeds > 3:
            differences.append(f"... and {mismatched_seeds - 3} more chain seed mismatches")
    
    # Compare chain by chain
    min_len = min(len(chains1), len(chains2))
    mismatched_chains = 0
    
    for i in range(min_len):
        c1, c2 = chains1[i], chains2[i]
        chain_diffs = []
        
        # Check for missing required fields
        for fld in CHAIN_FIELDS:
            if fld not in c1 and fld not in c2:
                chain_diffs.append(f"MISSING {fld}: not in either")
            elif fld not in c1:
                chain_diffs.append(f"MISSING {fld}: not in dir1 (dir2={c2[fld]})")
            elif fld not in c2:
                chain_diffs.append(f"MISSING {fld}: not in dir2 (dir1={c1[fld]})")
            elif c1[fld] != c2[fld]:
                chain_diffs.append(f"{fld}: {c1[fld]} vs {c2[fld]}")
        
        if chain_diffs:
            mismatched_chains += 1
            if mismatched_chains <= 5:
                differences.append(f"chain[{i}]: {'; '.join(chain_diffs)}")
    
    if mismatched_chains > 5:
        differences.append(f"... and {mismatched_chains - 5} more chain mismatches")
    
    return differences


def compare_alignments(data1: Dict, data2: Dict) -> List[str]:
    """Compare alignment data between two JSON objects."""
    differences = compare_meta_fields(data1, data2, ALIGNMENT_META_FIELDS, stage="alignments")
    
    # Compare alignments array
    alns1 = data1.get('alignments', [])
    alns2 = data2.get('alignments', [])
    
    if len(alns1) != len(alns2):
        differences.append(f"alignments array length: {len(alns1)} vs {len(alns2)}")
    
    # Compare alignment by alignment
    min_len = min(len(alns1), len(alns2))
    mismatched_alns = 0
    
    for i in range(min_len):
        a1, a2 = alns1[i], alns2[i]
        aln_diffs = []
        
        # Core alignment fields (from ALIGNMENT_FIELDS, excluding cigar)
        for fld in ALIGNMENT_FIELDS:
            if fld == 'cigar':
                continue
            if fld not in a1 and fld not in a2:
                aln_diffs.append(f"MISSING {fld}: not in either")
            elif fld not in a1:
                aln_diffs.append(f"MISSING {fld}: not in dir1 (dir2={a2[fld]})")
            elif fld not in a2:
                aln_diffs.append(f"MISSING {fld}: not in dir2 (dir1={a1[fld]})")
            elif a1[fld] != a2[fld]:
                aln_diffs.append(f"{fld}: {a1[fld]} vs {a2[fld]}")
        
        # CIGAR comparison (may differ slightly)
        cigar1 = a1.get('cigar', '')
        cigar2 = a2.get('cigar', '')
        if cigar1 != cigar2:
            # Check if lengths are very different
            len_diff = abs(len(cigar1) - len(cigar2))
            if len_diff > 10:
                aln_diffs.append(f"cigar length: {len(cigar1)} vs {len(cigar2)}")
            else:
                aln_diffs.append("cigar differs")
        
        if aln_diffs:
            mismatched_alns += 1
            if mismatched_alns <= 5:
                differences.append(f"alignment[{i}]: {'; '.join(aln_diffs)}")
    
    if mismatched_alns > 5:
        differences.append(f"... and {mismatched_alns - 5} more alignment mismatches")
    
    return differences


def compare_query_data(name: str, data1: Optional[Dict], data2: Optional[Dict],
                       stage: str) -> ComparisonResult:
    """Compare two parsed JSON dicts for a given stage."""
    result = ComparisonResult(name=name, stage=stage)

    if data1 is None and data2 is None:
        result.match = False
        result.missing_in_dir1 = True
        result.missing_in_dir2 = True
        return result

    if data1 is None:
        result.match = False
        result.missing_in_dir1 = True
        return result

    if data2 is None:
        result.match = False
        result.missing_in_dir2 = True
        return result

    # Compare based on stage
    if stage == 'seeds':
        result.differences = compare_seeds(data1, data2)
    elif stage == 'chains':
        result.differences = compare_chains(data1, data2)
    elif stage == 'alignments':
        result.differences = compare_alignments(data1, data2)
    else:
        if data1 != data2:
            result.differences = ["JSON content differs"]

    result.match = len(result.differences) == 0
    return result


def compare_file(file1: Path, file2: Path, stage: str) -> ComparisonResult:
    """Compare two JSON files for a given stage."""
    return compare_query_data(file1.name, load_json(file1), load_json(file2), stage)


def compare_directories(dir1: Path, dir2: Path, stages: List[str], 
                       read_filter: Optional[str] = None,
                       verbose: bool = False,
                       max_workers: Optional[int] = None) -> Tuple[Dict[str, SummaryStats],
                                                                     Dict[str, List[ComparisonResult]]]:
    """Compare all JSON files between two directories using a thread pool."""
    stats = {stage: SummaryStats() for stage in stages}
    results = {stage: [] for stage in stages}
    
    for stage in stages:
        stage_dir1 = dir1 / stage
        stage_dir2 = dir2 / stage
        
        if not stage_dir1.exists():
            print(f"Warning: {stage_dir1} does not exist")
            continue
        if not stage_dir2.exists():
            print(f"Warning: {stage_dir2} does not exist")
            continue
        
        # Get all JSON files from both directories
        files1 = set(f.name for f in stage_dir1.glob("*.json"))
        files2 = set(f.name for f in stage_dir2.glob("*.json"))
        all_files = sorted(files1 | files2)
        
        # Filter files
        if read_filter:
            all_files = [f for f in all_files if read_filter in f]
        
        # Compare files in parallel using a thread pool
        future_to_filename = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for filename in all_files:
                file1 = stage_dir1 / filename
                file2 = stage_dir2 / filename
                future = executor.submit(compare_file, file1, file2, stage)
                future_to_filename[future] = filename
            
            # Collect results keyed by filename to preserve sorted order
            result_map = {}
            for future in as_completed(future_to_filename):
                filename = future_to_filename[future]
                result_map[filename] = future.result()
        
        # Process results in sorted order
        for filename in all_files:
            result = result_map[filename]
            stats[stage].add(result)
            results[stage].append(result)
            
            if verbose or not result.match:
                print(str(result))
                if verbose and result.differences:
                    for diff in result.differences:
                        print(f"    {diff}")
    
    return stats, results


def print_summary(stats: Dict[str, SummaryStats]):
    """Print summary statistics."""
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
        print(f"  Total files:    {s.total}")
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
        print(f"OVERALL: {total_matches}/{total} files match ({overall_pct:.1f}%)")
        if total_mismatches == 0:
            print("✓ All files match!")
        else:
            print(f"✗ {total_mismatches} files have differences")


def export_mismatches(results: Dict[str, List[ComparisonResult]], 
                      output_file: Path):
    """Export mismatches to a JSON file for further analysis."""
    mismatches = {}
    for stage, stage_results in results.items():
        stage_mismatches = []
        for r in stage_results:
            if not r.match:
                stage_mismatches.append({
                    'name': r.name,
                    'missing_in_dir1': r.missing_in_dir1,
                    'missing_in_dir2': r.missing_in_dir2,
                    'differences': r.differences
                })
        if stage_mismatches:
            mismatches[stage] = stage_mismatches
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(mismatches, f, indent=2)
    print(f"\nMismatches exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare JSON intermediate files between two minimap2 runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('dir1', type=Path, 
                        help='First directory with JSON intermediates')
    parser.add_argument('dir2', type=Path,
                        help='Second directory with JSON intermediates')
    parser.add_argument('-s', '--stage', choices=['seeds', 'chains', 'alignments'],
                        action='append', dest='stages',
                        help='Stage(s) to compare (default: all)')
    parser.add_argument('-r', '--read', dest='read_filter',
                        help='Filter to specific read name (substring match)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show all comparisons including matches')
    parser.add_argument('-o', '--output', type=Path,
                        help='Export mismatches to JSON file')
    parser.add_argument('--summary-only', action='store_true',
                        help='Only show summary, no per-file output')
    
    args = parser.parse_args()
    
    # Validate directories
    if not args.dir1.exists():
        print(f"Error: {args.dir1} does not exist")
        sys.exit(1)
    if not args.dir2.exists():
        print(f"Error: {args.dir2} does not exist")
        sys.exit(1)
    
    # Default to all stages
    stages = args.stages if args.stages else list(STAGE_DIRS)
    
    print(f"Comparing JSON intermediates:")
    print(f"  Dir 1: {args.dir1}")
    print(f"  Dir 2: {args.dir2}")
    print(f"  Stages: {', '.join(stages)}")
    if args.read_filter:
        print(f"  Filter: {args.read_filter}")
    print()
    
    # Run comparison
    verbose = args.verbose and not args.summary_only
    stats, results = compare_directories(
        args.dir1, args.dir2, stages,
        read_filter=args.read_filter,
        verbose=verbose
    )
    
    # Print summary
    print_summary(stats)
    
    # Export mismatches if requested
    if args.output:
        export_mismatches(results, args.output)


if __name__ == '__main__':
    main()
