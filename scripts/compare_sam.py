#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""Compare SAM alignment correctness between a baseline and one or more test runs.

Designed for oracle-vs-fork verification (e.g. upstream minimap2 vs the AMD
fork's CPU/GPU modes), but works on any pair of SAM outputs produced from the
same reads.

What it checks, per query (keyed by read name):
  - Primary alignment chosen by SAM FLAG (not file order): rname, pos, strand,
    CIGAR must all match for an "exact primary" hit.
  - Unmapped reads (FLAG 0x4) handled explicitly (both-unmapped vs mismatch).
  - Secondary/supplementary alignment SETS compared (a common GPU divergence).
  - AS score differences, with an optional tolerance window and signed deltas.

Diff categories are mutually exclusive for the primary hit, so each query is
counted once (exact | pos-diff | cigar-only-diff | unmapped-mismatch | missing).

Memory note: each SAM is loaded fully into memory (minimap2 output is not
sorted by read name, so streaming comparison is not possible). Peak RAM is
roughly proportional to total SAM size; very large outputs (e.g. all-vs-all
ava presets, >10 GB) need a machine with comparable free RAM.

Usage:
  scripts/compare_sam.py baseline.sam test.sam
  scripts/compare_sam.py oracle.sam cpu.sam gpu.sam --labels cpu gpu
  scripts/compare_sam.py oracle.sam gpu.sam --as-tol 5 --max-samples 10
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

# SAM FLAG bits
FLAG_UNMAPPED = 0x4
FLAG_REVERSE = 0x10
FLAG_SECONDARY = 0x100
FLAG_SUPPLEMENTARY = 0x800
FLAG_NOT_PRIMARY = FLAG_SECONDARY | FLAG_SUPPLEMENTARY


def parse_sam_line(line):
    """Parse a SAM alignment line into a compact dict, or None if malformed."""
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 11:
        return None
    flag = int(fields[1])
    as_score = None
    for f in fields[11:]:
        if f.startswith("AS:i:"):
            try:
                as_score = int(f[5:])
            except ValueError:
                as_score = None
            break
    return {
        "qname": fields[0],
        "flag": flag,
        "rname": fields[2],
        "pos": int(fields[3]) if fields[3] != "*" else 0,
        "strand": "-" if (flag & FLAG_REVERSE) else "+",
        "cigar": fields[5],
        "as": as_score,
        "unmapped": bool(flag & FLAG_UNMAPPED),
        "secondary": bool(flag & FLAG_SECONDARY),
        "supplementary": bool(flag & FLAG_SUPPLEMENTARY),
    }


def load_sam(sam_file):
    """Load a SAM file into a dict: qname -> list of parsed records."""
    aligns = defaultdict(list)
    with open(sam_file) as fh:
        for line in fh:
            if line.startswith("@"):
                continue
            rec = parse_sam_line(line)
            if rec is not None:
                aligns[rec["qname"]].append(rec)
    return aligns


def pick_primary(hits):
    """Return the primary alignment (FLAG without secondary/supplementary)."""
    for h in hits:
        if not (h["flag"] & FLAG_NOT_PRIMARY):
            return h
    return hits[0] if hits else None


def supp_key(h):
    """Identity tuple for a supplementary/secondary alignment."""
    return (h["rname"], h["pos"], h["strand"], h["cigar"])


def compare(baseline, test, as_tol, max_samples):
    stats = {
        "total": 0,
        "exact_primary": 0,
        "both_unmapped": 0,
        "missing_in_test": 0,
        "unmapped_mismatch": 0,
        "pos_diff": 0,
        "cigar_only_diff": 0,
        "as_within_tol": 0,
        "as_diff_samples": [],
        "supp_set_diff": 0,
    }

    for qname, b_hits in baseline.items():
        stats["total"] += 1
        t_hits = test.get(qname)
        if not t_hits:
            stats["missing_in_test"] += 1
            continue

        b = pick_primary(b_hits)
        t = pick_primary(t_hits)

        if b["unmapped"] and t["unmapped"]:
            stats["both_unmapped"] += 1
            continue
        if b["unmapped"] != t["unmapped"]:
            stats["unmapped_mismatch"] += 1
            continue

        same_place = (
            b["rname"] == t["rname"]
            and b["pos"] == t["pos"]
            and b["strand"] == t["strand"]
        )
        same_cigar = b["cigar"] == t["cigar"]

        if same_place and same_cigar:
            stats["exact_primary"] += 1
        elif not same_place:
            stats["pos_diff"] += 1
        else:
            stats["cigar_only_diff"] += 1

        if b["as"] is not None and t["as"] is not None and b["as"] != t["as"]:
            if abs(b["as"] - t["as"]) <= as_tol:
                stats["as_within_tol"] += 1
            elif len(stats["as_diff_samples"]) < max_samples:
                stats["as_diff_samples"].append((qname, b["as"], t["as"]))

        b_supp = {supp_key(h) for h in b_hits if h["secondary"] or h["supplementary"]}
        t_supp = {supp_key(h) for h in t_hits if h["secondary"] or h["supplementary"]}
        if b_supp != t_supp:
            stats["supp_set_diff"] += 1

    return stats


def print_stats(label, stats, size_gb):
    total = stats["total"] or 1
    pct = 100 * stats["exact_primary"] / total
    print(f"\n{label}:")
    print(f"  File size: {size_gb:.2f} GB")
    print(f"  Total baseline queries:    {stats['total']}")
    print(f"  Exact primary match:       {stats['exact_primary']} ({pct:.2f}%)")
    print(f"  Both unmapped:             {stats['both_unmapped']}")
    print(f"  Position/strand diff:      {stats['pos_diff']}")
    print(f"  CIGAR-only diff:           {stats['cigar_only_diff']}")
    print(f"  Unmapped mismatch:         {stats['unmapped_mismatch']}")
    print(f"  Missing in test:           {stats['missing_in_test']}")
    print(f"  AS within tol:             {stats['as_within_tol']}")
    print(f"  AS beyond tol (samples):   {len(stats['as_diff_samples'])} shown")
    print(f"  Supp/secondary set diff:   {stats['supp_set_diff']}")
    for qname, b_as, t_as in stats["as_diff_samples"]:
        print(f"      {qname}: baseline={b_as}, test={t_as} (delta={t_as - b_as})")


def main():
    ap = argparse.ArgumentParser(
        description="Compare SAM alignment correctness: baseline vs test run(s).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("baseline", type=Path, help="Baseline/oracle SAM file")
    ap.add_argument("test", type=Path, nargs="+", help="One or more test SAM files")
    ap.add_argument(
        "--labels",
        nargs="+",
        help="Labels for each test file (default: file stem). Must match count.",
    )
    ap.add_argument(
        "--as-tol",
        type=int,
        default=0,
        help="AS score tolerance; diffs within this window are not flagged (default 0).",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Max AS-diff example rows to print per test (default 5).",
    )
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.test):
        ap.error(
            f"--labels count ({len(args.labels)}) must match test file count "
            f"({len(args.test)})"
        )

    if not args.baseline.exists():
        print(f"ERROR: baseline SAM not found: {args.baseline}", file=sys.stderr)
        return 1

    print("=" * 80)
    print(f"SAM correctness comparison (AS tol = {args.as_tol})")
    print(f"Baseline: {args.baseline}")
    print("=" * 80)
    print(f"\nLoading baseline ({args.baseline.stat().st_size / 1e9:.2f} GB)...")
    baseline = load_sam(args.baseline)
    print(f"  Loaded {len(baseline)} queries")

    any_missing = False
    for i, test_path in enumerate(args.test):
        label = args.labels[i] if args.labels else test_path.stem
        if not test_path.exists():
            print(f"\n{label}: SAM not found ({test_path})")
            any_missing = True
            continue
        test = load_sam(test_path)
        stats = compare(baseline, test, args.as_tol, args.max_samples)
        print_stats(label, stats, test_path.stat().st_size / 1e9)

    return 1 if any_missing else 0


if __name__ == "__main__":
    sys.exit(main())
