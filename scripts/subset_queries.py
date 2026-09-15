#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""Extract a subset of reads/sequences from a FASTA or FASTQ file.

Usage:
  # Count queries in a file
  python3 subset_queries.py input.fastq

  # Extract queries 0..99 (first 100)
  python3 subset_queries.py input.fastq -o subset.fastq --end 100

  # Extract queries 50..149
  python3 subset_queries.py input.fastq -o subset.fastq --start 50 --end 150

  # Extract without counting first (faster for large files)
  python3 subset_queries.py input.fastq -o subset.fastq --start 0 --end 100 --no-count
"""

import argparse
import sys
import os


def detect_format(filepath):
    """Detect whether file is FASTA or FASTQ based on first character."""
    with open(filepath, "rb") as f:
        first_char = f.read(1)
    if first_char == b">":
        return "fasta"
    elif first_char == b"@":
        return "fastq"
    else:
        sys.exit(f"Error: Unrecognized format (first byte: {first_char!r}). Expected '>' (FASTA) or '@' (FASTQ).")


def count_fasta(filepath):
    """Count FASTA records using grep -c for speed, fallback to Python."""
    import subprocess
    try:
        result = subprocess.run(
            ["grep", "-c", "^>", filepath],
            capture_output=True, text=True, check=False
        )
        return int(result.stdout.strip())
    except Exception:
        count = 0
        with open(filepath, "rb") as f:
            for line in f:
                if line.startswith(b">"):
                    count += 1
        return count


def count_fastq(filepath):
    """Count FASTQ records (lines / 4) using wc -l for speed, fallback to Python."""
    import subprocess
    try:
        result = subprocess.run(
            ["wc", "-l", filepath],
            capture_output=True, text=True, check=False
        )
        lines = int(result.stdout.strip().split()[0])
        return lines // 4
    except Exception:
        lines = 0
        with open(filepath, "rb") as f:
            for _ in f:
                lines += 1
        return lines // 4


def count_records(filepath, fmt):
    if fmt == "fastq":
        return count_fastq(filepath)
    else:
        return count_fasta(filepath)


def extract_fasta(filepath, output, start, end):
    """Stream FASTA records and write [start, end) to output."""
    written = 0
    idx = 0
    with open(filepath, "rb") as fin, open(output, "wb") as fout:
        in_range = False
        for line in fin:
            if line.startswith(b">"):
                if idx >= end:
                    break
                in_range = idx >= start
                if in_range:
                    fout.write(line)
                    written += 1
                idx += 1
            elif in_range:
                fout.write(line)
    return written


def extract_fastq(filepath, output, start, end):
    """Stream FASTQ records (4 lines each) and write [start, end) to output."""
    written = 0
    idx = 0
    with open(filepath, "rb") as fin, open(output, "wb") as fout:
        while True:
            header = fin.readline()
            if not header:
                break
            seq = fin.readline()
            plus = fin.readline()
            qual = fin.readline()
            if idx >= end:
                break
            if idx >= start:
                fout.write(header)
                fout.write(seq)
                fout.write(plus)
                fout.write(qual)
                written += 1
            idx += 1
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Count or extract a subset of reads from a FASTA/FASTQ file."
    )
    parser.add_argument("input", help="Input FASTA or FASTQ file")
    parser.add_argument("-o", "--output", help="Output file for the subset (omit to just count)")
    parser.add_argument("--start", type=int, default=0, help="Start index, 0-based inclusive (default: 0)")
    parser.add_argument("--end", type=int, default=None, help="End index, exclusive (default: total count)")
    parser.add_argument("--no-count", action="store_true",
                        help="Skip counting total records (faster for large files, requires --end)")

    args = parser.parse_args()

    fmt = detect_format(args.input)
    size = os.path.getsize(args.input)
    size_str = f"{size / (1024**3):.2f} GB" if size >= 1024**3 else f"{size / (1024**2):.1f} MB" if size >= 1024**2 else f"{size / 1024:.1f} KB"
    print(f"File:   {args.input}")
    print(f"Format: {fmt.upper()}")
    print(f"Size:   {size_str}")

    if args.no_count and args.output and args.end is not None:
        # Skip counting, just extract
        start = args.start
        end = args.end
        print(f"Extracting queries [{start}, {end}) (skipping full count) ...")
        if fmt == "fastq":
            written = extract_fastq(args.input, args.output, start, end)
        else:
            written = extract_fasta(args.input, args.output, start, end)
        print(f"Wrote {written} queries to {args.output}")
        return

    print("Counting records ...")
    total = count_records(args.input, fmt)
    print(f"Total queries: {total}")

    if args.output:
        start = args.start
        end = args.end if args.end is not None else total

        if start < 0 or start >= total:
            sys.exit(f"Error: --start {start} out of range [0, {total})")
        if end <= start or end > total:
            sys.exit(f"Error: --end {end} out of range ({start}, {total}]")

        print(f"Extracting queries [{start}, {end}) ...")
        if fmt == "fastq":
            written = extract_fastq(args.input, args.output, start, end)
        else:
            written = extract_fasta(args.input, args.output, start, end)
        print(f"Wrote {written} queries to {args.output}")


if __name__ == "__main__":
    main()
