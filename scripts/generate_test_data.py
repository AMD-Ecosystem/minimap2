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
Generate JSON test data files from minimap2 intermediate dump output.

Runs minimap2 with --extra-out-dir and --dump-intermediates FLATBUF to dump
structured intermediate data (seeds, chains, alignments) into FlatBuffer
binary files, then reads them and produces a single test data JSON file
consumable by unit tests.

No special build flags are required -- any standard minimap2 build
supports the --extra-out-dir and --dump-intermediates options.

Arguments:
  --ref PATH          Reference FASTA file (required)
  --query PATH        Query FASTA file (required)
  --output PATH       Output JSON file (required)
  --minimap2 PATH     minimap2 binary (auto-detected from build tree if omitted)
  -x, --preset NAME   minimap2 preset (e.g. map-ont, map-pb, map-hifi, sr)
  --name TEXT         Test name (auto-generated from file names if omitted)
  --description TEXT  Human-readable test description
  --dry-run           Print combined JSON to stdout without writing file

Output JSON schema:
  {
    "name": "...",
    "description": "...",
    "reference": "ref.fa",
    "query": "query.fa",
    "preset": "map-ont",
    "queries": [
      {
        "query_name": "read1",
        "seeds": [...],
        "chains": [...],
        "alignments": [
          {
            "query_name", "query_start", "query_end",
            "target_name", "target_start", "target_end",
            "strand", "mapq", "cigar", "score",
            "y_position", "NM", "blen"
          }
        ]
      }
    ]
  }

Key behaviors:
  - Auto-detects minimap2 binary in out/*/bin/ build tree
  - Forces -t 1 -a for deterministic single-threaded output
  - Combines per-query JSON files in natural sort order (read2 < read10)
  - Extracts y-position from the r2 field of each chain anchor
  - Alignment records include 13 fields from SAM output

Usage:
    ./generate_test_data.py --ref reference.fa --query query.fa --output test_data.json

    # Or with explicit minimap2 binary:
    ./generate_test_data.py --ref reference.fa --query query.fa --output test_data.json \
        --minimap2 /path/to/minimap2
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "generated"))


def run_minimap2_flatbuf_dump(minimap2_bin: str, ref_path: str, query_path: str,
                              preset: Optional[str] = None) -> str:
    """Run minimap2 with --dump-intermediates FLATBUF and return the dump directory path."""
    dump_dir = tempfile.mkdtemp(prefix="mm2_flatbuf_dump_")

    cmd = [minimap2_bin, "-t", "1", "-a", "--extra-out-dir", dump_dir,
           "--dump-intermediates=FLATBUF"]
    if preset:
        cmd.extend(["-x", preset])
    cmd.extend([ref_path, query_path])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        bin_files = [f for f in os.listdir(dump_dir) if f.endswith(".bin")]
        if not bin_files:
            print("Warning: No .bin files produced. Make sure minimap2 supports "
                  "--dump-intermediates FLATBUF.", file=sys.stderr)
            print(f"stderr: {result.stderr}", file=sys.stderr)
        return dump_dir
    except subprocess.CalledProcessError as e:
        print(f"Error running minimap2: {e}", file=sys.stderr)
        print(f"stdout: {e.stdout}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        shutil.rmtree(dump_dir, ignore_errors=True)
        sys.exit(1)


def parse_flatbuf_dump(dump_dir: str) -> Tuple[List[Dict], Optional[Dict]]:
    """Parse FlatBuffers .bin files from --dump-intermediates FLATBUF output."""
    from mm2.dump.QueryDump import QueryDump

    bin_files = sorted(f for f in os.listdir(dump_dir) if f.endswith(".bin"))

    index_params = None
    queries = []

    for fname in bin_files:
        filepath = os.path.join(dump_dir, fname)
        with open(filepath, "rb") as f:
            buf = f.read()
        q = QueryDump.GetRootAs(bytearray(buf))

        qname = q.Name().decode() if q.Name() else fname[:-4]

        meta = q.Meta()
        if meta and index_params is None:
            k = meta.K()
            w = meta.W()
            if k and w:
                index_params = {"k": k, "w": w}

        query_data = {
            "name": qname,
            "expected_anchors": [],
            "expected_chains": [],
            "expected_alignments": [],
        }

        for i in range(q.SeedsLength()):
            s = q.Seeds(i)
            query_pos = s.Y() & 0xFFFFFFFF
            query_data["expected_anchors"].append({
                "x": str(s.X()),
                "y": str(query_pos),
            })

        for i in range(q.ChainsLength()):
            c = q.Chains(i)
            query_data["expected_chains"].append({
                "score": c.Score(),
                "length": c.Length(),
                "first_qpos": c.FirstQpos(),
                "first_tpos": c.FirstTpos(),
                "last_qpos": c.LastQpos(),
                "last_tpos": c.LastTpos(),
            })

        for i in range(q.AlignmentsLength()):
            a = q.Alignments(i)
            query_data["expected_alignments"].append({
                "score": a.Score(),
                "dp_score": a.DpScore() or 0,
                "qs": a.Qs(),
                "qe": a.Qe(),
                "rs": a.Rs(),
                "re": a.Re(),
                "mapq": a.Mapq(),
                "rev": a.Rev(),
                "n_cigar_ops": a.NCigarOps() or 0,
                "blen": a.Blen() or 0,
                "mlen": a.Mlen() or 0,
                "dp_max": a.DpMax() or 0,
                "n_ambi": a.NAmbi() or 0,
            })

        queries.append(query_data)

    return queries, index_params


def generate_test_name(ref_path: str, query_path: str) -> str:
    """Generate a test name from file paths."""
    ref_name = Path(ref_path).stem
    query_name = Path(query_path).stem
    return f"{ref_name}_vs_{query_name}"


def get_relative_path(file_path: str, json_dir: str) -> str:
    """Get relative path from JSON file location to FASTA file."""
    try:
        json_path = Path(json_dir).resolve()
        fasta_path = Path(file_path).resolve()
        return os.path.relpath(fasta_path, json_path)
    except ValueError:
        # If on different drives (Windows), use absolute path
        return str(Path(file_path).resolve())


def generate_json(
    queries: List[Dict],
    ref_path: str,
    query_path: str,
    output_path: str,
    test_name: Optional[str] = None,
    description: Optional[str] = None,
    preset: Optional[str] = None,
    index_params: Optional[Dict] = None
) -> Dict:
    """Generate JSON test data structure with per-query expectations."""
    if not test_name:
        test_name = generate_test_name(ref_path, query_path)
    
    if not description:
        description = f"Test data for {test_name}"
    
    # Get relative paths from JSON output directory
    json_dir = os.path.dirname(output_path)
    rel_ref_path = get_relative_path(ref_path, json_dir)
    rel_query_path = get_relative_path(query_path, json_dir)
    
    # Sort queries by name using natural sorting to match FASTA file order
    # e.g., q1, q2, q3, ... q10, q11, q12 (not q1, q10, q11, q12, q2, ...)
    def natural_sort_key(query):
        """Natural sort key for query names like q1, q2, ..., q10, q11."""
        parts = re.split(r'(\d+)', query["name"])
        return [int(part) if part.isdigit() else part for part in parts]
    
    sorted_queries = sorted(queries, key=natural_sort_key)
    
    # Use index_params from the JSON dump (actual values used by minimap2)
    # Fall back to defaults if not available
    idx_params = index_params if index_params else {"k": 15, "w": 10}
    
    data = {
        "name": test_name,
        "description": description,
        "reference": {
            "file": rel_ref_path
        },
        "query": {
            "file": rel_query_path
        },
        "index_params": idx_params,
        "queries": sorted_queries  # Per-query expectations sorted by name
    }
    
    if preset:
        data["preset"] = preset
    
    return data


def find_minimap2_binary() -> Optional[str]:
    """Try to find minimap2 binary.
    
    Searches common build output locations. Any standard minimap2 build
    supports --extra-out-dir and --dump-intermediates, so no special build flags are required.
    """
    # Try common locations in order of preference
    candidates = [
        # CMake preset builds (new structure)
        "./out/amd-default/build/bin/minimap2",
        "./out/coverage/build/bin/minimap2",
        "./out/test-cpu/build/bin/minimap2",
        # Legacy locations
        "./out/coverage/build/minimap2",
        "./out/test-cpu/build/minimap2",
        "./build/minimap2",
        "./minimap2",
        "minimap2"  # In PATH
    ]
    
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return candidate
        except FileNotFoundError:
            continue
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON test data from minimap2 --dump-intermediates FLATBUF output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate test data for medium dataset
  ./generate_test_data.py \\
      --ref test_suite/medium/t0.fa \\
      --query test_suite/medium/q0.fa \\
      --output test_suite/medium/expected/medium_t0_q0.json

  # With custom test name and description
  ./generate_test_data.py \\
      --ref test_suite/medium/t0.fa \\
      --query test_suite/medium/q0.fa \\
      --output test_suite/medium/expected/medium_t0_q0.json \\
      --name "medium-test-case" \\
      --description "Medium-sized test case for performance testing"
        """
    )
    
    parser.add_argument("--ref", required=True, help="Reference FASTA file")
    parser.add_argument("--query", required=True, help="Query FASTA file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--minimap2", help="Path to minimap2 binary (auto-detect if not provided)")
    parser.add_argument("-x", "--preset", help="minimap2 preset (e.g. map-ont, map-pb, map-hifi, sr)")
    parser.add_argument("--dump-format", choices=["FLATBUF"], default="FLATBUF",
                        help="Dump format (default: FLATBUF)")
    parser.add_argument("--name", help="Test name (auto-generate if not provided)")
    parser.add_argument("--description", help="Test description")
    parser.add_argument("--dry-run", action="store_true", help="Print JSON without writing file")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.ref):
        print(f"Error: Reference file not found: {args.ref}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.query):
        print(f"Error: Query file not found: {args.query}", file=sys.stderr)
        sys.exit(1)
    
    # Find minimap2 binary
    minimap2_bin = args.minimap2
    if not minimap2_bin:
        minimap2_bin = find_minimap2_binary()
        if not minimap2_bin:
            print("Error: Could not find minimap2 binary. Please specify with --minimap2", file=sys.stderr)
            sys.exit(1)
        print(f"Using minimap2: {minimap2_bin}", file=sys.stderr)
    
    # Run minimap2 with --extra-out-dir --dump-intermediates <format>
    print(f"Running minimap2 with --dump-intermediates {args.dump_format}...", file=sys.stderr)
    if args.dump_format == "FLATBUF":
        dump_dir = run_minimap2_flatbuf_dump(minimap2_bin, args.ref, args.query, args.preset)
    
    try:
        # Parse dump
        print(f"Parsing {args.dump_format} dump from {dump_dir}...", file=sys.stderr)
        if args.dump_format == "FLATBUF":
            queries, index_params = parse_flatbuf_dump(dump_dir)
        else:
            queries, index_params = None, None
        
        # Count totals across all queries
        total_anchors = sum(len(q["expected_anchors"]) for q in queries)
        total_chains = sum(len(q["expected_chains"]) for q in queries)
        total_alignments = sum(len(q["expected_alignments"]) for q in queries)
        
        print(f"Found {len(queries)} query sequences", file=sys.stderr)
        print(f"Found {total_anchors} total anchors", file=sys.stderr)
        print(f"Found {total_chains} total chains", file=sys.stderr)
        print(f"Found {total_alignments} total alignments", file=sys.stderr)
        
        # Generate JSON
        json_data = generate_json(
            queries,
            args.ref,
            args.query,
            args.output,
            args.name,
            args.description,
            args.preset,
            index_params
        )
        
        # Output
        json_str = json.dumps(json_data, indent=2)
        
        if args.dry_run:
            print(json_str)
        else:
            # Create output directory if needed
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            
            with open(args.output, 'w') as f:
                f.write(json_str)
            
            print(f"Generated: {args.output}", file=sys.stderr)
    finally:
        # Clean up temporary dump directory
        shutil.rmtree(dump_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
