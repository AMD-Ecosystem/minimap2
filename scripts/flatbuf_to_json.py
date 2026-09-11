#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Convert a FlatBuffers intermediate dump (.bin) to JSON.

Usage:
    python3 flatbuf_to_json.py query.bin              # prints to stdout
    python3 flatbuf_to_json.py query.bin -o query.json
    python3 flatbuf_to_json.py dir_of_bins/ -o out/   # batch convert
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "generated"))

from mm2.dump.QueryDump import QueryDump


def parse_bin(filepath):
    with open(filepath, "rb") as f:
        buf = bytearray(f.read())
    q = QueryDump.GetRootAs(buf)

    meta = q.Meta()
    metadata = {}
    if meta:
        metadata = {
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
        seeds.append({"x": s.X(), "y": s.Y()})

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
            "rname": a.Rname().decode() if a.Rname() else "",
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
            "split": a.Split(),
            "parent": a.Parent(),
            "inv": a.Inv(),
            "subsc": a.Subsc(),
            "anchor_offset": a.AnchorOffset(),
            "cigar": a.Cigar().decode() if a.Cigar() else "",
        })

    return {
        "name": q.Name().decode() if q.Name() else "",
        "metadata": metadata,
        "seeds": seeds,
        "chained_seed_indices": chained_seed_indices,
        "chains": chains,
        "alignments": alignments,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert FlatBuffers .bin dump to JSON")
    parser.add_argument("input", help="A .bin file or directory of .bin files")
    parser.add_argument("-o", "--output",
                        help="Output file or directory (default: stdout)")
    parser.add_argument("--indent", type=int, default=2,
                        help="JSON indent level (default: 2)")
    args = parser.parse_args()

    inp = Path(args.input)

    if inp.is_file():
        data = parse_bin(inp)
        text = json.dumps(data, indent=args.indent)
        if args.output:
            Path(args.output).write_text(text + "\n")
        else:
            print(text)

    elif inp.is_dir():
        bins = sorted(inp.glob("*.bin"))
        if not bins:
            print(f"No .bin files found in {inp}", file=sys.stderr)
            sys.exit(1)

        out_dir = Path(args.output) if args.output else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        for bf in bins:
            data = parse_bin(bf)
            text = json.dumps(data, indent=args.indent)
            if out_dir:
                out_path = out_dir / (bf.stem + ".json")
                out_path.write_text(text + "\n")
            else:
                print(f"--- {bf.name} ---")
                print(text)

        if out_dir:
            print(f"Converted {len(bins)} files to {out_dir}/",
                  file=sys.stderr)
    else:
        print(f"Error: {inp} not found", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
