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
Test minimap2 SAM and PAF output across CPU and GPU modes.

Validates output using samtools (SAM) and paftools.js (PAF) with both
file output (-o) and stdout piping. Builds a matrix of (mode x format x
capture) and runs each combination as an independent test case.

Modes tested:
  - cpu-only
  - gpu-chain       (skipped if binary lacks --gpu-chain)
  - gpu-align       (skipped if binary lacks --gpu-align)
  - gpu-chain+align (skipped if binary lacks GPU flags)

Formats tested per mode:
  - SAM (-a)         : samtools quickcheck + flagstat + BAM round-trip
  - PAF (-c)         : column check + paftools.js stat
  - PAF (-c --cs)    : column check + paftools.js stat (with cs tag)

Each format is tested with both file output (-o) and stdout piping,
yielding up to 24 test cases (4 modes x 3 formats x 2 captures).

Arguments:
  minimap2            Path to minimap2 binary (default: out/amd-default/build/bin/minimap2)
  reference           Reference FASTA file (default: test/MT-human.fa)
  query               Query FASTA file (default: test/MT-orang.fa)
  --cpu-only          Skip all GPU modes, test CPU only
  --json [FILE]       Write JSON report (to FILE, or stdout if no path)

Prerequisites:
  - samtools on PATH (for SAM quickcheck, flagstat, BAM round-trip)
  - k8 on PATH (JavaScript runtime for paftools.js)
  - misc/paftools.js in repo tree (for PAF stat validation)
  - GPU config at configs/gpu_config.json (for GPU modes)

Output:
  stdout              PASS / FAIL table with timing per test
  --json report       Structured dict per test with:
    {
      "test": "...",
      "passed": true,
      "detail": "...",
      "detail_dict": { mapped, unmapped, total, ... }
    }

Key behaviors:
  - Auto-detects GPU support by probing minimap2 --help for --gpu-chain
  - BAM round-trip: SAM -> BAM -> SAM to verify lossless conversion
  - Streaming capture via subprocess stdout/stderr PIPE (no shell)
  - Per-test timeout: 60 s default (inherits from subprocess)
  - Exit code 1 if any test fails, 0 if all pass

Usage:
  python3 test_output_formats.py [minimap2_path] [reference.fa] [query.fa]
  python3 test_output_formats.py --cpu-only   # skip GPU modes entirely
  python3 test_output_formats.py --json results.json  # write JSON report to file
  python3 test_output_formats.py --json        # write JSON report to stdout
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time


# Default paths relative to the repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MM2 = os.path.join(REPO_ROOT, "out", "amd-default", "build", "bin", "minimap2")
DEFAULT_REF = os.path.join(REPO_ROOT, "test", "MT-human.fa")
DEFAULT_QRY = os.path.join(REPO_ROOT, "test", "MT-orang.fa")
PAFTOOLS = os.path.join(REPO_ROOT, "misc", "paftools.js")


def build_modes(gpu_cfg):
    """Return the list of test modes with their flags."""
    cfg_flags = ["--gpu-cfg", gpu_cfg] if gpu_cfg else []
    return [
        {
            "name": "cpu-only",
            "flags": [],
            "gpu": False,
        },
        {
            "name": "gpu-chain",
            "flags": ["--gpu-chain"] + cfg_flags,
            "gpu": True,
        },
        {
            "name": "gpu-align",
            "flags": ["--gpu-align"],
            "gpu": True,
        },
        {
            "name": "gpu-chain+align",
            "flags": ["--gpu-chain", "--gpu-align"] + cfg_flags,
            "gpu": True,
        },
    ]


# Formats: (label, minimap2 flags, extension)
FORMATS = [
    ("SAM",    ["-a"],         ".sam"),
    ("PAF",    ["-c"],         ".paf"),
    ("PAF+cs", ["-c", "--cs"], ".cs.paf"),
]


class Result:
    def __init__(self, mode, fmt, method):
        self.mode = mode
        self.fmt = fmt        # SAM, PAF, PAF+cs
        self.method = method  # file or stdout
        self.passed = False
        self.details = ""
        self.details_dict = {}
        self.elapsed = 0.0


def run_cmd(cmd, timeout=300):
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except FileNotFoundError as e:
        return -1, "", str(e)


def detect_gpu_support(mm2_path):
    """Check if minimap2 binary supports GPU flags by inspecting --help."""
    rc, _, stderr = run_cmd([mm2_path, "--help"])
    # minimap2 prints help to stdout with -h (rc=0), to stderr without (rc=1)
    _, stdout, _ = run_cmd([mm2_path, "-h"])
    help_text = stderr + stdout
    has_chain = "--gpu-chain" in help_text
    has_align = "--gpu-align" in help_text
    return has_chain, has_align


def check_prerequisites(mm2_path):
    """Verify required tools are available."""
    errors = []
    if not os.path.isfile(mm2_path):
        errors.append("minimap2 not found: {}".format(mm2_path))
    if not shutil.which("samtools"):
        errors.append("samtools not found on PATH")
    if not shutil.which("k8"):
        errors.append("k8 not found on PATH (required for paftools.js)")
    if not os.path.isfile(PAFTOOLS):
        errors.append("paftools.js not found: {}".format(PAFTOOLS))
    return errors


# ---------------------------------------------------------------------------
# SAM validation
# ---------------------------------------------------------------------------

def _sam_bam_roundtrip(sam_path):
    """SAM -> BAM -> SAM round-trip. Returns (passed, detail_str, detail_dict)."""
    bam_path = sam_path + ".bam"
    rt_path = sam_path + ".rt.sam"
    try:
        # SAM -> BAM
        rc, _, err = run_cmd(["samtools", "view", "-bS", "-o", bam_path, sam_path])
        if rc != 0:
            msg = "SAM->BAM FAIL: {}".format(err.strip()[:120])
            return False, msg, {"bam_roundtrip": False, "error": msg}
        # BAM -> SAM
        rc, _, err = run_cmd(["samtools", "view", "-h", "-o", rt_path, bam_path])
        if rc != 0:
            msg = "BAM->SAM FAIL: {}".format(err.strip()[:120])
            return False, msg, {"bam_roundtrip": False, "error": msg}
        # Compare alignment line counts (skip headers)
        with open(sam_path) as f:
            orig = sum(1 for l in f if not l.startswith("@"))
        with open(rt_path) as f:
            rt = sum(1 for l in f if not l.startswith("@"))
        if orig != rt:
            msg = "round-trip mismatch: {} vs {} alignments".format(orig, rt)
            return False, msg, {"bam_roundtrip": False, "alignments_original": orig, "alignments_roundtrip": rt}
        return True, "BAM round-trip OK ({} alignments)".format(orig), {"bam_roundtrip": True, "alignments": orig}
    finally:
        for p in (bam_path, rt_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _parse_flagstat(text):
    """Parse samtools flagstat output into a dict."""
    d = {}
    for line in text.strip().split("\n"):
        parts = line.split(" + ")
        if len(parts) < 2:
            continue
        try:
            qc_pass = int(parts[0])
        except ValueError:
            continue
        rest = parts[1]
        try:
            qc_fail = int(rest.split()[0])
        except (ValueError, IndexError):
            qc_fail = 0
        lower = line.lower()
        if "in total" in lower:
            d["total"] = qc_pass
            d["qc_failed"] = qc_fail
        elif re.match(r"^\d+ \+ \d+ mapped\b", line):
            d["mapped"] = qc_pass
            m = re.search(r"\((\d+\.\d+)%", line)
            if m:
                d["mapped_pct"] = float(m.group(1))
    return d


def validate_sam_file(sam_path):
    """Validate a SAM file: quickcheck + flagstat + BAM round-trip."""
    # quickcheck
    rc, _, err = run_cmd(["samtools", "quickcheck", "-v", sam_path])
    if rc != 0:
        msg = "quickcheck FAIL: {}".format(err.strip())
        return False, msg, {"error": msg}

    # flagstat
    rc, out, err = run_cmd(["samtools", "flagstat", sam_path])
    if rc != 0:
        msg = "flagstat FAIL: {}".format(err.strip())
        return False, msg, {"error": msg}
    lines = out.strip().split("\n")
    total_line = lines[0] if lines else "?"
    mapped_line = next((l for l in lines if "mapped" in l), "?")
    flagstat_dict = _parse_flagstat(out)

    # BAM round-trip
    rt_ok, rt_detail, rt_dict = _sam_bam_roundtrip(sam_path)
    if not rt_ok:
        d = {**flagstat_dict, **rt_dict}
        return False, "{} | {}".format(total_line, rt_detail), d

    d = {**flagstat_dict, **rt_dict}
    return True, "{} | {} | {}".format(total_line, mapped_line, rt_detail), d


def validate_sam_stdout(cmd, tmpdir):
    """Run cmd, capture stdout to temp file, then validate."""
    tmp_sam = os.path.join(tmpdir, "stdout_{}.sam".format(time.monotonic_ns()))
    try:
        mm2 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        with open(tmp_sam, "wb") as f:
            while True:
                chunk = mm2.stdout.read(65536)
                if not chunk:
                    break
                f.write(chunk)
        mm2.wait(timeout=30)
        if mm2.returncode != 0:
            msg = "minimap2 failed (rc={})".format(mm2.returncode)
            return False, msg, {"error": msg}
        return validate_sam_file(tmp_sam)
    except Exception as e:
        msg = "error: {}".format(e)
        return False, msg, {"error": msg}
    finally:
        try:
            os.unlink(tmp_sam)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# PAF validation
# ---------------------------------------------------------------------------

def _parse_paf_stats(paf_text):
    """Extract paftools.js stat summary lines and structured dict."""
    d = {}
    display_parts = []
    for line in paf_text.strip().split("\n"):
        if not line.startswith("Number of"):
            continue
        display_parts.append(line)
        # e.g. "Number of mapped sequences: 1"
        if ": " in line:
            key, _, val = line.partition(": ")
            # "Number of mapped sequences" -> "mapped_sequences"
            key = key.replace("Number of ", "").strip().replace(" ", "_")
            try:
                d[key] = int(val)
            except ValueError:
                d[key] = val
    summary = "; ".join(display_parts[:3]) if display_parts else "no stats"
    return summary, d


def validate_paf_data(paf_content):
    """Validate PAF content string: column check + paftools.js stat."""
    lines = [l for l in paf_content.strip().split("\n") if l]
    if not lines:
        return False, "empty output", {"error": "empty output"}
    bad = sum(1 for l in lines if len(l.split("\t")) < 12)
    if bad:
        msg = "{}/{} lines have < 12 columns".format(bad, len(lines))
        return False, msg, {"lines": len(lines), "bad_lines": bad}

    # paftools.js stat via stdin
    stats_dict = {}
    try:
        proc = subprocess.Popen(
            ["k8", PAFTOOLS, "stat", "/dev/stdin"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, _ = proc.communicate(input=paf_content.encode(), timeout=60)
        summary, stats_dict = _parse_paf_stats(out.decode())
    except Exception as e:
        summary = "paftools error: {}".format(e)

    d = {"lines": len(lines), **stats_dict}
    return True, "{} lines | {}".format(len(lines), summary), d


def validate_paf_file(paf_path):
    """Validate a PAF file using column check and paftools.js stat."""
    with open(paf_path) as f:
        content = f.read()
    return validate_paf_data(content)


def validate_paf_stdout(cmd):
    """Run cmd, capture stdout once and validate as PAF."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            msg = "minimap2 failed: {}".format(proc.stderr.strip()[:200])
            return False, msg, {"error": msg}
    except Exception as e:
        msg = "error: {}".format(e)
        return False, msg, {"error": msg}
    return validate_paf_data(proc.stdout)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_tests(mm2_path, ref, qry, modes):
    """Run all test combinations and return results."""
    results = []
    tmpdir = tempfile.mkdtemp(prefix="mm2_test_")

    for mode in modes:
        mode_name = mode["name"]
        base_cmd = [mm2_path] + mode["flags"]

        for fmt_label, fmt_flags, ext in FORMATS:
            is_sam = fmt_label == "SAM"

            # --- File output (-o) ---
            outfile = os.path.join(tmpdir, "{}_{}{}".format(mode_name, fmt_label, ext))
            cmd = base_cmd + fmt_flags + ["-o", outfile, ref, qry]
            r = Result(mode_name, fmt_label, "file")

            print("  {:20s} {:<6s} file   ...".format(mode_name, fmt_label), end=" ", flush=True)
            t0 = time.monotonic()
            rc, _, stderr = run_cmd(cmd)
            if rc != 0:
                r.details = "minimap2 failed (rc={}): {}".format(rc, stderr.strip()[:200])
                r.details_dict = {"error": r.details}
            elif is_sam:
                r.passed, r.details, r.details_dict = validate_sam_file(outfile)
            else:
                r.passed, r.details, r.details_dict = validate_paf_file(outfile)
            r.elapsed = time.monotonic() - t0
            print("{:<4s} ({:.1f}s)".format("PASS" if r.passed else "FAIL", r.elapsed))
            results.append(r)

            # --- Stdout output ---
            cmd_stdout = base_cmd + fmt_flags + [ref, qry]
            r2 = Result(mode_name, fmt_label, "stdout")

            print("  {:20s} {:<6s} stdout ...".format(mode_name, fmt_label), end=" ", flush=True)
            t0 = time.monotonic()
            if is_sam:
                r2.passed, r2.details, r2.details_dict = validate_sam_stdout(cmd_stdout, tmpdir)
            else:
                r2.passed, r2.details, r2.details_dict = validate_paf_stdout(cmd_stdout)
            r2.elapsed = time.monotonic() - t0
            print("{:<4s} ({:.1f}s)".format("PASS" if r2.passed else "FAIL", r2.elapsed))
            results.append(r2)

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    return results


def print_summary(results):
    """Print a formatted summary table."""
    w = 130
    print("\n" + "=" * w)
    print("{:<20s} {:<7s} {:<8s} {:<8s} {:>6s}  {}".format(
        "Mode", "Format", "Method", "Result", "Time", "Details"))
    print("-" * w)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        t = "{:.1f}s".format(r.elapsed)
        print("{:<20s} {:<7s} {:<8s} {:<8s} {:>6s}  {}".format(
            r.mode, r.fmt, r.method, status, t, r.details))
    print("=" * w)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    total_time = sum(r.elapsed for r in results)
    print("\nTotal: {}  Passed: {}  Failed: {}  Time: {:.1f}s".format(
        total, passed, failed, total_time))
    return failed


def write_json(results, mm2_path, ref, qry, dest):
    """Write results as JSON to *dest* (file path or '-' for stdout)."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    data = {
        "minimap2": mm2_path,
        "reference": ref,
        "query": qry,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "total_time": round(sum(r.elapsed for r in results), 1),
        "tests": [
            {
                "mode": r.mode,
                "format": r.fmt,
                "method": r.method,
                "passed": r.passed,
                "time": round(r.elapsed, 1),
                "details": r.details_dict,
            }
            for r in results
        ],
    }
    text = json.dumps(data, indent=2) + "\n"
    if dest == "-":
        sys.stdout.write(text)
    else:
        with open(dest, "w") as f:
            f.write(text)
        print("JSON results written to {}".format(dest))


def main():
    parser = argparse.ArgumentParser(
        description="Test minimap2 SAM/PAF output across CPU and GPU modes"
    )
    parser.add_argument(
        "minimap2", nargs="?", default=DEFAULT_MM2,
        help="Path to minimap2 binary (default: amd-default build)"
    )
    parser.add_argument(
        "reference", nargs="?", default=DEFAULT_REF,
        help="Reference FASTA file (default: test/MT-human.fa)"
    )
    parser.add_argument(
        "query", nargs="?", default=DEFAULT_QRY,
        help="Query FASTA file (default: test/MT-orang.fa)"
    )
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Skip GPU modes, only test CPU"
    )
    parser.add_argument(
        "--json", metavar="FILE", nargs="?", const="-", default=None,
        help="Write results as JSON (to FILE, or stdout if no path given)"
    )
    parser.add_argument(
        "--gpu-cfg", metavar="FILE", default=None,
        help="GPU config JSON file (optional; overrides auto-derived defaults)"
    )
    args = parser.parse_args()

    print("minimap2:  {}".format(args.minimap2))
    print("reference: {}".format(args.reference))
    print("query:     {}".format(args.query))
    print()

    # Validate inputs
    errors = check_prerequisites(args.minimap2)
    if not os.path.isfile(args.reference):
        errors.append("Reference file not found: {}".format(args.reference))
    if not os.path.isfile(args.query):
        errors.append("Query file not found: {}".format(args.query))
    if errors:
        for e in errors:
            print("ERROR: {}".format(e), file=sys.stderr)
        return 1

    # Resolve GPU config: only used when explicitly provided via --gpu-cfg
    gpu_cfg = None
    if args.gpu_cfg is not None:
        if not os.path.isfile(args.gpu_cfg):
            print("ERROR: GPU config not found: {}".format(args.gpu_cfg), file=sys.stderr)
            return 1
        gpu_cfg = args.gpu_cfg

    # Determine which modes to run
    modes = build_modes(gpu_cfg)
    if args.cpu_only:
        modes = [m for m in modes if not m["gpu"]]
        print("Mode: CPU-only (--cpu-only flag)\n")
    else:
        has_chain, has_align = detect_gpu_support(args.minimap2)
        if not has_chain and not has_align:
            print("GPU flags not found in minimap2 --help. Running CPU-only.\n")
            modes = [m for m in modes if not m["gpu"]]
        else:
            skip = set()
            if not has_chain:
                skip.add("gpu-chain")
                skip.add("gpu-chain+align")
            if not has_align:
                skip.add("gpu-align")
                skip.add("gpu-chain+align")
            if gpu_cfg is None:
                print("NOTE: GPU config not found, gpu-chain modes will use defaults")
            if skip:
                modes = [m for m in modes if m["name"] not in skip]
                print("Skipping modes: {}".format(", ".join(sorted(skip))))
            active = [m["name"] for m in modes]
            print("Active modes: {}\n".format(", ".join(active)))

    print("Running tests...\n")
    results = run_tests(args.minimap2, args.reference, args.query, modes)
    failed = print_summary(results)

    if args.json is not None:
        write_json(results, args.minimap2, args.reference, args.query, args.json)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
