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
generate_all_mode_benchmark_data.py
Run generate_benchmark_data.py across multiple minimap2 execution modes
and collect timing + intermediate data for each.

Base modes:
  cpu             CPU only
  gpu_chain       GPU chaining  (--gpu-chain --gpu-cfg <cfg>)
  gpu_align       GPU alignment (--gpu-align)
  gpu_both        GPU chain + align (--gpu-chain --gpu-align --gpu-cfg <cfg>)

Thread counts are given as a list via -t.  Each base mode is run at every
thread count, producing directories like cpu_t1/, cpu_t32/, gpu_chain_t8/, etc.

Usage:
  python3 scripts/generate_all_mode_benchmark_data.py -r ref.fa -q reads.fq [OPTIONS]

Examples:
  # All 4 modes at t=1 and t=32, 3 timing runs each
  python3 scripts/generate_all_mode_benchmark_data.py -r ref.fa -q reads.fq -t 1 32 -n 3

  # CPU-only at multiple thread counts
  python3 scripts/generate_all_mode_benchmark_data.py -r ref.fa -q reads.fq --modes cpu -t 1 4 8 32

  # GPU chain + align at 32 threads, intermediates only (no timing)
  python3 scripts/generate_all_mode_benchmark_data.py -r ref.fa -q reads.fq --modes gpu_both -t 32 -n 0

  # Pass extra minimap2 flags
  python3 scripts/generate_all_mode_benchmark_data.py -r ref.fa -q reads.fq -- --cs --MD
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Mode definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Mode:
    name: str
    description: str
    threads: int
    extra_flags: List[str] = field(default_factory=list)
    is_gpu: bool = False
    needs_gpu_cfg: bool = False


BASE_MODES = ["cpu", "gpu_chain", "gpu_align", "gpu_both"]

BASE_MODE_DEFS = {
    "cpu":       {"desc": "CPU",              "flags": [],                              "gpu": False, "cfg": False},
    "gpu_chain": {"desc": "GPU Chain",         "flags": ["--gpu-chain"],                  "gpu": True,  "cfg": True},
    "gpu_align": {"desc": "GPU Align",         "flags": ["--gpu-align"],                  "gpu": True,  "cfg": False},
    "gpu_both":  {"desc": "GPU Chain + Align", "flags": ["--gpu-chain", "--gpu-align"],   "gpu": True,  "cfg": True},
}


def expand_modes(base_names: List[str], thread_counts: List[int]) -> List[Mode]:
    """Expand base mode names x thread counts into concrete Mode objects."""
    modes = []
    for t in sorted(thread_counts):
        for name in base_names:
            d = BASE_MODE_DEFS[name]
            modes.append(Mode(
                name=f"{name}_t{t}",
                description=f"{d['desc']} ({t} threads)",
                threads=t,
                extra_flags=list(d["flags"]),
                is_gpu=d["gpu"],
                needs_gpu_cfg=d["cfg"],
            ))
    return modes


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


def detect_gpu_count() -> int:
    """Detect the number of AMD GPUs available.

    Detection order:
      1. MM2_GPU_COUNT env var (explicit override)
      2. HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES env var
      3. hipGetDeviceCount via libamdhip64.so (most accurate, respects
         ROCR_VISIBLE_DEVICES, works in containers with HIP runtime)
      4. rocminfo agent count (fallback when HIP runtime unavailable)
    """
    override = os.environ.get("MM2_GPU_COUNT")
    if override is not None:
        return int(override)

    for env_var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        val = os.environ.get(env_var)
        if val is not None:
            return len([x for x in val.split(",") if x.strip()])

    try:
        import ctypes
        hip = None
        for lib in ("libamdhip64.so", "/opt/rocm/lib/libamdhip64.so"):
            try:
                hip = ctypes.CDLL(lib)
                break
            except OSError:
                continue
        if hip is not None:
            count = ctypes.c_int(0)
            if hip.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value > 0:
                return count.value
    except (OSError, AttributeError):
        pass

    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            # Split rocminfo output into agent blocks (delimited by *****)
            # and count blocks that have a Vendor Name other than "CPU".
            blocks = re.split(r'\*{7,}', result.stdout)
            gpu_count = 0
            for block in blocks:
                if re.search(r'Marketing Name:', block):
                    vendor = re.search(r'Vendor Name:\s+(\S+)', block)
                    if vendor and vendor.group(1) != 'CPU':
                        gpu_count += 1
            if gpu_count > 0:
                return gpu_count
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return 0


def print_header(msg: str):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 72}{Colors.NC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{msg}{Colors.NC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 72}{Colors.NC}\n")


def print_mode_header(msg: str):
    print(f"\n{Colors.YELLOW}--- {msg} ---{Colors.NC}")


def load_avg_json(path: Path) -> Optional[dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Run a single mode
# ─────────────────────────────────────────────────────────────────────────────

def run_mode(
    mode: Mode,
    *,
    script: Path,
    ref: str,
    query: str,
    outdir: Path,
    n_runs: int,
    preset: Optional[str],
    build_preset: str,
    binary: Optional[str],
    skip_build: bool,
    skip_intermediates: bool,
    compress: bool,
    gpu_cfg: Optional[str],
    extra_mm2_args: List[str],
    gpu_id: Optional[int] = None,
    query2: Optional[str] = None,
) -> Optional[dict]:
    """Invoke generate_benchmark_data.py for a single mode, return avg JSON."""
    mode_dir = outdir / mode.name
    print_mode_header(f"{mode.name}: {mode.description}")

    # Skip if averaged stats already exist
    avg_path = mode_dir / "timer_summary_avg.json"
    existing = load_avg_json(avg_path)
    if existing is not None:
        print(f"{Colors.GREEN}  SKIP – {avg_path} already exists "
              f"(wall {existing.get('wall_time_sec', 0):.4f}s){Colors.NC}")
        return existing

    cmd = [
        sys.executable, str(script),
        "-r", ref,
        "-q", query,
        "-n", str(n_runs),
        "-t", str(mode.threads),
        "-o", str(mode_dir),
    ]
    if query2:
        cmd += ["--query2", query2]
    if preset:
        cmd += ["-p", preset]
    if skip_build:
        cmd += ["-B"]
    if skip_intermediates:
        cmd += ["--skip-intermediates"]
    if compress:
        cmd += ["--compress"]
    cmd += ["--build-preset", build_preset]
    if binary:
        cmd += ["--binary", binary]

    # Build extra minimap2 args for this mode
    mm2_extra = list(mode.extra_flags)
    if mode.needs_gpu_cfg and gpu_cfg:
        mm2_extra += ["--gpu-cfg", gpu_cfg]
    mm2_extra += extra_mm2_args

    if mm2_extra:
        cmd += ["--"]
        cmd += mm2_extra

    env = os.environ.copy()
    gpu_label = ""
    if gpu_id is not None:
        if "HIP_VISIBLE_DEVICES" not in os.environ:
            env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_label = f"  [GPU {gpu_id}]"

    print(f"{gpu_label}  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"{Colors.RED}{gpu_label}  FAILED (exit code {result.returncode}){Colors.NC}")
        return None

    avg_path = mode_dir / "timer_summary_avg.json"
    avg = load_avg_json(avg_path)
    if avg is None:
        print(f"{gpu_label}  No timing data ({avg_path.name} not found)")
        return {}
    print(f"{Colors.GREEN}{gpu_label}  OK – avg wall time: {avg.get('wall_time_sec', 0):.4f}s{Colors.NC}")
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: Dict[str, Optional[dict]], modes: List[Mode]):
    """Print a transposed comparison table (modes as rows, stages as columns)."""
    print_header("DATA GENERATION SUMMARY")

    # Use the first mode as baseline
    baseline_name = modes[0].name if modes else ""
    baseline = results.get(baseline_name)
    baseline_wall = baseline.get("wall_time_sec", 0) if baseline else 0

    # Collect all stage names from any result
    stage_names: List[str] = []
    for avg in results.values():
        if avg and "stages" in avg:
            for sn in avg["stages"]:
                if sn not in stage_names:
                    stage_names.append(sn)

    # Column widths
    name_w = 22
    col_w = 12

    # ── Header row: mode | stage1 | stage2 | ... | wall | speedup ──
    cols = stage_names + ["wall_time"]
    if baseline_wall > 0:
        cols.append("speedup")
    hdr = f"{'mode':<{name_w}}"
    sep = f"{'─' * name_w}"
    for c in cols:
        hdr += f" | {c:>{col_w}}"
        sep += f"-+-{'─' * col_w}"
    print(hdr)
    print(sep)

    # ── One row per mode ──
    for m in modes:
        avg = results.get(m.name)
        if avg is None:
            row = f"{m.name:<{name_w}}"
            for _ in cols:
                row += f" | {'FAIL':>{col_w}}"
            print(f"{Colors.RED}{row}{Colors.NC}")
            continue

        row = f"{m.name:<{name_w}}"
        for sn in stage_names:
            if sn in avg.get("stages", {}):
                val = avg["stages"][sn].get("sum_sec", 0)
                row += f" | {val:>{col_w}.2f}"
            else:
                row += f" | {'N/A':>{col_w}}"

        wall = avg.get("wall_time_sec", 0)
        row += f" | {wall:>{col_w}.2f}"

        if baseline_wall > 0:
            if wall > 0:
                speedup = baseline_wall / wall
                color = Colors.GREEN if speedup >= 1.0 else Colors.YELLOW
                row += f" | {color}{speedup:>{col_w}.2f}x{Colors.NC}"
            else:
                row += f" | {'N/A':>{col_w}}"

        print(row)

    print(sep)
    print()


def export_json(results: Dict[str, Optional[dict]], modes: List[Mode], outdir: Path,
                ref: str, query: str):
    """Write combined results JSON."""
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "reference": ref,
            "query": query,
            "output_dir": str(outdir),
        },
        "modes": {},
    }
    for m in modes:
        avg = results.get(m.name)
        data["modes"][m.name] = {
            "description": m.description,
            "threads": m.threads,
            "is_gpu": m.is_gpu,
            "avg": avg,
        }

    out_path = outdir / "all_mode_results.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results exported to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def auto_fetch(target, script_dir: Path):
    """Fetch a missing ref/query input via download_preset_data.ensure_path
    (--auto-download). Returns True if ``target`` exists afterwards."""
    sys.path.insert(0, str(script_dir))
    try:
        import download_preset_data as dpd
    except ImportError as exc:
        print(f"  auto-download unavailable ({exc}); install 'requests'/'tqdm'",
              file=sys.stderr)
        return False
    return dpd.ensure_path(target)


def parse_args():
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Run generate_benchmark_data.py across multiple minimap2 modes "
                    "and collect timing + intermediate data for each.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
BASE MODES:
  cpu             CPU only
  gpu_chain       GPU chaining  (--gpu-chain)
  gpu_align       GPU alignment (--gpu-align)
  gpu_both        GPU chain + align

Each base mode is run at every thread count given via -t.
Output dirs: <outdir>/<mode>_t<N>/

EXAMPLES:
  python3 scripts/generate_all_mode_benchmark_data.py -r ref.fa -q reads.fq -t 1 32 -n 3
  python3 scripts/generate_all_mode_benchmark_data.py -r ref.fa -q reads.fq --modes cpu -t 1 4 8 32
  python3 scripts/generate_all_mode_benchmark_data.py -r ref.fa -q reads.fq --modes gpu_both -t 32 -n 0
  python3 scripts/generate_all_mode_benchmark_data.py -r ref.fa -q reads.fq -- --cs --MD
""",
    )

    parser.add_argument("-r", "--ref", required=True, help="Reference FASTA")
    parser.add_argument("-q", "--query", required=True, help="Query FASTA/FASTQ")
    parser.add_argument(
        "--query2", default=None,
        help="Second mate FASTQ for paired-end mapping (e.g. sr R2).",
    )
    parser.add_argument(
        "-n", "--runs", type=int, default=5,
        help="Number of timing runs per mode (default: 5; 0 to skip timing)",
    )
    parser.add_argument(
        "-t", "--threads", type=int, nargs="+", default=[32],
        help="Thread count(s) to run each mode at (default: 32)",
    )
    parser.add_argument(
        "-o", "--outdir", default=None,
        help="Root output directory (default: <repo>/all_mode_data)",
    )
    parser.add_argument(
        "-p", "--preset", default=None,
        help="Minimap2 preset, e.g. map-ont",
    )
    parser.add_argument(
        "--build-preset", default="release",
        help="CMake build preset (default: release)",
    )
    parser.add_argument(
        "--binary", default=None,
        help="Explicit path to minimap2 binary (overrides auto-detect)",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Skip the prebuild step (use an already-built binary)",
    )
    parser.add_argument(
        "--auto-download", action="store_true",
        help="fetch any missing --ref/--query/--query2 input via "
             "download_preset_data.py instead of failing",
    )
    parser.add_argument(
        "--skip-intermediates", action="store_true",
        help="Skip intermediate dump generation in each mode",
    )
    parser.add_argument(
        "--no-compress", action="store_true",
        help="Keep raw intermediate files (do not compress into .zip archive)",
    )
    parser.add_argument(
        "--gpu-cfg", default=None,
        help="GPU config file path (default: auto-detect from hardware)",
    )
    parser.add_argument(
        "--skip-gpu", action="store_true",
        help="Skip all GPU modes",
    )
    parser.add_argument(
        "--modes", nargs="+", default=None,
        choices=BASE_MODES,
        help="Base modes to run (default: all). "
             "Each is run at every -t thread count.",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "--export-json", action="store_true",
        help="Export combined results to all_mode_results.json",
    )

    args, extra_mm2_args = parser.parse_known_args()
    if extra_mm2_args and extra_mm2_args[0] == "--":
        extra_mm2_args = extra_mm2_args[1:]

    return args, extra_mm2_args, script_dir, workspace_dir


def select_modes(args) -> List[Mode]:
    """Expand base mode names x thread counts into concrete Mode list."""
    base_names = args.modes if args.modes else list(BASE_MODES)
    if args.skip_gpu:
        base_names = [n for n in base_names if not BASE_MODE_DEFS[n]["gpu"]]

    if not base_names:
        print("ERROR: No modes selected", file=sys.stderr)
        sys.exit(1)

    return expand_modes(base_names, args.threads)


def prebuild(workspace_dir: Path, build_preset: str):
    """Run CMake configure + build before concurrent mode runs."""
    print_mode_header("Pre-build step")
    build_dir = workspace_dir / "out" / build_preset / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    rc = subprocess.run(
        ["cmake", "--preset", build_preset],
        cwd=str(workspace_dir),
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        lines = (rc.stdout + rc.stderr).strip().splitlines()
        for line in lines[-5:]:
            print(f"  {line}")
        print(f"{Colors.RED}CMake configure failed{Colors.NC}", file=sys.stderr)
        sys.exit(1)

    rc = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "minimap2_tool",
         f"-j{os.cpu_count()}"],
        capture_output=True, text=True,
    )
    lines = (rc.stdout + rc.stderr).strip().splitlines()
    for line in lines[-3:]:
        print(f"  {line}")
    if rc.returncode != 0:
        print(f"{Colors.RED}Build failed{Colors.NC}", file=sys.stderr)
        sys.exit(1)
    print(f"{Colors.GREEN}  Build OK{Colors.NC}")
    print()


def run_all_modes(selected_modes: List[Mode], args, outdir: Path,
                  gen_script: Path, gpu_count: int,
                  extra_mm2_args: List[str]) -> Dict[str, Optional[dict]]:
    """Run all selected modes sequentially, return {mode_name: avg_json}."""
    results: Dict[str, Optional[dict]] = {}

    print_mode_header(f"Running {len(selected_modes)} modes sequentially")

    for mode in selected_modes:
        gpu_id = 0 if mode.is_gpu and gpu_count > 0 else None
        try:
            avg = run_mode(
                mode,
                script=gen_script,
                ref=args.ref,
                query=args.query,
                outdir=outdir,
                n_runs=args.runs,
                preset=args.preset,
                build_preset=args.build_preset,
                binary=args.binary,
                skip_build=True,
                skip_intermediates=args.skip_intermediates,
                compress=not args.no_compress,
                gpu_cfg=args.gpu_cfg,
                extra_mm2_args=extra_mm2_args,
                gpu_id=gpu_id,
                query2=args.query2,
            )
            results[mode.name] = avg
        except Exception as e:
            print(f"{Colors.RED}  {mode.name} raised: {e}{Colors.NC}")
            results[mode.name] = None

    return results


def main():
    args, extra_mm2_args, script_dir, workspace_dir = parse_args()

    if args.no_color:
        Colors.disable()

    for f in [args.ref, args.query] + ([args.query2] if args.query2 else []):
        if Path(f).exists():
            continue
        if args.auto_download and auto_fetch(f, script_dir):
            continue
        print(f"ERROR: input not found: {f}"
              + ("" if args.auto_download
                 else " (pass --auto-download to fetch a known preset dataset)"),
              file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir) if args.outdir else workspace_dir / "all_mode_data"
    outdir.mkdir(parents=True, exist_ok=True)

    gen_script = script_dir / "generate_benchmark_data.py"
    if not gen_script.exists():
        print(f"ERROR: {gen_script} not found", file=sys.stderr)
        sys.exit(1)

    selected_modes = select_modes(args)
    gpu_count = detect_gpu_count()
    gpu_modes = [m for m in selected_modes if m.is_gpu]
    cpu_modes = [m for m in selected_modes if not m.is_gpu]

    if gpu_modes and gpu_count == 0:
        print(f"{Colors.RED}WARNING: No GPUs detected but GPU modes requested.{Colors.NC}")
        print(f"  GPU modes will still run but without HIP_VISIBLE_DEVICES assignment.")

    print_header("Minimap2 Multi-Mode Data Generation")
    print(f"  Reference:  {args.ref}")
    print(f"  Query:      {args.query}")
    print(f"  Threads:    {', '.join(str(t) for t in sorted(args.threads))}")
    print(f"  Runs/mode:  {args.runs}")
    print(f"  Output:     {outdir}")
    print(f"  GPUs:       {gpu_count}")
    print(f"  Modes:      {', '.join(m.name for m in selected_modes)}")
    print(f"  CPU modes:  {', '.join(m.name for m in cpu_modes) or 'none'}")
    print(f"  GPU modes:  {', '.join(m.name for m in gpu_modes) or 'none'}")
    if extra_mm2_args:
        print(f"  Extra args: {' '.join(extra_mm2_args)}")
    print()

    if not args.skip_build and not args.binary:
        prebuild(workspace_dir, args.build_preset)

    results = run_all_modes(selected_modes, args, outdir, gen_script,
                            gpu_count, extra_mm2_args)

    active = [m for m in selected_modes if m.name in results]
    print_summary(results, active)

    if args.export_json:
        export_json(results, active, outdir, args.ref, args.query)

    failed = [m.name for m in active if results.get(m.name) is None]
    if failed:
        print(f"{Colors.RED}FAILED modes: {', '.join(failed)}{Colors.NC}")
        return 1

    print(f"{Colors.GREEN}All {len(active)} modes completed successfully.{Colors.NC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
