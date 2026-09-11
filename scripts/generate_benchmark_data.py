#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""
generate_benchmark_data.py - Build, generate intermediates, and collect averaged timing stats
for the AMD minimap2 fork.

Usage:
    python3 scripts/generate_benchmark_data.py -r <ref.fa> -q <query.fa> [options]

Options:
    -r, --ref REF           Reference FASTA (required)
    -q, --query QUERY       Query FASTA (required)
    -n, --runs N            Number of timing runs (default: 5; 0 to skip timing)
    -t, --threads THREADS   Number of threads (default: 3)
    -o, --outdir OUTDIR     Output directory (default: ./benchmark_output)
    -p, --preset PRESET     Minimap2 preset, e.g. map-ont (default: none)
    -B, --skip-build        Skip build step
    --skip-intermediates    Skip intermediate dump generation
    --build-preset PRESET   CMake build preset (default: amd-default)
    --binary PATH           Explicit path to minimap2 binary (overrides auto-detect)
    --dump-format FMT       Dump format: JSON or FLATBUF (default: FLATBUF)
    -- [MINIMAP2_ARGS]      Extra arguments passed directly to minimap2
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import zstandard as zstd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Resource Monitor
# ─────────────────────────────────────────────────────────────────────────────

class ResourceMonitor:
    """Lightweight resource monitor that samples CPU, RAM, and GPU VRAM usage
    in a daemon thread while a subprocess runs.

    Sampling uses /proc for CPU/RAM (zero subprocess overhead) and amdsmi for
    GPU VRAM (~0.01ms per query).  Falls back to sysfs if amdsmi is unavailable.
    """

    def __init__(self, pid, interval=0.5, csv_path=None):
        self._pid = pid
        self._interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._samples = []
        self._csv_path = csv_path
        self._csv_file = None
        self._csv_writer = None
        self._gpu_devices = []
        self._amdsmi = None
        self._amdsmi_initialized = False
        self._init_gpus()

    def _init_gpus(self):
        try:
            import amdsmi
            amdsmi.amdsmi_init()
            self._amdsmi = amdsmi
            self._amdsmi_initialized = True
            handles = amdsmi.amdsmi_get_processor_handles()
            for idx, handle in enumerate(handles):
                try:
                    total_vram = amdsmi.amdsmi_get_gpu_memory_total(
                        handle, amdsmi.AmdSmiMemoryType.VRAM)
                except Exception:
                    total_vram = 0
                self._gpu_devices.append({
                    "index": idx,
                    "label": f"gpu{idx}",
                    "handle": handle,
                    "total_bytes": int(total_vram),
                    "source": "amdsmi",
                })
            if self._gpu_devices:
                return
        except Exception:
            self._amdsmi = None
            self._amdsmi_initialized = False

        # Fallback: sysfs VRAM counters (utilization unavailable in this mode)
        for card_dir in sorted(Path("/sys/class/drm").glob("card*/device")):
            vram_file = card_dir / "mem_info_vram_used"
            total_file = card_dir / "mem_info_vram_total"
            if vram_file.exists() and total_file.exists():
                card_name = card_dir.parent.name
                try:
                    total_vram = int(total_file.read_text().strip())
                except (IOError, ValueError):
                    total_vram = 0
                self._gpu_devices.append({
                    "index": len(self._gpu_devices),
                    "label": card_name,
                    "vram_path": vram_file,
                    "total_bytes": total_vram,
                    "source": "sysfs",
                })

    def _read_proc_stat(self):
        """Read CPU ticks and RSS from /proc/[pid]/stat (all threads)."""
        try:
            total_rss = 0
            total_utime = 0
            total_stime = 0
            page_size = os.sysconf("SC_PAGE_SIZE")

            # RSS from main process (threads share address space)
            try:
                stat = Path(f"/proc/{self._pid}/stat").read_text()
                parts = stat[stat.rfind(')') + 2:].split()
                rss_pages = int(parts[21])  # field 24
                total_rss = rss_pages * page_size
            except (FileNotFoundError, IndexError, ValueError):
                pass

            # CPU ticks from all threads (each thread has its own /proc/pid/task/tid/stat)
            task_dir = Path(f"/proc/{self._pid}/task")
            if task_dir.exists():
                for tid_dir in task_dir.iterdir():
                    try:
                        stat = (tid_dir / "stat").read_text()
                        parts = stat[stat.rfind(')') + 2:].split()
                        total_utime += int(parts[11])
                        total_stime += int(parts[12])
                    except (FileNotFoundError, IndexError, ValueError):
                        continue
            else:
                # Fallback: just the main process
                try:
                    stat = Path(f"/proc/{self._pid}/stat").read_text()
                    parts = stat[stat.rfind(')') + 2:].split()
                    total_utime = int(parts[11])
                    total_stime = int(parts[12])
                except (FileNotFoundError, IndexError, ValueError):
                    pass

            return total_utime + total_stime, total_rss
        except (FileNotFoundError, PermissionError):
            return 0, 0

    def _read_gpu_metrics(self):
        """Read per-GPU process VRAM usage and device utilization.

        Returns a list aligned with self._gpu_devices entries:
            [{"vram_bytes": int, "util_pct": int}, ...]
        """
        metrics = []
        for gpu in self._gpu_devices:
            vram_bytes = 0
            util_pct = 0

            if gpu.get("source") == "amdsmi" and self._amdsmi:
                try:
                    procs = self._amdsmi.amdsmi_get_gpu_process_list(gpu["handle"])
                    for proc in procs:
                        if proc.get("pid") == self._pid:
                            vram_bytes = proc.get("memory_usage", {}).get("vram_mem", 0)
                            break
                except Exception:
                    vram_bytes = 0

                try:
                    activity = self._amdsmi.amdsmi_get_gpu_activity(gpu["handle"])
                    util_pct = activity.get("gfx_activity", 0)
                except Exception:
                    util_pct = 0
            else:
                vram_path = gpu.get("vram_path")
                if vram_path:
                    try:
                        vram_bytes = int(vram_path.read_text().strip())
                    except (IOError, ValueError):
                        vram_bytes = 0

            metrics.append({
                "vram_bytes": vram_bytes,
                "util_pct": util_pct,
            })

        return metrics

    def _sample_loop(self):
        """Background sampling loop. Streams CSV rows in real-time if csv_path was set."""
        import csv as csv_mod
        clk_tck = os.sysconf("SC_CLK_TCK")

        if self._csv_path:
            Path(self._csv_path).parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv_mod.writer(self._csv_file)
            header = [
                "elapsed_sec", "cpu_utilization_pct",
                "rss_bytes", "vram_bytes_total", "gpu_utilization_pct_avg",
            ]
            for gpu in self._gpu_devices:
                label = gpu["label"]
                header.append(f"{label}_vram_bytes")
                header.append(f"{label}_utilization_pct")
            self._csv_writer.writerow(header)
            self._csv_file.flush()

        prev_sample = None
        t0 = None

        while not self._stop.is_set():
            ts = time.monotonic()
            if t0 is None:
                t0 = ts
            cpu_ticks, rss_bytes = self._read_proc_stat()
            gpu_metrics = self._read_gpu_metrics()
            gpu_vram_bytes = [m["vram_bytes"] for m in gpu_metrics]
            gpu_util_pcts = [m["util_pct"] for m in gpu_metrics]
            vram_total = sum(gpu_vram_bytes)
            gpu_util_avg = (sum(gpu_util_pcts) / len(gpu_util_pcts)) if gpu_util_pcts else 0.0
            sample = {
                "t": ts,
                "cpu_ticks": cpu_ticks,
                "rss_bytes": rss_bytes,
                "vram_bytes_total": vram_total,
                "gpu_util_pct_avg": gpu_util_avg,
                "gpu_vram_bytes": gpu_vram_bytes,
                "gpu_util_pcts": gpu_util_pcts,
            }
            self._samples.append(sample)

            # Stream to CSV
            if self._csv_writer:
                elapsed = ts - t0
                cpu_pct = 0.0
                if prev_sample is not None:
                    dt_wall = ts - prev_sample["t"]
                    dt_ticks = cpu_ticks - prev_sample["cpu_ticks"]
                    if dt_wall > 0:
                        cpu_pct = (dt_ticks / clk_tck) / dt_wall * 100.0
                row = [
                    f"{elapsed:.3f}",
                    f"{cpu_pct:.1f}",
                    rss_bytes,
                    vram_total,
                    f"{gpu_util_avg:.1f}",
                ]
                for idx in range(len(self._gpu_devices)):
                    row.append(gpu_vram_bytes[idx] if idx < len(gpu_vram_bytes) else 0)
                    row.append(gpu_util_pcts[idx] if idx < len(gpu_util_pcts) else 0)
                self._csv_writer.writerow(row)
                self._csv_file.flush()

            prev_sample = sample
            self._stop.wait(self._interval)

    def start(self):
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

    def summary(self):
        """Compute summary statistics from samples."""
        if not self._samples:
            return {}

        clk_tck = os.sysconf("SC_CLK_TCK")
        n = len(self._samples)
        rss_values = [s["rss_bytes"] for s in self._samples]
        vram_values = [s["vram_bytes_total"] for s in self._samples]
        gpu_util_values = [s.get("gpu_util_pct_avg", 0) for s in self._samples]
        gpu_total_bytes = sum(g.get("total_bytes", 0) for g in self._gpu_devices)

        # CPU utilization: delta ticks / delta wall time
        cpu_pct = 0.0
        if n >= 2:
            dt_wall = self._samples[-1]["t"] - self._samples[0]["t"]
            dt_ticks = self._samples[-1]["cpu_ticks"] - self._samples[0]["cpu_ticks"]
            if dt_wall > 0:
                cpu_pct = (dt_ticks / clk_tck) / dt_wall * 100.0

        result = {
            "samples": n,
            "interval_sec": self._interval,
            "cpu_utilization_pct": round(cpu_pct, 1),
            "ram_peak_gb": round(max(rss_values) / (1024**3), 3),
            "ram_avg_gb": round(sum(rss_values) / n / (1024**3), 3),
            "gpu_vram_peak_gb": round(max(vram_values) / (1024**3), 3),
            "gpu_vram_avg_gb": round(sum(vram_values) / n / (1024**3), 3),
            "gpu_vram_total_gb": round(gpu_total_bytes / (1024**3), 3),
            "gpu_utilization_avg_pct": round(sum(gpu_util_values) / n, 1),
            "gpu_utilization_peak_pct": max(gpu_util_values) if gpu_util_values else 0,
            "gpu_count": len(self._gpu_devices),
        }

        if gpu_total_bytes > 0:
            result["gpu_vram_peak_pct"] = round(
                max(vram_values) / gpu_total_bytes * 100, 1)

        result["gpus"] = []
        for idx, gpu in enumerate(self._gpu_devices):
            gpu_vram_values = []
            gpu_util_pcts = []
            for sample in self._samples:
                if idx < len(sample.get("gpu_vram_bytes", [])):
                    gpu_vram_values.append(sample["gpu_vram_bytes"][idx])
                if idx < len(sample.get("gpu_util_pcts", [])):
                    gpu_util_pcts.append(sample["gpu_util_pcts"][idx])

            total_bytes = gpu.get("total_bytes", 0)
            gpu_entry = {
                "gpu_index": idx,
                "label": gpu.get("label", f"gpu{idx}"),
                "gpu_vram_total_gb": round(total_bytes / (1024**3), 3),
                "gpu_vram_peak_gb": round(max(gpu_vram_values) / (1024**3), 3)
                if gpu_vram_values else 0.0,
                "gpu_vram_avg_gb": round(sum(gpu_vram_values) / len(gpu_vram_values) / (1024**3), 3)
                if gpu_vram_values else 0.0,
                "gpu_utilization_avg_pct": round(sum(gpu_util_pcts) / len(gpu_util_pcts), 1)
                if gpu_util_pcts else 0.0,
                "gpu_utilization_peak_pct": max(gpu_util_pcts) if gpu_util_pcts else 0.0,
            }
            if total_bytes > 0 and gpu_vram_values:
                gpu_entry["gpu_vram_peak_pct"] = round(max(gpu_vram_values) / total_bytes * 100, 1)
            result["gpus"].append(gpu_entry)

        return result

    def write_csv(self, path):
        """Write all samples to a CSV file with raw data (bytes, percentages)."""
        import csv
        if not self._samples:
            return

        clk_tck = os.sysconf("SC_CLK_TCK")
        t0 = self._samples[0]["t"]

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "elapsed_sec", "cpu_utilization_pct",
                "rss_bytes", "vram_bytes_total", "gpu_utilization_pct_avg",
            ])
            prev = self._samples[0]
            for i, s in enumerate(self._samples):
                elapsed = s["t"] - t0

                # CPU util: instantaneous between consecutive samples
                cpu_pct = 0.0
                if i > 0:
                    dt_wall = s["t"] - prev["t"]
                    dt_ticks = s["cpu_ticks"] - prev["cpu_ticks"]
                    if dt_wall > 0:
                        cpu_pct = (dt_ticks / clk_tck) / dt_wall * 100.0

                writer.writerow([
                    f"{elapsed:.3f}",
                    f"{cpu_pct:.1f}",
                    s["rss_bytes"],
                    s.get("vram_bytes_total", 0),
                    f"{s.get('gpu_util_pct_avg', 0):.1f}",
                ])
                prev = s

    def shutdown_gpu(self):
        """Clean up amdsmi."""
        if self._amdsmi_initialized and self._amdsmi:
            try:
                self._amdsmi.amdsmi_shut_down()
            except Exception:
                pass


def run_cmd(cmd, check=True, capture=False):
    """Run a shell command, printing it first."""
    if capture:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        return result
    return subprocess.run(cmd, check=check)


def build(root_dir, build_preset):
    """Build minimap2 AMD fork with CMake."""
    print(f"=== Building minimap2 (preset: {build_preset}) ===")
    build_dir = root_dir / "out" / build_preset / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Configure
    result = subprocess.run(
        ["cmake", "--preset", build_preset],
        cwd=str(root_dir),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        lines = (result.stdout + result.stderr).strip().splitlines()
        for line in lines[-5:]:
            print(f"  {line}")
        print("ERROR: CMake configure failed", file=sys.stderr)
        sys.exit(1)

    # Build
    result = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "minimap2_tool",
         f"-j{os.cpu_count()}"],
        capture_output=True, text=True,
    )
    lines = (result.stdout + result.stderr).strip().splitlines()
    for line in lines[-3:]:
        print(f"  {line}")
    if result.returncode != 0:
        print("ERROR: Build failed", file=sys.stderr)
        sys.exit(1)
    print()


def find_binary(root_dir, build_preset):
    """Auto-detect minimap2 binary from the build tree."""
    candidates = [
        root_dir / "out" / build_preset / "build" / "bin" / "minimap2",
        root_dir / "out" / build_preset / "install" / "bin" / "minimap2",
        root_dir / "build" / "bin" / "minimap2",
    ]
    for c in candidates:
        if c.exists() and os.access(str(c), os.X_OK):
            return c
    return None


def run_minimap2(binary, ref, query, threads, outdir, preset=None,
                 intermediates=False, dump_format="FLATBUF", extra_args=None,
                 monitor_resources=True, sample_interval=0.5,
                 query2=None):
    """Run minimap2 with --extra-out-dir and optionally --dump-intermediates.
    If query2 is given, runs paired-end (two mate files).
    Returns resource_summary dict if monitor_resources=True, else None."""
    cmd = [str(binary)]
    if preset:
        cmd += ["-x", preset]
    cmd += ["-t", str(threads)]
    cmd += ["--extra-out-dir", str(outdir)]
    if intermediates:
        cmd += [f"--dump-intermediates={dump_format}"]
    if extra_args:
        cmd += extra_args
    cmd += ["-a", str(ref), str(query)]
    if query2:
        cmd += [str(query2)]
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    stdout_file = outdir_path / "minimap2.stdout"
    stderr_file = outdir_path / "minimap2.stderr"

    resource_summary = None
    with open(stdout_file, "w") as fout, open(stderr_file, "w") as ferr:
        proc = subprocess.Popen(cmd, stdout=fout, stderr=ferr)
        monitor = None
        if monitor_resources:
            stats_dir = outdir_path / "stats"
            stats_dir.mkdir(parents=True, exist_ok=True)
            csv_path = stats_dir / "resource_usage.csv"
            monitor = ResourceMonitor(proc.pid, interval=sample_interval,
                                      csv_path=str(csv_path))
            monitor.start()
        try:
            retcode = proc.wait()
        finally:
            if monitor:
                monitor.stop()
                resource_summary = monitor.summary()
                # Always write summary JSON (even on failure)
                res_path = stats_dir / "resource_usage.json"
                with open(res_path, "w") as f:
                    json.dump(resource_summary, f, indent=2)
                monitor.shutdown_gpu()
        if retcode != 0:
            raise subprocess.CalledProcessError(retcode, cmd)

    return resource_summary


def collect_summaries(timing_dir, n_runs):
    """Load timer_summary.json and resource_usage.json from each run directory."""
    summaries = []
    for i in range(1, n_runs + 1):
        path = timing_dir / f"run_{i}" / "stats" / "timer_summary.json"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        with open(path) as f:
            entry = json.load(f)
        # Attach resource usage if available
        res_path = timing_dir / f"run_{i}" / "stats" / "resource_usage.json"
        if res_path.exists():
            with open(res_path) as f:
                entry["_resource_usage"] = json.load(f)
        summaries.append(entry)
    return summaries


def compute_average(summaries):
    """Compute averaged timing stats across runs."""
    n = len(summaries)
    if n == 0:
        return None

    root_numeric = [
        "wall_time_sec", "cpu_time_sec", "peak_rss_gb",
        "stage_total_sum_sec", "stage_per_thread_equiv_sec",
        "avg_total_per_thread_per_batch_sec", "avg_total_per_query_sec",
        "approx_non_stage_overhead_sec",
    ]
    stage_numeric = [
        "sum_sec", "total_per_thread_sec", "min_thread_batch_sec",
        "max_thread_batch_sec", "avg_thread_batch_sec",
    ]

    avg = {}
    for key in root_numeric:
        vals = [s[key] for s in summaries if key in s]
        avg[key] = sum(vals) / len(vals) if vals else 0.0

    avg["n_threads"] = summaries[0].get("n_threads", 0)
    avg["batch_count"] = summaries[0].get("batch_count", "0")
    avg["query_count"] = summaries[0].get("query_count", "0")
    avg["n_runs"] = n

    stages_template = summaries[0].get("stages", {})
    avg["stages"] = {}
    for stage_name in stages_template:
        avg["stages"][stage_name] = {}
        for sk in stage_numeric:
            vals = [
                s["stages"][stage_name][sk]
                for s in summaries
                if stage_name in s.get("stages", {})
                and sk in s["stages"][stage_name]
            ]
            avg["stages"][stage_name][sk] = (
                sum(vals) / len(vals) if vals else 0.0
            )

    per_run = []
    for i, s in enumerate(summaries):
        entry = {"run": i + 1, "wall_time_sec": s.get("wall_time_sec", 0)}
        for stage_name in stages_template:
            entry[f"{stage_name}_sum"] = (
                s.get("stages", {}).get(stage_name, {}).get("sum_sec", 0)
            )
        per_run.append(entry)
    avg["per_run"] = per_run

    # Average resource usage across runs
    res_runs = [s["_resource_usage"] for s in summaries if "_resource_usage" in s]
    if res_runs:
        res_keys = [
            "cpu_utilization_pct", "ram_peak_gb", "ram_avg_gb",
            "gpu_vram_peak_gb", "gpu_vram_avg_gb",
            "gpu_utilization_avg_pct",
        ]
        avg["resource_usage"] = {}
        for key in res_keys:
            vals = [r[key] for r in res_runs if key in r]
            if vals:
                avg["resource_usage"][key] = round(sum(vals) / len(vals), 3)
        # Peak values: take max across runs (not average)
        for key in ["ram_peak_gb", "gpu_vram_peak_gb", "gpu_utilization_peak_pct"]:
            vals = [r[key] for r in res_runs if key in r]
            if vals:
                avg["resource_usage"][f"{key}_max"] = round(max(vals), 3)
        gpu_counts = [r.get("gpu_count", 0) for r in res_runs if "gpu_count" in r]
        if gpu_counts:
            avg["resource_usage"]["gpu_count"] = max(gpu_counts)
        # Copy total VRAM from first run
        if "gpu_vram_total_gb" in res_runs[0]:
            avg["resource_usage"]["gpu_vram_total_gb"] = res_runs[0]["gpu_vram_total_gb"]
        if "gpu_vram_peak_pct" in res_runs[0]:
            peak_pcts = [r["gpu_vram_peak_pct"] for r in res_runs if "gpu_vram_peak_pct" in r]
            avg["resource_usage"]["gpu_vram_peak_pct"] = round(max(peak_pcts), 1)

        # Average per-GPU metrics by GPU index across runs.
        per_gpu = {}
        for run in res_runs:
            for gpu in run.get("gpus", []):
                idx = gpu.get("gpu_index")
                if idx is None:
                    continue
                if idx not in per_gpu:
                    per_gpu[idx] = {
                        "label": gpu.get("label", f"gpu{idx}"),
                        "gpu_vram_total_gb": gpu.get("gpu_vram_total_gb", 0),
                        "gpu_vram_peak_gb": [],
                        "gpu_vram_avg_gb": [],
                        "gpu_utilization_avg_pct": [],
                        "gpu_utilization_peak_pct": [],
                        "gpu_vram_peak_pct": [],
                    }
                per_gpu[idx]["gpu_vram_peak_gb"].append(gpu.get("gpu_vram_peak_gb", 0))
                per_gpu[idx]["gpu_vram_avg_gb"].append(gpu.get("gpu_vram_avg_gb", 0))
                per_gpu[idx]["gpu_utilization_avg_pct"].append(gpu.get("gpu_utilization_avg_pct", 0))
                per_gpu[idx]["gpu_utilization_peak_pct"].append(gpu.get("gpu_utilization_peak_pct", 0))
                if "gpu_vram_peak_pct" in gpu:
                    per_gpu[idx]["gpu_vram_peak_pct"].append(gpu["gpu_vram_peak_pct"])

        if per_gpu:
            avg["resource_usage"]["gpus"] = []
            for idx in sorted(per_gpu):
                item = per_gpu[idx]
                out_item = {
                    "gpu_index": idx,
                    "label": item["label"],
                    "gpu_vram_total_gb": item["gpu_vram_total_gb"],
                    "gpu_vram_peak_gb": round(max(item["gpu_vram_peak_gb"]), 3),
                    "gpu_vram_avg_gb": round(sum(item["gpu_vram_avg_gb"]) / len(item["gpu_vram_avg_gb"]), 3),
                    "gpu_utilization_avg_pct": round(
                        sum(item["gpu_utilization_avg_pct"]) / len(item["gpu_utilization_avg_pct"]), 1),
                    "gpu_utilization_peak_pct": round(max(item["gpu_utilization_peak_pct"]), 1),
                }
                if item["gpu_vram_peak_pct"]:
                    out_item["gpu_vram_peak_pct"] = round(max(item["gpu_vram_peak_pct"]), 1)
                avg["resource_usage"]["gpus"].append(out_item)

    return avg


def print_summary(avg):
    """Print a formatted summary table."""
    print(f"\n  Runs: {avg['n_runs']}")
    print(f"  Threads: {avg['n_threads']}")
    print(f"  {'':15s} {'Avg Sum (s)':>12s} {'Avg/Thread (s)':>15s}")
    print(f"  {'─' * 44}")
    for stage_name, stage_data in avg["stages"].items():
        print(
            f"  {stage_name:15s} "
            f"{stage_data['sum_sec']:12.4f} "
            f"{stage_data['total_per_thread_sec']:15.4f}"
        )
    print(f"  {'─' * 44}")
    print(
        f"  {'Total':15s} "
        f"{avg['stage_total_sum_sec']:12.4f} "
        f"{avg['stage_per_thread_equiv_sec']:15.4f}"
    )
    print(f"  Avg wall time:  {avg['wall_time_sec']:.4f} s")
    print(f"  Avg overhead:   {avg['approx_non_stage_overhead_sec']:.4f} s")

    # Resource usage
    res = avg.get("resource_usage")
    if res:
        print(f"\n  Resource Usage (avg across runs):")
        print(f"    GPU count:         {res.get('gpu_count', 0)}")
        print(f"    CPU utilization:   {res.get('cpu_utilization_pct', 0):.1f}%")
        print(f"    RAM peak:          {res.get('ram_peak_gb', 0):.2f} GB "
              f"(max: {res.get('ram_peak_gb_max', 0):.2f} GB)")
        print(f"    RAM avg:           {res.get('ram_avg_gb', 0):.2f} GB")
        vram_total = res.get('gpu_vram_total_gb', 0)
        if vram_total > 0:
            print(f"    GPU VRAM peak:     {res.get('gpu_vram_peak_gb', 0):.2f} GB / "
                  f"{vram_total:.1f} GB "
                  f"({res.get('gpu_vram_peak_pct', 0):.1f}%)")
            print(f"    GPU VRAM avg:      {res.get('gpu_vram_avg_gb', 0):.2f} GB")
            print(f"    GPU compute avg:   {res.get('gpu_utilization_avg_pct', 0):.1f}%"
                  f"  (peak: {res.get('gpu_utilization_peak_pct_max', 0):.0f}%)")
        for gpu in res.get("gpus", []):
            line = (f"    {gpu.get('label', 'gpu')} (idx {gpu.get('gpu_index', -1)}): "
                    f"VRAM peak {gpu.get('gpu_vram_peak_gb', 0):.2f} GB")
            if gpu.get("gpu_vram_total_gb", 0) > 0:
                line += f"/{gpu.get('gpu_vram_total_gb', 0):.1f} GB"
            if "gpu_vram_peak_pct" in gpu:
                line += f" ({gpu.get('gpu_vram_peak_pct', 0):.1f}%)"
            line += (f", compute avg {gpu.get('gpu_utilization_avg_pct', 0):.1f}%"
                     f" (peak {gpu.get('gpu_utilization_peak_pct', 0):.0f}%)")
            print(line)


def generate_intermediates(binary, ref, query, threads, inter_dir, preset,
                           dump_format, extra_args, compress=False,
                           query2=None):
    """Step 2: Generate intermediate dumps (seeds/chains/alignments)."""
    print(f"=== Generating intermediate dump ({dump_format}) ===")
    if (inter_dir / "stats" / "timer_summary.json").exists():
        print(f"  Intermediates already exist at {inter_dir}, skipping.")
        return

    if inter_dir.exists():
        shutil.rmtree(inter_dir)
    run_minimap2(
        binary, ref, query, threads, inter_dir,
        preset=preset, intermediates=True,
        dump_format=dump_format, extra_args=extra_args,
        query2=query2,
    )
    if compress:
        _compress_intermediates(inter_dir, dump_format)
    print(f"  Intermediates written to {inter_dir}")


def _compress_one(src_path):
    """Read file and zstd-compress in memory.  Return (src_path, orig_size, compressed_bytes)."""
    src = Path(src_path)
    orig_size = src.stat().st_size
    with open(src, "rb") as f:
        raw = f.read()
    cctx = zstd.ZstdCompressor(level=3)
    return src, orig_size, cctx.compress(raw)


def _compress_intermediates(inter_dir, dump_format):
    """Parallel compress individual files, pack into a ZIP (store mode) archive."""
    if dump_format == "FLATBUF":
        files = sorted(inter_dir.glob("*.bin"))
    elif dump_format == "JSON":
        files = []
        for subdir in ("seeds", "chains", "alignments"):
            d = inter_dir / subdir
            if d.is_dir():
                files.extend(sorted(d.glob("*.json")))
    else:
        return

    if not files:
        return

    n_workers = min(os.cpu_count() or 4, len(files))
    print(f"  Compressing {len(files)} files (zstd) with {n_workers} parallel workers ...")
    t0 = time.time()

    zip_path = inter_dir / "intermediates.zip"
    total_orig = 0
    total_cmp = 0
    n_packed = 0
    to_delete = []

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf, \
         ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_compress_one, f): str(f.relative_to(inter_dir)) + ".zst"
            for f in files
        }

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Compressing", unit="file"):
            arcname = futures[future]
            src, orig_size, cmp_data = future.result()
            total_orig += orig_size
            total_cmp += len(cmp_data)
            n_packed += 1
            zf.writestr(arcname, cmp_data)
            to_delete.append(src)

    # Delete originals only after zip is fully written and closed
    for src in to_delete:
        src.unlink()

    elapsed = time.time() - t0
    orig_mb = total_orig / 1e6
    cmp_mb = total_cmp / 1e6
    zip_mb = zip_path.stat().st_size / 1e6
    ratio = (cmp_mb / orig_mb * 100) if orig_mb > 0 else 0
    print(f"  Compressed {orig_mb:.0f} MB -> {cmp_mb:.0f} MB "
          f"({ratio:.0f}%) in {elapsed:.1f}s")
    print(f"  Archive: {zip_mb:.0f} MB ({n_packed} files)")

    # Remove empty subdirs (JSON case)
    for subdir in ("seeds", "chains", "alignments"):
        d = inter_dir / subdir
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


def run_timing(binary, ref, query, threads, outdir, n_runs, preset, extra_args,
               sample_interval=0.5, query2=None):
    """Steps 3-4: Run timing iterations and compute averaged stats."""
    timing_dir = outdir / "timing_runs"
    timing_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Running {n_runs} timing iterations ===")
    for i in range(1, n_runs + 1):
        run_dir = timing_dir / f"run_{i}"
        stat_file = run_dir / "stats" / "timer_summary.json"
        if stat_file.exists():
            print(f"  Run {i}/{n_runs} already has stats, skipping.")
            continue
        run_minimap2(
            binary, ref, query, threads, run_dir,
            preset=preset, intermediates=False, extra_args=extra_args,
            sample_interval=sample_interval, query2=query2,
        )
        print(f"  Run {i}/{n_runs} done")
    print()

    print(f"=== Computing average timing stats across {n_runs} runs ===")
    summaries = collect_summaries(timing_dir, n_runs)
    if not summaries:
        print("ERROR: No timer_summary.json files found", file=sys.stderr)
        sys.exit(1)

    avg = compute_average(summaries)
    avg_path = outdir / "timer_summary_avg.json"
    with open(avg_path, "w") as f:
        json.dump(avg, f, indent=2)
    print(f"  Averaged stats written to {avg_path}")
    print_summary(avg)

    return timing_dir, avg_path


def auto_fetch(target):
    """Fetch a missing ref/query input via download_preset_data.ensure_path
    (--auto-download). Returns True if ``target`` exists afterwards."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import download_preset_data as dpd
    except ImportError as exc:
        print(f"  auto-download unavailable ({exc}); install 'requests'/'tqdm'",
              file=sys.stderr)
        return False
    return dpd.ensure_path(target)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark AMD minimap2 fork: build, generate intermediates, "
                    "and collect averaged timing stats.",
    )
    parser.add_argument("-r", "--ref", required=True, help="Reference FASTA")
    parser.add_argument("-q", "--query", required=True, help="Query FASTA")
    parser.add_argument(
        "--query2", default=None,
        help="Second mate FASTQ for paired-end mapping (e.g. sr R2). "
             "When set, minimap2 runs paired-end: -a ref query query2.",
    )
    parser.add_argument(
        "-n", "--runs", type=int, default=5,
        help="Number of timing runs (default: 5)",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=3,
        help="Number of threads (default: 3)",
    )
    parser.add_argument(
        "-o", "--outdir", default=None,
        help="Output directory (default: <repo>/benchmark_output)",
    )
    parser.add_argument(
        "-p", "--preset", default=None,
        help="Minimap2 preset, e.g. map-ont",
    )
    parser.add_argument(
        "-B", "--skip-build", action="store_true",
        help="Skip the build step",
    )
    parser.add_argument(
        "--build-preset", default="amd-default",
        help="CMake build preset (default: amd-default)",
    )
    parser.add_argument(
        "--binary", default=None,
        help="Explicit path to minimap2 binary (overrides auto-detect)",
    )
    parser.add_argument(
        "--auto-download", action="store_true",
        help="fetch any missing --ref/--query/--query2 input via "
             "download_preset_data.py instead of failing",
    )
    parser.add_argument(
        "--dump-format", choices=["JSON", "FLATBUF"], default="FLATBUF",
        help="Dump format for intermediates: JSON or FLATBUF (default: FLATBUF)",
    )
    parser.add_argument(
        "--skip-intermediates", action="store_true",
        help="Skip intermediate dump generation (Step 2)",
    )
    parser.add_argument(
        "--compress", action="store_true",
        help="Compress intermediate dumps after generation (zstd + .zip archive)",
    )
    parser.add_argument(
        "--sample-interval", type=float, default=0.5,
        help="Resource monitor sampling interval in seconds (default: 0.5)",
    )
    args, extra_mm2_args = parser.parse_known_args()
    # Strip leading '--' separator if present
    if extra_mm2_args and extra_mm2_args[0] == "--":
        extra_mm2_args = extra_mm2_args[1:]

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    outdir = Path(args.outdir) if args.outdir else root_dir / "benchmark_output"

    # ── Step 1: Build ──────────────────────────────────────────────────────
    if not args.skip_build:
        build(root_dir, args.build_preset)

    if args.binary:
        binary = Path(args.binary)
    else:
        binary = find_binary(root_dir, args.build_preset)

    if not binary or not binary.exists():
        print(f"ERROR: minimap2 binary not found (tried build preset "
              f"'{args.build_preset}')", file=sys.stderr)
        sys.exit(1)

    print(f"Using binary: {binary}")
    if extra_mm2_args:
        print(f"Extra minimap2 args: {' '.join(extra_mm2_args)}")
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Step 1b: Ensure inputs are present (optionally auto-download) ───────
    for f in [args.ref, args.query] + ([args.query2] if args.query2 else []):
        if Path(f).exists():
            continue
        if args.auto_download and auto_fetch(f):
            continue
        print(f"ERROR: input not found: {f}"
              + ("" if args.auto_download
                 else " (pass --auto-download to fetch a known preset dataset)"),
              file=sys.stderr)
        sys.exit(1)
    # ── Step 2: Generate intermediates (single run) ────────────────────────
    inter_dir = outdir / "intermediates"
    if args.skip_intermediates:
        print("=== Skipping intermediate dump generation (--skip-intermediates) ===")
    else:
        generate_intermediates(
            binary, args.ref, args.query, args.threads, inter_dir,
            args.preset, args.dump_format, extra_mm2_args,
            compress=args.compress, query2=args.query2,
        )

    # ── Steps 3-4: Timing runs + averaging ────────────────────────
    timing_dir = None
    avg_path = None
    if args.runs > 0:
        timing_dir, avg_path = run_timing(
            binary, args.ref, args.query, args.threads, outdir,
            args.runs, args.preset, extra_mm2_args,
            sample_interval=args.sample_interval, query2=args.query2,
        )
    else:
        print("=== Skipping timing runs (--runs 0) ===")

    print()
    print("=== Benchmark complete ===")
    print(f"  Intermediates: {inter_dir}")
    if timing_dir:
        print(f"  Timing runs:   {timing_dir}")
        print(f"  Average stats: {avg_path}")


if __name__ == "__main__":
    main()
