#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""GIAB HG002 small-variant concordance (hap.py) across one or more minimap2 builds.

Aligns HG002 HiFi reads with each build/mode, calls variants with Clair3 (hifi
model), and scores the calls against the GIAB HG002 v4.2.1 benchmark with hap.py
— reporting SNP/INDEL recall, precision and F1 per build-mode. Comparing a dev
build to the upstream oracle shows whether a change (e.g. the v2.31 merge)
regresses variant-calling-relevant alignment; comparing a build's GPU modes to
its own CPU shows whether GPU divergence reaches downstream variant calls.

Builds are passed generically as ``--binary NAME=PATH[=MODES]`` (same convention
as scripts/run_accuracy_eval.py), so any labelled build can act as the baseline.

Tooling (no sudo required if docker is usable without sudo):
  - Clair3 : ``hkubal/clair3`` docker image (hifi model bundled)
  - hap.py : ``hap.py`` on PATH (e.g. a docker-backed wrapper) or the
             ``jmcdani20/hap.py`` docker image
  - samtools (local) for sort/index

Data prep (one-time):
  scripts/download_verification_data.py --components giab          # truth VCF + BED
  scripts/download_verification_data.py --components giab-reads     # HG002 HiFi reads

Example:
python3 scripts/run_giab_concordance.py --ref ref.fa \\
  --binary upstream=/path/to/upstream/minimap2=cpu \\
  --binary dev=out/release-gfx942/build/bin/minimap2 \\
  --reads     <preset_data>/giab/hg002.hifi.chr20.fastq.gz \\
  --truth-vcf <preset_data>/giab/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz \\
  --truth-bed <preset_data>/giab/HG002_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed \\
  --region chr20 --modes cpu gpu_chain gpu_align -t 48
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"
NO_COLOR = os.environ.get("NO_COLOR") is not None

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEV = str(REPO_ROOT / "out" / "release-gfx942" / "build" / "bin" / "minimap2")

# mode -> extra minimap2 flags (mirrors run_accuracy_eval.py / generate_all_mode).
MODE_FLAGS = {
    "cpu": [],
    "gpu_chain": ["--gpu-chain"],
    "gpu_align": ["--gpu-align"],
    "gpu_both": ["--gpu-chain", "--gpu-align"],
}


def c(code, text):
    return text if NO_COLOR else f"{code}{text}{RESET}"


def have(tool):
    return shutil.which(tool) is not None


def docker_usable():
    return have("docker") and subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def auto_fetch(target, kind, region):
    """Fetch a missing input via the download_*_data.py helpers (--auto-download).

    kind 'ref' uses download_preset_data; 'reads'/'truth' use
    download_verification_data. Returns True if ``target`` exists afterwards.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        if kind == "ref":
            import download_preset_data as dpd
            return dpd.ensure_path(target)
        import download_verification_data as dvd
        return dvd.ensure_path(target, region=region)
    except ImportError as exc:
        print(c(RED, f"  auto-download unavailable ({exc}); install 'requests'/'tqdm'"))
        return False


def parse_binaries(specs, default_modes):
    """NAME=PATH[=MODES] -> list of (name, path, [modes]). Mirrors run_accuracy_eval.py."""
    out = []
    for spec in specs:
        parts = spec.split("=")
        if len(parts) < 2:
            sys.exit(f"error: --binary must be NAME=PATH[=MODES], got '{spec}'")
        name, path = parts[0], parts[1]
        modes = parts[2].split(",") if len(parts) > 2 and parts[2] else list(default_modes)
        for m in modes:
            if m not in MODE_FLAGS:
                sys.exit(f"error: unknown mode '{m}' in --binary '{spec}'")
        out.append((name, path, modes))
    return out


def run(cmd, **kw):
    print(c(DIM, "    $ " + " ".join(str(x) for x in cmd)))
    return subprocess.run(cmd, check=True, **kw)


def mount_args(paths):
    """-v host:host for the parent dir of each path, collapsed to top-level dirs
    (a dir nested under another in the set is dropped to avoid overlapping binds)."""
    dirs = sorted({str(Path(p).resolve().parent) for p in paths})
    top = [d for d in dirs if not any(d != o and (d + "/").startswith(o + "/") for o in dirs)]
    args = []
    for d in top:
        args += ["-v", f"{d}:{d}"]
    return args


def align(binary, mode, ref, reads, preset, out_bam, threads):
    flags = MODE_FLAGS[mode]
    print(f"  {c(CYAN, 'align')} ({preset} {' '.join(flags) or 'cpu'})")
    mm = subprocess.Popen([binary, "-ax", preset, *flags, "-t", str(threads), ref, reads],
                          stdout=subprocess.PIPE, stderr=open(str(out_bam) + ".mm2.log", "wb"))
    sort = subprocess.Popen(["samtools", "sort", "-@", str(threads), "-o", str(out_bam), "-"],
                            stdin=mm.stdout)
    mm.stdout.close()
    sort.communicate()
    mm.wait()
    if mm.returncode or sort.returncode:
        raise RuntimeError(f"alignment failed (mm2={mm.returncode}, sort={sort.returncode})")
    run(["samtools", "index", "-@", str(threads), str(out_bam)])


def call_clair3(bam, ref, region, out_dir, threads, image, model, mounts):
    print(f"  {c(CYAN, 'Clair3')} ({model}, {region})")
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cmd = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
           *mount_args(mounts), image, "/opt/bin/run_clair3.sh",
           f"--bam_fn={bam}", f"--ref_fn={ref}", f"--threads={threads}",
           "--platform=hifi", f"--model_path=/opt/models/{model}",
           f"--ctg_name={region}", f"--output={out_dir}"]
    with open(out_dir.parent / "clair3.log", "wb") as log:
        if subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode != 0:
            raise RuntimeError(f"Clair3 failed (see {out_dir.parent / 'clair3.log'})")
    return out_dir / "merge_output.vcf.gz"


def run_happy(truth_vcf, truth_bed, calls_vcf, ref, region, out_prefix, threads, image, mounts):
    print(f"  {c(CYAN, 'hap.py')}")
    out_prefix = Path(out_prefix)
    base = ["", truth_vcf, str(calls_vcf), "-f", truth_bed, "-r", ref,
            "-o", str(out_prefix), "--location", region, "--threads", str(threads)]
    if have("hap.py"):
        cmd = ["hap.py"] + base[1:]
    else:
        cmd = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
               *mount_args(mounts), image, "/opt/hap.py/bin/hap.py"] + base[1:]
    with open(str(out_prefix) + ".run.log", "wb") as log:
        if subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode != 0:
            raise RuntimeError(f"hap.py failed (see {out_prefix}.run.log)")
    return Path(str(out_prefix) + ".summary.csv")


def parse_happy_summary(csv_path):
    """Return {'SNP': {...}, 'INDEL': {...}} from a hap.py *.summary.csv (PASS rows)."""
    res = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("Filter") != "PASS" or row.get("Type") not in ("SNP", "INDEL"):
                continue
            res[row["Type"]] = {
                "recall": float(row["METRIC.Recall"]),
                "precision": float(row["METRIC.Precision"]),
                "f1": float(row["METRIC.F1_Score"]),
                "tp": int(row["TRUTH.TP"]),
                "fn": int(row["TRUTH.FN"]),
                "fp": int(row["QUERY.FP"]),
            }
    return res


def main():
    p = argparse.ArgumentParser(
        description="GIAB HG002 small-variant concordance (hap.py) across minimap2 builds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--ref", required=True, help="reference FASTA (must match the GIAB truth assembly)")
    p.add_argument("--binary", action="append", default=[], metavar="NAME=PATH[=MODES]",
                   help="minimap2 build to evaluate; repeatable (default modes via --modes)")
    p.add_argument("--reads", required=True,
                   help="HG002 reads (from download_verification_data.py --components giab-reads)")
    p.add_argument("--truth-vcf", required=True, help="GIAB HG002 benchmark VCF (.vcf.gz)")
    p.add_argument("--truth-bed", required=True, help="GIAB HG002 high-confidence BED")
    p.add_argument("--region", default="chr20", help="region to call + score (default: %(default)s)")
    p.add_argument("--preset", default="map-hifi", help="minimap2 preset (default: %(default)s)")
    p.add_argument("--modes", nargs="+", default=["cpu"], choices=list(MODE_FLAGS),
                   help="default modes per build unless overridden in --binary (default: cpu)")
    p.add_argument("-t", "--threads", type=int, default=48, help="threads (default: %(default)s)")
    p.add_argument("--workdir", default="giab_concordance", help="output dir (default: %(default)s)")
    p.add_argument("--clair3-image", default="hkubal/clair3:latest")
    p.add_argument("--clair3-model", default="hifi")
    p.add_argument("--happy-image", default="jmcdani20/hap.py:v0.3.12")
    p.add_argument("--auto-download", action="store_true",
                   help="fetch any missing --ref/--reads/--truth-* input via the "
                        "download_*_data.py helpers instead of failing")
    p.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)  # stream progress, not just at exit
    except (AttributeError, ValueError):
        pass

    if not args.binary:
        sys.exit("error: need at least one --binary NAME=PATH[=MODES]")
    for f, kind in ((args.ref, "ref"), (args.reads, "reads"),
                    (args.truth_vcf, "truth"), (args.truth_bed, "truth")):
        if Path(f).exists():
            continue
        if args.auto_download:
            print(c(YELLOW, f"  auto-download: fetching missing {kind} → {f}"))
            if auto_fetch(f, kind, args.region):
                continue
        sys.exit(f"error: missing input: {f}\n  (prep via scripts/download_verification_data.py "
                 f"--components giab,giab-reads, or pass --auto-download)")
    if not have("samtools"):
        sys.exit("error: samtools not on PATH (needed for sort/index)")
    if not docker_usable():
        sys.exit("error: docker not usable (needed for Clair3" +
                 ("" if have("hap.py") else " and hap.py") + ")")
    if not Path(args.ref + ".fai").exists():
        run(["samtools", "faidx", args.ref])

    binaries = parse_binaries(args.binary, args.modes)
    for name, path, _ in binaries:
        if not Path(path).exists():
            sys.exit(f"error: binary not found for '{name}': {path}")

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    mounts = [args.ref, args.reads, args.truth_vcf, args.truth_bed, str(work.resolve())]

    results = []  # (label, summary dict)
    for name, path, modes in binaries:
        for mode in modes:
            label = name if mode == "cpu" else f"{name}/{mode}"
            out = work / label.replace("/", "_")
            out.mkdir(parents=True, exist_ok=True)
            summ = out / "happy.summary.csv"
            print(c(BOLD, f"\n=== {label} ==="))
            if summ.exists() and not args.force:
                print(f"  {c(GREEN, '✓ cached')} ({summ})")
                results.append((label, parse_happy_summary(summ)))
                continue
            try:
                bam = out / "aln.bam"
                align(path, mode, args.ref, args.reads, args.preset, bam, args.threads)
                calls = call_clair3(bam, args.ref, args.region, out / "clair3",
                                    args.threads, args.clair3_image, args.clair3_model, mounts)
                run_happy(args.truth_vcf, args.truth_bed, calls, args.ref, args.region,
                          out / "happy", args.threads, args.happy_image, mounts)
                results.append((label, parse_happy_summary(summ)))
                print(f"  {c(GREEN, 'DONE')}")
            except (RuntimeError, subprocess.CalledProcessError) as exc:
                print(f"  {c(RED, 'FAILED:')} {exc}")
                results.append((label, {}))

    # Summary table
    print(c(BOLD, "\n" + "=" * 78))
    print(c(BOLD, f"GIAB HG002 concordance — {args.region} (Clair3 {args.clair3_model})"))
    print(c(BOLD, "=" * 78))
    hdr = f"{'build/mode':<22} {'SNP R':>7} {'SNP P':>7} {'SNP F1':>8} {'INDEL R':>8} {'INDEL P':>8} {'INDEL F1':>9}"
    print(hdr)
    print("-" * len(hdr))
    for label, s in results:
        if not s:
            print(f"{label:<22} {c(RED, 'FAILED'):>7}")
            continue
        snp, ind = s.get("SNP", {}), s.get("INDEL", {})
        print(f"{label:<22} {snp.get('recall', 0):>7.4f} {snp.get('precision', 0):>7.4f} "
              f"{snp.get('f1', 0):>8.4f} {ind.get('recall', 0):>8.4f} {ind.get('precision', 0):>8.4f} "
              f"{ind.get('f1', 0):>9.4f}")
    print(c(DIM, f"\nresults under {work}/  (per build-mode: aln.bam, clair3/, happy.*)"))


if __name__ == "__main__":
    main()
