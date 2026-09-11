#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""Accuracy-floor evaluation: mapeval (simulated reads) + junceval (splice vs GENCODE).

Evaluates one or more minimap2 builds/versions across execution modes, scoring
with ``misc/paftools.js``. Builds are passed generically as ``--binary
NAME=PATH[=MODES]`` (e.g. a stock release vs a development build), so the script
makes no assumption about which build is the "reference" — any labelled build can
act as the baseline. Closes the "Accuracy floor met" verification item.

Two evaluations:

  mapeval   Simulate reads from a chosen reference region (pbsim3 for long reads,
            mason2 for short reads), map them back to the *full* reference with
            each build/mode, and score with ``paftools.js mapeval`` (fraction of
            wrong mappings at mapQ thresholds). Truth coordinates ride in the read
            names via ``paftools.js pbsim2fq`` (long) or ``mason2fq`` (short).

  junceval  Map real RNA reads (``-ax splice``) with each binary/mode and score
            predicted splice junctions against a GENCODE annotation with
            ``paftools.js junceval`` (junctions found / missing / wrong).

Scope notes:
  * Short-read (sr) mapeval uses mason2 (``mason_simulator`` + ``paftools.js
    mason2fq``); ``paftools.js mapeval`` parses pbsim and mason2 read names but
    NOT dwgsim. The sr sim point is skipped if ``mason_simulator`` is not on PATH.
  * GIAB small-variant concordance (hap.py) is a separate, optional pass and is
    not run here (hap.py/bgzip/tabix are typically unavailable).

Everything is idempotent where practical: simulated reads and per-mode SAMs are
reused if already present (use --force to regenerate).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAFTOOLS = REPO_ROOT / "misc" / "paftools.js"
DEFAULT_DEV = str(REPO_ROOT / "out" / "release-gfx942" / "build" / "bin" / "minimap2")
DEFAULT_PBSIM_DATA = str(Path.home() / ".local" / "src" / "pbsim3" / "data")

# Mode -> extra minimap2 flags (cfg is appended only when the mode uses the GPU
# chainer). Mirrors scripts/generate_all_mode_benchmark_data.py.
MODE_FLAGS = {
    "cpu": {"flags": [], "cfg": False},
    "gpu_chain": {"flags": ["--gpu-chain"], "cfg": True},
    "gpu_align": {"flags": ["--gpu-align"], "cfg": False},
    "gpu_both": {"flags": ["--gpu-chain", "--gpu-align"], "cfg": True},
}

# mapeval sim points: label -> {simulator, pbsim3 model basename, minimap2 preset}.
# HiFi uses the highest-accuracy ONT-HQ model as a high-accuracy long-read proxy
# (true CCS needs multi-pass + the ccs tool); documented as such in the report.
# map-pb uses the PacBio RS II ERRHMM model (classic CLR). "sr" simulates
# Illumina-style short reads with mason2 instead of pbsim3.
SIM_MODELS = {
    "map-ont": {"sim": "pbsim", "model": "ERRHMM-ONT", "preset": "map-ont"},
    "map-hifi": {"sim": "pbsim", "model": "ERRHMM-ONT-HQ", "preset": "map-hifi"},
    "map-pb": {"sim": "pbsim", "model": "ERRHMM-RSII", "preset": "map-pb"},
    "sr": {"sim": "mason", "model": None, "preset": "sr"},
}

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
RESET = "\033[0m"
NO_COLOR = os.environ.get("NO_COLOR") is not None


def c(code: str, text: str) -> str:
    return text if NO_COLOR else f"{code}{text}{RESET}"


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def auto_fetch(target, kind):
    """Fetch a missing input via the download_*_data.py helpers (--auto-download).

    kind 'ref'/'reads' use download_preset_data; 'gtf' uses
    download_verification_data. Returns True if ``target`` exists afterwards.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        if kind == "gtf":
            import download_verification_data as dvd
            return dvd.ensure_path(target)
        import download_preset_data as dpd
        return dpd.ensure_path(target)
    except ImportError as exc:
        print(c(RED, f"  auto-download unavailable ({exc}); install 'requests'/'tqdm'"))
        return False


def parse_binaries(specs, default_modes):
    """Parse ``NAME=PATH[=MODES]`` build specs into (name, path, modes) tuples.

    MODES is an optional comma-separated subset of MODE_FLAGS; when omitted the
    global default_modes apply. This keeps the script version-agnostic: any
    labelled build can be the baseline (e.g. a cpu-only stock release vs a GPU
    development build).
    """
    out = []
    for spec in specs:
        parts = spec.split("=", 2)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            print(c(RED, f"error: bad --binary spec '{spec}' (want NAME=PATH[=MODES])"))
            sys.exit(2)
        name, path = parts[0], parts[1]
        if len(parts) == 3 and parts[2].strip():
            modes = [m.strip() for m in parts[2].split(",") if m.strip()]
            bad = [m for m in modes if m not in MODE_FLAGS]
            if bad:
                print(c(RED, f"error: unknown mode(s) {bad} in --binary '{spec}' "
                             f"(choose from {list(MODE_FLAGS)})"))
                sys.exit(2)
        else:
            modes = list(default_modes)
        out.append((name, path, modes))
    return out


def run(cmd, **kw):
    """Run a command, raising on failure unless check=False."""
    printable = " ".join(str(x) for x in cmd)
    print(f"  {c(DIM, '$ ' + printable)}")
    return subprocess.run(cmd, **kw)


def map_cmd(binary, mode, preset, ref, reads, threads, gpu_cfg, extra=None):
    """Build a minimap2 command line for one binary/mode."""
    cmd = [binary, "-ax", preset, "-t", str(threads)]
    if extra:
        cmd += list(extra)
    mf = MODE_FLAGS[mode]
    cmd += mf["flags"]
    if mf["cfg"] and gpu_cfg:
        cmd += ["--gpu-cfg", gpu_cfg]
    cmd += [str(ref)] + [str(r) for r in (reads if isinstance(reads, (list, tuple)) else [reads])]
    return cmd


def do_map(binary, mode, preset, ref, reads, out_sam, threads, gpu_cfg, force, extra=None):
    """Map reads to a SAM file.

    Cached unless --force. The cache is invalidated automatically when a
    dependency (the minimap2 binary or the input reads) is newer than the SAM,
    so swapping in a rebuilt binary or regenerating sims transparently re-maps.
    """
    out_sam = Path(out_sam)
    if out_sam.exists() and out_sam.stat().st_size > 0 and not force:
        rlist = reads if isinstance(reads, (list, tuple)) else [reads]
        deps = [binary, *[str(r) for r in rlist]]
        newest_dep = max((Path(p).stat().st_mtime for p in deps if Path(p).exists()),
                         default=0.0)
        if out_sam.stat().st_mtime >= newest_dep:
            print(f"  {c(GREEN, 'reuse')} {out_sam.name}")
            return out_sam
        print(f"  {c(YELLOW, 'stale')} {out_sam.name} (binary/reads newer) → re-map")
    cmd = map_cmd(binary, mode, preset, ref, reads, threads, gpu_cfg, extra)
    err = out_sam.with_suffix(".stderr")
    with open(out_sam, "wb") as so, open(err, "wb") as se:
        cp = run(cmd, stdout=so, stderr=se)
    if cp.returncode != 0:
        tail = err.read_text(errors="replace").splitlines()[-5:]
        print(f"  {c(RED, 'map FAILED')} (rc={cp.returncode}) -- last stderr:")
        for ln in tail:
            print(f"    {ln}")
        return None
    return out_sam


def ensure_index(name, binary, preset, ref, idx_dir, threads, force):
    """Build (once) and reuse a per-(binary, preset) minimizer index .mmi.

    Mapping the FASTA re-indexes the whole reference on every run (~60% of wall
    time for a 3 GB genome); a prebuilt .mmi is loaded instead. The index is
    keyed by build label + preset (indexing params k/w/H come from the preset),
    cached unless --force or the binary/ref is newer. Returns the .mmi path, or
    None to fall back to mapping straight from the FASTA.
    """
    idx_dir.mkdir(parents=True, exist_ok=True)
    safe = preset.replace(":", "_")
    idx = idx_dir / f"idx.{name}.{safe}.mmi"
    if idx.exists() and idx.stat().st_size > 0 and not force:
        deps = [p for p in (binary, ref) if Path(p).exists()]
        newest = max((Path(p).stat().st_mtime for p in deps), default=0.0)
        if idx.stat().st_mtime >= newest:
            print(f"  {c(GREEN, 'reuse idx')} {idx.name}")
            return idx
    log = idx.with_suffix(".log")
    with open(log, "wb") as se:
        cp = run([str(binary), "-x", preset, "-t", str(threads), "-d", str(idx), str(ref)],
                 stdout=se, stderr=se)
    if cp.returncode != 0:
        print(c(YELLOW, f"  index build failed for {name}/{preset}; mapping from FASTA"))
        return None
    return idx


# ── mapeval ──────────────────────────────────────────────────────────────────


@dataclass
class MapevalResult:
    label: str
    binary: str
    mode: str
    mapped: int = 0          # cumulative reads mapped (mapQ >= 0)
    wrong: int = 0           # cumulative wrong mappings
    err_frac: float = float("nan")   # wrong / mapped
    unmapped: int = 0
    raw: str = ""
    sam: str = ""          # path to the scored SAM (for cross-mode divergence)
    # per-mapQ cumulative rows, descending mapQ: (mapQ, cum_total, cum_wrong)
    q_rows: list = field(default_factory=list)


def parse_mapeval(text: str, label: str, binary: str, mode: str) -> MapevalResult:
    """Parse paftools.js mapeval output.

    mapeval prints 'Q' rows (high mapQ → low mapQ); each row is:
        Q <mapQ> <grp_tot> <grp_err> <cum_err_frac> <cum_tot>
    where columns 3/4 are per-group counts but columns 5/6 are CUMULATIVE
    (col5 = cumulative wrong/total, col6 = cumulative reads mapped down to this
    mapQ). The LAST row (lowest mapQ) therefore holds the overall totals:
        mapped   = col6 (cumulative total)
        err_frac = col5 (cumulative error fraction)
        wrong    = sum of col4 over all rows  (= err_frac * mapped)
    A trailing 'U' row counts unmapped reads.
    """
    res = MapevalResult(label=label, binary=binary, mode=mode, raw=text)
    last_q = None
    wrong = 0
    for line in text.splitlines():
        t = line.split()
        if len(t) >= 6 and t[0] == "Q":
            last_q = t
            try:
                wrong += int(t[3])
                res.q_rows.append((int(t[1]), int(t[5]), wrong))
            except ValueError:
                pass
        elif len(t) >= 2 and t[0] == "U":
            try:
                res.unmapped = int(t[1])
            except ValueError:
                pass
    if last_q is not None:
        try:
            res.mapped = int(last_q[5])
            res.err_frac = float(last_q[4])
            res.wrong = wrong
        except (ValueError, IndexError):
            pass
    return res


def cum_at(q_rows, qmin):
    """Cumulative (total, wrong) over reads with mapQ >= qmin.

    q_rows are in descending mapQ with cumulative counts, so the last row whose
    mapQ >= qmin holds the cumulative totals for that confidence band.
    """
    tot = wrong = 0
    for mapq, ct, cw in q_rows:
        if mapq >= qmin:
            tot, wrong = ct, cw
        else:
            break
    return tot, wrong


def simulate_pbsim(pbsim, src_fa, src_fai, sim_dir, label, spec, args, region_tag):
    """pbsim3 wgs/errhmm -> per-contig .maf -> truth-named FASTQ (pbsim2fq)."""
    model_path = Path(args.pbsim_data) / f"{spec['model']}.model"
    if not model_path.exists():
        print(c(YELLOW, f"  skip {label}: pbsim model {model_path} missing"))
        return None
    simtag = f"{region_tag}.d{args.sim_depth}"
    prefix = sim_dir / f"sim.{label}.{simtag}"
    maf_files = sorted(sim_dir.glob(f"sim.{label}.{simtag}_*.maf"))
    if not maf_files or args.force:
        for old in sim_dir.glob(f"sim.{label}.{simtag}_*"):
            old.unlink()
        cp = run([pbsim, "--strategy", "wgs", "--method", "errhmm",
                  "--errhmm", str(model_path), "--depth", str(args.sim_depth),
                  "--genome", str(src_fa), "--prefix", str(prefix)])
        if cp.returncode != 0:
            print(c(RED, f"  pbsim failed for {label}"))
            return None
        # Some pbsim3 builds gzip their output (.maf.gz / .fq.gz); decompress.
        for gz in sorted(sim_dir.glob(f"sim.{label}.{simtag}_*.maf.gz")):
            run(["gunzip", "-f", str(gz)])
        maf_files = sorted(sim_dir.glob(f"sim.{label}.{simtag}_*.maf"))
    if not maf_files:
        print(c(YELLOW, f"  no .maf produced for {label}"))
        return None
    # pbsim2fq embeds truth coords in read names. Use the simulation-source fai
    # so contig names resolve by index correctly (not the full-genome fai).
    sim_fq = sim_dir / f"sim.{label}.{simtag}.fastq"
    if not sim_fq.exists() or args.force:
        cp = run(["k8", str(PAFTOOLS), "pbsim2fq", str(src_fai), *[str(m) for m in maf_files]],
                 stdout=open(sim_fq, "wb"))
        if cp.returncode != 0:
            print(c(RED, f"  pbsim2fq failed for {label}"))
            return None
    return sim_fq, simtag


def simulate_mason(src_fa, sim_dir, label, args, region_tag):
    """mason2 Illumina sim -> alignment SAM -> truth-named FASTQ (mason2fq).

    mason_simulator writes the ground-truth alignment via -oa; paftools.js
    mason2fq converts that SAM into an interleaved FASTQ whose read names encode
    the true coordinates (the format paftools.js mapeval parses).
    """
    simtag = f"{region_tag}.n{args.sr_num_reads}.l{args.sr_read_length}"
    mason_sam = sim_dir / f"sim.{label}.{simtag}.mason.sam"
    r1 = sim_dir / f"sim.{label}.{simtag}.r1.fq"
    r2 = sim_dir / f"sim.{label}.{simtag}.r2.fq"
    if not mason_sam.exists() or args.force:
        cp = run(["mason_simulator", "-ir", str(src_fa),
                  "-n", str(args.sr_num_reads),
                  "-o", str(r1), "-or", str(r2), "-oa", str(mason_sam),
                  "--illumina-read-length", str(args.sr_read_length),
                  "--num-threads", str(args.threads)])
        if cp.returncode != 0:
            print(c(RED, f"  mason_simulator failed for {label}"))
            return None
    sim_fq = sim_dir / f"sim.{label}.{simtag}.fastq"
    if not sim_fq.exists() or args.force:
        cp = run(["k8", str(PAFTOOLS), "mason2fq", str(mason_sam)],
                 stdout=open(sim_fq, "wb"))
        if cp.returncode != 0:
            print(c(RED, f"  mason2fq failed for {label}"))
            return None
    return sim_fq, simtag


def eval_mapeval(args, binaries, work):
    print()
    print(c(BOLD, "═══ mapeval (simulated-read mapping accuracy) ═══"))

    sim_dir = work / "sim"
    sim_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.sim_models.split(",") if m.strip()]
    need_pbsim = any(SIM_MODELS.get(m, {}).get("sim") == "pbsim" for m in models)
    need_mason = any(SIM_MODELS.get(m, {}).get("sim") == "mason" for m in models)
    pbsim = "pbsim" if have("pbsim") else ("pbsim3" if have("pbsim3") else None)
    if need_pbsim and pbsim is None:
        print(c(YELLOW, "  note: pbsim/pbsim3 not on PATH — long-read sim points skipped"))
    if need_mason and not have("mason_simulator"):
        print(c(YELLOW, "  note: mason_simulator not on PATH — sr sim point skipped"))

    # Extract the simulation source region into a small FASTA, and index it.
    # pbsim2fq resolves the truth contig name *by index* into this .fai, so it
    # MUST be the simulation-source fai (not the full-genome fai), otherwise the
    # truth names are mislabelled (e.g. chr21 reads tagged as chr1).
    region = args.sim_region.split(",")
    region_tag = "_".join(region)
    src_fa = sim_dir / ("src_" + region_tag + ".fa")
    if not src_fa.exists() or args.force:
        cp = run(["samtools", "faidx", str(args.ref), *region], stdout=open(src_fa, "wb"))
        if cp.returncode != 0:
            print(c(RED, "  samtools faidx (region) failed"))
            return []
    src_fai = Path(str(src_fa) + ".fai")
    if not src_fai.exists() or args.force:
        cp = run(["samtools", "faidx", str(src_fa)])
        if cp.returncode != 0:
            print(c(RED, "  samtools faidx (source index) failed"))
            return []

    results = []
    idx_dir = work / "index"
    idx_cache = {}

    # Phase 1: generate all simulated read sets in parallel (independent per model).
    def make_sim(label):
        spec = SIM_MODELS.get(label)
        if spec is None:
            print(c(YELLOW, f"  skip unknown sim model '{label}'"))
            return None
        if spec["sim"] == "pbsim":
            if pbsim is None:
                return None
            made = simulate_pbsim(pbsim, src_fa, src_fai, sim_dir, label, spec, args, region_tag)
        elif spec["sim"] == "mason":
            if not have("mason_simulator"):
                return None
            made = simulate_mason(src_fa, sim_dir, label, args, region_tag)
        else:
            return None
        if made is None:
            return None
        sim_fq, simtag = made
        return (label, spec["preset"], sim_fq, simtag)

    with ThreadPoolExecutor(max_workers=max(1, len(models))) as ex:
        sims = [s for s in ex.map(make_sim, models) if s]

    # Phase 1.5: prebuild every (build, preset) index in parallel. Indexing a
    # large reference is largely single-threaded, so building the distinct
    # indexes concurrently fills the cores a single sequential build leaves idle.
    needed, seen = [], set()
    for _label, preset, _fq, _tag in sims:
        for name, binary, _modes in binaries:
            if (name, preset) not in seen:
                seen.add((name, preset))
                needed.append((name, binary, preset))
    if needed:
        idx_threads = max(4, args.threads // len(needed))

        def build_idx(item):
            name, binary, preset = item
            return (name, preset), ensure_index(name, binary, preset, args.ref,
                                                 idx_dir, idx_threads, args.force)

        with ThreadPoolExecutor(max_workers=len(needed)) as ex:
            for key, idx in ex.map(build_idx, needed):
                idx_cache[key] = idx

    # Phase 2: map + score concurrently. CPU-only maps overlap freely; GPU maps
    # (gpu_chain/gpu_align/gpu_both) serialize on a semaphore since they share the
    # single GPU. mapeval scoring (single-threaded) is folded into each worker so
    # it overlaps too. Concurrency is capped so the per-map threads don't
    # oversubscribe the host.
    tasks = []
    for label, preset, sim_fq, simtag in sims:
        for name, binary, bmodes in binaries:
            target = idx_cache.get((name, preset)) or args.ref
            for mode in bmodes:
                out_sam = sim_dir / f"aln.{label}.{simtag}.{name}.{mode}.sam"
                tasks.append((label, preset, sim_fq, simtag, name, binary, mode, target, out_sam))

    gpu_sem = threading.Semaphore(1)
    cores = os.cpu_count() or args.threads
    map_conc = max(1, min(len(tasks), cores // max(1, args.threads)))

    def run_task(t):
        label, preset, sim_fq, simtag, name, binary, mode, target, out_sam = t
        uses_gpu = bool(MODE_FLAGS[mode]["flags"])
        if uses_gpu:
            gpu_sem.acquire()
        try:
            sam = do_map(binary, mode, preset, target, sim_fq, out_sam,
                         args.threads, args.gpu_cfg, args.force)
        finally:
            if uses_gpu:
                gpu_sem.release()
        if sam is None:
            return None
        cp = run(["k8", str(PAFTOOLS), "mapeval", str(sam)], stdout=subprocess.PIPE, text=True)
        res = parse_mapeval(cp.stdout, label, name, mode)
        res.sam = str(sam)
        print(f"  {c(CYAN, label + '.' + name + '.' + mode)}: mapped={res.mapped} "
              f"wrong={res.wrong} err={res.err_frac:.3g}")
        return res

    with ThreadPoolExecutor(max_workers=map_conc) as ex:
        results = [r for r in ex.map(run_task, tasks) if r is not None]

    # Stable order for the summary tables (model, then build, then mode).
    model_ix = {m: i for i, m in enumerate(models)}
    ver_ix = {n: i for i, (n, _b, _m) in enumerate(binaries)}
    mode_ix = {m: i for i, m in enumerate(["cpu", "gpu_chain", "gpu_align", "gpu_both"])}
    results.sort(key=lambda r: (model_ix.get(r.label, 99), ver_ix.get(r.binary, 99),
                                mode_ix.get(r.mode, 99)))
    return results


# ── junceval ─────────────────────────────────────────────────────────────────


@dataclass
class JuncevalResult:
    binary: str
    mode: str
    mapped: int = 0
    predicted: int = 0      # total predicted introns
    novel: int = 0          # non-overlapping (not in annotation)
    correct: int = 0        # introns matching annotation
    pct: float = float("nan")
    raw: str = ""


def parse_junceval(text: str, binary: str, mode: str) -> JuncevalResult:
    """Parse paftools.js junceval stdout summary.

    junceval prints (exact strings, paftools.js):
        # unmapped reads: N
        # mapped reads: N
        # primary alignments: N
        # singletons: N
        # predicted introns: N
        # non-overlapping introns: N     (novel / not in annotation)
        # correct introns: N (PCT%)      (match the annotation)
    """
    res = JuncevalResult(binary=binary, mode=mode, raw=text)
    for line in text.splitlines():
        m = re.search(r":\s*(\d+)", line)
        if m is None:
            continue
        n = int(m.group(1))
        if line.startswith("# mapped reads"):
            res.mapped = n
        elif line.startswith("# predicted introns"):
            res.predicted = n
        elif line.startswith("# non-overlapping introns"):
            res.novel = n
        elif line.startswith("# correct introns"):
            res.correct = n
            pm = re.search(r"\(([\d.]+)%\)", line)
            if pm:
                res.pct = float(pm.group(1))
    return res


def eval_junceval(args, binaries, work):
    print()
    print(c(BOLD, "═══ junceval (splice-junction accuracy vs GENCODE) ═══"))
    if not args.gtf or not args.splice_reads:
        print(c(RED, "  junceval needs --gtf and --splice-reads"))
        return []
    gtf = Path(args.gtf)
    if not gtf.exists() and getattr(args, "auto_download", False):
        print(c(YELLOW, f"  auto-download: fetching missing annotation → {gtf}"))
        auto_fetch(str(gtf), "gtf")
    if not gtf.exists():
        print(c(RED, f"  junceval needs annotation {gtf}"))
        return []
    reads = Path(args.splice_reads)
    if not reads.exists() and getattr(args, "auto_download", False):
        print(c(YELLOW, f"  auto-download: fetching missing splice reads → {reads}"))
        auto_fetch(str(reads), "reads")
    if not reads.exists():
        print(c(RED, f"  junceval needs splice reads {reads}"))
        return []

    jdir = work / "junceval"
    jdir.mkdir(parents=True, exist_ok=True)

    # Optional read subset for speed.
    if args.junc_subset and args.junc_subset > 0:
        sub = jdir / f"reads.{args.junc_subset}.fastq"
        if not sub.exists() or args.force:
            cp = run(["python3", str(REPO_ROOT / "scripts" / "subset_queries.py"),
                      str(reads), "-o", str(sub), "--start", "0",
                      "--end", str(args.junc_subset), "--no-count"])
            if cp.returncode != 0:
                print(c(RED, "  subset_queries failed"))
                return []
        reads = sub

    results = []
    for name, binary, bmodes in binaries:
        for mode in bmodes:
            tag = f"{name}.{mode}"
            out_sam = jdir / f"aln.{tag}.sam"
            sam = do_map(binary, mode, "splice",
                         args.ref, reads, out_sam, args.threads, args.gpu_cfg, args.force,
                         extra=["-uf"])
            if sam is None:
                continue
            # junceval: 1st positional is the annotation GTF, 2nd is the SAM.
            # -c restricts to /^(chr)?([0-9]+|X|Y)$/ (autosomes + X/Y).
            cp = run(["k8", str(PAFTOOLS), "junceval", "-c", str(gtf), str(sam)],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            res = parse_junceval(cp.stdout, name, mode)
            results.append(res)
            print(f"  {c(CYAN, tag)}: predicted={res.predicted} correct={res.correct} "
                  f"({res.pct:.2f}%) novel={res.novel}")
            print(c(DIM, "    " + cp.stdout.strip().replace("\n", "\n    ")))
    return results


# ── report ───────────────────────────────────────────────────────────────────


Q_THRESHOLDS = [60, 30, 10, 1, 0]


def primary_alns(sam_path):
    """Index primary alignments (flag & 0x900 == 0): qname -> (rname, pos, strand, cigar, mapq)."""
    m = {}
    try:
        with open(sam_path) as f:
            for ln in f:
                if not ln or ln[0] == "@":
                    continue
                t = ln.split("\t")
                if len(t) < 11:
                    continue
                flag = int(t[1])
                if flag & 0x900:
                    continue
                m[t[0]] = (t[2], t[3], flag & 16, t[5], int(t[4]))
    except OSError:
        pass
    return m


def print_mode_divergence(mapeval_res):
    """Report GPU<->CPU primary-alignment divergence per build.

    Compares each GPU mode against the same build's cpu mode at the primary locus
    (rname/pos/strand, CIGAR, mapQ). Supplementary/secondary-set and AS
    differences are additional; scripts/compare_sam.py does the full SAM diff.
    """
    groups = {}
    for r in mapeval_res:
        groups.setdefault((r.label, r.binary), []).append(r)
    rows = []
    for (label, ver), rs in groups.items():
        cpu = next((x for x in rs if x.mode == "cpu" and x.sam), None)
        others = [x for x in rs if x.mode != "cpu" and x.sam]
        if cpu is None or not others:
            continue
        base = primary_alns(cpu.sam)
        for o in others:
            g = primary_alns(o.sam)
            pos = cig = mq = tot = 0
            for q, b in base.items():
                x = g.get(q)
                if x is None:
                    continue
                tot += 1
                if (b[0], b[1], b[2]) != (x[0], x[1], x[2]):
                    pos += 1
                elif b[3] != x[3]:
                    cig += 1
                if b[4] != x[4]:
                    mq += 1
            rows.append((label, ver, o.mode, tot, pos, cig, mq))
    if not rows:
        return
    print()
    print(c(BOLD, "GPU↔CPU primary divergence (each GPU mode vs the same build's cpu mode):"))
    print(f"  {'model':<10} {'version':<9} {'mode':<10} {'pos/strand':>11} "
          f"{'cigar-only':>11} {'mapQ':>8}   of primaries")
    for label, ver, mode, tot, pos, cig, mq in rows:
        mark = c(YELLOW, "  ⚠ diverges") if (pos or cig) else c(GREEN, "  = cpu")
        print(f"  {label:<10} {ver:<9} {mode:<10} {pos:>11} {cig:>11} {mq:>8}   of {tot}{mark}")
    print(c(DIM, "  Primary-locus moves shown here; supplementary/secondary-set and CIGAR/AS "
                 "differences are additional — see scripts/compare_sam.py for the full "
                 "SAM-level comparison."))


def print_q_breakdown(mapeval_res):
    """Per-mapQ accuracy floor: cumulative err% for reads with mapQ >= Q.

    The aggregate err% is dominated by deliberately-ambiguous low-mapQ reads; the
    high-confidence bands (Q>=60/30) are where a real mapping regression would
    show. Printing the curve per model lets each version/mode be compared at the
    same confidence threshold.
    """
    if not mapeval_res:
        return
    print()
    print(c(BOLD, "mapeval per-mapQ accuracy floor (cumulative err%/wrong at mapQ >= Q):"))
    by_model = {}
    for r in mapeval_res:
        by_model.setdefault(r.label, []).append(r)
    for model, rows in by_model.items():
        print()
        print(c(BOLD, f"  {model}:"))
        header = f"    {'version.mode':<22}" + "".join(f"{'Q>=' + str(q):>16}" for q in Q_THRESHOLDS)
        print(header)
        for r in rows:
            cells = ""
            for q in Q_THRESHOLDS:
                tot, wrong = cum_at(r.q_rows, q)
                ef = (wrong / tot * 100) if tot else 0.0
                cells += f"{ef:.4f}%/{wrong}".rjust(16)
            print(f"    {(r.binary + '.' + r.mode):<22}{cells}")


def print_summary(mapeval_res, junceval_res):
    print()
    print(c(BOLD, "════════════════════ ACCURACY SUMMARY ════════════════════"))
    if mapeval_res:
        print()
        print(c(BOLD, "mapeval (lower err% = better; mapped/wrong over all mapQ):"))
        print(f"  {'model':<10} {'version':<9} {'mode':<10} {'mapped':>9} {'wrong':>7} {'err%':>8} {'unmapped':>9}")
        for r in mapeval_res:
            print(f"  {r.label:<10} {r.binary:<9} {r.mode:<10} {r.mapped:>9} "
                  f"{r.wrong:>7} {r.err_frac*100:>7.3f}% {r.unmapped:>9}")
        print_q_breakdown(mapeval_res)
        print_mode_divergence(mapeval_res)
    if junceval_res:
        print()
        print(c(BOLD, "junceval (higher correct%% = better; novel = not in annotation):"))
        print(f"  {'version':<9} {'mode':<10} {'predicted':>10} {'correct':>9} {'correct%':>9} {'novel':>8}")
        for r in junceval_res:
            print(f"  {r.binary:<9} {r.mode:<10} {r.predicted:>10} {r.correct:>9} "
                  f"{r.pct:>8.2f}% {r.novel:>8}")
    print()
    print(c(DIM, "Interpretation: each version's cpu mode should match the others within "
                 "noise; GPU modes may diverge slightly (chaining/tie order). "
                 "Large gaps = regression."))


def main():
    p = argparse.ArgumentParser(
        description="Accuracy-floor evaluation (mapeval + junceval) across one or more minimap2 builds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ref", required=True, help="reference FASTA (mapping target + simulation source)")
    p.add_argument("--binary", action="append", default=[], metavar="NAME=PATH[=MODES]",
                   help="minimap2 build to evaluate; repeatable. NAME is a free-form "
                        "label (e.g. v2.31, dev, pr110), PATH is the binary, and the "
                        "optional =MODES is a comma-separated subset of "
                        "{cpu,gpu_chain,gpu_align,gpu_both} to run for that build "
                        "(default: the global --modes). A cpu-only reference build is "
                        "specified as NAME=PATH=cpu. Example: "
                        f"--binary v2.31=/path/to/minimap2=cpu --binary dev={DEFAULT_DEV}")
    p.add_argument("--gpu-cfg", default=None,
                   help="GPU config JSON for GPU chain modes (default: none — use the "
                        "binary's built-in GPU defaults)")
    p.add_argument("-t", "--threads", type=int, default=32, help="threads (default: %(default)s)")
    p.add_argument("--modes", nargs="+", default=["cpu", "gpu_chain", "gpu_align"],
                   choices=list(MODE_FLAGS),
                   help="default modes evaluated per build unless overridden in --binary "
                        "(default: cpu gpu_chain gpu_align)")
    p.add_argument("--eval", nargs="+", default=["mapeval", "junceval"],
                   choices=["mapeval", "junceval"], help="which evaluations to run")
    p.add_argument("--workdir", default="accuracy_eval", help="output workdir for sims/SAMs (default: %(default)s)")
    p.add_argument("--force", action="store_true", help="regenerate sims/SAMs even if present")

    # mapeval
    p.add_argument("--sim-region", default="chr21,chr22",
                   help="comma-separated ref contigs to simulate from (default: %(default)s)")
    p.add_argument("--sim-depth", type=float, default=10.0, help="simulated depth (default: %(default)s)")
    p.add_argument("--sim-models", default="map-ont,map-hifi,map-pb,sr",
                   help="comma-separated sim points from {map-ont,map-hifi,map-pb,sr} "
                        "(map-* use pbsim3, sr uses mason2; default: %(default)s)")
    p.add_argument("--pbsim-data", default=DEFAULT_PBSIM_DATA,
                   help="dir with pbsim3 ERRHMM-*.model files (default: %(default)s)")
    p.add_argument("--sr-num-reads", type=int, default=2000000,
                   help="mason2 fragment pairs for the sr sim point (default: %(default)s)")
    p.add_argument("--sr-read-length", type=int, default=150,
                   help="mason2 Illumina read length for the sr sim point (default: %(default)s)")

    # junceval
    p.add_argument("--splice-reads", default=None,
                   help="real RNA reads for junceval (required when --eval junceval)")
    p.add_argument("--gtf", default=None,
                   help="GENCODE annotation GTF for junceval, exon rows with transcript_id "
                        "(required when --eval junceval)")
    p.add_argument("--junc-subset", type=int, default=100000,
                   help="map only the first N RNA reads for junceval (0 = all; default: %(default)s)")

    p.add_argument("--auto-download", action="store_true",
                   help="fetch missing --ref/--gtf/--splice-reads inputs via the "
                        "download_*_data.py helpers instead of failing")

    args = p.parse_args()

    if not PAFTOOLS.exists() or not have("k8"):
        print(c(RED, "error: need k8 + misc/paftools.js for scoring"))
        sys.exit(2)
    if not args.binary:
        print(c(RED, "error: need at least one --binary NAME=PATH[=MODES]"))
        sys.exit(2)
    binaries = parse_binaries(args.binary, args.modes)
    for name, path, _modes in binaries:
        if not Path(path).exists():
            print(c(RED, f"error: binary not found for '{name}': {path}"))
            sys.exit(2)

    if not Path(args.ref).exists():
        if args.auto_download:
            print(c(YELLOW, f"  auto-download: fetching missing reference → {args.ref}"))
            auto_fetch(args.ref, "ref")
        if not Path(args.ref).exists():
            print(c(RED, f"error: reference not found: {args.ref}"
                         + ("" if args.auto_download else " (pass --auto-download to fetch)")))
            sys.exit(2)

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    mapeval_res, junceval_res = [], []
    if "mapeval" in args.eval:
        mapeval_res = eval_mapeval(args, binaries, work)
    if "junceval" in args.eval:
        junceval_res = eval_junceval(args, binaries, work)

    print_summary(mapeval_res, junceval_res)


if __name__ == "__main__":
    main()
