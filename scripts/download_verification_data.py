#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""Acquire the extra ground-truth datasets for minimap2 correctness verification.

This complements ``download_preset_data.py`` (real preset reads + reference) by
fetching the ground-truth and feature-flag inputs:

  anno   GENCODE GRCh38 annotation (GTF -> BED12 via paftools.js gff2bed).
         Unblocks --junc-bed/-j/--jump-pass1, derived .spsc, and junceval.
  giab   GIAB HG002 v4.2.1 benchmark VCF + high-confidence BED (variant-calling
         concordance ground truth for the genomic presets).
  giab-reads  HG002 HiFi reads for one chromosome (default chr20), streamed from
         the remote PrecisionFDA BAM via byte-range — the read input for the
         hap.py concordance (scripts/run_giab_concordance.py). Heavy (~1.5 GB);
         NOT part of 'all', opt in with --components giab-reads.

Simulated reads for paftools.js mapeval are generated on demand by
``scripts/run_accuracy_eval.py`` itself (pbsim3/mason2), so they are not fetched
here.

Real per-preset input reads (including the splice RNA sets: SG-NEx ONT cDNA /
direct-RNA and ENCODE short-read RNA-seq) are downloaded by
``download_preset_data.py`` instead — this script only fetches ground-truth and
feature-flag inputs.

Any other accession can be fetched with --url/--into instead of guessing a path.

Everything is idempotent: existing/extracted outputs are skipped. Steps whose
required tool is missing are skipped with an actionable message rather than
failing the whole run.
"""

import argparse
import gzip
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover - clearer message than a raw traceback
    sys.stderr.write("error: the 'requests' package is required (pip install requests)\n")
    raise

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
RESET = "\033[0m"
NO_COLOR = os.environ.get("NO_COLOR") is not None
CHUNK_SIZE = 8 * 1024 * 1024
SEP = "━" * 72

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = os.environ.get("MM2_PRESET_DATA", str(REPO_ROOT / "test" / "preset_data"))
PAFTOOLS = REPO_ROOT / "misc" / "paftools.js"


def _c(code: str, text: str) -> str:
    return text if NO_COLOR else f"{code}{text}{RESET}"


def human_size(nbytes: int) -> str:
    n = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


# ── Dataset registry ─────────────────────────────────────────────────────────


@dataclass
class Item:
    component: str           # anno | giab
    label: str
    subdir: str
    filename: str            # downloaded artifact (under dest/subdir)
    url: str
    size_hint: str = ""
    decompress_to: str | None = None  # gunzip target (under dest/subdir)


# GENCODE: comprehensive annotation on the primary assembly (EBI mirror is the
# canonical GENCODE FTP). Release pinned for reproducibility; bump deliberately.
GENCODE_RELEASE = "46"
GENCODE_BASE = (
    f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{GENCODE_RELEASE}"
)

# GIAB HG002 (NA24385/son) v4.2.1 small-variant benchmark on GRCh38.
GIAB_BASE = (
    "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/"
    "AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh38"
)

ITEMS: list[Item] = [
    Item(
        component="anno",
        label=f"GENCODE v{GENCODE_RELEASE} GRCh38 comprehensive annotation (GTF)",
        subdir="anno",
        filename=f"gencode.v{GENCODE_RELEASE}.annotation.gtf.gz",
        url=f"{GENCODE_BASE}/gencode.v{GENCODE_RELEASE}.annotation.gtf.gz",
        size_hint="~50 MB",
        decompress_to=f"gencode.v{GENCODE_RELEASE}.annotation.gtf",
    ),
    Item(
        component="giab",
        label="GIAB HG002 v4.2.1 benchmark VCF (GRCh38)",
        subdir="giab",
        filename="HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz",
        url=f"{GIAB_BASE}/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz",
        size_hint="~60 MB",
    ),
    Item(
        component="giab",
        label="GIAB HG002 v4.2.1 high-confidence regions (BED)",
        subdir="giab",
        filename="HG002_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed",
        url=f"{GIAB_BASE}/HG002_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed",
        size_hint="~6 MB",
    ),
    Item(
        component="giab",
        label="GIAB HG002 v4.2.1 benchmark VCF index (tbi)",
        subdir="giab",
        filename="HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi",
        url=f"{GIAB_BASE}/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi",
        size_hint="~2 MB",
    ),
]


# ── Download / post-process helpers ──────────────────────────────────────────


def download_file(url: str, dest: Path, dry_run: bool = False) -> bool:
    if dry_run:
        print(f"    {_c(DIM, 'dry-run:')} {url}")
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            done = 0
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    done += len(chunk)
                    if total and not NO_COLOR:
                        pct = 100 * done / total
                        print(f"\r    {dest.name}: {pct:5.1f}% "
                              f"({human_size(done)}/{human_size(total)})", end="", flush=True)
            if total and not NO_COLOR:
                print()
        tmp.rename(dest)
        print(f"    {_c(GREEN, '✓ downloaded')} ({human_size(dest.stat().st_size)})")
        return True
    except (requests.RequestException, OSError) as exc:
        print(f"    {_c(RED, 'ERROR:')} {exc}")
        tmp.unlink(missing_ok=True)
        return False


def decompress_gz(src: Path, dst: Path) -> bool:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        print(f"    decompressing → {dst.name} ...", end=" ", flush=True)
        with gzip.open(src, "rb") as gz, open(tmp, "wb") as out:
            shutil.copyfileobj(gz, out, CHUNK_SIZE)
        tmp.rename(dst)
        print(_c(GREEN, f"OK ({human_size(dst.stat().st_size)})"))
        return True
    except (OSError, gzip.BadGzipFile) as exc:
        print(_c(RED, f"FAILED: {exc}"))
        tmp.unlink(missing_ok=True)
        return False


def run_gff2bed(gtf: Path, bed12: Path, dry_run: bool) -> bool:
    """GTF → BED12 via `k8 paftools.js gff2bed` (splice junction truth)."""
    if bed12.exists():
        print(f"    {_c(GREEN, '✓ already converted')} ({bed12.name})")
        return True
    if not have("k8"):
        print(f"    {_c(YELLOW, 'skip gff2bed:')} 'k8' not found "
              f"(install k8 to run paftools.js, then re-run --components anno)")
        return True
    if not PAFTOOLS.exists():
        print(f"    {_c(YELLOW, 'skip gff2bed:')} {PAFTOOLS} missing")
        return True
    if dry_run:
        print(f"    {_c(DIM, 'dry-run:')} k8 {PAFTOOLS} gff2bed {gtf.name} > {bed12.name}")
        return True
    tmp = bed12.with_suffix(".tmp")
    try:
        print(f"    gff2bed → {bed12.name} ...", end=" ", flush=True)
        with open(tmp, "wb") as out:
            subprocess.run(["k8", str(PAFTOOLS), "gff2bed", str(gtf)],
                           stdout=out, check=True)
        tmp.rename(bed12)
        print(_c(GREEN, f"OK ({human_size(bed12.stat().st_size)})"))
        return True
    except (subprocess.CalledProcessError, OSError) as exc:
        print(_c(RED, f"FAILED: {exc}"))
        tmp.unlink(missing_ok=True)
        return False


# ── GIAB concordance reads (HG002 HiFi, one chromosome) ───────────────────────

# HG002 PacBio HiFi aligned to GRCh38 (PrecisionFDA Truth Challenge V2). The
# whole-genome BAM is ~73 GB but supports HTTP byte-range, so we stream just one
# chromosome (~1.5 GB) using its remote .bai. The GIAB hap.py concordance
# (scripts/run_giab_concordance.py) re-aligns the extracted reads with each
# minimap2 build.
GIAB_READS_BAM = (
    "https://storage.googleapis.com/deepvariant/pacbio-case-study-testdata/"
    "HG002.pfda_challenge.grch38.phased.bam"
)
SAMTOOLS_IMAGE = "staphb/samtools:latest"


def _local_samtools_has_libcurl() -> bool:
    try:
        out = subprocess.run(["samtools", "--version"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return "libcurl=yes" in out.stdout


def _docker_usable() -> bool:
    return have("docker") and subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def _host_ca_bundle() -> str | None:
    for f in ("/etc/ssl/certs/ca-certificates.crt",
              "/etc/pki/tls/certs/ca-bundle.crt",
              "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem"):
        if Path(f).is_file():
            return f
    return None


def acquire_giab_reads(dest: Path, region: str, dry_run: bool) -> bool:
    """Stream HG002 HiFi reads for one chromosome from the remote PrecisionFDA BAM.

    Produces dest/giab/hg002.hifi.<region>.fastq.gz (the read input for the GIAB
    hap.py concordance). Only the requested region is transferred via the remote
    .bai, not the 73 GB whole-genome BAM. Needs a libcurl-enabled samtools (the
    local binary if built with libcurl, else the staphb/samtools docker image);
    skips with guidance if neither is available.
    """
    giab = dest / "giab"
    giab.mkdir(parents=True, exist_ok=True)
    out_fq = giab / f"hg002.hifi.{region}.fastq.gz"
    print()
    print(f"  {_c(BOLD, '[giab-reads]')} HG002 HiFi {region} reads (for hap.py concordance)")
    if out_fq.exists():
        print(f"    {_c(GREEN, '✓ already prepared')} ({human_size(out_fq.stat().st_size)})")
        return True

    bai = giab / "HG002.pfda_challenge.grch38.phased.bam.bai"
    if not bai.exists():
        print("    fetching remote index (.bai) ...")
        if not download_file(GIAB_READS_BAM + ".bai", bai, dry_run=dry_run):
            return False
    if dry_run:
        print(f"    {_c(DIM, 'dry-run:')} stream {region} from PrecisionFDA HiFi BAM → {out_fq.name}")
        return True

    region_bam = giab / f"hg002.hifi.{region}.bam"
    # 1) Stream the region (needs a libcurl-enabled samtools).
    if _local_samtools_has_libcurl():
        try:
            with open(region_bam, "wb") as fo:
                subprocess.run(["samtools", "view", "-b", "-@", "8", "-X",
                                GIAB_READS_BAM, str(bai), region], stdout=fo, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"    {_c(RED, 'FAILED (stream):')} {exc}")
            return False
    elif _docker_usable():
        ca = _host_ca_bundle()
        if not ca:
            print(f"    {_c(YELLOW, 'skip giab-reads:')} no host CA bundle found for docker-samtools TLS")
            return True
        cmd = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
               "-e", "CURL_CA_BUNDLE=/ca.crt", "-v", f"{ca}:/ca.crt:ro",
               "-v", f"{giab}:{giab}", "-w", str(giab), "--entrypoint", "sh",
               SAMTOOLS_IMAGE, "-c",
               f"samtools view -b -@ 8 -X '{GIAB_READS_BAM}' '{bai}' {region} -o '{region_bam}'"]
        if subprocess.run(cmd).returncode != 0:
            print(f"    {_c(RED, 'FAILED (docker stream)')}")
            return False
    else:
        print(f"    {_c(YELLOW, 'skip giab-reads:')} need a libcurl-enabled samtools "
              f"(local samtools built with libcurl, or the '{SAMTOOLS_IMAGE}' docker image)")
        return True

    # 2) BAM → FASTQ (primary reads only; operates on the local file).
    print(f"    region BAM {human_size(region_bam.stat().st_size)}; extracting reads → {out_fq.name} ...")
    try:
        if have("samtools"):
            p1 = subprocess.Popen(["samtools", "fastq", "-@", "8", "-F", "0x900", str(region_bam)],
                                  stdout=subprocess.PIPE)
            with open(out_fq, "wb") as fo:
                p2 = subprocess.Popen(["gzip"], stdin=p1.stdout, stdout=fo)
                p1.stdout.close()
                p2.communicate()
            p1.wait()
            if p1.returncode or p2.returncode:
                raise subprocess.CalledProcessError(p1.returncode or p2.returncode, "samtools fastq | gzip")
        else:
            cmd = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
                   "-v", f"{giab}:{giab}", "-w", str(giab), "--entrypoint", "sh",
                   SAMTOOLS_IMAGE, "-c",
                   f"samtools fastq -@ 8 -F 0x900 '{region_bam}' | gzip > '{out_fq}'"]
            if subprocess.run(cmd).returncode != 0:
                raise subprocess.CalledProcessError(1, "docker samtools fastq")
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"    {_c(RED, 'FAILED (fastq):')} {exc}")
        return False

    print(f"    {_c(GREEN, '✓ prepared')} {out_fq.name} ({human_size(out_fq.stat().st_size)})")
    return True


def ensure_path(target, *, region: str | None = None, dry_run: bool = False) -> bool:
    """Fetch a known verification artifact to ``target`` if it is missing.

    Matches ``target``'s basename against the dataset registry (GIAB truth
    VCF/BED — the VCF's .tbi index is pulled alongside it — and the GENCODE GTF
    or its decompressed form) or the HG002 HiFi region-reads pattern
    ``hg002.hifi.<region>.fastq.gz``. Returns True if the file exists afterwards
    (or under ``dry_run``), else False. Lets consumer scripts (e.g.
    run_giab_concordance.py, run_accuracy_eval.py) auto-fetch inputs on demand.
    """
    target = Path(target)
    if target.exists():
        return True
    name = target.name

    m = re.fullmatch(r"hg002\.hifi\.(.+)\.fastq\.gz", name)
    if m:
        if target.parent.name != "giab":
            print(f"  {_c(RED, 'auto-download:')} giab-reads must live under a 'giab/' "
                  f"directory; got {target}")
            return False
        if not acquire_giab_reads(target.parent.parent, region or m.group(1), dry_run):
            return False
        return dry_run or target.exists()

    for it in ITEMS:
        if it.filename == name:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not download_file(it.url, target, dry_run=dry_run):
                return False
            tbi = next((x for x in ITEMS if x.filename == name + ".tbi"), None)
            if tbi:
                download_file(tbi.url, target.with_name(tbi.filename), dry_run=dry_run)
            return dry_run or target.exists()
        if it.decompress_to == name:
            target.parent.mkdir(parents=True, exist_ok=True)
            gz = target.with_name(it.filename)
            if not gz.exists() and not download_file(it.url, gz, dry_run=dry_run):
                return False
            return True if dry_run else decompress_gz(gz, target)

    print(f"  {_c(RED, 'auto-download:')} no known verification source for {name}")
    return False


# ── Main ──────────────────────────────────────────────────────

# 'all' deliberately excludes giab-reads (a heavy ~1.5 GB docker/byte-range
# stream); request it explicitly with --components giab-reads.
COMPONENTS = ("anno", "giab", "giab-reads")
ALL_COMPONENTS = ("anno", "giab")


def run(args: argparse.Namespace) -> int:
    dest = Path(args.dest)
    selected = (
        set(ALL_COMPONENTS) if args.components == "all"
        else {c.strip() for c in args.components.split(",")}
    )
    unknown = selected - set(COMPONENTS)
    if unknown:
        print(f"{_c(RED, 'unknown components:')} {', '.join(sorted(unknown))}")
        print(f"  valid: {', '.join(COMPONENTS)}")
        return 2

    print()
    print(_c(DIM, SEP))
    print(f"  {_c(BOLD, 'v2.31 verification data acquisition')}")
    print(f"  Destination: {_c(CYAN, str(dest))}")
    print(f"  Components:  {_c(CYAN, ', '.join(sorted(selected)))}")
    if args.dry_run:
        print(f"  Mode:        {_c(YELLOW, 'DRY RUN')}")
    print(_c(DIM, SEP))

    failed = 0

    # Download-based components (anno, giab)
    dl_items = [it for it in ITEMS if it.component in selected]
    for i, it in enumerate(dl_items, 1):
        target = dest / it.subdir / it.filename
        extracted = dest / it.subdir / it.decompress_to if it.decompress_to else None
        target.parent.mkdir(parents=True, exist_ok=True)

        print()
        print(f"  {_c(BOLD, f'[{it.component}]')} {it.label}")
        print(f"    file: {_c(DIM, it.filename)}  ({it.size_hint})")

        if extracted and extracted.exists():
            print(f"    {_c(GREEN, '✓ already extracted')} ({extracted.name})")
        elif target.exists():
            print(f"    {_c(GREEN, '✓ already downloaded')} ({human_size(target.stat().st_size)})")
            if extracted and not args.dry_run and not decompress_gz(target, extracted):
                failed += 1
        else:
            if download_file(it.url, target, dry_run=args.dry_run):
                if extracted and not args.dry_run and not decompress_gz(target, extracted):
                    failed += 1
            else:
                failed += 1

    # GENCODE GTF → BED12 (after the GTF is in place)
    if "anno" in selected:
        gtf = dest / "anno" / f"gencode.v{GENCODE_RELEASE}.annotation.gtf"
        bed12 = dest / "anno" / "gencode.bed12"
        if gtf.exists() or args.dry_run:
            print()
            print(f"  {_c(BOLD, '[anno]')} GTF → BED12 (paftools.js gff2bed)")
            if not run_gff2bed(gtf, bed12, args.dry_run):
                failed += 1

    # GIAB concordance reads (opt-in; heavy remote stream)
    if "giab-reads" in selected:
        if not acquire_giab_reads(dest, args.giab_region, args.dry_run):
            failed += 1

    print()
    print(_c(DIM, SEP))
    if failed:
        print(f"  {_c(RED, f'{failed} step(s) failed')}")
    else:
        print(f"  {_c(GREEN, 'done')}")
    # Pointer for any additional accession-specific pulls.
    if not args.quiet:
        print()
        print(f"  {_c(BOLD, 'Need a different accession?')} fetch one explicitly, e.g.:")
        print(f"        {_c(DIM, '$')} {sys.argv[0]} --url <FASTQ_URL> "
              f"--into splice-sr/extra_R1.fastq.gz --decompress")
    print(_c(DIM, SEP))
    return 1 if failed else 0


def fetch_url(args: argparse.Namespace) -> int:
    """--url mode: fetch one explicit file into dest/<--into>."""
    dest = Path(args.dest)
    rel = args.into or Path(args.url).name
    target = dest / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetch {_c(CYAN, args.url)}")
    print(f"     → {_c(CYAN, str(target))}")
    if target.exists():
        print(f"    {_c(GREEN, '✓ already present')} ({human_size(target.stat().st_size)})")
        return 0
    ok = download_file(args.url, target, dry_run=args.dry_run)
    if ok and not args.dry_run and target.suffix == ".gz" and args.decompress:
        decompress_gz(target, target.with_suffix(""))
    return 0 if ok else 1


def main() -> None:
    p = argparse.ArgumentParser(
        description="Acquire ground-truth verification datasets (annotation, GIAB).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "components: anno (GENCODE→BED12), giab (HG002 v4.2.1),\n"
            "            giab-reads (HG002 HiFi region reads, opt-in)\n"
            "examples:\n"
            "  %(prog)s --components anno,giab\n"
            "  %(prog)s --components giab-reads --giab-region chr20\n"
            "  %(prog)s --url <FASTQ_URL> --into splice-sr/encode_R1.fastq.gz\n"
            "  %(prog)s --dry-run"
        ),
    )
    p.add_argument("--dest", default=DEFAULT_DEST,
                   help="destination preset_data root (default: %(default)s)")
    p.add_argument("--components", default="all",
                   help="comma-separated: anno,giab,giab-reads or 'all' "
                        "(default: all; 'all' excludes the heavy giab-reads)")
    p.add_argument("--giab-region", default="chr20",
                   help="chromosome to stream for --components giab-reads (default: %(default)s)")
    p.add_argument("--url", default=None,
                   help="fetch a single explicit URL (for SG-NEx/ENCODE accessions)")
    p.add_argument("--into", default=None,
                   help="with --url: destination path relative to --dest")
    p.add_argument("--decompress", action="store_true",
                   help="with --url: gunzip a .gz after download")
    p.add_argument("--dry-run", action="store_true",
                   help="show actions without downloading/generating")
    p.add_argument("--list", action="store_true",
                   help="list registered items and exit")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="suppress the manual-items hint")
    args = p.parse_args()

    if args.list:
        print("components:", ", ".join(COMPONENTS))
        for it in ITEMS:
            print(f"  [{it.component}] {it.subdir}/{it.filename}  ({it.size_hint})")
        sys.exit(0)

    if args.url:
        sys.exit(fetch_url(args))

    sys.exit(run(args))


if __name__ == "__main__":
    main()
