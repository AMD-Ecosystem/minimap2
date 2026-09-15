#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""Download representative real-read datasets for every minimap2 preset.

Provides one canonical input dataset per preset/flag combination so each
``-x`` mode in USAGE.md can be exercised end to end: HPRC/GIAB long reads and
assemblies, plus the splice RNA inputs (SG-NEx ONT cDNA / direct-RNA and ENCODE
short-read RNA-seq). Ground-truth / accuracy inputs (GENCODE annotation, GIAB
benchmark VCF, simulated reads) are *not* here — those live in
``download_verification_data.py``.
"""

import argparse
import gzip
import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import requests
from tqdm import tqdm

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

BASE_URL = "https://human-pangenomics.s3.amazonaws.com"


def _c(code: str, text: str) -> str:
    return text if NO_COLOR else f"{code}{text}{RESET}"


def human_size(nbytes: int) -> str:
    n = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Download:
    preset: str
    label: str
    subdir: str
    filename: str
    size_hint: str
    sha256: str = ""
    s3_path: str = ""
    url_override: str | None = None
    alt_existing: str | None = None
    decompress_to: str | None = None
    bam_to_fastq: str | None = None
    extra_flags: str = ""
    extra_files: list["Download"] = field(default_factory=list)

    @property
    def url(self) -> str:
        if self.url_override:
            return self.url_override
        if not self.s3_path:
            raise ValueError(f"{self.preset}: no s3_path or url_override")
        return f"{BASE_URL}/{self.s3_path}"


NCBI_REF_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz"

# SG-NEx (Singapore Nanopore Expression) public AWS Open Data bucket — long-read
# RNA inputs for the splice presets (no upstream checksums published).
SGNEX_BASE = "https://sg-nex-data.s3.amazonaws.com/data/sequencing_data_ont/fastq"
# ENCODE paired-end polyA+ RNA-seq (76 bp) — short-read RNA preset (splice:sr).
ENCODE_BASE = "https://www.encodeproject.org/files"

DOWNLOADS: list[Download] = [
    Download(
        preset="ref",
        label="GRCh38 no-alt analysis set (reference genome)",
        subdir="",
        filename="GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz",
        url_override=NCBI_REF_URL,
        size_hint="~833 MB compressed, ~3 GB decompressed",
        sha256="fb4243ebb014caf27111f24dd62b7ce42160f28581da6f8fcd6cba5977778d02",
        decompress_to="ref.fa",
    ),
    Download(
        preset="map-hifi",
        label="PacBio HiFi reads (HG02486)",
        subdir="map-hifi",
        filename="HG02486.m64076_200211_192227.dc.q20.fastq.gz",
        s3_path="submissions/3A25CF8A-1F36-42EE-BC9F-D29CECAA2A99--HPRC_DEEPCONSENSUS_v1pt2_2023_12_q20/HG02486/raw_data/PacBio_HiFi/deepconsensus/v1pt2/HG02486.m64076_200211_192227.dc.q20.fastq.gz",
        size_hint="~898 MB",
        sha256="c20a8969f567eabb81cab310f5ca35678d7eab69b80a5e939683db2588750d8c",
        decompress_to="HG02486.m64076_200211_192227.dc.q20.fastq",
    ),
    Download(
        preset="map-pb",
        label="PacBio CLR subreads (HG002, RS II)",
        subdir="map-pb",
        filename="m141224_220915_42177.subreads.fasta.gz",
        url_override="https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/PacBio_MtSinai_NIST/PacBio_fasta/m141224_220915_42177.subreads.fasta.gz",
        size_hint="~370 MB",
        sha256="02d3b919851f98c8d33802220b15af082202a610e95784d5bf2140dc9ceed793",
        decompress_to="m141224_220915_42177.subreads.fasta",
    ),
    Download(
        preset="map-ont",
        label="Oxford Nanopore reads (NA18565)",
        subdir="map-ont",
        filename="10_10_23_R1041_HPRC_NA18565_1_dorado0.6.0_sup4.3.0_5mCG_5hmCG.bam",
        s3_path="working/HPRC/NA18565/raw_data/nanopore/dorado0.6.0_sup4.3.0_5mCG_5hmCG/10_10_23_R1041_HPRC_NA18565_1_dorado0.6.0_sup4.3.0_5mCG_5hmCG.bam",
        size_hint="~973 MB",
        sha256="99058b64581a49071830345bee9f2748b1d610a3aa8d4c7f3684d55d588efc58",
        bam_to_fastq="10_10_23_R1041_HPRC_NA18565_1_dorado0.6.0_sup4.3.0_5mCG_5hmCG.fastq",
    ),
    Download(
        preset="sr",
        label="Illumina Hi-C R1 (HG03098)",
        subdir="sr",
        filename="SE5138_NWM047-3_S1_L002_R1_001.trimmed.fastq.gz",
        s3_path="working/HPRC_PLUS/HG03098/raw_data/hic/SE5138_NWM047-3_S1_L002_R1_001.trimmed.fastq.gz",
        size_hint="~806 MB",
        sha256="a71894c6e3d33cca014a87d64e9a543b4b700768fa538e054c69610eec4825b3",
        decompress_to="SE5138_NWM047-3_S1_L002_R1_001.trimmed.fastq",
        extra_files=[
            Download(
                preset="sr",
                label="Illumina Hi-C R2 (HG03098)",
                subdir="sr",
                filename="SE5138_NWM047-3_S1_L002_R2_001.trimmed.fastq.gz",
                s3_path="working/HPRC_PLUS/HG03098/raw_data/hic/SE5138_NWM047-3_S1_L002_R2_001.trimmed.fastq.gz",
                size_hint="~920 MB",
                sha256="4a2b249df4b543a106b2b4420a8c9d291aa08fb53791ddbca9037ef87929e8b4",
                decompress_to="SE5138_NWM047-3_S1_L002_R2_001.trimmed.fastq",
            ),
        ],
    ),
    Download(
        preset="splice:hq",
        label="PacBio IsoSeq/Kinnex FLNC (HG00097)",
        subdir="splice",
        filename="HG00097.lymph.m84203_240914_042802_s4-m84203_240914_022843_s3.flnc.bam",
        s3_path="working/HPRC/HG00097/raw_data/PacBio_Kinnex/HG00097.lymph.m84203_240914_042802_s4-m84203_240914_022843_s3.flnc.bam",
        size_hint="~15 GB",
        sha256="caa72115f286c5154d79c81de92f9edf75984b50a78dadd50d07f540fd605748",
        extra_flags="-uf",
        bam_to_fastq="HG00097.lymph.m84203_240914_042802_s4-m84203_240914_022843_s3.flnc.fastq",
    ),
    Download(
        preset="asm5",
        label="Assembly FASTA (HG00438 paternal)",
        subdir="asm",
        filename="HG00438.paternal.f1_assembly_v2_genbank.fa.gz",
        s3_path="working/HPRC/HG00438/assemblies/year1_f1_assembly_v2_genbank/HG00438.paternal.f1_assembly_v2_genbank.fa.gz",
        size_hint="~834 MB",
        sha256="a5840157a6995c5fbc08698e52b614fbdf6f57a7245db944edb328a0b3e23ff8",
        alt_existing="HG00438.paternal.f1_assembly_v2_genbank.fa",
        decompress_to="HG00438.paternal.f1_assembly_v2_genbank.fa",
    ),
    Download(
        preset="splice",
        label="SG-NEx ONT cDNA (A549 directcDNA rep3 run1) → plain `splice`",
        subdir="splice-rna",
        filename="SGNex_A549_directcDNA_replicate3_run1.fastq.gz",
        url_override=f"{SGNEX_BASE}/SGNex_A549_directcDNA_replicate3_run1/"
                     "SGNex_A549_directcDNA_replicate3_run1.fastq.gz",
        size_hint="~102 MB",
        sha256="ff773a56038ffd62777b7a7c4f289f3c129ac58a57811ee682c8598f97259a7a",
        decompress_to="SGNex_A549_directcDNA_replicate3_run1.fastq",
    ),
    Download(
        preset="splice",
        label="SG-NEx ONT direct-RNA (Hct116 rep1 run1) → `splice -uf -k14`",
        subdir="splice-rna",
        filename="SGNex_Hct116_directRNA_replicate1_run1.fastq.gz",
        url_override=f"{SGNEX_BASE}/SGNex_Hct116_directRNA_replicate1_run1/"
                     "SGNex_Hct116_directRNA_replicate1_run1.fastq.gz",
        size_hint="~97 MB",
        sha256="956556902a1ef5433ac32e6e60857324aad8a759d62d9d439c76409862ed1f4a",
        extra_flags="-uf -k14",
        decompress_to="SGNex_Hct116_directRNA_replicate1_run1.fastq",
    ),
    Download(
        preset="splice:sr",
        label="ENCODE polyA RNA-seq R1 (ENCFF336WMA, 76 bp) → `splice:sr`",
        subdir="splice-sr",
        filename="ENCFF336WMA.fastq.gz",
        url_override=f"{ENCODE_BASE}/ENCFF336WMA/@@download/ENCFF336WMA.fastq.gz",
        size_hint="~272 MB",
        decompress_to="ENCFF336WMA.fastq",
        extra_files=[
            Download(
                preset="splice:sr",
                label="ENCODE polyA RNA-seq R2 (ENCFF732TMT, 76 bp)",
                subdir="splice-sr",
                filename="ENCFF732TMT.fastq.gz",
                url_override=f"{ENCODE_BASE}/ENCFF732TMT/@@download/ENCFF732TMT.fastq.gz",
                size_hint="~276 MB",
                decompress_to="ENCFF732TMT.fastq",
            ),
        ],
    ),
]


def download_file(url: str, dest: Path, expected_sha256: str,
                  dry_run: bool = False) -> bool:
    if dry_run:
        print(f"    {_c(DIM, 'dry-run:')} {url}")
        return True

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        resp = requests.get(url, stream=True, timeout=(15, 60))
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        h = hashlib.sha256()

        with open(tmp, "wb") as f, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"    {dest.name}",
            bar_format="    {l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
            disable=NO_COLOR,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                h.update(chunk)
                bar.update(len(chunk))

        actual = h.hexdigest()
        if expected_sha256 and actual != expected_sha256:
            print(f"    {_c(RED, '✗ SHA256 mismatch!')}")
            print(f"      expected: {expected_sha256}")
            print(f"      got:      {actual}")
            tmp.unlink(missing_ok=True)
            return False
        if expected_sha256:
            print(f"    {_c(GREEN, '✓ SHA256 verified')}")
        else:
            print(f"    {_c(YELLOW, '⚠ no SHA256 on record — skipped verification')}")

        tmp.rename(dest)
        return True

    except (requests.RequestException, OSError) as exc:
        print(f"    {_c(RED, 'ERROR:')} {exc}")
        tmp.unlink(missing_ok=True)
        return False


def decompress_gz(src: Path, dst: Path) -> bool:
    tmp = dst.with_suffix(".tmp")
    try:
        print(f"    Decompressing to {dst.name}...", end=" ", flush=True)
        with gzip.open(src, "rb") as gz, open(tmp, "wb") as out:
            shutil.copyfileobj(gz, out)
        tmp.rename(dst)
        print(_c(GREEN, f"OK ({human_size(dst.stat().st_size)})"))
        return True
    except (OSError, gzip.BadGzipFile) as exc:
        print(_c(RED, f"FAILED: {exc}"))
        tmp.unlink(missing_ok=True)
        return False


def convert_bam_to_fastq(src: Path, dst: Path) -> bool:
    tmp = dst.with_suffix(".tmp")
    try:
        print(f"    Converting BAM → FASTQ ({dst.name})...", end=" ", flush=True)
        with open(tmp, "wb") as out:
            subprocess.run(
                ["samtools", "fastq", "-T", "*", str(src)],
                stdout=out, stderr=subprocess.DEVNULL, check=True,
            )
        tmp.rename(dst)
        print(_c(GREEN, f"OK ({human_size(dst.stat().st_size)})"))
        return True
    except FileNotFoundError:
        print(_c(RED, "FAILED: samtools not found"))
        tmp.unlink(missing_ok=True)
        return False
    except subprocess.CalledProcessError as exc:
        print(_c(RED, f"FAILED: samtools exit code {exc.returncode}"))
        tmp.unlink(missing_ok=True)
        return False


# ── Main ─────────────────────────────────────────────────────────────────────


def ensure_path(target, *, dry_run: bool = False) -> bool:
    """Fetch a known preset artifact to ``target`` if it is missing.

    Matches ``target``'s basename against the download registry by its
    post-processed name (decompressed / BAM→FASTQ) or its raw filename, then
    downloads and post-processes so the final file lands at ``target``. Returns
    True if the file exists afterwards (or under ``dry_run``), else False. Lets
    consumer scripts auto-fetch a missing reference/reads input on demand.
    """
    target = Path(target)
    if target.exists():
        return True
    name = target.name
    for dl in DOWNLOADS:
        final = dl.bam_to_fastq or dl.decompress_to or dl.filename
        if name not in (final, dl.filename):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = target.with_name(dl.filename)
        if not raw.exists() and not download_file(dl.url, raw, dl.sha256, dry_run=dry_run):
            return False
        if dry_run:
            return True
        if dl.decompress_to and name == dl.decompress_to:
            return decompress_gz(raw, target)
        if dl.bam_to_fastq and name == dl.bam_to_fastq:
            return convert_bam_to_fastq(raw, target)
        return target.exists()
    print(f"  {_c(RED, 'auto-download:')} no known preset source for {name}")
    return False


def run(args: argparse.Namespace) -> int:
    dest = Path(args.dest)
    presets = {p.strip() for p in args.presets.split(",")} if args.presets != "all" else None

    all_downloads: list[Download] = []
    for dl in DOWNLOADS:
        if presets and dl.preset not in presets:
            continue
        all_downloads.append(dl)
        all_downloads.extend(dl.extra_files)

    if not all_downloads:
        print(f"{_c(YELLOW, 'Nothing to download.')} Check --presets filter.")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    for sd in {dl.subdir for dl in all_downloads}:
        if sd:
            (dest / sd).mkdir(parents=True, exist_ok=True)

    print()
    print(_c(DIM, SEP))
    print(f"  {_c(BOLD, 'Minimap2 Preset Data Downloader')}")
    print(f"  Destination: {_c(CYAN, str(dest))}")
    print(f"  Presets:     {_c(CYAN, ', '.join(sorted(presets)) if presets else 'all')}")
    if args.dry_run:
        print(f"  Mode:        {_c(YELLOW, 'DRY RUN')}")
    if args.verify:
        print(f"  Verify:      {_c(CYAN, 'SHA256 check on existing files')}")
    print(_c(DIM, SEP))

    downloaded = skipped = failed = hash_ok = hash_fail = 0
    total = len(all_downloads)

    for i, dl in enumerate(all_downloads, 1):
        target = dest / dl.subdir / dl.filename
        decompressed = dest / dl.subdir / dl.decompress_to if dl.decompress_to else None
        fastq_out = dest / dl.subdir / dl.bam_to_fastq if dl.bam_to_fastq else None
        alt_path = dest / dl.alt_existing if dl.alt_existing else None
        extracted = decompressed or fastq_out  # the post-processed output, if any

        print()
        print(f"  {_c(BOLD, f'[{i}/{total}]')} {_c(CYAN, dl.preset)}: {dl.label}")
        print(f"    File: {_c(DIM, dl.filename)}")
        print(f"    Size: {dl.size_hint}")

        # Already fully extracted?
        if extracted and extracted.exists():
            print(f"    {_c(GREEN, '✓ Already extracted')} ({extracted.name}, {human_size(extracted.stat().st_size)})")
            skipped += 1
            continue
        if alt_path and alt_path.exists():
            print(f"    {_c(GREEN, '✓ Already exists')} (alt: {dl.alt_existing}, {human_size(alt_path.stat().st_size)})")
            skipped += 1
            continue

        # Raw download exists but needs post-processing?
        if target.exists():
            if extracted:
                print(f"    {_c(GREEN, '✓ Already downloaded')} ({human_size(target.stat().st_size)})")
                if args.verify and dl.sha256:
                    print("    Verifying SHA256...", end=" ", flush=True)
                    actual = sha256_file(target)
                    if actual == dl.sha256:
                        print(_c(GREEN, "OK"))
                        hash_ok += 1
                    else:
                        print(_c(RED, "MISMATCH"))
                        print(f"      expected: {dl.sha256}")
                        print(f"      got:      {actual}")
                        hash_fail += 1
                ok = True
                if not args.dry_run:
                    if decompressed:
                        ok = decompress_gz(target, decompressed)
                    elif fastq_out:
                        ok = convert_bam_to_fastq(target, fastq_out)
                if ok:
                    skipped += 1
                else:
                    failed += 1
                continue
            else:
                print(f"    {_c(GREEN, '✓ Already exists')} ({human_size(target.stat().st_size)})")
                if args.verify and dl.sha256:
                    print("    Verifying SHA256...", end=" ", flush=True)
                    actual = sha256_file(target)
                    if actual == dl.sha256:
                        print(_c(GREEN, "OK"))
                        hash_ok += 1
                    else:
                        print(_c(RED, "MISMATCH"))
                        print(f"      expected: {dl.sha256}")
                        print(f"      got:      {actual}")
                        hash_fail += 1
                skipped += 1
                continue

        # Download
        if download_file(dl.url, target, dl.sha256, dry_run=args.dry_run):
            if not args.dry_run:
                print(f"    {_c(GREEN, '✓ Downloaded')} ({human_size(target.stat().st_size)})")
                # Post-process
                if decompressed:
                    if not decompress_gz(target, decompressed):
                        failed += 1
                        continue
                elif fastq_out:
                    if not convert_bam_to_fastq(target, fastq_out):
                        failed += 1
                        continue
            downloaded += 1
        else:
            failed += 1

    print()
    print(_c(DIM, SEP))
    print(f"  {_c(BOLD, 'Summary')}")
    print(f"    Downloaded: {_c(GREEN, str(downloaded))}")
    print(f"    Skipped:    {_c(YELLOW, str(skipped))}")
    if failed:
        print(f"    Failed:     {_c(RED, str(failed))}")
    if args.verify:
        print(f"    Hash OK:    {_c(GREEN, str(hash_ok))}")
        if hash_fail:
            print(f"    Hash FAIL:  {_c(RED, str(hash_fail))}")
    print(_c(DIM, SEP))

    # Example commands
    if not args.quiet:
        examples = [
            dl for dl in DOWNLOADS
            if dl.preset != "ref" and (not presets or dl.preset in presets)
        ]
        if examples:
            ref = dest / "ref.fa"
            print()
            print(f"  {_c(BOLD, 'Example commands:')}")
            for dl in examples:
                def _fname(d: Download) -> str:
                    name = d.bam_to_fastq or d.decompress_to or d.filename
                    return str(dest / d.subdir / name)
                files = " ".join(_fname(d) for d in [dl] + dl.extra_files)
                extra = f" {dl.extra_flags}" if dl.extra_flags else ""
                print(f"    {_c(DIM, '$')} minimap2 -ax {dl.preset}{extra} {ref} {files}")
            print()

    return 1 if (failed or hash_fail) else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download representative HPRC datasets for minimap2 presets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "presets: map-hifi, map-pb, map-ont, sr, splice, splice:hq, splice:sr, asm5\n"
            "example: %(prog)s --presets map-hifi,map-pb --dest /data/input"
        ),
    )
    parser.add_argument(
        "--dest",
        default=DEFAULT_DEST,
        help="destination directory (default: %(default)s)",
    )
    parser.add_argument(
        "--presets",
        default="all",
        help="comma-separated preset names to download, or 'all' (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify SHA256 checksums of existing files",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="suppress example commands in output",
    )

    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
