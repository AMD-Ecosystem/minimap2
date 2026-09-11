# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""Tests for mappy GPU mapping (per-read + batch) and ThreadPoolExecutor.

These exercise the AMD fork additions on top of upstream mappy:
  * backward-compatible CPU ``Aligner.map()``
  * ``Aligner.map_batch()``
  * GPU modes (gpu_chain / gpu_align / gpu_both), skipped when no GPU
  * concurrent mapping with multiple Aligners via ThreadPoolExecutor

Run from the repository root::

    pytest python/tests/test_mappy_gpu.py

By default the test imports ``mappy`` from the environment. To point at a
freshly built module, set ``MAPPY_SO`` to the built ``.so`` path.
"""
import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
REF = os.path.join(REPO_ROOT, "test", "MT-human.fa")
QRY = os.path.join(REPO_ROOT, "test", "MT-orang.fa")
GPU_CFG = os.path.join(REPO_ROOT, "configs", "gpu_config.json")


def _load_mappy():
    so = os.environ.get("MAPPY_SO")
    if so:
        spec = importlib.util.spec_from_file_location("mappy", so)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["mappy"] = mod
        spec.loader.exec_module(mod)
        return mod
    return __import__("mappy")


mappy = _load_mappy()


@pytest.fixture(scope="module")
def queries():
    return [(name, seq) for name, seq, _qual in mappy.fastx_read(QRY)]


def _hit_keys(hits):
    return [(h.ctg, h.r_st, h.r_en, h.strand, h.mapq, h.cigar_str) for h in hits]


def _gpu_available(**kwargs):
    try:
        a = mappy.Aligner(REF, preset="map-ont", **kwargs)
        return bool(a)
    except Exception:
        return False


GPU_MODES = {
    "gpu_chain": dict(gpu_chain=True, gpu_cfg=GPU_CFG),
    "gpu_align": dict(gpu_align=True),
    "gpu_both": dict(gpu_chain=True, gpu_align=True, gpu_cfg=GPU_CFG),
}
# All four mapping modes, CPU first (always runs) then the three GPU modes.
ALL_MODES = {"cpu": dict(), **GPU_MODES}
HAVE_GPU = _gpu_available(gpu_chain=True, gpu_cfg=GPU_CFG)


def _requires_gpu(mode):
    return mode != "cpu" and not HAVE_GPU



def test_cpu_map_backward_compat(queries):
    a = mappy.Aligner(REF, preset="map-ont")
    assert a
    total = 0
    for _name, seq in queries:
        for hit in a.map(seq):
            total += 1
            assert hit.ctg == "MT_human"
            assert hit.r_en > hit.r_st
    assert total >= 1


def test_map_batch_matches_map(queries):
    a = mappy.Aligner(REF, preset="map-ont")
    per_read = [_hit_keys(a.map(seq)) for _name, seq in queries]
    batch = a.map_batch([s for _n, s in queries], names=[n for n, _s in queries])
    assert [_hit_keys(h) for h in batch] == per_read


def test_context_reuse(queries):
    a = mappy.Aligner(REF, preset="map-ont")
    first = [_hit_keys(h) for h in a.map_batch([s for _n, s in queries])]
    second = [_hit_keys(h) for h in a.map_batch([s for _n, s in queries])]
    assert first == second


@pytest.fixture(scope="module")
def cpu_baseline(queries):
    a = mappy.Aligner(REF, preset="map-ont")
    return [_hit_keys(a.map(seq)) for _name, seq in queries]


@pytest.mark.parametrize("mode", list(ALL_MODES))
def test_modes_match_cpu(queries, cpu_baseline, mode):
    """For every mode (cpu, gpu-chain, gpu-align, gpu-both), both the per-read
    map() and the batched map_batch() must match the CPU baseline."""
    if _requires_gpu(mode):
        pytest.skip("no GPU available")
    a = mappy.Aligner(REF, preset="map-ont", **ALL_MODES[mode])

    map_keys = [_hit_keys(a.map(seq)) for _name, seq in queries]
    batch_keys = [_hit_keys(h) for h in a.map_batch([s for _n, s in queries], names=[n for n, _s in queries])]

    assert map_keys == cpu_baseline
    assert batch_keys == cpu_baseline



@pytest.mark.skipif(not HAVE_GPU, reason="no GPU available")
def test_threadpool_concurrent_aligners(queries):
    from concurrent.futures import ThreadPoolExecutor

    seqs = [s for _n, s in queries]

    def worker(_i):
        a = mappy.Aligner(REF, preset="map-ont", gpu_chain=True, gpu_align=True, gpu_cfg=GPU_CFG)
        res = a.map_batch(seqs)
        return sum(len(h) for h in res)

    expected = sum(len(list(mappy.Aligner(REF, preset="map-ont").map(s))) for s in seqs)
    with ThreadPoolExecutor(max_workers=4) as pool:
        counts = list(pool.map(worker, range(4)))
    assert counts == [expected] * 4
