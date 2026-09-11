#!/usr/bin/env python
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import sys
import getopt
import mappy as mp

def main(argv):
	long_opts = ["gpu-chain", "gpu-align", "gpu-cfg=", "gpu-devices=", "batch"]
	opts, args = getopt.getopt(argv[1:], "x:n:m:k:w:r:t:cdM", long_opts)
	if len(args) < 2:
		print("Usage: minimap2.py [options] <ref.fa>|<ref.mmi> <query.fq>")
		print("Options:")
		print("  -x STR          preset: sr, map-pb, map-ont, asm5, asm10 or splice")
		print("  -n INT          mininum number of minimizers")
		print("  -m INT          mininum chaining score")
		print("  -k INT          k-mer length")
		print("  -w INT          minimizer window length")
		print("  -r INT          band width")
		print("  -t INT          number of threads (for GPU worker pools)")
		print("  -c              output the cs tag")
		print("  -d              output the ds tag")
		print("  -M              output the MD tag")
		print("  --gpu-chain     run chaining on the GPU")
		print("  --gpu-align     run alignment on the GPU")
		print("  --gpu-cfg STR   GPU chaining config JSON (default: auto-detect)")
		print("  --gpu-devices STR  comma-separated GPU device ids (e.g. 0,1)")
		print("  --batch         map all reads in one batched call (GPU throughput)")
		sys.exit(1)

	preset = min_cnt = min_chain_score = k = w = bw = None
	n_threads = 3
	out_cs = out_ds = out_MD = False
	gpu_chain = gpu_align = use_batch = False
	gpu_cfg = gpu_devices = None
	for opt, arg in opts:
		if opt == '-x': preset = arg
		elif opt == '-n': min_cnt = int(arg)
		elif opt == '-m': min_chain_score = int(arg)
		elif opt == '-r': bw = int(arg)
		elif opt == '-k': k = int(arg)
		elif opt == '-w': w = int(arg)
		elif opt == '-t': n_threads = int(arg)
		elif opt == '-c': out_cs = True
		elif opt == '-d': out_ds = True
		elif opt == '-M': out_MD = True
		elif opt == '--gpu-chain': gpu_chain = True
		elif opt == '--gpu-align': gpu_align = True
		elif opt == '--gpu-cfg': gpu_cfg = arg
		elif opt == '--gpu-devices': gpu_devices = [int(x) for x in arg.split(',') if x != '']
		elif opt == '--batch': use_batch = True

	a = mp.Aligner(args[0], preset=preset, min_cnt=min_cnt, min_chain_score=min_chain_score,
		k=k, w=w, bw=bw, n_threads=n_threads,
		gpu_chain=gpu_chain, gpu_align=gpu_align, gpu_cfg=gpu_cfg, gpu_devices=gpu_devices)
	if not a: raise Exception("ERROR: failed to load/build index file '{}'".format(args[0]))

	if use_batch:
		names, seqs = [], []
		for name, seq, qual in mp.fastx_read(args[1]):
			names.append(name); seqs.append(seq)
		results = a.map_batch(seqs, names=names, cs=out_cs, ds=out_ds, MD=out_MD)
		for name, seq, hits in zip(names, seqs, results):
			for h in hits:
				print('{}\t{}\t{}'.format(name, len(seq), h))
	else:
		for name, seq, qual in mp.fastx_read(args[1]): # read one sequence
			for h in a.map(seq, cs=out_cs, ds=out_ds, MD=out_MD): # traverse hits
				print('{}\t{}\t{}'.format(name, len(seq), h))

if __name__ == "__main__":
	main(sys.argv)
