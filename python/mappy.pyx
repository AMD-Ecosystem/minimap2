# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
from libc.stdint cimport uint8_t, int8_t
from libc.stdlib cimport free, malloc
cimport cmappy
import sys

def version():
	"""Return the PEP 440 version string (e.g. '2.24.0', '2.25.0a1')."""
	cdef const cmappy.mm_version_t *vi = cmappy.mm_version_info()
	base = f"{vi.major}.{vi.minor}.{vi.patch}"
	pre = vi.prerelease
	if isinstance(pre, bytes):
		pre = pre.decode('utf-8')
	if pre:
		import re as _re
		m = _re.match(r'^(alpha|beta|rc)\.?(\d*)$', pre)
		if m:
			_pep = {'alpha': 'a', 'beta': 'b', 'rc': 'rc'}[m.group(1)]
			base += f"{_pep}{m.group(2) or '0'}"
	return base

def git_version():
	"""Return the full SemVer version string with git metadata (e.g. '2.24.0+g34cc1a3')."""
	cdef const char *v = cmappy.mm_version()
	return v.decode('utf-8') if isinstance(v, bytes) else v

def version_info():
	"""Return a dict with detailed version information."""
	cdef const cmappy.mm_version_t *vi = cmappy.mm_version_info()
	return {
		'major': vi.major,
		'minor': vi.minor,
		'patch': vi.patch,
		'full': vi.full.decode('utf-8') if isinstance(vi.full, bytes) else vi.full,
		'prerelease': vi.prerelease.decode('utf-8') if isinstance(vi.prerelease, bytes) else vi.prerelease,
		'git_hash': vi.git_hash.decode('utf-8') if isinstance(vi.git_hash, bytes) else vi.git_hash,
		'git_short': vi.git_short.decode('utf-8') if isinstance(vi.git_short, bytes) else vi.git_short,
		'git_dirty': bool(vi.git_dirty),
	}

__version__ = version()
__git_version__ = git_version()

cmappy.mm_reset_timer()

cdef class Alignment:
	cdef int _ctg_len, _r_st, _r_en
	cdef int _q_st, _q_en
	cdef int _NM, _mlen, _blen
	cdef int8_t _strand, _trans_strand
	cdef uint8_t _mapq, _is_primary
	cdef int _seg_id
	cdef _ctg, _cigar, _cs, _ds, _MD # these are python objects

	def __cinit__(self, ctg, cl, cs, ce, strand, qs, qe, mapq, cigar, is_primary, mlen, blen, NM, trans_strand, seg_id, cs_str, ds_str, MD_str):
		self._ctg = ctg if isinstance(ctg, str) else ctg.decode()
		self._ctg_len, self._r_st, self._r_en = cl, cs, ce
		self._strand, self._q_st, self._q_en = strand, qs, qe
		self._NM, self._mlen, self._blen = NM, mlen, blen
		self._mapq = mapq
		self._cigar = cigar
		self._is_primary = is_primary
		self._trans_strand = trans_strand
		self._seg_id = seg_id
		self._cs = cs_str
		self._ds = ds_str
		self._MD = MD_str

	@property
	def ctg(self): return self._ctg

	@property
	def ctg_len(self): return self._ctg_len

	@property
	def r_st(self): return self._r_st

	@property
	def r_en(self): return self._r_en

	@property
	def strand(self): return self._strand

	@property
	def trans_strand(self): return self._trans_strand

	@property
	def blen(self): return self._blen

	@property
	def mlen(self): return self._mlen

	@property
	def NM(self): return self._NM

	@property
	def is_primary(self): return (self._is_primary != 0)

	@property
	def q_st(self): return self._q_st

	@property
	def q_en(self): return self._q_en

	@property
	def mapq(self): return self._mapq

	@property
	def cigar(self): return self._cigar

	@property
	def read_num(self): return self._seg_id + 1

	@property
	def cs(self): return self._cs

	@property
	def ds(self): return self._ds

	@property
	def MD(self): return self._MD

	@property
	def cigar_str(self):
		return "".join(map(lambda x: str(x[0]) + 'MIDNSHP=XB'[x[1]], self._cigar))

	def __str__(self):
		if self._strand > 0: strand = '+'
		elif self._strand < 0: strand = '-'
		else: strand = '?'
		if self._is_primary != 0: tp = 'tp:A:P'
		else: tp = 'tp:A:S'
		if self._trans_strand > 0: ts = 'ts:A:+'
		elif self._trans_strand < 0: ts = 'ts:A:-'
		else: ts = 'ts:A:.'
		a = [str(self._q_st), str(self._q_en), strand, self._ctg, str(self._ctg_len), str(self._r_st), str(self._r_en),
			str(self._mlen), str(self._blen), str(self._mapq), tp, ts, "cg:Z:" + self.cigar_str]
		if self._cs != "": a.append("cs:Z:" + self._cs)
		if self._ds != "": a.append("ds:Z:" + self._ds)
		if self._MD != "": a.append("MD:Z:" + self._MD)
		return "\t".join(a)

cdef class ThreadBuffer:
	cdef cmappy.mm_tbuf_t *_b

	def __cinit__(self):
		self._b = cmappy.mm_tbuf_init()

	def __dealloc__(self):
		cmappy.mm_tbuf_destroy(self._b)

cdef class Aligner:
	cdef cmappy.mm_idx_t *_idx
	cdef cmappy.mm_idxopt_t idx_opt
	cdef cmappy.mm_mapopt_t map_opt
	cdef cmappy.mm_logger_t *_logger
	cdef cmappy.mm_mapctx_t *_ctx
	cdef int *_gpu_dev_ids
	cdef int _n_threads
	cdef object _gpu_cfg_bytes

	def __cinit__(self, fn_idx_in=None, preset=None, k=None, w=None, min_cnt=None, min_chain_score=None, min_dp_score=None, bw=None, bw_long=None, best_n=None, n_threads=3, fn_idx_out=None, max_frag_len=None, extra_flags=None, seq=None, scoring=None, sc_ambi=None, max_chain_skip=None, gpu_chain=False, gpu_align=False, gpu_cfg=None, gpu_chain_workers=None, gpu_devices=None, gpu_flush_threshold=None, gpu_batch_max_align=None, gpu_batch_max_mem=None, gpu_accum_pool_size=None, gpu_batch_max_mem_cap=None):
		self._idx = NULL
		self._logger = NULL
		self._ctx = NULL
		self._gpu_dev_ids = NULL
		self._gpu_cfg_bytes = None
		self._n_threads = n_threads if n_threads and n_threads > 0 else 1
		cmappy.mm_set_opt(NULL, &self.idx_opt, &self.map_opt) # set the default options
		if preset is not None:
			cmappy.mm_set_opt(str.encode(preset), &self.idx_opt, &self.map_opt) # apply preset
		self.map_opt.flag |= 4 # always perform alignment
		self.idx_opt.batch_size = 0x7fffffffffffffffL # always build a uni-part index
		if k is not None: self.idx_opt.k = k
		if w is not None: self.idx_opt.w = w
		if min_cnt is not None: self.map_opt.min_cnt = min_cnt
		if min_chain_score is not None: self.map_opt.min_chain_score = min_chain_score
		if min_dp_score is not None: self.map_opt.min_dp_max = min_dp_score
		if bw is not None: self.map_opt.bw = bw
		if bw_long is not None: self.map_opt.bw_long = bw_long
		if best_n is not None: self.map_opt.best_n = best_n
		if max_frag_len is not None: self.map_opt.max_frag_len = max_frag_len
		if extra_flags is not None: self.map_opt.flag |= extra_flags
		if scoring is not None and len(scoring) >= 4:
			self.map_opt.a, self.map_opt.b = scoring[0], scoring[1]
			self.map_opt.q, self.map_opt.e = scoring[2], scoring[3]
			self.map_opt.q2, self.map_opt.e2 = self.map_opt.q, self.map_opt.e
			if len(scoring) >= 6:
				self.map_opt.q2, self.map_opt.e2 = scoring[4], scoring[5]
				if len(scoring) >= 7:
					self.map_opt.sc_ambi = scoring[6]
		if sc_ambi is not None: self.map_opt.sc_ambi = sc_ambi
		if max_chain_skip is not None: self.map_opt.max_chain_skip = max_chain_skip

		# GPU options (additive; defaults reproduce the CPU behavior)
		cdef int ndev, di
		if gpu_chain: self.map_opt.flag |= cmappy.MM_F_GPU_CHAIN
		if gpu_align: self.map_opt.flag |= cmappy.MM_F_GPU_ALIGN
		if gpu_cfg is not None:
			self._gpu_cfg_bytes = gpu_cfg if isinstance(gpu_cfg, bytes) else str(gpu_cfg).encode()
			self.map_opt.gpu_config_file = self._gpu_cfg_bytes
		if gpu_chain_workers is not None: self.map_opt.gpu_chain_workers = gpu_chain_workers
		if gpu_flush_threshold is not None: self.map_opt.gpu_flush_threshold = gpu_flush_threshold
		if gpu_batch_max_align is not None: self.map_opt.gpu_batch_max_align = gpu_batch_max_align
		if gpu_batch_max_mem is not None: self.map_opt.gpu_batch_max_mem = gpu_batch_max_mem
		if gpu_accum_pool_size is not None: self.map_opt.gpu_accum_pool_size = gpu_accum_pool_size
		if gpu_batch_max_mem_cap is not None: self.map_opt.gpu_batch_max_mem_cap = gpu_batch_max_mem_cap
		if gpu_devices is not None:
			ndev = len(gpu_devices)
			if ndev > 0:
				self._gpu_dev_ids = <int*>malloc(ndev * sizeof(int))
				for di in range(ndev):
					self._gpu_dev_ids[di] = int(gpu_devices[di])
				self.map_opt.gpu_device_ids = self._gpu_dev_ids
				self.map_opt.gpu_num_devices = ndev
		elif gpu_chain or gpu_align:
			# Default to single-GPU device 0, mirroring the CLI default.
			self._gpu_dev_ids = <int*>malloc(sizeof(int))
			self._gpu_dev_ids[0] = 0
			self.map_opt.gpu_device_ids = self._gpu_dev_ids
			self.map_opt.gpu_num_devices = 1

		cdef cmappy.mm_idx_reader_t *r;

		if seq is None:
			if fn_idx_out is None:
				r = cmappy.mm_idx_reader_open(str.encode(fn_idx_in), &self.idx_opt, NULL)
			else:
				r = cmappy.mm_idx_reader_open(str.encode(fn_idx_in), &self.idx_opt, str.encode(fn_idx_out))
			if r is not NULL:
				self._idx = cmappy.mm_idx_reader_read(r, n_threads) # NB: ONLY read the first part
				cmappy.mm_idx_reader_close(r)
				cmappy.mm_mapopt_update(&self.map_opt, self._idx)
				cmappy.mm_idx_index_name(self._idx)
		else:
			self._idx = cmappy.mappy_idx_seq(self.idx_opt.w, self.idx_opt.k, self.idx_opt.flag&1, self.idx_opt.bucket_bits, str.encode(seq), len(seq))
			cmappy.mm_mapopt_update(&self.map_opt, self._idx)
			self.map_opt.mid_occ = 1000 # don't filter high-occ seeds

		# One shared mapping context per Aligner. Holds the core context for
		# every run and, when GPU flags are set, the GPU chain/align worker
		# pools and device list. The logger is borrowed by the context.
		self._logger = cmappy.mm_logger_create(0, NULL)
		self._ctx = cmappy.mm_mapctx_create(self._logger, &self.map_opt, self._n_threads)
		if self._ctx == NULL and (gpu_chain or gpu_align):
			# GPU sub-context creation failed (e.g. no usable device). Fall back
			# to a CPU-only context so mapping still works, and tell the caller.
			import warnings
			warnings.warn("GPU mapping context could not be created; falling back to CPU", RuntimeWarning)
			self.map_opt.flag &= ~(cmappy.MM_F_GPU_CHAIN | cmappy.MM_F_GPU_ALIGN)
			self._ctx = cmappy.mm_mapctx_create(self._logger, &self.map_opt, self._n_threads)

	def __dealloc__(self):
		if self._ctx is not NULL:
			cmappy.mm_mapctx_destroy(self._ctx)
			self._ctx = NULL
		if self._logger is not NULL:
			cmappy.mm_logger_destroy(self._logger)
			self._logger = NULL
		if self._gpu_dev_ids is not NULL:
			free(self._gpu_dev_ids)
			self._gpu_dev_ids = NULL
		if self._idx is not NULL:
			cmappy.mm_idx_destroy(self._idx)
			self._idx = NULL

	def __bool__(self):
		return (self._idx != NULL)

	def map(self, seq, seq2=None, name=None, buf=None, cs=False, ds=False, MD=False, max_frag_len=None, extra_flags=None):
		cdef cmappy.mm_reg1_t *regs
		cdef cmappy.mm_hitpy_t h
		cdef ThreadBuffer b
		cdef int n_regs
		cdef char *cs_str = NULL
		cdef int l_cs_str, m_cs_str = 0
		cdef void *km
		cdef cmappy.mm_mapopt_t map_opt

		if self._idx == NULL: return
		if ((self.map_opt.flag & 4) and (self._idx.flag & 2)): return
		map_opt = self.map_opt
		if max_frag_len is not None: map_opt.max_frag_len = max_frag_len
		if extra_flags is not None: map_opt.flag |= extra_flags

		if self._idx is NULL: return None
		if buf is None: b = ThreadBuffer()
		else: b = buf

		_seq = seq if isinstance(seq, bytes) else seq.encode()
		if name is not None:
			_name = name if isinstance(name, bytes) else name.encode()

		if seq2 is None:
			if name is None:
				regs = cmappy.mm_map_aux_ctx(self._idx, NULL, _seq, NULL,  &n_regs, b._b, &map_opt, self._ctx, 0)
			else:
				regs = cmappy.mm_map_aux_ctx(self._idx, _name, _seq, NULL,  &n_regs, b._b, &map_opt, self._ctx, 0)
		else:
			_seq2 = seq2 if isinstance(seq2, bytes) else seq2.encode()
			if name is None:
				regs = cmappy.mm_map_aux_ctx(self._idx, NULL, _seq, _seq2, &n_regs, b._b, &map_opt, self._ctx, 0)
			else:
				regs = cmappy.mm_map_aux_ctx(self._idx, _name, _seq, _seq2, &n_regs, b._b, &map_opt, self._ctx, 0)

		try:
			i = 0
			while i < n_regs:
				cmappy.mm_reg2hitpy(self._idx, &regs[i], &h)
				cigar, _cs, _ds, _MD = [], '', '', ''
				for k in range(h.n_cigar32): # convert the 32-bit CIGAR encoding to Python array
					c = h.cigar32[k]
					cigar.append([c>>4, c&0xf])
				if cs or ds or MD: # generate the cs/ds and/or the MD tag, if requested
					km = cmappy.mm_tbuf_get_km(b._b)
					_cur_seq = _seq2 if h.seg_id > 0 and seq2 is not None else _seq
					if cs:
						l_cs_str = cmappy.mm_gen_cs(km, &cs_str, &m_cs_str, self._idx, &regs[i], _cur_seq, 1)
						_cs = cs_str[:l_cs_str] if isinstance(cs_str, str) else cs_str[:l_cs_str].decode()
					if ds:
						l_cs_str = cmappy.mm_gen_ds(km, &cs_str, &m_cs_str, self._idx, &regs[i], _cur_seq, 1)
						_ds = cs_str[:l_cs_str] if isinstance(cs_str, str) else cs_str[:l_cs_str].decode()
					if MD:
						l_cs_str = cmappy.mm_gen_MD(km, &cs_str, &m_cs_str, self._idx, &regs[i], _cur_seq)
						_MD = cs_str[:l_cs_str] if isinstance(cs_str, str) else cs_str[:l_cs_str].decode()
				yield Alignment(h.ctg, h.ctg_len, h.ctg_start, h.ctg_end, h.strand, h.qry_start, h.qry_end, h.mapq, cigar, h.is_primary, h.mlen, h.blen, h.NM, h.trans_strand, h.seg_id, _cs, _ds, _MD)
				cmappy.mm_free_reg1(&regs[i])
				i += 1
		finally:
			while i < n_regs:
				cmappy.mm_free_reg1(&regs[i])
				i += 1
			free(regs)
			free(cs_str)

	def map_batch(self, seqs, names=None, n_threads=None, cs=False, ds=False, MD=False):
		"""Map a batch of single-segment sequences in one call.

		Runs the same batched pipeline used for files, so when the Aligner was
		created with gpu_chain/gpu_align the GPU stages process the whole batch
		together (real cross-read batching) instead of one launch per read.

		seqs   : list of query sequences (str or bytes)
		names  : optional list of query names (same length as seqs)
		Returns a list (one entry per input sequence) of lists of Alignment.

		Not thread-safe on a single Aligner: this call drives the Aligner's
		shared mapping context (and, for GPU, its worker pool) internally, so
		concurrent map()/map_batch() calls on the *same* Aligner must not
		overlap. For parallelism give each worker its own Aligner.
		"""
		cdef int n = len(seqs)
		cdef int nt
		cdef cmappy.mm_hitpy_t h
		cdef ThreadBuffer b
		cdef char *cs_str = NULL
		cdef int l_cs_str, m_cs_str = 0
		cdef void *km
		cdef cmappy.mm_mapopt_t map_opt
		cdef const char **cseqs = NULL
		cdef const char **cnames = NULL
		cdef int *out_n = NULL
		cdef cmappy.mm_reg1_t **out_regs = NULL
		cdef cmappy.mm_idx_t *idx = self._idx
		cdef cmappy.mm_mapctx_t *ctx = self._ctx
		cdef int i, j, c, k

		if self._idx == NULL or n <= 0:
			return []
		if ((self.map_opt.flag & 4) and (self._idx.flag & 2)):
			return [[] for _ in range(n)]
		nt = n_threads if (n_threads is not None and n_threads > 0) else self._n_threads
		map_opt = self.map_opt

		# Encode inputs and keep the bytes objects alive for the C call.
		enc_seqs = [s if isinstance(s, bytes) else s.encode() for s in seqs]
		if names is not None:
			enc_names = [(nm if isinstance(nm, bytes) else str(nm).encode()) for nm in names]
		else:
			enc_names = [b"" for _ in range(n)]

		cseqs = <const char**>malloc(n * sizeof(char*))
		cnames = <const char**>malloc(n * sizeof(char*))
		out_n = <int*>malloc(n * sizeof(int))
		out_regs = <cmappy.mm_reg1_t**>malloc(n * sizeof(void*))
		if cseqs == NULL or cnames == NULL or out_n == NULL or out_regs == NULL:
			free(cseqs)
			free(cnames)
			free(out_n)
			free(out_regs)
			raise MemoryError("mappy: failed to allocate map_batch buffers")
		for i in range(n):
			cseqs[i] = <const char*>enc_seqs[i]
			cnames[i] = <const char*>enc_names[i]

		cdef int ret = 0
		with nogil:
			ret = cmappy.mm_map_batch_ctx(idx, n, cnames, cseqs, &map_opt, nt, ctx, out_n, out_regs)
		if ret < 0:
			# mm_map_batch_ctx left out_regs[i]/out_n[i] zeroed on failure.
			free(cseqs)
			free(cnames)
			free(out_n)
			free(out_regs)
			raise RuntimeError("mappy: mm_map_batch_ctx failed (allocation error?)")

		# Only a tag request (cs/ds/MD) needs a thread buffer for km scratch.
		if cs or ds or MD:
			b = ThreadBuffer()
		results = []
		for i in range(n):
			hits = []
			_cur_seq = enc_seqs[i]
			for j in range(out_n[i]):
				cmappy.mm_reg2hitpy(self._idx, &out_regs[i][j], &h)
				cigar, _cs, _ds, _MD = [], '', '', ''
				for k in range(h.n_cigar32):
					c = h.cigar32[k]
					cigar.append([c>>4, c&0xf])
				if cs or ds or MD:
					km = cmappy.mm_tbuf_get_km(b._b)
					if cs:
						l_cs_str = cmappy.mm_gen_cs(km, &cs_str, &m_cs_str, self._idx, &out_regs[i][j], _cur_seq, 1)
						_cs = cs_str[:l_cs_str] if isinstance(cs_str, str) else cs_str[:l_cs_str].decode()
					if ds:
						l_cs_str = cmappy.mm_gen_ds(km, &cs_str, &m_cs_str, self._idx, &out_regs[i][j], _cur_seq, 1)
						_ds = cs_str[:l_cs_str] if isinstance(cs_str, str) else cs_str[:l_cs_str].decode()
					if MD:
						l_cs_str = cmappy.mm_gen_MD(km, &cs_str, &m_cs_str, self._idx, &out_regs[i][j], _cur_seq)
						_MD = cs_str[:l_cs_str] if isinstance(cs_str, str) else cs_str[:l_cs_str].decode()
				hits.append(Alignment(h.ctg, h.ctg_len, h.ctg_start, h.ctg_end, h.strand, h.qry_start, h.qry_end, h.mapq, cigar, h.is_primary, h.mlen, h.blen, h.NM, h.trans_strand, h.seg_id, _cs, _ds, _MD))
				cmappy.mm_free_reg1(&out_regs[i][j])
			if out_regs[i] != NULL:
				free(out_regs[i])
			results.append(hits)

		free(cs_str)
		free(cseqs)
		free(cnames)
		free(out_n)
		free(out_regs)
		return results

	def seq(self, str name, int start=0, int end=0x7fffffff):
		cdef int l
		cdef char *s
		if self._idx == NULL: return
		if ((self.map_opt.flag & 4) and (self._idx.flag & 2)): return
		s = cmappy.mappy_fetch_seq(self._idx, name.encode(), start, end, &l)
		if l == 0: return None
		r = s[:l] if isinstance(s, str) else s[:l].decode()
		free(s)
		return r

	@property
	def k(self): return self._idx.k

	@property
	def w(self): return self._idx.w

	@property
	def n_seq(self): return self._idx.n_seq

	@property
	def seq_names(self):
		cdef char *p
		if self._idx == NULL: return
		sn = []
		for i in range(self._idx.n_seq):
			p = self._idx.seq[i].name
			s = p if isinstance(p, str) else p.decode()
			sn.append(s)
		return sn

def fastx_read(fn, read_comment=False):
	cdef cmappy.kseq_t *ks
	ks = cmappy.mm_fastx_open(str.encode(fn))
	if ks is NULL: return None
	while cmappy.kseq_read(ks) >= 0:
		if ks.qual.l > 0: qual = ks.qual.s if isinstance(ks.qual.s, str) else ks.qual.s.decode()
		else: qual = None
		name = ks.name.s if isinstance(ks.name.s, str) else ks.name.s.decode()
		seq = ks.seq.s if isinstance(ks.seq.s, str) else ks.seq.s.decode()
		if read_comment:
			if ks.comment.l > 0: comment = ks.comment.s if isinstance(ks.comment.s, str) else ks.comment.s.decode()
			else: comment = None
			yield name, seq, qual, comment
		else:
			yield name, seq, qual
	cmappy.mm_fastx_close(ks)

def revcomp(seq):
	l = len(seq)
	bseq = seq if isinstance(seq, bytes) else seq.encode()
	cdef char *s = cmappy.mappy_revcomp(l, bseq)
	r = s[:l] if isinstance(s, str) else s[:l].decode()
	free(s)
	return r

def verbose(v=None):
	if v is None: v = -1
	return cmappy.mm_verbose_level(v)
