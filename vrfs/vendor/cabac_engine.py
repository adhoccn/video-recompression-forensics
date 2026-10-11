#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os as _os, sys as _sys
_p = _os.path.dirname(_os.path.abspath(__file__))
while True:
    _c = _os.path.join(_p, 'script', '_common')
    if _os.path.isdir(_c):
        _sys.path.insert(0, _c)
        break
    _n = _os.path.dirname(_p)
    if _n == _p:
        break
    _p = _n
# ^ locate the canonical front-end copies in script/_common
"""H.264 CABAC arithmetic decoding engine (ITU-T H.264 9.3.1 / 9.3.3.2).

Pure Python, paired with cabac_tables.py (spec constant tables transcribed from
FFmpeg). Entropy decoding only, no pixel reconstruction -- used to extract
quantized coefficient levels in forensic research.

Three decoding primitives:
  decode_decision(ctxIdx)  regular context-adaptive (9.3.3.2.1)
  decode_bypass()          bypass, equiprobable     (9.3.3.2.3)
  decode_terminate()       terminate                (9.3.3.2.4)
"""
from cabac_tables import CTX_INIT_I, CTX_INIT_PB, RANGE_LPS, TRANS_MPS, TRANS_LPS

__all__ = ["Cabac", "init_contexts", "CabacError"]

class CabacError(Exception):
    pass

def _clip3(lo, hi, v):
    return lo if v < lo else (hi if v > hi else v)

def init_contexts(slice_type_i, cabac_init_idc, slice_qpy):
    """9.3.1.1 context variable initialization.

    slice_type_i : True for I/SI slices (CTX_INIT_I), else CTX_INIT_PB[cabac_init_idc]
    Returns pstate[1024] and vmps[1024] as two lists (much faster than a list
    of tuples).
    """
    tab = CTX_INIT_I if slice_type_i else CTX_INIT_PB[cabac_init_idc]
    q = _clip3(0, 51, slice_qpy)
    pstate = [0] * 1024
    vmps = [0] * 1024
    for i in range(1024):
        m, n = tab[i]
        pre = _clip3(1, 126, ((m * q) >> 4) + n)
        if pre <= 63:
            pstate[i] = 63 - pre
            vmps[i] = 0
        else:
            pstate[i] = pre - 64
            vmps[i] = 1
    return pstate, vmps

class Cabac:
    """Operates on an RBSP with emulation prevention bytes already removed.

    data      : bytes  RBSP of the whole NAL unit
    byte_pos  : int    start of slice_data (after cabac_alignment_one_bit,
                       byte aligned)
    """

    __slots__ = ("data", "n", "pos", "bitbuf", "bitcnt",
                 "range", "offset", "pstate", "vmps", "overrun")

    def __init__(self, data, byte_pos, pstate, vmps):
        self.data = data
        self.n = len(data)
        self.pos = byte_pos
        self.bitbuf = 0
        self.bitcnt = 0
        self.overrun = 0
        self.pstate = pstate
        self.vmps = vmps
        # 9.3.1.2 engine initialization
        self.range = 510
        self.offset = self._read_bits(9)

    def _read_bit(self):
        if self.bitcnt == 0:
            if self.pos < self.n:
                self.bitbuf = self.data[self.pos]
                self.pos += 1
            else:
                self.bitbuf = 0
                self.overrun += 1
                if self.overrun > 64:
                    raise CabacError("bitstream overrun")
            self.bitcnt = 8
        self.bitcnt -= 1
        return (self.bitbuf >> self.bitcnt) & 1

    def _read_bits(self, k):
        v = 0
        for _ in range(k):
            v = (v << 1) | self._read_bit()
        return v

    def decode_decision(self, ctx_idx):
        p = self.pstate[ctx_idx]
        rng = self.range
        r_lps = RANGE_LPS[p][(rng >> 6) & 3]
        rng -= r_lps
        off = self.offset
        if off >= rng:
            # LPS
            bit = 1 - self.vmps[ctx_idx]
            off -= rng
            rng = r_lps
            if p == 0:
                self.vmps[ctx_idx] = 1 - self.vmps[ctx_idx]
            self.pstate[ctx_idx] = TRANS_LPS[p]
        else:
            bit = self.vmps[ctx_idx]
            self.pstate[ctx_idx] = TRANS_MPS[p]
        # RenormD
        while rng < 256:
            rng <<= 1
            off = (off << 1) | self._read_bit()
        self.range = rng
        self.offset = off
        return bit

    def decode_bypass(self):
        off = ((self.offset << 1) | self._read_bit())
        if off >= self.range:
            self.offset = off - self.range
            return 1
        self.offset = off
        return 0

    def decode_terminate(self):
        rng = self.range - 2
        if self.offset >= rng:
            self.range = rng
            return 1
        off = self.offset
        while rng < 256:
            rng <<= 1
            off = (off << 1) | self._read_bit()
        self.range = rng
        self.offset = off
        return 0

    def unary_ctx(self, ctx_first, ctx_rest, cmax=None):
        """Unary/truncated unary: bin0 uses ctx_first, the rest use ctx_rest
        (may be a function)."""
        v = 0
        while True:
            c = ctx_first if v == 0 else (
                ctx_rest(v) if callable(ctx_rest) else ctx_rest)
            if self.decode_decision(c) == 0:
                return v
            v += 1
            if cmax is not None and v >= cmax:
                return v

    def bypass_egk(self, k):
        """EGk suffix (bypass). Returns the suffix value sufS."""
        s = 0
        guard = 0
        while self.decode_bypass():
            s += 1 << k
            k += 1
            guard += 1
            if guard > 32:
                raise CabacError("EGk runaway")
        while k > 0:
            k -= 1
            if self.decode_bypass():
                s += 1 << k
        return s

    def bypass_fl(self, nbits):
        v = 0
        for _ in range(nbits):
            v = (v << 1) | self.decode_bypass()
        return v
