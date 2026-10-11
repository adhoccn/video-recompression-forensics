#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQ comb axis: reads the H.264 quantized |level| histogram.

Single compression gives a smooth, near-monotonically decreasing AC histogram.
Double compression with Q1 != Q2 piles coefficients onto the first-generation
lattice c*Qstep1/Qstep2, producing comb peaks and monotonicity violations. Two
parameter-free statistics:

    peak = sum_k max(0, h(k) - (h(k-1)+h(k+1))/2)
    viol = sum_k max(0, h(k+1) - h(k))
    score = peak + viol

Coefficient extraction uses the CAVLC front end (vendor/bs_extract), so CABAC
streams return available=False rather than a guessed number.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))

KMAX = 24          # extractor emits 1..24
KOBS = 23          # theory uses 1..23
MAX_IDR = 13
MIN_COEFF = 200    # statistics unstable below this

_bs_extract = None


def _get_extractor():
    global _bs_extract
    if _bs_extract is None:
        import bs_extract as m
        _bs_extract = m
    return _bs_extract


def coeff_available(err=None):
    return {"available": False, "entropy": "unknown",
            "reason": err or "coefficient extraction unavailable"}


def file_hist(path, max_idr=MAX_IDR):
    """
    Extract the AC |level| histogram (1..24) from an Annex-B elementary stream file.
    Returns (hist, n_idr, bad_frames).
    """
    m = _get_extractor()
    r = m.extract_idr_coeffs(path, max_idr=max_idr)
    lv = r["levels"].astype(np.int64)
    # drop zigzag position 0 (DC/placeholder); count 1..15 only, so AC only
    ac = lv[:, 1:16].ravel()
    ac = np.abs(ac[ac != 0])
    h = np.bincount(np.clip(ac, 0, KMAX), minlength=KMAX + 1)[1:KMAX + 1]
    return h.tolist(), int(r["n_idr"]), int(r["check"]["bad_frames"])


def dq_score(h):
    """Comb score = convex peak energy + monotonicity violations."""
    h = np.asarray(h, float)
    tot = h.sum()
    if tot < MIN_COEFF:
        return None, None, None
    h = h / tot
    peak = 0.0
    for k in range(1, KMAX - 1):
        peak += max(0.0, h[k] - (h[k - 1] + h[k + 1]) / 2.0)
    viol = 0.0
    for k in range(KMAX - 1):
        viol += max(0.0, h[k + 1] - h[k])
    return float(peak + viol), float(peak), float(viol)


def profile_idc_of(data):
    """Read profile_idc from the SPS (first byte). Returns None if unavailable."""
    try:
        i = 0
        while True:
            j = data.find(b"\x00\x00\x01", i)
            if j < 0:
                return None
            if (data[j + 3] & 0x1F) == 7:
                return int(data[j + 4])
            i = j + 3
    except Exception:
        return None


def _bitreader(buf):
    """Minimal Exp-Golomb bit reader."""

    class R:
        def __init__(s, b):
            s.b, s.p = b, 0

        def u1(s):
            if s.p >= len(s.b) * 8:
                raise EOFError
            v = (s.b[s.p >> 3] >> (7 - (s.p & 7))) & 1
            s.p += 1
            return v

        def ue(s):
            lz = 0
            while s.u1() == 0:
                lz += 1
                if lz > 32:
                    raise EOFError
            v = 1
            for _ in range(lz):
                v = (v << 1) | s.u1()
            return v - 1
    return R(buf)


def _find_nal(data, ntype):
    """Scan the Annex-B stream; return the RBSP start (start code excluded) of the first NAL of this type."""
    i, n = 0, len(data)
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0:
            return None
        if (data[j + 3] & 0x1F) == ntype:
            return j + 4
        i = j + 3


def entropy_coding_of(data):
    """
    Determine the entropy coding mode, to state the capability boundary honestly
    and concretely.

    entropy_coding_mode_flag is in the PPS, not the SPS (nal_unit_type = 8; first
    bit after skipping pic_parameter_set_id and seq_parameter_set_id). Parsing it
    from the SPS never finds the flag and returns unknown for every stream, so
    CABAC streams run through the CAVLC decoder before failing.

    PPS syntax (first three fields only):
        pic_parameter_set_id      ue(v)
        seq_parameter_set_id      ue(v)
        entropy_coding_mode_flag  u(1)      <- 0 = CAVLC, 1 = CABAC
    No other field is parsed; unreadable input yields unknown rather than a guess.
    """
    p = profile_idc_of(data)
    if p == 66:                       # Baseline: CAVLC only
        return "cavlc"
    off = _find_nal(data, 8)          # PPS
    if off is not None:
        try:
            r = _bitreader(data[off:off + 64])
            r.ue()                    # pic_parameter_set_id
            r.ue()                    # seq_parameter_set_id
            return "cabac" if r.u1() else "cavlc"
        except Exception:
            pass
    return "unknown"
