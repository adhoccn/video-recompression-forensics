#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOP phase axis: uses only the frame-size series, no entropy decoding.

First-generation I frames sit at t = 0, GOP1, 2*GOP1, ...; re-encoding with
GOP2 != GOP1 turns those positions into P frames whose sizes deviate
systematically, so the series carries a phase anomaly at period GOP1.
Single-compressed video has no such I frames and no signal.

Pipeline (NAL byte counts only):
  1. split VCL NALs (type 1/5) -> frame sizes
  2. keep P frames, take log sizes
  3. detrend by subtracting a sliding median
  4. notch out the second-generation GOP period
  5. phase strength score(p) = max_j Welch t (phase j vs the rest)
  6. z-normalize against a permutation null; plain argmax_p biases toward large p
"""
import numpy as np

GOP2_DEFAULT = 15
DETREND_W = 31             # much larger than the tested period, so the signal survives
N_PERM = 500
MIN_FRAMES = 60
MIN_PER_PHASE = 4
RNG_SEED = 12345


def movmedian(x, w):
    h = w // 2
    out = np.empty_like(x)
    for i in range(len(x)):
        a, b = max(0, i - h), min(len(x), i + h + 1)
        out[i] = np.median(x[a:b])
    return out


def phase_score(idx, val, p, min_per_phase=MIN_PER_PHASE, return_phase=False):
    """
    Phase strength at period p = max_j Welch t (frames at phase j vs the rest).

    A directed comparison is used instead of a robust z of "largest phase vs
    median phase": the signal concentrates in a single phase (where
    first-generation I frames land) and the latter is diluted by the other phases.

    All phase counts, sums and sums of squares are obtained in one pass with
    bincount and the t values are then computed in batch; a per-phase loop would
    run 60 x n_candidates times inside the permutation test and dominate runtime.
    """
    idx = np.asarray(idx)
    val = np.asarray(val, float)
    ph = idx % p
    na = np.bincount(ph, minlength=p).astype(float)
    sa = np.bincount(ph, weights=val, minlength=p)
    sa2 = np.bincount(ph, weights=val * val, minlength=p)
    tot_n = float(len(val))
    tot_s = float(val.sum())
    tot_s2 = float(np.dot(val, val))
    nb = tot_n - na
    ma = sa / np.maximum(na, 1.0)
    mb = (tot_s - sa) / np.maximum(nb, 1.0)
    # ddof=1 sample variance = (sum(x^2) - n*mean^2)/(n-1)
    va = np.maximum((sa2 - na * ma * ma) / np.maximum(na - 1.0, 1.0), 0.0)
    vb = np.maximum(((tot_s2 - sa2) - nb * mb * mb) / np.maximum(nb - 1.0, 1.0), 0.0)
    se = np.sqrt(va / np.maximum(na, 1.0) + vb / np.maximum(nb, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ma - mb) / np.where(se < 1e-12, np.nan, se)
    ok = (na >= min_per_phase) & (nb >= min_per_phase) & np.isfinite(t)
    if not ok.any():
        return (0.0, -1) if return_phase else 0.0
    t = np.where(ok, t, -np.inf)
    j = int(np.argmax(t))
    best = float(t[j])
    return (best, j) if return_phase else best


def candidates(gop2, lo=6, hi=25):
    """Candidate periods: exclude integer-multiple relations with the second-generation GOP (spurious periods of the second pass)."""
    return [p for p in range(lo, hi)
            if p % gop2 != 0 and gop2 % p != 0]


def analyze_series(sizes, types, gop2=GOP2_DEFAULT, n_perm=N_PERM,
                   gop1_known=None):
    """
    sizes: bytes per frame (float); types: matching NAL type (1=P, 5=IDR).
    Returns a dict covering both the known-period and blind-search views.
    """
    sizes = np.asarray(sizes, float)
    types = np.asarray(types, int)
    n = len(sizes)
    if n < MIN_FRAMES:
        return {"ok": False, "reason": "too few frames (%d < %d)" % (n, MIN_FRAMES)}

    fidx = np.arange(n)
    keep = types == 1                                  # P frames only
    idx = fidx[keep]
    y = np.log(np.maximum(sizes[keep], 1.0))
    if len(y) < MIN_FRAMES:
        return {"ok": False, "reason": "too few P frames (%d)" % len(y)}

    y = y - movmedian(y, DETREND_W)                    # detrend
    ph = idx % gop2                                    # notch the second-generation period
    for j in range(gop2):
        s = ph == j
        if s.sum() >= 3:
            y[s] -= y[s].mean()

    cands = candidates(gop2)
    scores, phases = {}, {}
    for p in cands:
        s, j = phase_score(idx, y, p, return_phase=True)
        scores[p], phases[p] = s, j

    # max-T permutation test: the blind search takes a maximum over |cands|
    # candidates and each permutation takes a max too, so family-wise error is
    # controlled by construction. Bonferroni would be over-conservative here
    # because candidates share one frame series and are strongly correlated.
    # The observed statistic is the raw Welch t, not z: the z denominator comes
    # from the null, so maximizing z would reuse the same sample twice.
    obs_arr = np.array([scores[p] for p in cands], float)
    obs_max = float(obs_arr.max())
    best_p = int(cands[int(np.argmax(obs_arr))])

    rng = np.random.default_rng(RNG_SEED)
    null_max = np.empty(n_perm)
    for b in range(n_perm):
        yp = rng.permutation(y)
        m = -np.inf
        for p in cands:
            v = phase_score(idx, yp, p)
            if v > m:
                m = v
        null_max[b] = m
    # the +1 convention counts the observation itself, so p has a 1/(n_perm+1) floor.
    p_maxt = float((np.count_nonzero(null_max >= obs_max) + 1) / (n_perm + 1))

    # per-period z is diagnostic only and does not enter the verdict.
    z = {}
    for k, p in enumerate(cands):
        mu, sd = null_max.mean(), null_max.std() + 1e-9
        z[p] = float((obs_arr[k] - mu) / sd)
    out = {"ok": True, "n_frames": int(n), "n_p": int(keep.sum()),
           "gop2": int(gop2),
           "n_candidates": len(cands), "n_perm": int(n_perm),
           "blind_p": int(best_p), "blind_score": round(obs_max, 4),
           "blind_z": float(z[best_p]),
           "blind_phase": int(phases[best_p]),
           "p_perm_maxt": p_maxt,
           "null_max_mean": round(float(null_max.mean()), 4),
           "null_max_sd": round(float(null_max.std()), 4),
           "null_max_p95": round(float(np.percentile(null_max, 95)), 4),
           "z_curve": {str(k): round(v, 4) for k, v in z.items()}}
    if gop1_known is not None and gop1_known in z:
        out["known_p"] = int(gop1_known)
        out["known_z"] = float(z[gop1_known])
        out["known_phase"] = int(phases[gop1_known])
        out["known_hit"] = bool(int(best_p) == int(gop1_known))
    return out


def frame_series_annexb(data):
    """Parse (sizes, types) from an Annex-B byte string."""
    starts = []
    i, n = 0, len(data)
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0:
            break
        starts.append(j)
        i = j + 3
    sizes, types = [], []
    for k, s in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else n
        while end > s + 3 and data[end - 1] == 0:
            end -= 1
        nal = data[s + 3:end]
        if not nal:
            continue
        t = nal[0] & 0x1F
        if t in (1, 5):
            sizes.append(float(len(nal)))
            types.append(t)
    return sizes, types
