#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Completely monotone (CM) cone, cone distance, applicability gate, and the r axis.

* qstep(qp)         H.264 quantizer step (doubles every +6 QP)
* kl_to_cm(p)       information projection distance to the CM cone
* gate(hist)        applicability gate: large D_single => not applicable
* r_axis(qp1, qp2)  r = Qstep(QP1)/Qstep(QP2) and period-6 undetectable points
"""
import numpy as np

_QBASE = np.array([0.625, 0.6875, 0.8125, 0.875, 1.0, 1.125])

KOBS = 23      # observable bins: |level| in 1..23
KRAW = 24      # extractor emits 1..24; bin 24 is a tail aggregate, dropped

# empirical cutoff on single-compression cone distance
GATE_D_SINGLE = 1.5e-3


def qstep(qp):
    """H.264 quantizer step. Doubles every +6 QP; within a period, one of 6 base values."""
    qp = int(qp)
    return float(_QBASE[qp % 6] * (2.0 ** (qp // 6)))


def rho_grid(n=140, t_lo=2e-3, t_hi=28.0):
    """rho grid. Returns rho only; callers generally do not need t."""
    t = np.exp(np.linspace(np.log(t_lo), np.log(t_hi), n))
    return np.exp(-t)


def geo_basis(rhos, K=KOBS):
    """Extreme rays of the CM cone: truncated geometric pmf, one ray per row."""
    k = np.arange(1, K + 1)
    R = rhos[:, None] ** (k[None, :] - 1.0)
    G = R * ((1.0 - rhos) / (1.0 - rhos ** K))[:, None]
    return G / G.sum(axis=1, keepdims=True)


_RHOS = None
_G = None


def _basis():
    global _RHOS, _G
    if _G is None:
        _RHOS = rho_grid()
        _G = geo_basis(_RHOS, KOBS)
    return _RHOS, _G


def fit_mixture(p, G, iters=4000, tol=1e-13):
    """EM for the I-projection: w* = argmin_w KL(p || wG)."""
    p = np.asarray(p, float)
    p = p / p.sum()
    n = G.shape[0]
    w = np.full(n, 1.0 / n)
    for _ in range(iters):
        q = np.maximum(w @ G, 1e-300)
        w_new = ((w[:, None] * G) / q[None, :]) @ p
        s = w_new.sum()
        if s <= 0:
            break
        w_new /= s
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w, np.maximum(w @ G, 1e-300)


def kl(p, q):
    p = np.maximum(np.asarray(p, float), 1e-300)
    q = np.maximum(np.asarray(q, float), 1e-300)
    return float(np.sum(p * np.log(p / q)))


def kl_to_cm(p, G=None):
    """min_{q in CM} KL(p||q): distance to the completely monotone cone."""
    if G is None:
        _, G = _basis()
    w, q = fit_mixture(p, G)
    return kl(p, q), q, w


def norm_bins(hist):
    """The extractor emits 24 bins; truncate to 1..23 and normalize."""
    h = np.asarray(hist, float)[:KOBS]
    s = h.sum()
    if s <= 0:
        return None
    return h / s


def distance(hist):
    """
    Histogram distance to the CM cone, D = min_{q in CM} KL(p||q). A pure
    statistic; it carries no decision semantics.

    Double compression inflates D (rounding onto the first-generation lattice
    breaks complete monotonicity), so D is also a detection statistic; it is
    also resolution dependent, so D alone cannot separate "double" from
    "high-resolution single". That confound is resolved by the gate, which must
    be measured on a single-compression reference.
    """
    p = norm_bins(hist)
    if p is None:
        return None
    return float(kl_to_cm(p)[0])


def gate(hist):
    """
    Applicability gate. The input must be the histogram of a single-compression
    reference, not the sample under test.

    The gate asks whether the single-compression baseline of this material sits
    close to the CM cone. Measured on the sample itself, genuinely double
    compressed material has large D, which always closes the gate and refuses a
    verdict on every true positive.

    Returns dict:
      d_single   distance from the reference (single-compression) histogram to the cone
      applies    certificate applies when D_single < GATE_D_SINGLE
      note       human-readable explanation
    """
    p = norm_bins(hist)
    if p is None:
        return {"measured": False, "d_single": None, "applies": None,
                "note": "no non-zero AC coefficients recovered from the reference"}
    d = kl_to_cm(p)[0]
    applies = bool(d < GATE_D_SINGLE)
    if applies:
        note = ("Reference (single-compression) baseline sits inside the CM cone "
                "(D_single = %.2e < %.1e); the certificate applies."
                % (d, GATE_D_SINGLE))
    else:
        note = ("Reference (single-compression) baseline is already far from the CM "
                "cone (D_single = %.2e >= %.1e); the certificate does not cover this "
                "material, so no verdict is issued." % (d, GATE_D_SINGLE))
    return {"measured": True, "d_single": float(d), "applies": applies,
            "note": note, "threshold": GATE_D_SINGLE}


def gate_unmeasured(reason):
    """Explicit placeholder when the gate cannot be measured; never silently treated as applicable."""
    return {"measured": False, "d_single": None, "applies": None,
            "note": reason}


def r_axis(qp1, qp2):
    """
    r = Qstep(QP1)/Qstep(QP2) and the period-6 verdict.

    r = 1 (same QP) makes re-quantization idempotent and annihilates the DQ axis.
    1/r an exact positive integer means the first-generation lattice lies exactly
    on the second-generation lattice, also an exact undetectable point
    (period-6: Qstep doubles every +6 QP, so QP2 = QP1 + 6k gives 1/r = 2^k).
    """
    d1, d2 = qstep(qp1), qstep(qp2)
    r = d1 / d2
    inv = 1.0 / r
    exact = bool(abs(inv - round(inv)) < 1e-9 and round(inv) >= 1)
    L = int(round(inv)) if exact else None
    if abs(r - 1.0) < 1e-9:
        regime, why = "idempotent", ("QP1 = QP2 ⇒ re-quantization is idempotent; "
                                     "the DQ axis carries no information")
    elif exact:
        regime, why = "undetectable", ("1/r = %d is an exact integer ⇒ the first-pass "
                                       "grid lies on the second-pass grid (period-6 "
                                       "family); the DQ axis is annihilated" % L)
    elif r > 1.0:
        regime, why = "detectable", ("r = %.4f > 1 ⇒ second pass is finer; the comb is "
                                     "observable" % r)
    else:
        regime, why = "erasure", ("r = %.4f < 1 ⇒ second pass is coarser; the first-pass "
                                  "comb is largely erased" % r)
    return {"qp1": int(qp1), "qp2": int(qp2), "r": float(r), "inv_r": float(inv),
            "exact_undetectable": exact, "L": L, "Qstep1": d1, "Qstep2": d2,
            "regime": regime, "why": why}


def period6_family(qp1, kmax=4):
    """Given QP1, list the period-6 undetectable points QP2 = QP1 + 6k (k = 0..kmax)."""
    return [int(qp1) + 6 * k for k in range(kmax + 1)]
