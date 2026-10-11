#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verdict orchestration: gate -> per-axis measurement -> fusion.

  1. Gate first. D_single >= 1.5e-3 means not applicable; no score is emitted.
  2. Report each axis separately before fusing.
  3. State capability limits. The DQ axis is CAVLC-only; CABAC returns unavailable.
  4. Falsifiable. Given QP1/QP2 the r axis predicts the regime, then the
     prediction is checked against the measurement.
"""
import os
import tempfile
import numpy as np

from . import mp4box, gop_axis, dq_axis, cone, score, rde_bridge

__version__ = "1.1.0"

RAW_EXT = (".264", ".h264", ".annexb")


def load_elementary(path):
    """
    Return (annexb_bytes, sizes, types, how).
    Native .264 is read directly; containers go through lossless MP4 extraction
    (start codes swapped only, bitstream untouched).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXT:
        with open(path, "rb") as f:
            data = f.read()
        sizes, types = gop_axis.frame_series_annexb(data)
        return data, sizes, types, "annexb-raw"
    data = mp4box.to_annexb(path)
    sizes, types = mp4box.nal_frame_series(path)
    return data, sizes, types, "annexb-from-mp4"


def gop_measure(sizes, types, gop2=15, gop1=None, n_perm=500):
    """GOP phase axis, with permutation-test p (within-video null, not cross-video)."""
    r = gop_axis.analyze_series(sizes, types, gop2=gop2, n_perm=n_perm,
                                gop1_known=gop1)
    if not r.get("ok"):
        return {"available": False, "reason": r.get("reason", "n/a")}
    z = r["blind_z"]
    # the verdict uses gop_axis's max-T permutation p: each permutation also takes
    # a max, so family-wise error is controlled by construction.
    # pointwise p and z are still returned, diagnostic only.
    n_cands = r["n_candidates"]
    p_maxt = r["p_perm_maxt"]
    ALPHA = 0.05
    out = {"available": True, "n_frames": r["n_frames"], "n_p": r["n_p"],
           "gop2": r["gop2"], "blind_p": r["blind_p"],
           "blind_z": round(float(z), 4), "blind_phase": r["blind_phase"],
           "blind_score": r["blind_score"],
           "n_candidates": n_cands, "n_perm": r["n_perm"],
           "test": "max-T permutation (family-wise by construction)",
           "p_perm": p_maxt,
           "alpha": ALPHA,
           "significant": bool(p_maxt < ALPHA),
           "null_max_mean": r["null_max_mean"],
           "null_max_sd": r["null_max_sd"],
           "null_max_p95": r["null_max_p95"],
           "z_curve": r["z_curve"]}
    if gop1 is not None:
        out["known_p"] = r.get("known_p")
        out["known_z"] = round(float(r.get("known_z", 0.0)), 4)
        out["known_hit"] = r.get("known_hit")
    return out


def dq_measure(data, workdir=None):
    """
    DQ comb axis. The elementary stream has to land in a temp file for the CAVLC
    extractor to read. Anything other than CAVLC is unavailable.
    """
    ent = dq_axis.entropy_coding_of(data)
    if ent == "cabac":
        # CABAC cannot parse through the CAVLC front end; short-circuit.
        return {"available": False, "entropy": "cabac",
                "reason": "this stream uses CABAC entropy coding; the DQ comb "
                          "axis reads coefficients through a CAVLC front end, "
                          "so it is unavailable here. Re-encode with "
                          "-coder cavlc (Baseline) to enable it, or rely on the "
                          "GOP-phase axis, which does not need entropy decoding."}
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=".264", dir=workdir)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        hist, n_idr, bad = dq_axis.file_hist(tmp)
    except Exception as e:
        return {"available": False, "entropy": ent or "unknown",
                "reason": "coefficient extraction failed (%s: %s); "
                          "the DQ axis needs a CAVLC (Baseline) stream"
                          % (type(e).__name__, e)}
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    n_coeff = int(sum(hist))
    if n_coeff < dq_axis.MIN_COEFF:
        return {"available": False, "entropy": ent or "unknown",
                "n_coeff": n_coeff,
                "reason": "only %d non-zero AC coefficients recovered (< %d); "
                          "statistic unstable" % (n_coeff, dq_axis.MIN_COEFF)}

    score, peak, viol = dq_axis.dq_score(hist)
    if score is None:
        return {"available": False, "entropy": ent or "unknown",
                "n_coeff": n_coeff, "reason": "insufficient coefficients"}
    return {"available": True, "entropy": ent or "unknown",
            "n_idr": n_idr, "bad_frames": bad, "n_coeff": n_coeff,
            "score": round(float(score), 6),
            "peak": round(float(peak), 6), "viol": round(float(viol), 6),
            "hist": hist, "d_cone": cone.distance(hist)}


DQ_GATE_UNMEASURED = ("DQ comb axis unavailable, so the applicability gate "
                      "could not be measured")
NO_REF = ("No single-compression reference supplied; the applicability gate "
          "cannot be measured and applicability is unknown")

# DQ decision threshold, calibrated on 912 single/double pairs at QP1=23.
#   single compression        score in [4.0e-5, 3.24e-2]   (n=20)
#   detectable double (r>1)   score in [4.20e-1, 8.73e-1]  (n=8)
# threshold is the geometric midpoint, ~3-4x margin on both sides.
#
# Two notes:
#   * absolute score only. A ratio (lift) form reaches 1049 on negatives, so a
#     QP-mismatched reference produces false positives.
#   * the axis can only emit positives. Below threshold is NO_EVIDENCE: samples
#     at QP2 in {23,28,34} land inside the single-compression band, which is the
#     predicted idempotent/erasure behaviour, not a miss.
DQ_HIT = 0.10
DQ_HIT_LO = 3.24e-2
DQ_HIT_HI = 4.20e-1

# homology criteria for reference vs suspect (see pairing_check)
PAIR_RHO_MIN = 0.80
PAIR_LEN_MIN = 0.80


def fuse(dq, gop, rinfo=None, gate=None, rde=None):
    """
    Fusion verdict: two bitstream axes plus a third pixel-domain axis. Rule priority:

      R0 gate measured and failed  => NOT_APPLICABLE (no score emitted)
      R1 any axis significant      => DOUBLE_COMPRESSION_INDICATED
      R2 none significant          => NO_EVIDENCE ("not detected" != "proven single")

    The pixel-domain axis decides at r* ~ 1.768 while the bitstream axes decide at
    r > 1, so the two are merged by rule rather than averaged:

      DETECTED    + hit    => INDICATED / high (cross-domain corroboration)
      DETECTED    + silent => INDICATED / moderate (check the r regime first; silence
                              in the annihilation band is expected)
      UNDECIDABLE + hit    => INDICATED / moderate (opposite side lacks resolution,
                              not a contradiction)
      UNDECIDABLE + silent => NO_EVIDENCE
      DIAGNOSTIC (no reference, not identification grade) => supporting evidence only,
                              does not change the verdict
      unavailable          => bitstream axes decide alone; the scorecard reports the
                              reduced evidence coverage

    gate=None means the gate was not measured: emit the verdict anyway, but state that
    applicability is unknown.
    """
    rde = rde or {}
    reasons = []
    if gate is None:
        gate_status = "not_measured"
        reasons.append(NO_REF if dq.get("available") else DQ_GATE_UNMEASURED)
    elif not gate.get("measured"):
        gate_status = "not_measured"
        reasons.append(gate.get("note", NO_REF))
    elif gate.get("pairing") == "mismatch":
        # the gate measures the reference's D_single. If reference and suspect differ in
        # source, that number describes another clip: still decide, but applicability is unknown.
        gate_status = "unreliable_reference"
        reasons.append(
            "The applicability gate was measured on a reference that is not the "
            "same content as the sample (%s). Its D_single = %s describes that "
            "other clip, not this one, so the gate result cannot be transferred "
            "and applicability stays UNKNOWN."
            % (gate.get("reference"), gate.get("d_single")))
    elif not gate.get("applies"):
        gate_status = "closed"
        rde_note = ""
        if rde.get("available"):
            rde_note = (" The pixel-domain engine separately reports %s (mode "
                        "%s); it is shown for reference only and does not "
                        "reopen the gate." % (rde.get("state"), rde.get("mode")))
        return {
            "verdict": "NOT_APPLICABLE",
            "confidence": "n/a",
            "gate_status": "closed",
            "reasons": [gate["note"],
                        "No score is reported because the certificate does not "
                        "cover this material." + rde_note],
            "scorecard": score.scorecard(dq=dq, gop=gop, rde=rde,
                                         rinfo=rinfo, gate=gate),
        }
    else:
        gate_status = "open"
        reasons.append(gate["note"])

    s = dq.get("score")
    dq_hit = bool(dq.get("available") and s is not None and s >= DQ_HIT)
    if dq.get("available"):
        if dq_hit:
            reasons.append(
                "DQ comb score = %.4f (peak %s, viol %s), %d non-zero AC "
                "coefficients from %d IDR frames. It sits above the calibrated "
                "threshold %.2f, in the band where single compression never "
                "reaches (observed single-compression maximum %.2e)."
                % (s, dq.get("peak"), dq.get("viol"),
                   dq.get("n_coeff", 0), dq.get("n_idr", 0),
                   DQ_HIT, DQ_HIT_LO))
        elif (rinfo or {}).get("exact_undetectable") or \
                (rinfo or {}).get("regime") in ("erasure", "idempotent",
                                                "undetectable"):
            reasons.append(
                "DQ comb score = %.4f, below threshold %.2f. The r-axis "
                "predicts annihilation at this operating point (%s), so this "
                "value is expected and carries no information either way."
                % (s, DQ_HIT, (rinfo or {}).get("why", "r <= 1")))
        else:
            reasons.append(
                "DQ comb score = %.4f, below the calibrated threshold %.2f "
                "(single-compression band reaches %.2e, detectable-region "
                "double compression starts at %.2e). The DQ axis yields no "
                "evidence here. Note that a low score is not evidence of "
                "single compression: at r <= 1 the comb is provably erased."
                % (s, DQ_HIT, DQ_HIT_LO, DQ_HIT_HI))

    gop_hit = False
    if gop.get("available"):
        p = gop.get("p_perm")
        gop_hit = bool(gop.get("significant"))
        reasons.append(
            "GOP-phase axis: strongest period p=%s (score %s, z=%s, phase %s) "
            "over %s candidates; max-T permutation p = %.4f over %s permutations "
            "(null max: mean %s, 95th pct %s) ⇒ %s."
            % (gop.get("blind_p"), gop.get("blind_score"), gop.get("blind_z"),
               gop.get("blind_phase"), gop.get("n_candidates"),
               p if p is not None else float("nan"),
               gop.get("n_perm"), gop.get("null_max_mean"),
               gop.get("null_max_p95"),
               "significant" if gop_hit else "not significant"))
        if p is not None and p <= 1.0 / (int(gop.get("n_perm") or 100) + 1) + 1e-12:
            reasons.append(
                "Note: p is at the resolution floor of %s permutations "
                "(no permutation matched or exceeded the observed maximum); "
                "read it as 'p <= %s', not as an exact value."
                % (gop.get("n_perm"), "%.4f" % p))
    else:
        reasons.append("GOP-phase axis unavailable (%s)."
                       % gop.get("reason", "n/a"))

    if not dq.get("available"):
        reasons.append("DQ axis unavailable (%s)." % dq.get("reason", "n/a"))

    rde_hit = False
    if rde.get("available"):
        st = rde.get("state")
        if st == "DETECTED":
            rde_hit = True
            reasons.append(
                "Pixel-domain axis (engine v%s, %s mode): %s, confidence %s, "
                "primary statistic %s (n=%s, dz=%s, p_sign=%s). This is an "
                "independent evidence chain: it reads re-encoding distortion, "
                "not the bitstream."
                % (rde.get("engine_version"), rde.get("mode"),
                   rde.get("state"), rde.get("confidence"),
                   rde.get("primary_name") or rde.get("primary_key"),
                   rde.get("n"), rde.get("dz"), rde.get("p_sign")))
        elif st == "UNDECIDABLE":
            reasons.append(
                "Pixel-domain axis (engine v%s, %s mode): UNDECIDABLE. Its own "
                "decidability boundary is r* ≈ %s; below it the pixel-domain "
                "response is not separable. That is a resolution bound of that "
                "engine, not evidence for single compression."
                % (rde.get("engine_version"), rde.get("mode"),
                   rde.get("r_star") or 1.768))
        elif st == "UNDERPOWERED":
            reasons.append(
                "Pixel-domain axis (engine v%s, %s mode): UNDERPOWERED. It reports "
                "its own sample-size budget falls short of the minimum "
                "detectable effect (MDE %s at alpha %s). That is a power limit "
                "on this clip, not evidence either way."
                % (rde.get("engine_version"), rde.get("mode"),
                   rde.get("mdes"), rde.get("n")))
        elif st == "DIAGNOSTIC":
            reasons.append(
                "Pixel-domain axis (engine v%s, profile mode, no reference): "
                "diagnostic index dip_z = %s against screening line tau = %s ⇒ "
                "%s. this engine declares the mode non-identification-grade, so it is "
                "carried as supporting evidence only and does not by itself "
                "decide the verdict."
                % (rde.get("engine_version"), rde.get("dip_z"),
                   rde.get("threshold_used"), rde.get("state_zh")))
        else:
            reasons.append("Pixel-domain axis returned state %r; not "
                           "scored." % st)
    else:
        reasons.append("Pixel-domain axis unavailable (%s). The verdict rests on "
                       "the bitstream axes only, and the scorecard reports the "
                       "reduced evidence coverage."
                       % rde.get("reason", "engine not reachable"))
    if rde.get("pairing") == "invalid":
        reasons.append(rde.get("pairing_note", "reference pairing rejected"))

    if gate_status == "not_measured":
        reasons.append("The verdict below is issued with applicability UNKNOWN; "
                       "supply a single-compression reference to close the gate.")

    sc = score.scorecard(dq=dq, gop=gop, rde=rde, rinfo=rinfo, gate=gate,
                         dq_hit=dq_hit, gop_hit=gop_hit)

    n_hit = sum([1 for x in (dq_hit, gop_hit, rde_hit) if x])
    if n_hit >= 2:
        return {"verdict": "DOUBLE_COMPRESSION_INDICATED",
                "confidence": "high", "axes_agree": True,
                "gate_status": gate_status, "reasons": reasons,
                "scorecard": sc, "engines_agree": True}
    if n_hit == 1:
        src = "pixel domain" if rde_hit else "bitstream"
        extra = ["Only one evidence chain is positive: %s." % src]
        if rde_hit and not (dq_hit or gop_hit):
            extra.append(
                "The pixel-domain chain is positive while the bitstream axes "
                "are silent; check the r regime first, because at r <= 1 the "
                "comb is provably erased and silence there is expected rather "
                "than contradictory.")
        elif not rde_hit and (dq_hit or gop_hit):
            extra.append(
                "The bitstream chain is positive while the pixel-domain engine "
                "is silent or undecidable; note that the pixel-domain response "
                "needs r >= r* ≈ 1.768 to decide at all, so silence on its side "
                "is expected in the band 1 < r < 1.768.")
        return {"verdict": "DOUBLE_COMPRESSION_INDICATED",
                "confidence": "moderate",
                "axes_agree": False, "gate_status": gate_status,
                "reasons": reasons + extra, "scorecard": sc,
                "engines_agree": not rde.get("available")}
    return {"verdict": "NO_EVIDENCE", "confidence": "n/a",
            "gate_status": gate_status, "scorecard": sc,
            "reasons": reasons + [
                "No axis reached significance. This is absence of evidence, "
                "not evidence of single compression."]}


def pairing_check(sizes, ref_sizes):
    """
    Whether reference and suspect share a source. Both the pixel-domain paired mode and
    the applicability gate treat the reference as the first generation of the same
    content; if the two are not the same source neither conclusion holds, so the case
    must be blocked rather than reported as usual.

    Criterion is the frame-size series alone: different encodes of the same content
    correlate strongly frame by frame (measured rho 0.983-0.997 across QPs); different
    content measures at rho ~ 0.24. The margin is an order of magnitude, so a 0.80
    threshold does not misjudge.
    """
    a = np.asarray(sizes, float)
    b = np.asarray(ref_sizes, float)
    n = min(len(a), len(b))
    if n < 30:
        return {"same_source": None, "status": "unknown", "rho": None,
                "note": "too few frames to compare (%d)" % n}
    ratio = n / max(len(a), len(b))
    rho = float(np.corrcoef(a[:n], b[:n])[0, 1])
    if ratio < PAIR_LEN_MIN:
        return {"same_source": False, "status": "mismatch",
                "rho": round(rho, 3), "len_ratio": round(ratio, 3),
                "note": ("frame counts differ by more than %d%% (%d vs %d)"
                         % (int(100 * (1 - PAIR_LEN_MIN)), len(a), len(b)))}
    if rho < PAIR_RHO_MIN:
        return {"same_source": False, "status": "mismatch",
                "rho": round(rho, 3), "len_ratio": round(ratio, 3),
                "note": ("frame-size sequences correlate at rho = %.3f; the same "
                         "content re-encoded at another QP stays above 0.98, so "
                         "these are not the same content" % rho)}
    return {"same_source": True, "status": "match",
            "rho": round(rho, 3), "len_ratio": round(ratio, 3),
            "note": "frame-size sequences correlate at rho = %.3f" % rho}


def analyze(path, reference=None, qp1=None, qp2=None, gop1=None, gop2=15,
            n_perm=60, workdir=None, use_rde=True):
    """
    Full analysis of one file (two bitstream axes plus an optional third pixel-domain axis).

    reference : single-compression reference (first generation of the same content).
                Supplying it is required to measure the gate; otherwise the verdict
                carries "applicability UNKNOWN". With a reference, the pixel domain
                also runs in paired identification-grade mode.
    use_rde   : whether to fold in the pixel-domain engine. One detect call measures
                5-20 s; disabling it drops one evidence chain and leaves the bitstream
                verdict unchanged.
    """
    data, sizes, types, how = load_elementary(path)
    out = {
        "engine_version": __version__,
        "file": os.path.basename(path),
        "stream": {"how": how, "bytes": len(data), "n_vcl": len(sizes)},
        "gop": gop_measure(sizes, types, gop2=gop2, gop1=gop1, n_perm=n_perm),
        "dq": dq_measure(data, workdir=workdir),
    }

    pairing = None
    gate = None
    if reference is not None:
        try:
            rdata, rsizes, _, _ = load_elementary(reference)
            pairing = pairing_check(sizes, rsizes)
            pairing["reference"] = os.path.basename(reference)
            rdq = dq_measure(rdata, workdir=workdir)
            if rdq.get("available"):
                gate = cone.gate(rdq["hist"])
                gate["reference"] = os.path.basename(reference)
                gate["pairing"] = pairing["status"]
            else:
                gate = cone.gate_unmeasured(
                    "reference DQ axis unavailable (%s)" % rdq.get("reason", "?"))
        except Exception as e:
            gate = cone.gate_unmeasured(
                "reference could not be analysed (%s: %s)" % (type(e).__name__, e))
    out["pairing"] = pairing
    out["gate"] = gate

    # the pixel domain is optional; failure must not block the verdict.
    # paired mode requires the same source; otherwise it does not measure recompression.
    rde = {"available": False, "reason": "disabled by caller"}
    if use_rde:
        try:
            ref_for_rde = reference
            if pairing is not None and pairing.get("status") == "mismatch":
                ref_for_rde = None
            rde = rde_bridge.detect(path, reference=ref_for_rde)
            if ref_for_rde is None:
                rde["pairing"] = "invalid"
                rde["pairing_note"] = (
                    "the reference is not the same content as the sample "
                    "(frame-size rho = %s), so paired mode would compare two "
                    "unrelated clips; the pixel-domain axis fell back to "
                    "diagnostic mode, which does not carry identification grade"
                    % pairing.get("rho"))
        except Exception as e:
            rde = {"available": False,
                   "reason": "pixel-domain bridge error (%s: %s)"
                             % (type(e).__name__, e)}
    if "profile" in rde:
        rde.pop("profile", None)      # profile curves are long; excluded by default
    out["rde"] = rde

    if qp1 is not None and qp2 is not None:
        rinfo = cone.r_axis(qp1, qp2)
        rinfo["period6_family"] = cone.period6_family(qp1)
        out["r_axis"] = rinfo
    else:
        rinfo = None
    out["decision"] = fuse(out["dq"], out["gop"], rinfo, gate, rde=rde)
    return out
