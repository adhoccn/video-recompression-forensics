#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evidence-strength scorecard (0-100): how strong this body of evidence is.

It is not a posterior probability, not a likelihood ratio, and not an accuracy
figure. Per-axis raw statistics and the weights are returned so anyone can
recompute it.

  1. No score when the gate fails.
  2. The DQ axis leaves the score at points where the comb is predicted to be
     annihilated (status=annihilated).
  3. Weights renormalize over the axes that could decide, and coverage is
     reported as-is.
"""
import math

W_DQ = 0.30
W_GOP = 0.30
W_RDE = 0.30
W_CONS = 0.10

# anchors from calibration; see pipeline.DQ_HIT_LO / DQ_HIT_HI
DQ_LO = 3.24e-2
DQ_HI = 4.20e-1
DQ_CAP = 95

# GOP axis: log10(1/p) caps at 3 (p=1e-3); more permutations only refine resolution.
GOP_LOG_CAP = 3.0
# Cap when not significant. p=0.10 would map to 34, which reads as weak evidence
# but is permutation noise; suppress it so the score does not speak for the p value.
GOP_NS_CAP = 25

# pixel-domain profile mode (no reference) is not identification-grade; capped here
RDE_BELOW_CAP = 30
RDE_HIT_CAP = 85

GRADES = [(85, "强证据"), (70, "较强证据"), (50, "中等证据"),
          (25, "弱证据"), (0, "无证据")]


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def dq_item(dq, rinfo=None):
    """DQ comb axis subscore. Returns (score|None, status, why)."""
    if not dq.get("available"):
        return None, "unavailable", dq.get("reason", "axis unavailable")
    s = dq.get("score")
    if s is None:
        return None, "unavailable", "no score produced"
    regime = (rinfo or {}).get("regime")
    if (rinfo or {}).get("exact_undetectable") or regime in ("idempotent",
                                                             "undetectable",
                                                             "erasure"):
        return None, "annihilated", (
            "r-axis says the comb is provably erased at this operating point "
            "(%s), so the measured %.3e carries no information and is not "
            "scored" % ((rinfo or {}).get("why", "r <= 1"), s))
    if s <= 0:
        return 5, "scored", "score at/below the single-compression floor"
    x = (math.log(s) - math.log(DQ_LO)) / (math.log(DQ_HI) - math.log(DQ_LO))
    v = int(round(5 + (DQ_CAP - 5) * _clamp(x)))
    why = ("score %.3e mapped on a log scale between the observed "
           "single-compression ceiling %.2e and the detectable-region floor "
           "%.2e" % (s, DQ_LO, DQ_HI))
    return v, "scored", why


def gop_item(gop):
    """GOP phase axis subscore. Uses the max-T permutation p, not z: the
    normal assumption behind z does not hold in the tail.
    """
    if not gop.get("available"):
        return None, "unavailable", gop.get("reason", "axis unavailable")
    p = gop.get("p_perm")
    if p is None:
        return None, "unavailable", "no permutation p value"
    floor = 1.0 / (int(gop.get("n_perm") or 0) + 1) if gop.get("n_perm") else 0.0
    at_floor = floor > 0 and p <= floor + 1e-12
    if p <= 0:
        v = 100
    else:
        v = int(round(100 * _clamp(math.log10(1.0 / p) / GOP_LOG_CAP)))
    why = ("max-T permutation p = %.4f over %s permutations ⇒ "
           "log10(1/p) = %.2f (capped at %s)"
           % (p, gop.get("n_perm"), math.log10(1.0 / p) if p > 0 else float("inf"),
              GOP_LOG_CAP))
    if not gop.get("significant"):
        # non-significant must not map into the thirties; that would be presenting
        # permutation noise as signal. Suppress it and give the reason in why.
        v = min(v, GOP_NS_CAP)
        why += ("; not significant at alpha = %s, so the subscore is capped at "
                "%d — read it as permutation noise, not as weak evidence"
                % (gop.get("alpha", 0.05), GOP_NS_CAP))
    elif at_floor:
        why += ("; p is at the resolution floor of %s permutations, so read it "
                "as 'p <= %.4f'" % (gop.get("n_perm"), p))
    return v, "scored", why


def rde_item(rde):
    """
    Pixel-domain axis subscore. UNDECIDABLE / UNDERPOWERED drop out of scoring
    (status=undecidable): both are limits of that engine — resolution or sample
    size — not negative evidence, and scoring them low would penalize the clip
    for the engine's own shortfall.
    """
    if not rde or not rde.get("available"):
        return None, "unavailable", (rde or {}).get("reason", "RDE engine not reachable")
    state = rde.get("state")
    if state == "DETECTED":
        c = rde.get("confidence")
        v = int(round(100 * c)) if c is not None else 90
        return v, "scored", ("RDE paired mode: DETECTED with declared "
                             "confidence %.2f" % (c if c is not None else float("nan")))
    if state == "UNDECIDABLE":
        return None, "undecidable", (
            "RDE paired mode: UNDECIDABLE. Its own decidability boundary is "
            "r* ≈ %s; below it the pixel-domain response is not separable. "
            "That is a resolution bound of that engine, not negative evidence, "
            "so it is excluded from the weighting rather than scored low."
            % (rde.get("r_star") if rde.get("r_star") else "1.77"))
    if state == "UNDERPOWERED":
        return None, "undecidable", (
            "RDE: UNDERPOWERED, i.e. its own sample-size budget is short of the "
            "minimum detectable effect (MDE %s at alpha %s). That is a power "
            "problem of that engine on this clip, not negative evidence."
            % (rde.get("mdes"), rde.get("n")))
    if state == "DIAGNOSTIC":
        dip = rde.get("dip_z")
        tau = rde.get("threshold_used")
        if dip is None or not tau:
            return None, "unavailable", "RDE diagnostic index incomplete"
        ratio = _clamp(dip / tau)
        if ratio < 1.0:
            v = int(round(RDE_BELOW_CAP * ratio))
            why = ("RDE profile mode (no reference): dip_z %.2f is %.0f%% of the "
                   "screening line tau %.2f, i.e. below it ⇒ %.0f%% of a small "
                   "allowance of %d points; this mode is diagnostic only"
                   % (dip, 100 * ratio, tau, 100 * ratio, RDE_BELOW_CAP))
        else:
            v = int(round(min(RDE_HIT_CAP, 60 + 25 * (ratio - 1.0))))
            why = ("RDE profile mode: dip_z %.2f crosses the screening line tau "
                   "%.2f ⇒ %d points, still capped at %d because RDE declares "
                   "this mode non-identification-grade"
                   % (dip, tau, v, RDE_HIT_CAP))
        return v, "scored", why
    return None, "unavailable", "RDE state %r not scorable" % state


def consistency_item(n_pos, n_silent):
    """
    Cross-axis consistency term. Consistency is counted only over decidable
    axes: undecidable ones have already left the weighting and must not come
    back here as the "silent side".

      ≥2 axes positive                      ⇒ 100
      1 positive, no decidable axis silent  ⇒ 85
      1 positive, decidable axes silent     ⇒ 70
      0 positive                            ⇒ 0
    """
    if n_pos >= 2:
        return 100, "scored", "%d axes positive ⇒ mutually corroborating" % n_pos
    if n_pos == 1:
        if n_silent == 0:
            return 85, "scored", ("one axis positive and no decidable axis "
                                  "contradicts it ⇒ single-source evidence")
        return 70, "scored", ("one axis positive while %d decidable axis/axes "
                              "stay silent ⇒ single-source evidence"
                              % n_silent)
    return 0, "scored", "no axis positive"


def grade(v):
    for lo, name in GRADES:
        if v >= lo:
            return name
    return "无证据"


def scorecard(dq=None, gop=None, rde=None, rinfo=None, gate=None,
              dq_hit=None, gop_hit=None):
    """
    Combine the subscores into a scorecard.

    Gate closed (applies=False) ⇒ not applicable, no score is issued.
    """
    dq = dq or {}
    gop = gop or {}
    rde = rde or {}

    if gate and gate.get("measured") and not gate.get("applies"):
        return {"score": None, "grade": "不适用（证书未覆盖该素材）",
                "coverage": 0.0, "items": [], "notes": [
                    "The applicability gate is closed (D_single = %s ≥ %s), so "
                    "no score is issued."
                    % (gate.get("d_single"), gate.get("threshold"))]}

    items = []

    v, st, why = dq_item(dq, rinfo)
    items.append({"key": "dq", "label": "DQ 梳齿轴（系数域）", "weight": W_DQ,
                  "score": v, "status": st, "why": why})
    v, st, why = gop_item(gop)
    items.append({"key": "gop", "label": "GOP 相位轴（帧大小域）", "weight": W_GOP,
                  "score": v, "status": st, "why": why})
    v, st, why = rde_item(rde)
    items.append({"key": "rde", "label": "像素域探针轴", "weight": W_RDE,
                  "score": v, "status": st, "why": why})

    if dq_hit is None:
        dq_hit = bool(dq.get("available") and (dq.get("score") or 0) >= 0.10)
    if gop_hit is None:
        gop_hit = bool(gop.get("available") and gop.get("significant"))
    hits = []
    if dq_hit:
        hits.append("dq")
    if gop_hit:
        hits.append("gop")
    if rde.get("available") and rde.get("state") == "DETECTED":
        hits.append("rde")

    n_silent = 0
    if dq.get("available") and not dq_hit and \
            items[0]["status"] == "scored":
        n_silent += 1
    if gop.get("available") and not gop_hit:
        n_silent += 1
    if rde.get("available") and rde.get("state") in ("DETECTED", "DIAGNOSTIC") \
            and "rde" not in hits:
        n_silent += 1
    v, st, why = consistency_item(len(hits), n_silent)
    items.append({"key": "consistency", "label": "轴间一致性（修正项）",
                  "weight": W_CONS, "score": v, "status": st, "why": why})

    scored = [it for it in items if it["status"] == "scored" and it["score"] is not None]
    wsum = sum(it["weight"] for it in scored)
    if wsum <= 0:
        return {"score": None, "grade": "无法评估（无可用证据轴）",
                "coverage": 0.0, "items": items, "positive_axes": hits,
                "notes": [
                    "Every axis is unavailable, undecidable or theoretically "
                    "annihilated, so no score can be formed."]}

    total = sum(it["weight"] * it["score"] for it in scored) / wsum
    total = int(round(_clamp(total, 0, 100)))

    notes = []
    if wsum < 0.999:
        miss = [it["label"] for it in items
                if it["status"] != "scored" and it["key"] != "consistency"]
        notes.append("证据覆盖度 %.2f：%s 未参与打分（不可用或该引擎在此素材上"
                     "不可判），权重已在可用轴上重新归一化。"
                     % (wsum, "、".join(miss) or "部分轴"))
    best = max((it["score"] for it in scored if it["key"] != "consistency"),
               default=None)
    if best is not None and total + 15 < best:
        notes.append("最强单轴 %d 分，合成后 %d 分：合成值低是因为并非所有可判轴"
                     "都指向同一方向，而不是最强那条轴的证据变弱了。"
                     % (best, total))
    if not (gate and gate.get("measured") and gate.get("applies")):
        notes.append("适用性门控**未测**（无单压参考）：分数照出，但证书在该素材上"
                     "是否适用未知，不能当作鉴定结论使用。")
    notes.append("分数是证据强度指数，不是后验概率，也不是似然比。")

    return {"score": total, "grade": grade(total),
            "coverage": round(wsum, 3), "items": items, "notes": notes,
            "positive_axes": hits}


__all__ = ["scorecard", "dq_item", "gop_item", "rde_item", "grade"]
