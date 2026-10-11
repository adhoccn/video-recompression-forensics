#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge to the pixel-domain engine (RDE), used as the third evidence axis.

The pixel domain reads the distortion residual across probe QPs 16-44; the
bitstream domain reads coefficient histograms and frame sizes. The evidence
sources do not overlap and their decision boundaries differ (pixel domain
r* = 1.768, bitstream domain r > 1).

  1. Only calls the RDE /v1/detect endpoint; never modifies the other side.
  2. Unreachable means the axis is missing; the other two still decide.
  3. The other side's wording is passed through unchanged.

Base URL resolution:
  1. VRFS_RDE_BASE (explicit)
  2. http://127.0.0.1:8080 (same host)
  3. VRFS_RDE_PUBLIC (fallback when no local process exists)
  4. none reachable => axis unavailable
VRFS_RDE_OFF=1 disables this axis.
"""
import os
import ssl
import json
import time
import uuid
import urllib.request
import urllib.error

DEFAULT_LOCAL = "http://127.0.0.1:8080"
DEFAULT_PUBLIC = os.environ.get("VRFS_RDE_PUBLIC", "")

_CANDIDATES = ()
_probe_cache = {"ts": 0.0, "base": None}
PROBE_TTL = 30.0
DETECT_TIMEOUT = 180

STATE_ZH = {
    "DETECTED": "检出",
    "UNDECIDABLE": "不可判",
    "DIAGNOSTIC": "诊断（非鉴定级）",
    "NOT_DETECTED": "未检出",
}


def _candidates():
    if os.environ.get("VRFS_RDE_OFF"):
        return ()
    explicit = os.environ.get("VRFS_RDE_BASE")
    if explicit:
        return (explicit.rstrip("/"),)
    return tuple(x for x in (DEFAULT_LOCAL, DEFAULT_PUBLIC) if x)


def _ssl_ctx():
    """Only verdict results are fetched, so self-signed endpoints are accepted."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _get(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "vrfs-bridge/1.0"})
    ctx = _ssl_ctx() if url.startswith("https://") else None
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.status, r.read().decode("utf-8", "replace")


def probe(force=False):
    """Find a live RDE. Returns base (without trailing slash) or None."""
    now = time.time()
    if not force and _probe_cache["base"] and now - _probe_cache["ts"] < PROBE_TTL:
        return _probe_cache["base"]
    found = None
    for base in _candidates():
        try:
            st, body = _get(base + "/health", 1.5)
            if st == 200 and '"status"' in body:
                try:
                    info = json.loads(body)
                except ValueError:
                    info = {}
                # only treat /health with calibration as the decision engine
                if info.get("calibration") or info.get("version"):
                    found = base
                    break
        except Exception:
            continue
    _probe_cache["ts"] = now
    _probe_cache["base"] = found
    return found


def status():
    """For /health and the page: which RDE is connected and its declared
    operating point.
    """
    base = probe()
    if not base:
        return {"available": False,
                "reason": "no RDE engine reachable (tried %s)"
                          % (", ".join(_candidates()) or "disabled by VRFS_RDE_OFF")}
    try:
        st, body = _get(base + "/health", 3.0)
        h = json.loads(body) if st == 200 else {}
    except Exception as e:
        return {"available": False, "base": base,
                "reason": "health check failed (%s: %s)" % (type(e).__name__, e)}
    cal = h.get("calibration") or {}
    out = {"available": True, "base": base, "version": h.get("version"),
           "tau_deploy": cal.get("tau_deploy"), "alpha": cal.get("alpha"),
           "r_star": cal.get("r_star"),
           "holdout_fpr_emp": (cal.get("holdout_fpr") or {}).get("emp"),
           "busy_slots": h.get("busy_slots")}
    return out


def _field(boundary, name, path):
    with open(path, "rb") as f:
        payload = f.read()
    return (("--%s\r\n" % boundary).encode()
            + ('Content-Disposition: form-data; name="%s"; filename="%s"\r\n'
               % (name, os.path.basename(path))).encode()
            + b"Content-Type: application/octet-stream\r\n\r\n"
            + payload + b"\r\n")


def _multipart(path, reference=None):
    """Build multipart by hand to avoid adding a requests dependency for one call."""
    boundary = "----vrfs%s" % uuid.uuid4().hex
    body = _field(boundary, "files", path)
    if reference:
        body += _field(boundary, "reference", reference)
    body += ("--%s--\r\n" % boundary).encode()
    return body, "multipart/form-data; boundary=%s" % boundary


def detect(path, reference=None, base=None, timeout=DETECT_TIMEOUT):
    """
    Call the pixel-domain engine /v1/detect.

    reference given ⇒ paired, identification-grade; no reference ⇒ profile,
    diagnostic-grade (the other side declares it is not an identification
    conclusion). That distinction has to reach the user.
    """
    base = base or probe()
    if not base:
        return {"available": False, "reason": "RDE engine not reachable"}
    body, ctype = _multipart(path, reference)
    req = urllib.request.Request(base + "/v1/detect", data=body,
                                 headers={"Content-Type": ctype,
                                          "User-Agent": "vrfs-bridge/1.0"})
    ctx = _ssl_ctx() if base.startswith("https://") else None
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            raw = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return {"available": False, "base": base,
                "reason": "RDE returned HTTP %s" % e.code}
    except Exception as e:
        return {"available": False, "base": base,
                "reason": "RDE call failed (%s: %s)" % (type(e).__name__, e)}
    try:
        d = json.loads(raw)
    except ValueError:
        return {"available": False, "base": base,
                "reason": "RDE returned non-JSON"}
    res = (d.get("results") or [{}])[0]
    st = res.get("statistic") or {}
    out = {
        "available": True,
        "base": base,
        "engine_version": res.get("engine_version") or d.get("engine_version"),
        "mode": res.get("mode") or d.get("mode"),
        "state": res.get("state"),
        "state_zh": res.get("state_zh") or STATE_ZH.get(res.get("state"), res.get("state")),
        "confidence": res.get("confidence"),
        "threshold_used": res.get("threshold_used"),
        "primary_key": st.get("primary_key"),
        "primary_name": st.get("name"),
        "n": st.get("n"),
        "dz": st.get("dz"),
        "p_sign": st.get("p_sign"),
        "dip_z": st.get("dip_z"),
        "p_star": st.get("p_star"),
        "baseline_drop": st.get("baseline_drop"),
        "n_frames": st.get("n_frames"),
        "probe_engine": st.get("probe_engine"),
        "r_star": (res.get("boundary") or {}).get("r_star"),
        "mdes": (res.get("budget") or {}).get("mdes"),
        "reasons": res.get("reasons") or [],
        "note": res.get("note") or "",
        "elapsed_s": round(time.time() - t0, 2),
        # profile kept as numbers for plotting; excluded from the main JSON
        "profile": st.get("profile") or {},    }
    return out


__all__ = ["probe", "status", "detect", "STATE_ZH"]
