#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTTP service.

  GET  /                     forensic station page (no external assets)
  GET  /health               engine self-check
  GET  /v1/r_axis            r axis regime query
  GET  /v1/samples           built-in sample list
  GET  /v1/rde               pixel-domain engine status
  POST /v1/analyze           single-file analysis
  POST /v1/analyze_sample    analyze a built-in sample by name
  POST /v1/batch             one reference plus many suspects

Uploads land in a work directory and are deleted after the verdict.
No score is produced when the gate fails.
"""
import os
import re
import sys
import json
import time
import shutil
import tempfile
import traceback

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vrfs import pipeline, cone, rde_bridge, __version__          # noqa: E402

APP_VERSION = "1.1.0"

WORK_DIR = os.environ.get("VRFS_WORK_DIR", "/var/lib/vrfs/work")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# sample dir: static/samples first, then env, then the deployed sample dir.
_LOCAL_SAMPLES = os.path.join(STATIC_DIR, "samples")
SAMPLES_DIR = os.environ.get("VRFS_SAMPLES_DIR") or (
    _LOCAL_SAMPLES if os.path.isdir(_LOCAL_SAMPLES)
    else "/opt/rde/app/static/samples")
MAX_UPLOAD_MB = int(os.environ.get("VRFS_MAX_UPLOAD_MB", "400"))
MAX_FILES = int(os.environ.get("VRFS_MAX_FILES", "8"))

os.makedirs(WORK_DIR, exist_ok=True)

app = FastAPI(title="VRFS", version=APP_VERSION,
              description="Video Recompression Forensic Station: "
                          "bitstream dual-axis analysis with an applicability gate.")

_SAMPLE_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.(mp4|264|h264|mkv|mov)$")


def _safe(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(name or "file"))


def _save_upload(f, dst_dir, used):
    """Save an upload; a duplicate name gets a suffix. Otherwise same-named
    files overwrite each other and two results actually judge one file.
    """
    base = _safe(f.filename)
    stem, ext = os.path.splitext(base)
    n, name = 1, base
    while name in used:
        n += 1
        name = "%s__%d%s" % (stem, n, ext)
    used.add(name)
    path = os.path.join(dst_dir, name)
    with open(path, "wb") as out:
        shutil.copyfileobj(f.file, out)
    return path, base


def _cleanup(d):
    shutil.rmtree(d, ignore_errors=True)


def _analyze(path, ref_path, qp1, qp2, gop1, gop2, n_perm, use_rde=True):
    # intermediates must go to WORK_DIR, not the sample dir: the extractor writes a
    # temporary .264 next to the input, which would litter static/samples and be
    # picked up as input by anything scanning that directory.
    d = tempfile.mkdtemp(dir=WORK_DIR, prefix="w_")
    try:
        return pipeline.analyze(path, reference=ref_path, qp1=qp1, qp2=qp2,
                                gop1=gop1, gop2=gop2, n_perm=n_perm,
                                workdir=d, use_rde=use_rde)
    finally:
        _cleanup(d)


def _strip_hist(d):
    """The histogram is returned only when plotting is requested; it is dropped
    by default to keep responses small.
    """
    if isinstance(d, dict) and "hist" in d:
        d.pop("hist", None)
    return d


@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
def index():
    p = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(p):
        return PlainTextResponse("VRFS API is running. Web UI not installed.")
    return FileResponse(p)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return PlainTextResponse("", status_code=204)


# built-in samples (with manifest.json); served directly here for local runs.
if os.path.isdir(SAMPLES_DIR):
    try:
        from fastapi.staticfiles import StaticFiles
        app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
    except Exception:                                   # pragma: no cover
        pass


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "engine_version": __version__,
        "work_dir": WORK_DIR,
        "samples_dir": SAMPLES_DIR if os.path.isdir(SAMPLES_DIR) else None,
        "samples_available": len(_sample_files()),
        "limits": {"max_upload_mb": MAX_UPLOAD_MB, "max_files": MAX_FILES},
        "axes": {
            "dq_comb": {"entropy": "CAVLC only",
                        "note": "CABAC streams return unavailable"},
            "gop_phase": {"entropy": "any",
                          "note": "needs >= 60 VCL frames"},
            "pixel_domain": {
                "engine": "RDE (read-only bridge)",
                **rde_bridge.status()},
        },
        "gate_threshold_D_single": cone.GATE_D_SINGLE,
    }


@app.get("/v1/rde")
def rde_status():
    """Report which engine the pixel-domain third axis reached and its declared
    operating point. Takes no file.
    """
    return rde_bridge.status()


@app.get("/v1/r_axis")
def r_axis(qp1: int, qp2: int):
    if not (0 <= qp1 <= 51 and 0 <= qp2 <= 51):
        raise HTTPException(400, "qp1/qp2 must be in 0..51")
    r = cone.r_axis(qp1, qp2)
    r["period6_family_from_qp1"] = cone.period6_family(qp1)
    return r


def _sample_files():
    if not os.path.isdir(SAMPLES_DIR):
        return []
    out = []
    for n in sorted(os.listdir(SAMPLES_DIR)):
        if _SAMPLE_RE.match(n) and n.endswith((".mp4", ".264", ".h264")):
            out.append(n)
    return out


@app.get("/v1/samples")
def samples():
    """Built-in sample list; samples are analyzed by name, without re-uploading."""
    return {"dir": SAMPLES_DIR if os.path.isdir(SAMPLES_DIR) else None,
            "files": _sample_files()}


@app.post("/v1/analyze_sample")
def analyze_sample(name: str = Form(...),
                   reference: str = Form(None),
                   qp1: int = Form(None), qp2: int = Form(None),
                   gop1: int = Form(None), gop2: int = Form(15),
                   n_perm: int = Form(500),
                   use_rde: bool = Form(True),
                   with_hist: bool = Form(False)):
    if not _SAMPLE_RE.match(name):
        raise HTTPException(400, "bad sample name")
    path = os.path.join(SAMPLES_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(404, "sample not found")
    ref_path = None
    if reference:
        if not _SAMPLE_RE.match(reference):
            raise HTTPException(400, "bad reference name")
        ref_path = os.path.join(SAMPLES_DIR, reference)
        if not os.path.exists(ref_path):
            raise HTTPException(404, "reference not found")
    t0 = time.time()
    try:
        r = _analyze(path, ref_path, qp1, qp2, gop1, gop2, n_perm, use_rde)
    except Exception as e:
        return JSONResponse({"error": "%s: %s" % (type(e).__name__, e),
                             "trace": traceback.format_exc()[-1500:]},
                            status_code=500)
    if not with_hist:
        _strip_hist(r.get("dq", {}))
    r["elapsed_s"] = round(time.time() - t0, 2)
    r["source"] = {"kind": "sample", "name": name}
    return r


@app.post("/v1/analyze")
def analyze(file: UploadFile = File(...),
            reference: UploadFile = File(None),
            qp1: int = Form(None), qp2: int = Form(None),
            gop1: int = Form(None), gop2: int = Form(15),
            n_perm: int = Form(500),
            use_rde: bool = Form(True),
            with_hist: bool = Form(False)):
    d = tempfile.mkdtemp(dir=WORK_DIR, prefix="a_")
    try:
        used = set()
        path, _ = _save_upload(file, d, used)
        ref_path = None
        if reference is not None:
            ref_path, _ = _save_upload(reference, d, used)
        t0 = time.time()
        r = _analyze(path, ref_path, qp1, qp2, gop1, gop2, n_perm, use_rde)
        if not with_hist:
            _strip_hist(r.get("dq", {}))
        r["elapsed_s"] = round(time.time() - t0, 2)
        r["source"] = {"kind": "upload"}
        return r
    except Exception as e:
        return JSONResponse({"error": "%s: %s" % (type(e).__name__, e),
                             "trace": traceback.format_exc()[-1500:]},
                            status_code=500)
    finally:
        _cleanup(d)


@app.post("/v1/batch")
def batch(reference: UploadFile = File(None),
          files: list[UploadFile] = File(...),
          qp1: int = Form(None), qp2: int = Form(None),
          gop1: int = Form(None), gop2: int = Form(15),
          n_perm: int = Form(40), use_rde: bool = Form(False)):
    if len(files) > MAX_FILES:
        raise HTTPException(400, "at most %d files" % MAX_FILES)
    d = tempfile.mkdtemp(dir=WORK_DIR, prefix="b_")
    try:
        used = set()
        ref_path = None
        if reference is not None:
            ref_path, _ = _save_upload(reference, d, used)
        out = []
        for f in files:
            p, orig = _save_upload(f, d, used)
            t0 = time.time()
            try:
                r = _analyze(p, ref_path, qp1, qp2, gop1, gop2, n_perm, use_rde)
                _strip_hist(r.get("dq", {}))
                r["elapsed_s"] = round(time.time() - t0, 2)
                r["label"] = orig
            except Exception as e:
                r = {"file": orig, "error": "%s: %s" % (type(e).__name__, e)}
            out.append(r)
        return {"count": len(out), "results": out}
    finally:
        _cleanup(d)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1",
                port=int(os.environ.get("VRFS_PORT", "8081")))
