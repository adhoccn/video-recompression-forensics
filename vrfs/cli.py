#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line entry point.

    python -m vrfs.cli analyze  <file> [--reference REF] [--qp1 23 --qp2 18]
    python -m vrfs.cli r-axis   23 18
    python -m vrfs.cli serve    [--port 8081]

Runs offline; nothing leaves the machine.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fmt(r):
    d = r["decision"]
    print("file    : %s" % r["file"])
    st = r["stream"]
    print("stream  : %s, %d bytes, %d VCL NAL"
          % (st["how"], st["bytes"], st["n_vcl"]))
    g = r.get("gate")
    if not g:
        print("gate    : NOT MEASURED (no single-compression reference)")
    elif not g.get("measured"):
        print("gate    : NOT MEASURED (%s)" % g.get("note", ""))
    else:
        print("gate    : %-6s D_single = %.3e  (threshold %.1e)"
              % ("OPEN" if g["applies"] else "CLOSED",
                 g["d_single"], g["threshold"]))
    dq = r["dq"]
    if dq.get("available"):
        print("dq axis : score %.6f (peak %.6f + viol %.6f), %d coeffs, "
              "%d IDR, entropy=%s, cone distance %.3e"
              % (dq["score"], dq["peak"], dq["viol"], dq["n_coeff"],
                 dq["n_idr"], dq["entropy"], dq.get("d_cone") or 0.0))
    else:
        print("dq axis : UNAVAILABLE (%s)" % dq.get("reason", ""))
    gp = r["gop"]
    if gp.get("available"):
        print("gop axis: p=%s z=%.4f  max-T p=%.4f (%s perms, %s cands)  %s"
              % (gp["blind_p"], gp["blind_z"],
                 gp["p_perm"] if gp.get("p_perm") is not None else float("nan"),
                 gp.get("n_perm"), gp.get("n_candidates"),
                 "SIGNIFICANT" if gp.get("significant") else "not significant"))
    else:
        print("gop axis: UNAVAILABLE (%s)" % gp.get("reason", ""))
    ra = r.get("r_axis")
    if ra:
        print("r axis  : QP1=%d QP2=%d r=%.4f 1/r=%.4f  regime=%s"
              % (ra["qp1"], ra["qp2"], ra["r"], ra["inv_r"], ra["regime"]))
    rd = r.get("rde") or {}
    if rd.get("available"):
        print("pixel   : engine v%s (%s mode) state=%s confidence=%s dip_z=%s"
              % (rd.get("engine_version"), rd.get("mode"), rd.get("state"),
                 rd.get("confidence"), rd.get("dip_z")))
    else:
        print("pixel   : UNAVAILABLE (%s)" % rd.get("reason", ""))
    print("verdict : %s (confidence %s, gate %s)"
          % (d["verdict"], d.get("confidence", "n/a"),
             d.get("gate_status", "n/a")))
    sc = d.get("scorecard") or {}
    if sc:
        print("score   : %s / 100  %s (coverage %s, positive axes: %s)"
              % (sc.get("score") if sc.get("score") is not None else "n/a",
                 sc.get("grade", ""), sc.get("coverage"),
                 ", ".join(sc.get("positive_axes") or []) or "none"))
        for it in sc.get("items", []):
            print("          %-12s w=%.2f  %-6s [%s]"
                  % (it["key"], it["weight"],
                     it["score"] if it["score"] is not None else "-",
                     it["status"]))
        for n in sc.get("notes", []):
            print("        * %s" % n)
    for t in d.get("reasons", []):
        print("        - %s" % t)


def main(argv=None):
    from vrfs import pipeline, cone
    ap = argparse.ArgumentParser(prog="vrfs", description=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    a = sub.add_parser("analyze")
    a.add_argument("file")
    a.add_argument("--reference", default=None,
                   help="single-compression reference (enables the gate)")
    a.add_argument("--qp1", type=int, default=None)
    a.add_argument("--qp2", type=int, default=None)
    a.add_argument("--gop1", type=int, default=None)
    a.add_argument("--gop2", type=int, default=15)
    a.add_argument("--n-perm", type=int, default=500)
    a.add_argument("--no-rde", action="store_true",
                   help="skip the pixel-domain axis; faster, fewer evidence chains")
    a.add_argument("--json", action="store_true")

    r = sub.add_parser("r-axis")
    r.add_argument("qp1", type=int)
    r.add_argument("qp2", type=int)

    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8081)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--open", action="store_true", help="open the console in a browser")

    args = ap.parse_args(argv)
    if args.cmd == "analyze":
        res = pipeline.analyze(args.file, reference=args.reference,
                               qp1=args.qp1, qp2=args.qp2,
                               gop1=args.gop1, gop2=args.gop2,
                               n_perm=args.n_perm,
                               use_rde=not args.no_rde)
        res.get("dq", {}).pop("hist", None)
        if args.json:
            print(json.dumps(res, indent=1, default=str))
        else:
            _fmt(res)
        return 0
    if args.cmd == "r-axis":
        out = cone.r_axis(args.qp1, args.qp2)
        out["period6_family_from_qp1"] = cone.period6_family(args.qp1)
        print(json.dumps(out, indent=1))
        return 0
    if args.cmd == "serve":
        os.environ.setdefault("VRFS_WORK_DIR",
                              os.path.join(os.path.expanduser("~"), ".vrfs", "work"))
        from vrfs.server import app                       # noqa: E402
        import uvicorn
        url = "http://127.0.0.1:%d/" % args.port
        print("VRFS local console: %s" % url)
        print("(offline: nothing leaves this machine)")
        if args.open:
            import threading
            import webbrowser
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
