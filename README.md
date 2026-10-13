# Video Recompression Forensics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23338056.svg)](https://doi.org/10.5281/zenodo.23338056)

Detect double compression in H.264 video from three independent evidence axes,
with an applicability gate and a 0-100 evidence-strength score.

No model weights, no GPU, no network calls.

## Install

```bash
pip install -r requirements.txt
```

## Usage

Command line:

```bash
python -m vrfs.cli analyze video.mp4
python -m vrfs.cli analyze suspect.mp4 --reference single.mp4
python -m vrfs.cli r-axis 23 18
```

Web UI (same engine, runs offline):

```bash
python -m vrfs.cli serve --port 8081 --open
```

HTTP service:

```bash
python -m vrfs.server
# GET  /health
# GET  /v1/r_axis?qp1=23&qp2=18
# POST /v1/analyze
# POST /v1/batch
```

## Three axes

| Axis | Signal | Entropy decoding | Limits |
|---|---|---|---|
| DQ comb | convex peaks and monotonicity violations in the AC `\|level\|` histogram | required (CAVLC) | CAVLC only; CABAC returns unavailable |
| GOP phase | periodicity of the frame-size series at candidate periods | not required | needs >= 60 VCL frames |
| Pixel domain (RDE) | distortion residual before/after re-encoding | not required | optional; needs a reachable RDE endpoint |

## Three verdicts

| Verdict | Meaning |
|---|---|
| `DOUBLE_COMPRESSION_INDICATED` | double compression detected |
| `NO_EVIDENCE` | no evidence; not proof of single compression |
| `NOT_APPLICABLE` | method does not apply to this input; no score emitted |

The second verdict matters: at the idempotent point `r = 1` and at the period-6
undetectable points, the comb is provably annihilated. Measuring nothing there
is the expected outcome, not a negative result.

The gate runs before any verdict. When the single-compression histogram is too
far from the completely-monotone cone, the tool reports not applicable instead
of producing an unsupported number.

## Score

The 0-100 score answers how strong the evidence is. It is not a posterior
probability, not a likelihood ratio, and not an accuracy figure. Weights and
per-axis raw statistics are returned alongside it.

`r = Qstep(QP1) / Qstep(QP2)` is the single degree of freedom driving the
coefficient-domain axis. `r = 1` and `r = 1/2, 1/4, 1/8` are the predicted
blind spots.

## Layout

```
vrfs/
  cone.py       completely-monotone cone, cone distance, applicability gate, r axis
  dq_axis.py    coefficient-domain axis
  gop_axis.py   frame-size-domain axis
  rde_bridge.py pixel-domain axis (optional)
  pipeline.py   gate -> per-axis measurement -> fusion
  score.py      evidence-strength scorecard
  mp4box.py     minimal MP4 demuxer (Annex-B extraction, no re-encode)
  server.py     FastAPI service
  cli.py        command-line entry point
  vendor/       H.264 CABAC parser (self-contained) and CAVLC extractor front end
```

## Dependencies

The CABAC parser under `vendor/` is self-contained and needs only numpy.

The CAVLC coefficient extractor (`vendor/bs_extract.py`) additionally imports
`bitstring` plus six modules (`h264bits`, `nalsps`, `nalpps`, `nalslice`,
`macroblock`, `block`) from the third-party project
[halochou/py-h264-decoder](https://github.com/halochou/py-h264-decoder). That
project is not redistributed here because it declares no open-source license.
To enable the DQ axis, obtain it yourself and put those modules on
`PYTHONPATH`, for example:

```bash
pip install bitstring
export PYTHONPATH=/path/to/py-h264-decoder
```

Without them the DQ axis reports `unavailable` and the remaining axes still
produce a verdict with reduced reported coverage.

## Cite

See `CITATION.cff`, or use the DOI badge at the top once a Zenodo DOI is minted.

## License

MIT. See `LICENSE`.
