#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight H.264 CAVLC transform coefficient extractor (A2 path, task #19).
Reuses the CAVLC tables and parsing algorithm of _ref/py-h264-decoder, except:
- only NAL types 7(SPS)/8(PPS)/5(IDR slice) are parsed; P frames/SEI are skipped
  (NAL entropy decoding is self-contained, no desync risk)
- no pixel reconstruction at all (no intra_pred/inter_pred/deblock/idct)
- 3 bugs in the reference code are fixed (all fatal for our streams):
  1) Block.ce_nc: chroma AC blocks of Intra_16x16 macroblocks had nC derived
     from luma neighbours, desyncing the whole stream.
     Rewritten: chroma AC always goes through the chroma_neighbor path
     (per spec 9.2.1).
  2) Slice.mb_to_slice_group_map: hardcoded [None]*100, out of range beyond
     100 macroblocks.
     Rewritten: num_slice_groups==1 produces an all-zero map directly.
  3) Slice default attributes: deblock offsets / num_ref_idx are absent in
     some streams -> AttributeError.
Output: luma quantized levels per IDR frame (I4x4 full 16 coefficients zigzag /
I16x16 AC 15 coefficients) + QP_Y per macroblock.
Baseline profile (profile_idc 66) + CAVLC only, guaranteed by the generation side.
"""
import os, sys
import numpy as np

def _find_ref_decoder():
    here = os.path.dirname(os.path.abspath(__file__))
    p = here
    while True:
        for name in ('py-h264-decoder', 'ref_cavlc_pyh264decoder'):
            for cand in (os.path.join(p, '_ref', name), os.path.join(p, name)):
                if os.path.isdir(cand):
                    return cand
        parent = os.path.dirname(p)
        if parent == p:
            return os.path.join(here, '_ref', 'py-h264-decoder')
        p = parent

_REF = _find_ref_decoder()
sys.path.insert(0, _REF)

from bitstring import BitArray, BitStream  # noqa: E402
from h264bits import H264Bits              # noqa: E402
from nalsps import SPS                     # noqa: E402
from nalpps import PPS                     # noqa: E402
import nalslice                            # noqa: E402
import macroblock as mbmod                 # noqa: E402
import block as blkmod                     # noqa: E402

def _patched_ce_nc(self):
    if self.color != "Y" and self.ad_type == "DC":
        return -1 if self.slice.sps.ChromaArrayType == 1 else -2
    if self.color == "Y":
        (mbAddrA, blkIdxA) = self.luma_neighbor("A")
        blkA = None if mbAddrA is None else self.slice.mbs[mbAddrA].luma_blocks[blkIdxA]
        (mbAddrB, blkIdxB) = self.luma_neighbor("B")
        blkB = None if mbAddrB is None else self.slice.mbs[mbAddrB].luma_blocks[blkIdxB]
    else:  # chroma AC: always use chroma neighbours (the reference code wrongly uses luma for Intra_16x16)
        c = 0 if self.color == "Cb" else 1
        (mbAddrA, blkIdxA) = self.chroma_neighbor("A")
        blkA = None if mbAddrA is None else self.slice.mbs[mbAddrA].chroma_ac_blocks[c][blkIdxA]
        (mbAddrB, blkIdxB) = self.chroma_neighbor("B")
        blkB = None if mbAddrB is None else self.slice.mbs[mbAddrB].chroma_ac_blocks[c][blkIdxB]
    nA = nB = 0
    availA = mbAddrA is not None
    availB = mbAddrB is not None
    if availA:
        mbA = self.slice.mbs[mbAddrA]
        if mbA.mb_type in ("P_Skip", "B_Skip") or \
           (mbA.mb_type != "I_PCM" and self.all_ac_zeros(mbAddrA, blkIdxA)):
            nA = 0
        else:
            nA = blkA.TotalCoeff
    if availB:
        mbB = self.slice.mbs[mbAddrB]
        if mbB.mb_type in ("P_Skip", "B_Skip") or \
           (mbB.mb_type != "I_PCM" and self.all_ac_zeros(mbAddrB, blkIdxB)):
            nB = 0
        else:
            nB = blkB.TotalCoeff
    if availA and availB:
        return (nA + nB + 1) >> 1
    if availA:
        return nA
    if availB:
        return nB
    return 0

blkmod.Block.ce_nc = _patched_ce_nc

def _patched_sg_map(self):
    if self.pps.num_slice_groups_minus1 == 0:
        self.MbToSliceGroupMap = [0] * int(self.PicSizeInMbs)
    else:
        raise NotImplementedError("multi slice-group not supported (x264 never emits)")

nalslice.Slice.mb_to_slice_group_map = _patched_sg_map

SPS.Extended_SAR = 255  # spec Table E-1; the reference code refers to an undefined self.Extended_SAR
nalslice.Slice.slice_alpha_c0_offset_div2 = 0
nalslice.Slice.slice_beta_offset_div2 = 0

_orig_header = nalslice.Slice.slice_header
def _patched_header(self):
    _orig_header(self)
    if not hasattr(self, "num_ref_idx_l0_active_minus1"):
        self.num_ref_idx_l0_active_minus1 = self.pps.num_ref_idx_l0_default_active_minus1
nalslice.Slice.slice_header = _patched_header

mbmod.Macroblock.calculate_mv = lambda self: None  # coefficient extraction does not need MVs

class _VPS:
    def __init__(self):
        self.active_sps = None
        self.active_pps = None
        self.last_has_mmco_5 = 0
        self.PreviousFrameNum = 0
        self.PreviousFrameNumOffset = 0
        self.FrameNumOffset = 0
        self.max_frame_num = 0

def extract_idr_coeffs(path, max_idr=None, verbose=False):
    """Parse an annexb .264 stream, return a dict:
       levels  (N,16) int16  luma quantized levels, I16x16 AC stored at [1:16]
                            ([0]=0 placeholder)
       is16    (N,)   uint8  1=I16x16 AC block, 0=I4x4 block
       qp      (N,)   int16  QP_Y of the owning macroblock
       frames  (N,)   int16  IDR frame index
       n_mbs_expected/parsed consistency check info
    """
    with open(path, "rb") as f:
        nalus = list(BitArray(f).split("0x000001", bytealigned=True))[1:]
    sps = None
    ppss = []
    vps = _VPS()
    lv_rows, is16_rows, is8_rows, qp_rows, fr_rows = [], [], [], [], []
    lv8_rows, qp8_rows, fr8_rows = [], [], []   # T2: 8x8 transform blocks (64 coefficients, interleaved)
    n_idr = 0
    check = {"expected_mbs": None, "bad_frames": 0}
    for ba in nalus:
        ba.replace("0x000003", "0x0000", bytealigned=True)
        nb = H264Bits(BitStream(ba))
        params = {"forbidden_zero_bit": nb.f(1),
                  "nal_ref_idc": nb.u(2),
                  "nal_unit_type": nb.u(5)}
        t = params["nal_unit_type"]
        if t == 7:
            sps = SPS(nb, params=params)
        elif t == 8:
            ppss.append(PPS(nb, sps=sps, params=params))
        elif t == 5:  # IDR only
            if max_idr is not None and n_idr >= max_idr:
                break
            sl = nalslice.Slice(nb, sps=sps, ppss=ppss, params=params, vps=vps)
            exp = int(sl.PicSizeInMbs)
            check["expected_mbs"] = exp
            if len(sl.mbs) != exp:  # strong check for CAVLC desync
                check["bad_frames"] += 1
                if verbose:
                    print(f"  [!] frame {n_idr}: parsed {len(sl.mbs)} of {exp} MBs")
                n_idr += 1
                continue
            for mb in sl.mbs:
                qp = mb.QP_Y
                # T2: 8x8 transform macroblock (I_NxN + transform_size_8x8_flag=1).
                # under CAVLC residuals are coded per 4x4 sub-block and residual_luma is interleaved into
                # four 8x8 blocks of 64 coefficients, kept in levels8 and not mixed into the 16-wide levels.
                is8 = 1 if getattr(mb, "transform_size_8x8_flag", 0) else 0
                if is8:
                    for lv8 in mb.luma8x8_levels:
                        lv8_rows.append(np.asarray(lv8, np.int16))
                        qp8_rows.append(qp); fr8_rows.append(n_idr)
                    continue
                if mb.pred_mode == "Intra_16x16":
                    for blk in mb.luma_blocks:  # 15 AC (zigzag 1..15)
                        row = np.zeros(16, np.int16)
                        row[1:16] = blk.coeffLevel[:15]
                        lv_rows.append(row); is16_rows.append(1); is8_rows.append(0)
                        qp_rows.append(qp);  fr_rows.append(n_idr)
                else:  # Intra_4x4 / Intra_8x8: 16 coefficients (zigzag 0..15)
                    for blk in mb.luma_blocks:
                        lv_rows.append(np.asarray(blk.coeffLevel[:16], np.int16))
                        is16_rows.append(0); is8_rows.append(is8)
                        qp_rows.append(qp); fr_rows.append(n_idr)
            if verbose:
                print(f"  frame {n_idr}: {exp} MBs ok, QP[0]={sl.mbs[0].QP_Y}")
            n_idr += 1
    return {
        "levels": np.vstack(lv_rows) if lv_rows else np.zeros((0, 16), np.int16),
        "is16": np.asarray(is16_rows, np.uint8),
        "is8": np.asarray(is8_rows, np.uint8),
        "qp": np.asarray(qp_rows, np.int16),
        "frames": np.asarray(fr_rows, np.int16),
        # T2: 8x8 transform blocks (64 coefficients/block, 4 blocks/MB)
        "levels8": np.vstack(lv8_rows) if lv8_rows else np.zeros((0, 64), np.int16),
        "qp8": np.asarray(qp8_rows, np.int16),
        "frames8": np.asarray(fr8_rows, np.int16),
        "n_idr": n_idr, "check": check,
    }

if __name__ == "__main__":
    import time
    path = sys.argv[1]
    max_idr = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    t0 = time.time()
    r = extract_idr_coeffs(path, max_idr=max_idr, verbose=True)
    dt = time.time() - t0
    lv = r["levels"]
    nz = lv[lv != 0]
    print(f"file={os.path.basename(path)}")
    print(f"IDR frames parsed: {r['n_idr']}, bad: {r['check']['bad_frames']}, "
          f"expected MBs/frame: {r['check']['expected_mbs']}, time: {dt:.1f}s")
    print(f"blocks: {lv.shape[0]}, nonzero coeffs: {nz.size}")
    print(f"QP range: {r['qp'].min()}..{r['qp'].max()}" if r["qp"].size else "no data")
    if nz.size:
        vals, cnts = np.unique(np.abs(nz), return_counts=True)
        top = sorted(zip(cnts, vals), reverse=True)[:10]
        print("top |level| counts:", [(int(v), int(c)) for c, v in top])
