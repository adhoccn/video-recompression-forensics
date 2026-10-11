#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os as _os, sys as _sys
_p = _os.path.dirname(_os.path.abspath(__file__))
while True:
    _c = _os.path.join(_p, 'script', '_common')
    if _os.path.isdir(_c):
        _sys.path.insert(0, _c)
        break
    _n = _os.path.dirname(_p)
    if _n == _p:
        break
    _p = _n
# ^ locate the canonical front-end copies in script/_common
"""H.264 CABAC transform coefficient extractor (task #57).

Standalone rewrite of the CAVLC path from _ref/py-h264-decoder: CABAC neighbour
contexts (coded_block_flag / mvd / ref_idx / cbp) are entirely different from
the CAVLC nC, and forcing the reference decoder's Block/Macroblock structures
is more error-prone, so a lightweight syntax walk is implemented here.

Scope (guaranteed by the data generation side; violations raise instead of
failing silently):
  Main profile, 4:2:0, frame-only (no MBAFF/field coding), no B slices,
  no 8x8 transform (not allowed in Main), single slice group, no data
  partitioning.

Strong self-check: every slice must hit end_of_slice_flag=1 exactly at the
PicSizeInMbs-th macroblock. Arithmetic decoding avalanches immediately once it
loses sync, so passing this check == bit-exact entropy decoding.
"""
import os
import numpy as np

from cabac_engine import Cabac, init_contexts, CabacError

# Table 9-11 ctxIdxOffset ----
CBF_BASE, SIG_BASE, LAST_BASE, ABS_BASE = 85, 105, 166, 227
CAT_OFF_CBF = (0, 4, 8, 12, 16)
CAT_OFF_SIG = (0, 15, 29, 44, 47)
CAT_OFF_ABS = (0, 10, 20, 30, 39)
MAXNUMCOEFF = (16, 15, 16, 4, 15)

class NotSupported(Exception):
    pass

# bitstream helpers

def split_nalus(data):
    """Split NAL units on start codes, strip emulation prevention bytes,
    return [(nal_unit_type, rbsp_bytes)]."""
    out = []
    n = len(data)
    starts = []
    i = 0
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0:
            break
        starts.append(j + 3)
        i = j + 3
    for k, s in enumerate(starts):
        e = starts[k + 1] - 3 if k + 1 < len(starts) else n
        # drop trailing 0x00 that may belong to the next start code
        while e > s and data[e - 1] == 0:
            e -= 1
        raw = data[s:e]
        if not raw:
            continue
        # remove emulation prevention (00 00 03 -> 00 00)
        if b"\x00\x00\x03" in raw:
            buf = bytearray()
            p = 0
            L = len(raw)
            while p < L:
                if p + 2 < L and raw[p] == 0 and raw[p + 1] == 0 and raw[p + 2] == 3:
                    buf += b"\x00\x00"
                    p += 3
                else:
                    buf.append(raw[p])
                    p += 1
            raw = bytes(buf)
        out.append((raw[0] & 0x1F, raw))
    return out

class BitReader:
    __slots__ = ("d", "n", "p")

    def __init__(self, data, bitpos=0):
        self.d = data
        self.n = len(data) * 8
        self.p = bitpos

    def u(self, k):
        v = 0
        d, p = self.d, self.p
        for _ in range(k):
            v = (v << 1) | ((d[p >> 3] >> (7 - (p & 7))) & 1)
            p += 1
        self.p = p
        return v

    def u1(self):
        p = self.p
        self.p = p + 1
        return (self.d[p >> 3] >> (7 - (p & 7))) & 1

    def ue(self):
        z = 0
        while self.u1() == 0:
            z += 1
            if z > 32:
                raise NotSupported("bad ue")
        return (1 << z) - 1 + (self.u(z) if z else 0)

    def se(self):
        k = self.ue()
        return (k + 1) >> 1 if (k & 1) else -(k >> 1)

    def more_rbsp_data(self):
        # find the last 1 (rbsp_stop_one_bit)
        d = self.d
        last = len(d) * 8 - 1
        while last >= 0 and ((d[last >> 3] >> (7 - (last & 7))) & 1) == 0:
            last -= 1
        return self.p < last

    def byte_align_pos(self):
        return (self.p + 7) >> 3

# parameter sets

class SPSMin:
    def __init__(self, r):
        self.profile_idc = r.u(8)
        r.u(8)                       # constraint flags + reserved
        self.level_idc = r.u(8)
        self.sps_id = r.ue()
        self.chroma_format_idc = 1
        self.separate_colour_plane_flag = 0
        if self.profile_idc in (100, 110, 122, 244, 44, 83, 86, 118,
                                128, 138, 139, 134, 135):
            self.chroma_format_idc = r.ue()
            if self.chroma_format_idc == 3:
                self.separate_colour_plane_flag = r.u1()
            r.ue(); r.ue()           # bit_depth luma/chroma minus8
            r.u1()                   # qpprime_y_zero_transform_bypass_flag
            if r.u1():               # seq_scaling_matrix_present_flag
                raise NotSupported("seq scaling matrix")
        self.log2_max_frame_num = r.ue() + 4
        self.poc_type = r.ue()
        self.log2_max_poc_lsb = 0
        if self.poc_type == 0:
            self.log2_max_poc_lsb = r.ue() + 4
        elif self.poc_type == 1:
            self.delta_pic_order_always_zero_flag = r.u1()
            r.se(); r.se()
            for _ in range(r.ue()):
                r.se()
        self.max_num_ref_frames = r.ue()
        r.u1()                       # gaps_in_frame_num_value_allowed_flag
        self.pic_width_in_mbs = r.ue() + 1
        self.pic_height_in_map_units = r.ue() + 1
        self.frame_mbs_only_flag = r.u1()
        self.mbaff = 0 if self.frame_mbs_only_flag else r.u1()
        if not self.frame_mbs_only_flag:
            raise NotSupported("field/MBAFF coding")
        self.PicHeightInMbs = self.pic_height_in_map_units
        self.PicSizeInMbs = self.pic_width_in_mbs * self.PicHeightInMbs
        self.ChromaArrayType = (0 if self.separate_colour_plane_flag
                                else self.chroma_format_idc)

class PPSMin:
    def __init__(self, r):
        self.pps_id = r.ue()
        self.sps_id = r.ue()
        self.entropy_coding_mode_flag = r.u1()
        self.bottom_field_pic_order_in_frame_present_flag = r.u1()
        self.num_slice_groups_minus1 = r.ue()
        if self.num_slice_groups_minus1 > 0:
            raise NotSupported("multiple slice groups")
        self.num_ref_idx_l0_default_active_minus1 = r.ue()
        self.num_ref_idx_l1_default_active_minus1 = r.ue()
        self.weighted_pred_flag = r.u1()
        self.weighted_bipred_idc = r.u(2)
        self.pic_init_qp = 26 + r.se()
        r.se()                       # pic_init_qs_minus26
        self.chroma_qp_index_offset = r.se()
        self.deblocking_filter_control_present_flag = r.u1()
        self.constrained_intra_pred_flag = r.u1()
        self.redundant_pic_cnt_present_flag = r.u1()
        self.transform_8x8_mode_flag = 0
        if r.more_rbsp_data():
            self.transform_8x8_mode_flag = r.u1()
            if r.u1():               # pic_scaling_matrix_present_flag
                raise NotSupported("pic scaling matrix")
            r.se()

# slice header

SLICE_NAME = {0: "P", 1: "B", 2: "I", 3: "SP", 4: "SI"}

class SliceHeader:
    def __init__(self, r, sps, pps_map, nal_type, nal_ref_idc):
        self.first_mb = r.ue()
        st = r.ue()
        self.slice_type = SLICE_NAME[st % 5]
        self.pps_id = r.ue()
        pps = pps_map[self.pps_id]
        self.pps = pps
        if sps.separate_colour_plane_flag:
            r.u(2)
        r.u(sps.log2_max_frame_num)          # frame_num
        idr = (nal_type == 5)
        if idr:
            r.ue()                            # idr_pic_id
        if sps.poc_type == 0:
            r.u(sps.log2_max_poc_lsb)
            if pps.bottom_field_pic_order_in_frame_present_flag:
                r.se()
        elif sps.poc_type == 1 and not sps.delta_pic_order_always_zero_flag:
            r.se()
            if pps.bottom_field_pic_order_in_frame_present_flag:
                r.se()
        if pps.redundant_pic_cnt_present_flag:
            r.ue()
        if self.slice_type == "B":
            raise NotSupported("B slice")
        self.num_ref_idx_l0_active_minus1 = pps.num_ref_idx_l0_default_active_minus1
        if self.slice_type in ("P", "SP"):
            if r.u1():                        # num_ref_idx_active_override_flag
                self.num_ref_idx_l0_active_minus1 = r.ue()
        # ref_pic_list_modification
        if self.slice_type not in ("I", "SI"):
            if r.u1():
                while True:
                    idc = r.ue()
                    if idc == 3:
                        break
                    r.ue()
        # pred_weight_table
        if pps.weighted_pred_flag and self.slice_type in ("P", "SP"):
            r.ue()                            # luma_log2_weight_denom
            if sps.ChromaArrayType != 0:
                r.ue()
            for _ in range(self.num_ref_idx_l0_active_minus1 + 1):
                if r.u1():
                    r.se(); r.se()
                if sps.ChromaArrayType != 0 and r.u1():
                    for _ in range(2):
                        r.se(); r.se()
        # dec_ref_pic_marking
        if nal_ref_idc != 0:
            if idr:
                r.u1(); r.u1()
            else:
                if r.u1():                    # adaptive_ref_pic_marking_mode_flag
                    while True:
                        op = r.ue()
                        if op == 0:
                            break
                        if op in (1, 3):
                            r.ue()
                        if op == 2:
                            r.ue()
                        if op in (3, 6):
                            r.ue()
                        if op == 4:
                            r.ue()
        self.cabac_init_idc = 0
        if pps.entropy_coding_mode_flag and self.slice_type not in ("I", "SI"):
            self.cabac_init_idc = r.ue()
        self.slice_qpy = pps.pic_init_qp + r.se()
        if pps.deblocking_filter_control_present_flag:
            idc = r.ue()
            if idc != 1:
                r.se(); r.se()

# macroblock layer

def _blk_xy(b):
    """luma4x4BlkIdx -> (x, y), in units of 4x4 blocks (6.4.3)."""
    return ((((b >> 2) & 1) << 1) | (b & 1),
            (((b >> 3) & 1) << 1) | ((b >> 1) & 1))

def _xy_blk(x, y):
    return (((((y >> 1) << 1) | (x >> 1)) << 2) |
            (((y & 1) << 1) | (x & 1)))

_BLK_XY = [_blk_xy(b) for b in range(16)]

class MBInfo:
    __slots__ = ("kind", "skip", "i16", "ipcm", "intra", "cbp_l", "cbp_c",
                 "cbf_ldc", "cbf_l", "cbf_cdc", "cbf_cac",
                 "mvd", "refidx", "predflag", "ichroma", "qpy")

    def __init__(self):
        self.kind = "NA"
        self.skip = False
        self.i16 = False
        self.ipcm = False
        self.intra = False
        self.cbp_l = 0
        self.cbp_c = 0
        self.cbf_ldc = 0
        self.cbf_l = [0] * 16
        self.cbf_cdc = [0, 0]
        self.cbf_cac = [[0] * 4, [0] * 4]
        self.mvd = [[0, 0] for _ in range(16)]
        self.refidx = [0, 0, 0, 0]
        self.predflag = [0, 0, 0, 0]
        self.ichroma = 0
        self.qpy = 26

class SliceDecoder:
    """CABAC syntax walk for one slice + luma coefficient collection."""

    def __init__(self, sh, sps, cab, collect):
        self.sh = sh
        self.sps = sps
        self.c = cab
        self.W = sps.pic_width_in_mbs
        self.total = sps.PicSizeInMbs
        self.first = sh.first_mb
        self.mbs = {}
        self.prev_qp_delta = 0
        self.qpy = sh.slice_qpy
        self.is_i = (sh.slice_type in ("I", "SI"))
        self.collect = collect          # callback: (kind, coeffs, qp)
        self.n_mb = 0
        self.stats = {}                 # macroblock type histogram (debug/sanity)

    # neighbours
    def mbA(self, addr):
        if addr % self.W == 0:
            return None
        return self.mbs.get(addr - 1)

    def mbB(self, addr):
        return self.mbs.get(addr - self.W)

    def nbr_luma4x4(self, addr, blk, d):
        """Return (MBInfo|None, blkIdxN)."""
        x, y = _BLK_XY[blk]
        if d == "A":
            x -= 1
            if x < 0:
                return self.mbA(addr), _xy_blk(3, y)
        else:
            y -= 1
            if y < 0:
                return self.mbB(addr), _xy_blk(x, 3)
        return self.mbs.get(addr), _xy_blk(x, y)

    def nbr_chroma4x4(self, addr, blk, d):
        x, y = blk & 1, blk >> 1
        if d == "A":
            x -= 1
            if x < 0:
                return self.mbA(addr), (y << 1) | 1
        else:
            y -= 1
            if y < 0:
                return self.mbB(addr), (1 << 1) | x
        return self.mbs.get(addr), (y << 1) | x

    # main loop
    def run(self):
        c = self.c
        addr = self.first
        while True:
            if addr >= self.total:
                raise CabacError("mb address overflow")
            mb = MBInfo()
            self.mbs[addr] = mb
            more = True
            if not self.is_i:
                # mb_skip_flag  ctxIdxOffset 11
                a, b = self.mbA(addr), self.mbB(addr)
                inc = ((1 if (a is not None and not a.skip) else 0) +
                       (1 if (b is not None and not b.skip) else 0))
                if c.decode_decision(11 + inc):
                    mb.skip = True
                    mb.kind = "P_Skip"
                    mb.predflag = [1, 1, 1, 1]
                    mb.qpy = self.qpy
                    self.prev_qp_delta = 0
                    more = False
            if more:
                self.macroblock_layer(addr, mb)
            self.stats[mb.kind] = self.stats.get(mb.kind, 0) + 1
            self.n_mb += 1
            if c.decode_terminate():
                break
            addr += 1
        return addr

    # mb_type
    def _mb_type_intra_suffix(self, off, addr, first_inc=None):
        """Intra mb_type tree (Table 9-36). off=3 (I slice) or 17 (P slice suffix).
        Returns (kind, i16_pred, cbp_l, cbp_c)."""
        c = self.c
        if off == 3:
            b0 = c.decode_decision(3 + first_inc)
        else:
            b0 = c.decode_decision(17)
        if b0 == 0:
            return "I_NxN", None, 0, 0
        if c.decode_terminate():
            return "I_PCM", None, 0, 0
        # Table 9-39 ctxIdxInc: binIdx2->3/1, binIdx3->4/2,
        #   binIdx4-> (b3?5:6)/(b3?2:3), binIdx5-> (b3?6:7)/3, binIdx6->7/3
        if off == 3:
            C2, C3, C4_y, C4_n, C5_y, C5_n, C6 = 6, 7, 8, 9, 9, 10, 10
        else:
            C2, C3, C4_y, C4_n, C5_y, C5_n, C6 = 18, 19, 19, 20, 20, 20, 20
        bl = c.decode_decision(C2)                 # binIdx2: cbp_luma
        b3 = c.decode_decision(C3)                 # binIdx3: cbp_chroma != 0
        if b3:
            cbpc = 1 + c.decode_decision(C4_y)     # binIdx4
            p1 = c.decode_decision(C5_y)           # binIdx5
            p2 = c.decode_decision(C6)             # binIdx6
        else:
            cbpc = 0
            p1 = c.decode_decision(C4_n)           # binIdx4
            p2 = c.decode_decision(C5_n)           # binIdx5
        return "I_16x16", (p1 << 1) | p2, (15 if bl else 0), cbpc

    def macroblock_layer(self, addr, mb):
        c = self.c
        sh = self.sh
        npart = 0
        sub = None
        if self.is_i:
            a, b = self.mbA(addr), self.mbB(addr)
            inc = ((1 if (a is not None and a.kind != "I_NxN") else 0) +
                   (1 if (b is not None and b.kind != "I_NxN") else 0))
            kind, i16p, cbpl, cbpc = self._mb_type_intra_suffix(3, addr, inc)
        else:
            if c.decode_decision(14) == 0:         # inter
                b1 = c.decode_decision(15)
                b2 = c.decode_decision(16 if b1 != 1 else 17)
                if b1 == 0:
                    kind = "P_8x8" if b2 else "P_16x16"
                else:
                    kind = "P_16x8" if b2 else "P_8x16"
                i16p, cbpl, cbpc = None, 0, 0
                npart = {"P_16x16": 1, "P_16x8": 2, "P_8x16": 2, "P_8x8": 4}[kind]
            else:
                kind, i16p, cbpl, cbpc = self._mb_type_intra_suffix(17, addr)
        mb.kind = kind
        mb.intra = kind.startswith("I")
        mb.i16 = (kind == "I_16x16")
        mb.ipcm = (kind == "I_PCM")
        if mb.ipcm:
            raise NotSupported("I_PCM macroblock")

        if mb.intra:
            if kind == "I_NxN":
                for _ in range(16):
                    if c.decode_decision(68) == 0:      # prev_intra4x4_pred_mode_flag
                        c.decode_decision(69)           # rem_intra4x4_pred_mode: 3 bin FL
                        c.decode_decision(69)
                        c.decode_decision(69)
            # intra_chroma_pred_mode, ctxIdxOffset 64
            a, b = self.mbA(addr), self.mbB(addr)
            inc = ((1 if (a is not None and a.intra and not a.ipcm and a.ichroma != 0) else 0) +
                   (1 if (b is not None and b.intra and not b.ipcm and b.ichroma != 0) else 0))
            v = 0
            if c.decode_decision(64 + inc):
                v = 1
                if c.decode_decision(67):
                    v = 2
                    if c.decode_decision(67):
                        v = 3
            mb.ichroma = v
        else:
            mb.predflag = [1, 1, 1, 1]
            if kind == "P_8x8":
                sub = []
                for _ in range(4):
                    if c.decode_decision(21):
                        sub.append("8x8")
                    elif c.decode_decision(22) == 0:
                        sub.append("8x4")
                    else:
                        sub.append("4x8" if c.decode_decision(23) else "4x4")
                self._inter_pred_8x8(addr, mb, sub)
            else:
                self._inter_pred(addr, mb, kind, npart)

        # coded_block_pattern
        if not mb.i16:
            cbpl, cbpc = self._cbp(addr, mb)
        mb.cbp_l, mb.cbp_c = cbpl, cbpc

        if cbpl or cbpc or mb.i16:
            d = self._mb_qp_delta()
            self.qpy = (self.qpy + d + 52) % 52
            self.prev_qp_delta = d
        else:
            self.prev_qp_delta = 0
        mb.qpy = self.qpy
        if cbpl or cbpc or mb.i16:
            self._residual(addr, mb)

    # inter prediction parameters
    def _part_blocks(self, kind, i):
        """Return the 4x4 coordinates (x, y) of the partition top-left block
        and the list of 4x4 blocks it covers."""
        if kind == "P_16x16":
            return (0, 0), list(range(16))
        if kind == "P_16x8":
            y = i * 2
            return (0, y), [_xy_blk(x, y + dy) for dy in (0, 1) for x in range(4)]
        if kind == "P_8x16":
            x = i * 2
            return (x, 0), [_xy_blk(x + dx, y) for y in range(4) for dx in (0, 1)]
        raise AssertionError

    def _ref_idx(self, addr, mb, x, y, blks):
        if self.sh.num_ref_idx_l0_active_minus1 <= 0:
            r = 0
        else:
            c = self.c
            inc = 0
            for d, w in (("A", 1), ("B", 2)):
                nb, nblk = self.nbr_luma4x4(addr, _xy_blk(x, y), d)
                if nb is None or nb.intra or nb.skip:
                    continue
                nx, ny = _BLK_XY[nblk]
                p8 = (ny >> 1) * 2 + (nx >> 1)
                if nb.predflag[p8] and nb.refidx[p8] > 0:
                    inc += w
            r = 0
            if c.decode_decision(54 + inc):
                r = 1
                while r < 32 and c.decode_decision(54 + (4 if r == 1 else 5)):
                    r += 1
        p8 = (y >> 1) * 2 + (x >> 1)
        mb.refidx[p8] = r
        if len(blks) >= 8:           # 16x16/16x8/8x16 span multiple 8x8
            for b in blks:
                bx, by = _BLK_XY[b]
                mb.refidx[(by >> 1) * 2 + (bx >> 1)] = r
        return r

    def _mvd(self, addr, mb, x, y, blks):
        c = self.c
        for comp in (0, 1):
            base = 40 + comp * 7
            s = 0
            for d in ("A", "B"):
                nb, nblk = self.nbr_luma4x4(addr, _xy_blk(x, y), d)
                if nb is None or nb.intra or nb.skip:
                    continue
                s += abs(nb.mvd[nblk][comp])
            inc = 0 if s < 3 else (1 if s <= 32 else 2)
            pre = 0
            if c.decode_decision(base + inc):
                pre = 1
                while pre < 9:
                    ci = base + (3 if pre == 1 else (4 if pre == 2 else
                                                     (5 if pre == 3 else 6)))
                    if c.decode_decision(ci) == 0:
                        break
                    pre += 1
            val = pre
            if pre == 9:
                val = 9 + c.bypass_egk(3)
            if val != 0 and c.decode_bypass():
                val = -val
            for b in blks:
                mb.mvd[b][comp] = val

    def _inter_pred(self, addr, mb, kind, npart):
        parts = [self._part_blocks(kind, i) for i in range(npart)]
        for (x, y), blks in parts:
            self._ref_idx(addr, mb, x, y, blks)
        for (x, y), blks in parts:
            self._mvd(addr, mb, x, y, blks)

    def _inter_pred_8x8(self, addr, mb, sub):
        for i in range(4):
            x, y = (i & 1) * 2, (i >> 1) * 2
            blks = [_xy_blk(x + dx, y + dy) for dy in (0, 1) for dx in (0, 1)]
            self._ref_idx(addr, mb, x, y, blks)
        for i in range(4):
            x0, y0 = (i & 1) * 2, (i >> 1) * 2
            s = sub[i]
            if s == "8x8":
                parts = [((x0, y0), [_xy_blk(x0 + dx, y0 + dy)
                                     for dy in (0, 1) for dx in (0, 1)])]
            elif s == "8x4":
                parts = [((x0, y0 + k), [_xy_blk(x0 + dx, y0 + k) for dx in (0, 1)])
                         for k in (0, 1)]
            elif s == "4x8":
                parts = [((x0 + k, y0), [_xy_blk(x0 + k, y0 + dy) for dy in (0, 1)])
                         for k in (0, 1)]
            else:
                parts = [((x0 + dx, y0 + dy), [_xy_blk(x0 + dx, y0 + dy)])
                         for dy in (0, 1) for dx in (0, 1)]
            for (x, y), blks in parts:
                self._mvd(addr, mb, x, y, blks)

    # coded_block_pattern (9.3.3.1.1.4) ----
    def _cbp(self, addr, mb):
        c = self.c
        cbpl = 0
        for b8 in range(4):
            x, y = (b8 & 1) * 2, (b8 >> 1) * 2
            inc = 0
            for d, w in (("A", 1), ("B", 2)):
                nb, nblk = self.nbr_luma4x4(addr, _xy_blk(x, y), d)
                if nb is None or nb.ipcm:
                    continue                      # condTermFlagN = 0
                nx, ny = _BLK_XY[nblk]
                n8 = (ny >> 1) * 2 + (nx >> 1)
                if nb is mb:
                    has = (cbpl >> n8) & 1        # bins already decoded in this macroblock
                else:
                    has = 0 if nb.skip else ((nb.cbp_l >> n8) & 1)
                if not has:
                    inc += w
            if c.decode_decision(73 + inc):
                cbpl |= (1 << b8)
        # chroma
        inc = 0
        for d, w in (("A", 1), ("B", 2)):
            nb = self.mbA(addr) if d == "A" else self.mbB(addr)
            if nb is None:
                continue
            if nb.ipcm or (not nb.skip and nb.cbp_c != 0):
                inc += w
        cbpc = 0
        if c.decode_decision(77 + inc):
            inc2 = 0
            for d, w in (("A", 1), ("B", 2)):
                nb = self.mbA(addr) if d == "A" else self.mbB(addr)
                if nb is None:
                    continue
                if nb.ipcm or (not nb.skip and nb.cbp_c == 2):
                    inc2 += w
            cbpc = 2 if c.decode_decision(81 + inc2) else 1
        return cbpl, cbpc

    # mb_qp_delta (9.3.3.1.1.5) ----
    def _mb_qp_delta(self):
        c = self.c
        inc = 1 if self.prev_qp_delta != 0 else 0
        if c.decode_decision(60 + inc) == 0:
            return 0
        k = 1
        while k < 88 and c.decode_decision(60 + (2 if k == 1 else 3)):
            k += 1
        return (k + 1) >> 1 if (k & 1) else -(k >> 1)

    # coded_block_flag ctxIdxInc (9.3.3.1.1.9) ----
    def _cbf_inc(self, addr, mb, cat, idx, comp=0):
        inc = 0
        for d, w in (("A", 1), ("B", 2)):
            if cat in (0, 3):
                nb = self.mbA(addr) if d == "A" else self.mbB(addr)
                nblk = 0
            elif cat in (1, 2):
                nb, nblk = self.nbr_luma4x4(addr, idx, d)
            else:
                nb, nblk = self.nbr_chroma4x4(addr, idx, d)
            if nb is None:
                if mb.intra:
                    inc += w          # unavailable and in the current frame -> 1
                continue
            if nb.ipcm:
                inc += w
                continue
            if nb.skip:
                continue
            # transBlockN availability
            if cat == 0:
                v = nb.cbf_ldc if nb.i16 else 0
                if not nb.i16:
                    continue
            elif cat in (1, 2):
                n8 = nblk >> 2
                if not ((nb.cbp_l >> n8) & 1):
                    continue
                v = nb.cbf_l[nblk]
            elif cat == 3:
                if nb.cbp_c == 0:
                    continue
                v = nb.cbf_cdc[comp]
            else:
                if nb.cbp_c != 2:
                    continue
                v = nb.cbf_cac[comp][nblk]
            if v:
                inc += w
        return inc

    # residual_block_cabac (7.3.5.3.3) ----
    def _residual_block(self, addr, mb, cat, idx, comp=0):
        c = self.c
        maxc = MAXNUMCOEFF[cat]
        inc = self._cbf_inc(addr, mb, cat, idx, comp)
        cbf = c.decode_decision(CBF_BASE + CAT_OFF_CBF[cat] + inc)
        # record cbf for neighbours
        if cat == 0:
            mb.cbf_ldc = cbf
        elif cat in (1, 2):
            mb.cbf_l[idx] = cbf
        elif cat == 3:
            mb.cbf_cdc[comp] = cbf
        else:
            mb.cbf_cac[comp][idx] = cbf
        if not cbf:
            return None
        sigb = SIG_BASE + CAT_OFF_SIG[cat]
        lastb = LAST_BASE + CAT_OFF_SIG[cat]
        sig = [0] * maxc
        numc = maxc
        i = 0
        while i < numc - 1:
            ci = min(i, 2) if cat == 3 else i
            if c.decode_decision(sigb + ci):
                sig[i] = 1
                if c.decode_decision(lastb + ci):
                    numc = i + 1
                    break
            i += 1
        sig[numc - 1] = 1
        # decode levels in reverse order
        absb = ABS_BASE + CAT_OFF_ABS[cat]
        cap = 4 - (1 if cat == 3 else 0)
        n_gt1 = 0
        n_eq1 = 0
        coeffs = [0] * maxc
        for i in range(numc - 1, -1, -1):
            if not sig[i]:
                continue
            ci = (0 if n_gt1 else min(4, 1 + n_eq1))
            pre = 0
            if c.decode_decision(absb + ci):
                pre = 1
                while pre < 14:
                    if c.decode_decision(absb + 5 + min(cap, n_gt1)) == 0:
                        break
                    pre += 1
            val = pre + 1
            if pre == 14:
                val = 15 + c.bypass_egk(0)
            if c.decode_bypass():
                val = -val
            coeffs[i] = val
            if abs(val) > 1:
                n_gt1 += 1
            elif n_gt1 == 0:
                n_eq1 += 1
        return coeffs

    def _residual(self, addr, mb):
        cbpl, cbpc = mb.cbp_l, mb.cbp_c
        if mb.i16:
            self._residual_block(addr, mb, 0, 0)
        for i8 in range(4):
            for i4 in range(4):
                b = i8 * 4 + i4
                if (cbpl >> i8) & 1:
                    if mb.i16:
                        cf = self._residual_block(addr, mb, 1, b)
                        self.collect(1, cf, mb.qpy)
                    else:
                        cf = self._residual_block(addr, mb, 2, b)
                        self.collect(2, cf, mb.qpy)
                else:
                    mb.cbf_l[b] = 0
        if cbpc:
            for comp in (0, 1):
                self._residual_block(addr, mb, 3, 0, comp)
        if cbpc == 2:
            for comp in (0, 1):
                for i4 in range(4):
                    self._residual_block(addr, mb, 4, i4, comp)

# top level

def extract_all_coeffs(path, max_frames=None, verbose=False):
    """Parse a CABAC annexb .264 stream; return a dict with the same layout as
    bs_pframe.extract_all_coeffs."""
    with open(path, "rb") as f:
        data = f.read()
    sps = None
    pps_map = {}
    lv_rows, is16_rows, qp_rows, fr_rows = [], [], [], []
    cur_frame = -1
    n_slices = 0
    check = {"expected_mbs": None, "bad_slices": 0, "frames_info": []}

    for ntype, rbsp in split_nalus(data):
        if ntype == 7:
            sps = SPSMin(BitReader(rbsp, 8))
        elif ntype == 8:
            p = PPSMin(BitReader(rbsp, 8))
            pps_map[p.pps_id] = p
            if not p.entropy_coding_mode_flag:
                raise NotSupported("stream is CAVLC, use bs_pframe instead")
        elif ntype in (1, 5):
            if sps is None or not pps_map:
                continue
            r = BitReader(rbsp, 8)
            nal_ref_idc = (rbsp[0] >> 5) & 3
            sh = SliceHeader(r, sps, pps_map, ntype, nal_ref_idc)
            if sh.first_mb == 0:
                cur_frame += 1
                if max_frames is not None and cur_frame >= max_frames:
                    cur_frame -= 1
                    break
            # cabac_alignment_one_bit -> byte align
            bytepos = r.byte_align_pos()
            pstate, vmps = init_contexts(sh.slice_type in ("I", "SI"),
                                         sh.cabac_init_idc, sh.slice_qpy)
            cab = Cabac(rbsp, bytepos, pstate, vmps)

            rows = []

            def collect(kind, cf, qp, _rows=rows):
                if cf is None:
                    return
                row = np.zeros(16, np.int16)
                if kind == 1:                 # I16x16 AC: 15 coefficients into [1:16]
                    row[1:16] = cf[:15]
                    _rows.append((row, 1, qp))
                else:                          # 4x4 luma: 16 coefficients
                    row[:] = cf[:16]
                    _rows.append((row, 0, qp))

            dec = SliceDecoder(sh, sps, cab, collect)
            exp = sps.PicSizeInMbs
            check["expected_mbs"] = exp
            try:
                last = dec.run()
                ok = (last == exp - 1) and (dec.n_mb == exp - sh.first_mb)
            except (CabacError, NotSupported, IndexError) as e:
                ok = False
                if verbose:
                    print(f"  [!] slice @frame {cur_frame}: {e}")
            if not ok:
                check["bad_slices"] += 1
                if verbose:
                    print(f"  [!] frame {cur_frame}: parsed {dec.n_mb} MBs "
                          f"(expected {exp - sh.first_mb})")
                continue
            for row, i16, qp in rows:
                lv_rows.append(row)
                is16_rows.append(i16)
                qp_rows.append(qp)
                fr_rows.append(cur_frame)
            n_slices += 1
            if verbose:
                check["frames_info"].append(
                    (cur_frame, sh.slice_type, dec.n_mb, sh.slice_qpy,
                     dict(sorted(dec.stats.items()))))

    return {
        "levels": (np.vstack(lv_rows) if lv_rows
                   else np.zeros((0, 16), np.int16)),
        "is16": np.asarray(is16_rows, np.uint8),
        "qp": np.asarray(qp_rows, np.int16),
        "frame": np.asarray(fr_rows, np.int16),
        "levels8": np.zeros((0, 64), np.int16),
        "n_frames": cur_frame + 1,
        "n_slices": n_slices,
        "check": check,
    }

if __name__ == "__main__":
    import sys, time
    p = sys.argv[1]
    mf = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    t0 = time.time()
    r = extract_all_coeffs(p, max_frames=mf, verbose=True)
    lv = r["levels"]
    nz = lv[lv != 0]
    print(f"file={os.path.basename(p)}  {time.time()-t0:.1f}s")
    print(f"frames={r['n_frames']} slices_ok={r['n_slices']} "
          f"bad={r['check']['bad_slices']} mbs/frame={r['check']['expected_mbs']}")
    print(f"blocks={lv.shape[0]} nonzero={nz.size}")
    if nz.size:
        v, cnt = np.unique(np.abs(nz), return_counts=True)
        print("top |level|:", [(int(a), int(b)) for b, a in
                               sorted(zip(cnt, v), reverse=True)[:10]])
    for fi in r["check"]["frames_info"][:8]:
        print("  frame", fi)
