#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal MP4 demuxer: recovers the H.264 elementary stream as Annex-B.

Extraction only replaces the AVCC 4-byte length prefix with a 00 00 00 01 start
code, so the bitstream stays byte-identical and no re-encode or external ffmpeg
is needed.

Non-fragmented MP4 only (moov up front), which covers x264/ffmpeg output.
Fragmented MP4 (fMP4, HLS) raises instead of returning wrong data.
"""
import struct
import os

def _boxes(data, start, end):
    """Yield (type, body_start, body_end) for each box in [start, end)."""
    i = start
    while i + 8 <= end:
        size = struct.unpack(">I", data[i:i + 4])[0]
        typ = data[i + 4:i + 8]
        head = 8
        if size == 1:                       # 64-bit largesize
            if i + 16 > end:
                return
            size = struct.unpack(">Q", data[i + 8:i + 16])[0]
            head = 16
        elif size == 0:                     # extends to EOF
            size = end - i
        if size < head or i + size > end:
            return
        yield typ, i + head, i + size
        i += size


def _find(data, start, end, want):
    for typ, bs, be in _boxes(data, start, end):
        if typ == want:
            return bs, be
    return None, None


def _find_path(data, start, end, path):
    """Descend a box path level by level; return (body_start, body_end) of the last box."""
    cur_s, cur_e = start, end
    for want in path:
        if isinstance(want, tuple):         # first match among the candidates
            for w in want:
                s, e = _find(data, cur_s, cur_e, w)
                if s is not None:
                    cur_s, cur_e = s, e
                    break
            else:
                return None, None
        else:
            s, e = _find(data, cur_s, cur_e, want)
            if s is None:
                return None, None
            cur_s, cur_e = s, e
    return cur_s, cur_e


def _video_trak(data, moov_s, moov_e):
    """Return (body_start, body_end) of the first trak whose hdlr is 'vide'."""
    for typ, bs, be in _boxes(data, moov_s, moov_e):
        if typ != b"trak":
            continue
        hs, he = _find_path(data, bs, be, [b"mdia", b"hdlr"])
        if hs is None:
            continue
        # hdlr body: version/flags(4) + pre_defined(4) + handler_type(4)
        if data[hs + 8:hs + 12] == b"vide":
            return bs, be
    return None, None


def _stsd_info(data, stbl_s, stbl_e):
    """Return (nal_length_size, codec_tag)."""
    ss, se = _find_path(data, stbl_s, stbl_e, [b"stsd"])
    if ss is None:
        return 4, None
    # stsd body: version/flags(4) + entry_count(4) + entries
    n = struct.unpack(">I", data[ss + 4:ss + 8])[0]
    i = ss + 8
    for _ in range(max(n, 1)):
        if i + 8 > se:
            break
        size = struct.unpack(">I", data[i:i + 4])[0]
        tag = data[i + 4:i + 8]
        if tag in (b"avc1", b"avc3", b"hev1", b"hvc1"):
            # 78 bytes from the avc1 body start to avcC
            j = i + 8 + 78
            cs, ce = _find(data, j, min(i + size, se), b"avcC")
            if cs is None:
                cs, ce = _find(data, j, min(i + size, se), b"hvcC")
            if cs is None:
                return 4, tag
            # avcC: configurationVersion(1) AVCProfileIndication(1) ...
            #      lengthSizeMinusOne is in the low 2 bits of byte 5
            return (data[cs + 4] & 0x03) + 1, tag
        i += size
    return 4, None


def parameter_sets(path):
    """
    Extract SPS / PPS from avcC as a list of Annex-B fragments (start codes included).

    Why this has to be fetched separately: x264 and most encoders keep SPS/PPS in the
    container's avcC (out-of-band), so they are absent from the sample data. Splicing
    samples alone yields a bare stream with no parameter sets, which breaks in two
    observed ways:
      1. entropy-coding detection cannot read the PPS, so CABAC is not identified and
         the CAVLC decoder runs to no purpose before dying on an IndexError, leaving
         the user with an uninformative error;
      2. the coefficient extractor has no SPS to build picture parameters from, so
         parsing is unreliable.
    MP4 files muxed from a raw stream with -c copy usually carry in-band parameter
    sets, which is why this path went unnoticed.

    avcC layout (ISO/IEC 14496-15):
      1B version, 1B profile, 1B compat, 1B level, 1B lengthSizeMinusOne (low 2 bits),
      1B numSPS (low 5 bits), [2B len + NAL]..., 1B numPPS, [2B len + NAL]...
    """
    samples, nal_len, tag = sample_table(path)
    if tag in (b"hev1", b"hvc1"):
        return []
    with open(path, "rb") as f:
        data = f.read()
    moov_s, moov_e = _find(data, 0, len(data), b"moov")
    if moov_s is None:
        return []
    tr = _video_trak(data, moov_s, moov_e)
    if tr is None:
        return []
    ss, se = _find_path(data, tr[0], tr[1], [b"mdia", b"minf", b"stbl"])
    if ss is None:
        return []
    xs, xe = _find_path(data, ss, se, [b"stsd"])
    if xs is None:
        return []
    n = struct.unpack(">I", data[xs + 4:xs + 8])[0]
    i = xs + 8
    for _ in range(max(n, 1)):
        if i + 8 > xe:
            break
        size = struct.unpack(">I", data[i:i + 4])[0]
        if data[i + 4:i + 8] in (b"avc1", b"avc3"):
            cs, ce = _find(data, i + 8 + 78, min(i + size, xe), b"avcC")
            if cs is None:
                return []
            out, p = [], cs + 5
            # SPS
            try:
                nsps = data[p] & 0x1F
                p += 1
                for _ in range(nsps):
                    ln = struct.unpack(">H", data[p:p + 2])[0]
                    p += 2
                    out.append(b"\x00\x00\x00\x01" + data[p:p + ln])
                    p += ln
                npps = data[p] & 0xFF
                p += 1
                for _ in range(npps):
                    ln = struct.unpack(">H", data[p:p + 2])[0]
                    p += 2
                    out.append(b"\x00\x00\x00\x01" + data[p:p + ln])
                    p += ln
            except Exception:
                return out
            return out
        i += size
    return []


def _stsz(data, stbl_s, stbl_e):
    """Return the byte size of each sample."""
    ss, se = _find_path(data, stbl_s, stbl_e, [b"stsz"])
    if ss is None:
        ss, se = _find_path(data, stbl_s, stbl_e, [b"stz2"])
        if ss is None:
            return None
        # stz2: field_size(1) at body+4
        fs = data[ss + 4]
        cnt = struct.unpack(">I", data[ss + 8:ss + 12])[0]
        out = []
        p = ss + 12
        if fs == 16:
            for _ in range(cnt):
                out.append(struct.unpack(">H", data[p:p + 2])[0]); p += 2
        elif fs == 8:
            out = list(data[p:p + cnt]); p += cnt
        else:
            raise ValueError("unsupported stz2 field_size=%d" % fs)
        return out
    sample_size = struct.unpack(">I", data[ss + 4:ss + 8])[0]
    cnt = struct.unpack(">I", data[ss + 8:ss + 12])[0]
    if sample_size != 0:                    # fixed sample size
        return [sample_size] * cnt
    return [struct.unpack(">I", data[ss + 12 + 4 * k:ss + 16 + 4 * k])[0]
            for k in range(cnt)]


def _chunk_offsets(data, stbl_s, stbl_e):
    ss, se = _find_path(data, stbl_s, stbl_e, [b"stco"])
    if ss is not None:
        cnt = struct.unpack(">I", data[ss + 4:ss + 8])[0]
        return [struct.unpack(">I", data[ss + 8 + 4 * k:ss + 12 + 4 * k])[0]
                for k in range(cnt)]
    ss, se = _find_path(data, stbl_s, stbl_e, [b"co64"])
    if ss is not None:
        cnt = struct.unpack(">I", data[ss + 4:ss + 8])[0]
        return [struct.unpack(">Q", data[ss + 8 + 8 * k:ss + 16 + 8 * k])[0]
                for k in range(cnt)]
    return None


def _stsc(data, stbl_s, stbl_e):
    """Return [(first_chunk, samples_per_chunk, sample_desc_index), ...]"""
    ss, se = _find_path(data, stbl_s, stbl_e, [b"stsc"])
    if ss is None:
        return None
    cnt = struct.unpack(">I", data[ss + 4:ss + 8])[0]
    out = []
    for k in range(cnt):
        p = ss + 8 + 12 * k
        out.append(struct.unpack(">III", data[p:p + 12]))
    return out


def sample_table(path):
    """Return (samples, nal_len_size, codec_tag); samples is [(offset, size), ...]."""
    with open(path, "rb") as f:
        data = f.read()
    moov_s, moov_e = _find(data, 0, len(data), b"moov")
    if moov_s is None:
        raise ValueError("no moov box (fragmented mp4 or not an mp4)")
    ts, te = _video_trak(data, moov_s, moov_e)
    if ts is None:
        raise ValueError("no video track")
    ss, se = _find_path(data, ts, te, [b"mdia", b"minf", b"stbl"])
    if ss is None:
        raise ValueError("no stbl")
    sizes = _stsz(data, ss, se)
    offs = _chunk_offsets(data, ss, se)
    stsc = _stsc(data, ss, se)
    if not sizes or not offs or not stsc:
        raise ValueError("incomplete sample table")
    nal_len, tag = _stsd_info(data, ss, se)

    # expand chunks into samples
    samples = []
    si = 0
    ci = 0
    n_chunk = len(offs)
    for c in range(n_chunk):
        while ci + 1 < len(stsc) and (c + 1) >= stsc[ci + 1][0]:
            ci += 1
        spc = stsc[ci][1]
        base = offs[c]
        pos = base
        for _ in range(spc):
            if si >= len(sizes):
                break
            samples.append((pos, sizes[si]))
            pos += sizes[si]
            si += 1
    return samples, nal_len, tag


def to_annexb(path):
    """
    Recover the H.264 stream inside an MP4 as Annex-B elementary bytes.
    Bitstream content is unchanged byte for byte; only length prefixes become
    start codes.
    """
    samples, nal_len, tag = sample_table(path)
    if tag in (b"hev1", b"hvc1"):
        raise ValueError("HEVC track: not supported by this H.264 pipeline")
    with open(path, "rb") as f:
        data = f.read()
    out = bytearray()
    for nal in parameter_sets(path):          # SPS/PPS from avcC must come first
        out += nal
    for off, size in samples:
        p = off
        end = off + size
        while p + nal_len <= end:
            if nal_len == 4:
                n = struct.unpack(">I", data[p:p + 4])[0]
            elif nal_len == 2:
                n = struct.unpack(">H", data[p:p + 2])[0]
            elif nal_len == 1:
                n = data[p]
            else:
                raise ValueError("bad nal_length_size=%d" % nal_len)
            p += nal_len
            if n <= 0 or p + n > end:
                break
            out += b"\x00\x00\x00\x01" + data[p:p + n]
            p += n
    return bytes(out)


def nal_frame_series(path):
    """
    Lightweight variant: return only the (size, nal_type) series, not the full
    elementary stream. The GOP phase axis needs frame sizes and types only, so this
    uses about half the memory of building the full stream.
    """
    samples, nal_len, tag = sample_table(path)
    with open(path, "rb") as f:
        data = f.read()
    sizes, types = [], []
    for off, size in samples:
        p = off
        end = off + size
        # one sample holds one slice NAL; the largest VCL NAL is taken as the frame
        best = None
        while p + nal_len <= end:
            if nal_len == 4:
                n = struct.unpack(">I", data[p:p + 4])[0]
            elif nal_len == 2:
                n = struct.unpack(">H", data[p:p + 2])[0]
            else:
                n = data[p]
            p += nal_len
            if n <= 0 or p + n > end:
                break
            t = data[p] & 0x1F
            if t in (1, 5):
                if best is None or n > best[0]:
                    best = (n, t)
            p += n
        if best is not None:
            sizes.append(float(best[0]))
            types.append(best[1])
    return sizes, types
