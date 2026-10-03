# -*- coding: utf-8 -*-
"""读取 Houdini 旧版 BGEO v5（.bhclassic / nonNewtonCode 导出的 .bgeo.gz）点数据，与 nonNewtonCode/extern/my_partio/io/BGEO.cpp 一致。"""
import gzip
import struct

import numpy as np


def read_bhclassic(path):
    """返回 dict：'position' -> (N,3) float32，以及其余点属性（float/int/vector）。"""
    with open(path, "rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    with (gzip.open if compressed else open)(path, "rb") as f:
        buf = f.read()
    o = 0

    def rd(fmt):
        nonlocal o
        v = struct.unpack_from(">" + fmt, buf, o)
        o += struct.calcsize(">" + fmt)
        return v

    magic = buf[:4]
    o = 4
    if magic != b"Bgeo":
        raise ValueError(f"{path}: not a BGEO v5 file (magic={magic!r})")
    _, version, n_pts, _n_prims, _n_pgrp = rd("ciiii")
    _n_prgrp, n_pattr, _n_vattr, _n_prattr, _n_attr = rd("iiiii")
    if version != 5:
        raise ValueError(f"{path}: BGEO version {version} != 5")
    attrs = [("position", 0, 4, "f")]  # P 为齐次 4 分量
    size = 4
    for _ in range(n_pattr):
        (ln,) = rd("H")
        name = buf[o:o + ln].decode()
        o += ln
        cnt, htype = rd("Hi")
        if htype not in (0, 1, 5):
            raise ValueError(f"{path}: unsupported attribute type {htype} for '{name}'")
        o += 4 * cnt  # 默认值
        attrs.append((name, size, cnt, "i" if htype == 1 else "f"))
        size += cnt
    raw = np.frombuffer(buf, ">i4", n_pts * size, o).reshape(n_pts, size)
    out = {}
    for name, off, cnt, kind in attrs:
        col = raw[:, off:off + cnt]
        col = col.view(">f4").astype(np.float32) if kind == "f" else col.astype(np.int32)
        if name == "position":
            col = col[:, :3]
        out[name] = col[:, 0] if cnt == 1 and name != "position" else col
    return out
