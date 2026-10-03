# -*- coding: utf-8 -*-
"""Taichi（constraint_solver.py 导出的 usdc）与 nonNewtonCode（MyPartio 导出的 bgeo.gz 序列）逐帧数值对比。

用法见 SPH/ALIGN_CPP.md。两边都按 25 fps 导出，坐标系一致（C++ 场景坐标），对比每个物体的
质心、包围盒、平均速度、平均黏度、平均温度、平均应变率随时间的曲线，输出 csv 与 png。
"""
import argparse
import csv
import glob
import os
import re

import numpy as np

from bgeo_io import read_bhclassic


def cpp_frames(cpp_dir, ids):
    """Stream (frame, (position, attributes, body)) for the shared C++ frames."""
    per_id = []
    for fid in ids:
        files = glob.glob(os.path.join(cpp_dir, f"ParticleData_{fid}_*.bgeo.gz"))
        if not files:
            raise FileNotFoundError(f"{cpp_dir} 下没有 ParticleData_{fid}_*.bgeo.gz")
        per_id.append({int(re.search(r"_(\d+)\.bgeo\.gz$", f).group(1)): f for f in files})
    common = sorted(set.intersection(*[set(d) for d in per_id]))
    for fr in common:
        pos, attrs, body = [], {}, []
        for b, d in enumerate(per_id):
            a = read_bhclassic(d[fr])
            pos.append(a["position"])
            body.append(np.full(len(a["position"]), b, np.int32))
            for k, v in a.items():
                if k != "position":
                    attrs.setdefault(k, []).append(np.asarray(v))
        yield fr, (np.concatenate(pos), {k: np.concatenate(v) for k, v in attrs.items()}, np.concatenate(body))


def taichi_frames(usd_path):
    from pxr import Usd, UsdGeom

    st = Usd.Stage.Open(usd_path)
    prim = st.GetPrimAtPath("/World/particles")
    if not prim:
        raise ValueError(f"{usd_path}: missing /World/particles; wait for the simulation to finish")
    pts = UsdGeom.Points(prim)
    pv = UsdGeom.PrimvarsAPI(prim)
    body = np.array(pv.GetPrimvar("body").Get(), np.int32)
    names = {"mu": "mu", "T": "T", "strainRateNorm": "strainRateNorm"}
    class Frames:
        stage = st  # Keep the owning stage alive while reading lazy frames.
        def __contains__(self, fr):
            return fr in pts.GetPointsAttr().GetTimeSamples()

        def __getitem__(self, fr):
            return read_frame(fr)

    def read_frame(fr):
        attrs = {"velocity": np.array(pts.GetVelocitiesAttr().Get(fr))}
        for k, n in names.items():
            p = pv.GetPrimvar(n)
            if p and p.Get(fr) is not None:
                attrs[k] = np.array(p.Get(fr))
        return np.array(pts.GetPointsAttr().Get(fr)), attrs, body
    return Frames(), st.GetTimeCodesPerSecond()


# C++ 属性名 -> 统一名。C++ 的 nonNewtonViscosity 为运动黏度，Taichi 的 mu 为动力黏度（= 运动黏度 * 1000）
CPP_ALIAS = {"velocity": "velocity", "nonNewtonViscosity": "nu", "temperature": "T", "strainRateNorm": "strainRateNorm"}


def stats(pos, attrs, body, nb, src):
    rows = []
    for b in range(nb):
        m = body == b
        if not m.any():
            rows.append(None)
            continue
        p = np.asarray(pos[m], np.float64)
        r = dict(com_x=p[:, 0].mean(), com_y=p[:, 1].mean(), com_z=p[:, 2].mean(), y_max=p[:, 1].max(),
                 x_min=p[:, 0].min(), x_max=p[:, 0].max(), y_min=p[:, 1].min(), z_min=p[:, 2].min(), z_max=p[:, 2].max())
        if not np.isfinite(p).all():
            raise ValueError(f"{src} body {b}: non-finite positions")
        a = {CPP_ALIAS.get(k, k): v for k, v in attrs.items()} if src == "cpp" else dict(attrs)
        if src == "taichi" and "mu" in a:
            a["nu"] = a.pop("mu") / 1000.0
        if "velocity" in a:
            r["v_mean"] = np.linalg.norm(a["velocity"][m], axis=1).mean()
        for k in ("nu", "T", "strainRateNorm"):
            if k in a:
                values = a[k][m]
                r[k + "_mean"] = float(np.mean(values, dtype=np.float64))
                r[k + "_finite_fraction"] = float(np.isfinite(values).mean())
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp_dir", required=True, help="C++ 输出目录下的 mypartio 文件夹")
    ap.add_argument("--ids", required=True, help="C++ fluid id，逗号分隔，顺序须与 Taichi 的 body 编号一致")
    ap.add_argument("--taichi_usd", required=True)
    ap.add_argument("--fps", type=float, default=25.0, help="C++ dataExportFPS（默认 25）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--times", help="C++ diagnostic frame_times.csv; defaults to <cpp_dir>/frame_times.csv")
    args = ap.parse_args()
    ids = args.ids.split(",")
    os.makedirs(args.out, exist_ok=True)

    cpp = cpp_frames(args.cpp_dir, ids)
    tai, tai_fps = taichi_frames(args.taichi_usd)
    nb = len(ids)
    # C++：第 k 次导出（k 从 1 起）发生在 t = (k-1)/fps；Taichi：第 f 帧在 t = f/tai_fps
    rows = []
    times_path = args.times or os.path.join(args.cpp_dir, "frame_times.csv")
    times = {}
    if os.path.exists(times_path):
        with open(times_path, newline="") as tf:
            times = {int(r["frame"]): float(r["time"]) for r in csv.DictReader(tf)}
    for k, (pos, attrs, body) in cpp:
        t = times.get(k, (k - 1) / args.fps)
        f = int(round(t * tai_fps))
        if f not in tai:
            continue
        tp, ta, tb = tai[f]
        sc = stats(pos, attrs, body, nb, "cpp")
        st = stats(tp, ta, tb, nb, "taichi")
        for b in range(nb):
            if sc[b] is None or st[b] is None:
                continue
            row = dict(t=round(t, 7), t_taichi=f / tai_fps, time_error=t - f / tai_fps,
                       body=ids[b], n_cpp=int((body == b).sum()), n_taichi=int((tb == b).sum()))
            for key in sc[b]:
                if key in st[b]:
                    row[key + "_cpp"] = sc[b][key]
                    row[key + "_taichi"] = st[b][key]
            rows.append(row)
    if not rows:
        raise SystemExit("两边没有时间重合的帧：检查 --fps 与两边的模拟时长")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("t", "body", "n_cpp", "n_taichi"), k))
    with open(os.path.join(args.out, "compare.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [k[:-4] for k in keys if k.endswith("_cpp") and k != "n_cpp"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 2.2 * len(metrics)), sharex=True)
    for ax, mtr in zip(np.atleast_1d(axes), metrics):
        for b, fid in enumerate(ids):
            rb = [r for r in rows if r["body"] == fid and mtr + "_cpp" in r]
            t = [r["t"] for r in rb]
            ln = ax.plot(t, [r[mtr + "_cpp"] for r in rb], "-", label=f"{fid} C++")[0]
            ax.plot(t, [r[mtr + "_taichi"] for r in rb], "--", color=ln.get_color(), label=f"{fid} Taichi")
        ax.set_ylabel(mtr)
        if mtr.startswith("nu"):
            ax.set_yscale("symlog")
    np.atleast_1d(axes)[0].legend(fontsize=6, ncol=4)
    np.atleast_1d(axes)[-1].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "compare.png"), dpi=110)
    print(f"wrote {len(rows)} rows -> {args.out}/compare.csv, compare.png")


if __name__ == "__main__":
    main()
