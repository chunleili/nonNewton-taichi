"""Reproduce the five Windows alignment runs without changing source scenes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CASES = {
    "ramp1": ("ramp", ["--group", "ramp1"], "Newtonain,PowerLaw1,PowerLaw2", 5, "ramp_ramp1_dfsph_bender_cpp"),
    "ramp2": ("ramp", ["--group", "ramp2"], "Cross,Casson,Carreau", 5, "ramp_ramp2_dfsph_bender_cpp"),
    "ramp3": ("ramp", ["--group", "ramp3"], "Bingham,HerschelBulkley", 5, "ramp_ramp3_dfsph_bender_cpp"),
    "ice-cream": ("icecream", [], "Fluid", 10, "icecream_cpp"),
    "hotcut": ("hotcut", [], "Fluid", 20, "hotcut_cpp"),
}


def run(cmd, cwd, log, env=None):
    print("Running:", subprocess.list2cmdline(list(map(str, cmd))), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        subprocess.run(list(map(str, cmd)), cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                       env=env, check=True)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--cpp_repo", type=Path, default=HERE.parents[1] / "nonNewtonCode")
    ap.add_argument("--case", choices=["all", *CASES], default="all")
    ap.add_argument("--only", choices=["all", "cpp", "taichi", "compare"], default="all")
    ap.add_argument("--arch", default="cuda")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--visc_solver", choices=["auto", "xpbd", "cg", "cpp_cg"], default="auto")
    ap.add_argument("--pressure_solver", choices=["auto", "xpbd", "cpp_dfsph"], default="auto")
    ap.add_argument("--output_dir", type=Path, default=HERE / "output/quant/full")
    ap.add_argument("--result_dir", type=Path, default=HERE / "align_results/quantitative")
    args = ap.parse_args()
    cpp = args.cpp_repo.resolve()
    out = args.output_dir.resolve()
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
               TI_OFFLINE_CACHE_FILE_PATH=str(HERE / "output/ti-cache"))
    for name, (scene, extra, ids, stop, tag) in CASES.items():
        if args.case != "all" and args.case != name:
            continue
        dst = cpp / "data/MyScenes" / f"align_{name}.json"
        if args.only in ("all", "cpp"):
            data = json.loads((cpp / "data/MyScenes" / f"{name}.json").read_text(encoding="utf-8-sig"))
            cfg = data["Configuration"]
            fields = "strainRateNorm" if name.startswith("ramp") else "temperature"
            cfg.update(enableMyPartioExport=True, dataExportFPS=25, stopAt=stop,
                       particleAttributes=f"velocity;nonNewtonViscosity;{fields};density",
                       enableObjectSplitting=False)
            cfg.pop("pauseAt", None)
            dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
            run([cpp / "bin/SPHSimulator.exe", "--no-gui", "--no-initial-pause",
                 "--output-dir", f"output/align/{name}", dst], cpp / "bin", out / f"cpp-{name}.log", env)
        if args.only in ("all", "taichi"):
            mode = args.visc_solver
            if mode == "auto":
                mode = "cpp_cg"
            visc_args = {"xpbd": [], "cg": ["--visc_cg"], "cpp_cg": ["--cpp_visc_cg"]}[mode]
            pressure = args.pressure_solver
            if pressure == "auto":
                pressure = "cpp_dfsph" if mode == "cpp_cg" else "xpbd"
            if pressure == "cpp_dfsph":
                if mode != "cpp_cg":
                    ap.error("cpp_dfsph requires cpp_cg")
                visc_args += ["--cpp_dfsph"]
                if scene == "ramp":
                    # C++ ramp 用 Bender2019 体积图边界；表由 C++ .cdm 采样，见 ALIGN_CPP.md 5.3
                    visc_args += ["--bender_table", HERE / "data/models/cpp/ramp_bender_table.npz"]
            elif scene == "ramp":
                visc_args += ["--noslip"]
            tag = tag if pressure == "cpp_dfsph" or scene != "ramp" else tag.replace("_dfsph_bender", "_noslip")
            run([sys.executable, "-u", HERE / "constraint_solver.py", "--scene", scene, *extra,
                 *visc_args,
                 "--cpp_compat", "--export", "usd", "--fps", "25", "--arch", args.arch,
                 "--iters", args.iters, "--out", out,
                 "--cpp_boundary_cache", cpp / "data/MyScenes/Cache"], HERE, out / f"taichi-{name}.log", env)
        if args.only in ("all", "compare"):
            run([sys.executable, HERE / "compare_cpp.py", "--cpp_dir",
                 cpp / f"bin/output/align/{name}/mypartio", "--ids", ids,
                 "--taichi_usd", out / f"{tag}.usdc", "--out", args.result_dir / name],
                HERE, out / f"compare-{name}.log", env)


if __name__ == "__main__":
    main()
