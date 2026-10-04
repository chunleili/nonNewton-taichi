"""Build the second-round thermal alignment report from measured frame statistics."""
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "align_results"
NEW = OUT / "quantitative"
CASES = {"ice-cream": (10, 251, 6250), "hotcut": (20, 501, 24872)}


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize(data):
    last = data[-1]
    result = {"frames": len(data), "time_cpp": float(last["t"]),
              "time_taichi": float(last["t_taichi"]),
              "max_time_error": max(abs(float(r["time_error"])) for r in data),
              "particles": int(last["n_cpp"]), "metrics": {}}
    for metric in ("y_max", "T_mean", "nu_mean"):
        valid = [r for r in data if metric != "nu_mean" or float(r["t"]) > 0]
        cpp = np.array([float(r[metric + "_cpp"]) for r in valid])
        tai = np.array([float(r[metric + "_taichi"]) for r in valid])
        if not np.isfinite(cpp).all() or not np.isfinite(tai).all():
            raise ValueError(f"Non-finite {metric}")
        diff = tai - cpp
        result["metrics"][metric] = dict(cpp_final=float(cpp[-1]), taichi_final=float(tai[-1]),
            final_abs_error=float(abs(diff[-1])), final_relative_percent=float(abs(diff[-1] / cpp[-1]) * 100),
            rmse=float(np.sqrt(np.mean(diff**2))), max_abs_error=float(abs(diff).max()))
    com = np.array([[float(r[f"com_{a}_taichi"]) - float(r[f"com_{a}_cpp"])
                    for a in "xyz"] for r in data])
    result["com_final_distance"] = float(np.linalg.norm(com[-1]))
    result["com_distance_rmse"] = float(np.sqrt(np.mean(np.square(com).sum(1))))
    return result


def main():
    summaries = {}
    table = ["|场景 / 指标|C++ 末帧|Taichi 末帧|首轮末帧绝对误差|本轮末帧绝对误差|本轮相对误差|全程 RMSE|全程最大绝对误差|",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case, (duration, frames, count) in CASES.items():
        data = rows(NEW / case / "compare.csv")
        if len(data) != frames or float(data[-1]["t_taichi"]) < duration:
            raise ValueError(f"{case}: incomplete output")
        if any(int(r["n_cpp"]) != count or int(r["n_taichi"]) != count for r in data):
            raise ValueError(f"{case}: changed particle count")
        result, baseline = summarize(data), summarize(rows(OUT / case / "compare.csv"))
        result["baseline"] = baseline
        tag = "icecream_cpp" if case == "ice-cream" else "hotcut_cpp"
        runtime = HERE / "output/quant/full" / f"{tag}.json"
        if runtime.exists():
            run = json.loads(runtime.read_text())
            result["solver_parameters"] = run["parameters"]
            result["runtime_seconds"] = run["seconds"]
            result["sampled_diagnostics"] = {key: max(r.get(key, 0) for r in run["log"])
                for key in ("cell_overflow", "neighbor_cap", "outside_grid", "pressure_iters", "divergence_iters", "visc_cg_iters", "visc_cg_error")}
        summaries[case] = result
        for metric, label in (("y_max", "最大高度 (m)"), ("T_mean", "平均温度"), ("nu_mean", "平均运动黏度")):
            v, old = result["metrics"][metric], baseline["metrics"][metric]
            table.append(f"|{case} / {label}|{v['cpp_final']:.6f}|{v['taichi_final']:.6f}|{old['final_abs_error']:.6f}|{v['final_abs_error']:.6f}|{v['final_relative_percent']:.3f}%|{v['rmse']:.6f}|{v['max_abs_error']:.6f}|")
    NEW.mkdir(exist_ok=True)
    (NEW / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    report = """# C++ / Taichi 第二轮定量对齐

2026-10-04。本轮针对两个热场景继续量化修复，C++ 参考仍是 `nonNewtonCode/align-windows` 的 `83a4acb`，
没有为贴合曲线改变 C++ 热场景、重力、粒子分辨率、热源或材料参数。
首轮 Taichi 基线是 `11dfc47`，其数据和结论保留在 [REPORT.md](REPORT.md)。

## 完整验证结果

冰淇淋跑满 10 秒、251 帧、6250 粒子；热切割跑满 20 秒、501 帧、24872 粒子。
所有比较帧粒子数一致，位置均有限；两个场景采样日志的哈希桶溢出、邻居容量截断均为 0。
相对误差以 C++ 末帧为分母。RMSE 与最大误差覆盖全部比较帧，ν 排除尚未计算黏度的 C++ t=0 帧。

""" + "\n".join(table) + "\n\n"
    for case, result in summaries.items():
        report += (f"**{case}**：末帧质心距离 {result['com_final_distance']:.6f} m，"
                   f"全程质心距离 RMSE {result['com_distance_rmse']:.6f} m；"
                   f"真实导出时间与所匹配 Taichi 帧的最大偏差 {result['max_time_error']:.6f} s。\n\n"
                   f"[逐帧 CSV](quantitative/{case}/compare.csv) · [全部指标曲线](quantitative/{case}/compare.png)\n\n")
    report += """[本轮与首轮曲线总览](quantitative/overview.png) · [机器可读统计](quantitative/summary.json)

## 修复依据与受控短测

1. **补齐 Casson 矩阵中的边界黏性**。`Viscosity_Casson::matrixVecProd` 的边界分支仍然有效，
   构造函数默认 `viscosityBoundary=0.1`。源码中的“删掉 boundary”注释仅适用于部分 RHS / applyForces 代码，
   不代表矩阵中的 Akinci 边界项被删除。首轮 Taichi CG 遗漏了这项，这是保形差异的重要来源。
   本轮矩阵与 3×3 块预条件器同时加入 `−10 dt ν_b ρ0² V_b / ρ_i² · gradW⊗r/(r²+0.01h²)`。
   静态边界的速度为零，故只增加对角块，不增加非零 RHS。
2. **匹配分步顺序与压力线性化**。增加显式选择的 `--cpp_dfsph`：步初邻域/密度与梯度 → 散度求解 →
   热扩散/摩擦/黏度刷新 → Casson CG → 重力 → 线性化预测密度压力求解 → 更新位置。
   各求解均使用步初位置，不在压力循环中重算移动后的非线性密度。原 XPBD/PBF 路径仍可独立选择。
3. **匹配参考停止规则**。密度误差上限 0.05%，最少 2 次、最多 100 次；散度误差上限 0.1%/dt，最少 1 次、最多 100 次。
   按全体流体粒子数取平均，邻居总数不足 20 时不修正散度。
   压力/散度使用参考的 0.5 倍 warmstart 与对应截断；CG 使用上一时间步的速度差作为初值。
   ice-cream 的 C++ `viscoMaxIter` 默认是 100；hotcut JSON 是 1000，本轮分别匹配。

热切割 3 秒受控短测，保持全部物理参数与粒子数不变：

|求解设置|最大高度 (m)|与 C++ 高度的绝对差 (m)|
|---|---:|---:|
|C++ 参考|2.610920|—|
|首轮：流体 CG + PBF|1.257788|1.353132|
|补齐边界项，保留 PBF|2.690494|0.079574|
|边界项 + 参考 DFSPH 分步|2.609142|0.001779|

[边界项短测](quant_probe/boundary-only/compare.csv) · [分步短测](quant_probe/dfsph/compare.csv)

另以 200 fps 导出 C++ 与 Taichi 前 0.1 秒进行单步检查。
0.005 秒时最大高度仅差约 0.48 μm，平均速度差约 2.9e-6 m/s；0.1 秒时质心距离约 1.25 μm。
这项检查使用同一初始粒子和时间，支持实现一致性，不能替代上面的长时全程统计。
[单步统计](quant_probe/step/compare.csv)

## 验证与复现

10 项 CPU 数值测试经分组验证通过：原 7 项、含静态边界的 Casson 独立稠密矩阵测试、
DFSPH 单次压力迭代与独立约束 Jacobian 的比较、低邻居数情况下的刚体自由落体。
短测与完整 CUDA 运行均完成。C++ 基准仍使用 Windows 10 / VS2022 Release / MSVC 19.44，
Taichi 环境为独立 Python 3.11.17 / Taichi 1.7.2 / NumPy 1.26.4 / usd-core 26.3 / RTX 4090。

```powershell
.\\.venv311\\Scripts\\python.exe SPH/run_alignment.py --case ice-cream --only taichi
.\\.venv311\\Scripts\\python.exe SPH/run_alignment.py --case ice-cream --only compare
.\\.venv311\\Scripts\\python.exe SPH/run_alignment.py --case hotcut --only taichi
.\\.venv311\\Scripts\\python.exe SPH/run_alignment.py --case hotcut --only compare
.\\.venv311\\Scripts\\python.exe SPH/build_quantitative_report.py
.\\.venv311\\Scripts\\python.exe -m unittest discover -s SPH -p test_alignment.py -v
```

默认热场景使用 `cpp_cg + cpp_dfsph`。可用 `--pressure_solver xpbd` 保留位置约束，
或者直接调用求解器并用 `--cpp_visc_boundary 0` 隔离边界项。新输出分别在 `SPH/output/quant/full/` 与
`SPH/align_results/quantitative/`；原始模拟输出不提交。

## ramp 的新基准问题与尚存限制

本轮没有替换三组 ramp 的首轮结果，也没有声称 ramp 已定量通过。
重新核查实际 MSVC 工程：`USE_AVX` 已开启。`NonNewton_Weiler2018::matrixVecProd` 的 AVX 分支从
`m_viscosity` 和 `m_boundaryViscosity` 取固定系数，而标量分支从 `m_viscosity_nonNewton[i]` 与
`m_boundaryViscosity_nonNewton[i]` 取逐粒子非牛顿黏度。
AVX 算子与用逐粒子黏度构造的预条件器/RHS 也不一致。ramp JSON 只有 `viscosity0`，
未设置基础 `viscosity`，后者默认 0.01。不能再将标量路径的量纲核查直接解释为当前 AVX 基准的完整力学行为。
PowerLaw1 / Casson 的零应变率还产生 inf，运行日志出现 NaN CG 残差；这两个模型不构成健康的有限数值基准。

继续 ramp 对齐需要选择：修复 C++ 的 AVX/标量差别与零应变率，建立新基准，或刻意复刻现有数值问题。
这个选择会改变参照行为，已向用户询问；在确定前保留原 C++ 分支与数据。
已经准备可复用的 C++ Discregrid 只读桥接器，读取同一 `.cdm` 体积图；在 20 个真实近边界点验证 SDF 梯度与有限差分一致。
它用于后续消除 Bender 体积图和 Taichi 粒子边界的差别，尚未接入 ramp 时间积分。

热场景仍有固定/自适应时间步、采样时刻、位置精度与浮点求和顺序的差别。
C++ 热扩散原地并行更新，Taichi 使用双缓冲；没有为逐位复现数据竞争而引入同类竞争。
两个实现的 CG 在温度不均匀时分别运行于 v 和对称化后的 q 空间，误差停止量并非完全相同。
ice-cream 参考限制 100 次迭代，部分步在达到残差目标前触及上限，本轮按同样上限运行并记录实际残差。
这些限制意味着当前结论是上述全程统计指标已明显接近，不能外推为逐粒子、逐位相同。
"""
    (OUT / "QUANTITATIVE_REPORT.md").write_text(report, encoding="utf-8")
    overview()
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def overview():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
    for col, case in enumerate(CASES):
        new, old = rows(NEW / case / "compare.csv"), rows(OUT / case / "compare.csv")
        for row, (metric, label) in enumerate((("y_max", "Maximum height (m)"), ("T_mean", "Mean temperature"), ("nu_mean", "Mean exported viscosity"))):
            ax = axes[row, col]
            for data, src, legend, style, color in ((new, "cpp", "C++ reference", "-", "#2563eb"),
                                                   (new, "taichi", "Taichi corrected", "--", "#16a34a"),
                                                   (old, "taichi", "Taichi first round", ":", "#ea580c")):
                data = [r for r in data if metric != "nu_mean" or float(r["t"]) > 0]
                tkey = "t" if src == "cpp" else "t_taichi"
                ax.plot([float(r[tkey]) for r in data], [float(r[f"{metric}_{src}"]) for r in data],
                        style, label=legend, color=color, linewidth=1.8)
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
            if row == 0:
                ax.set_title(case)
                ax.legend(fontsize=9)
            if row == 2:
                ax.set_xlabel("Time (s)")
    fig.suptitle("Quantitative alignment: full-duration verification", fontsize=16)
    fig.savefig(NEW / "overview.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
