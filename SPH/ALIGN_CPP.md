# Taichi 版与 nonNewtonCode（C++）数值对齐：交接说明

> 2026-10-03 Windows 验证更新：远程实际含求解器和本文的分支是 `v2`，不是下表历史记载的 `main`。
> 当前可复现入口为 `python run_alignment.py`，依赖见 `requirements-align.in`。
> 热场景兼容模式须先运行 C++，随后读取同一份 `data/MyScenes/Cache` 边界采样；也可用 `--cpp_boundary_cache` 指定。
> 已核实的质量、边界黏度、热源、C++ Casson 预条件器错误及最终逐帧结果，见 `align_results/REPORT.md`。
> 第 5 节保留交接时的问题清单；已修复项与被源码/实验否定的旧推断，以报告为准。
> 2026-10-04 后续量化修复与完整验证见 `align_results/QUANTITATIVE_REPORT.md`；
> 热场景 runner 现默认补齐 Casson 边界黏性并使用 `--cpp_dfsph`。ramp AVX 与标量实现的差别单列在新报告中。
> 2026-10-05 Mac 端更新（提交 `0ddc769`，已合并进 `main`）：
> - 上一条"ramp AVX 用固定黏度"不成立：`NonNewton_Weiler2018.cpp:20` 有 `#undef USE_AVX`，ramp 实际运行标量分支，
>   矩阵、预条件器、RHS 都用逐粒子非牛顿黏度，C++ ramp 基准内部是一致的。
> - ramp 现可用 `--cpp_visc_cg --cpp_dfsph --bender_table data/models/cpp/ramp_bender_table.npz`：DFSPH + Weiler2018 隐式黏性 +
>   C++ Bender2019 体积图边界，`run_alignment.py` 已默认使用。修正后的第 5.1、5.3 节与第 6 节结果见下文。
> - 分支：`v2` 与 `bicep` 已合并进 `main`，以后统一用 `main`。

给在 Windows 机器上接手的 agent。目标：在 Windows 上跑 C++ 原版，导出逐帧粒子数据，与本仓库 Taichi 版
（`SPH/constraint_solver.py`）逐帧比较，定位并缩小差异。

## 0. 两个仓库

| | 仓库 | 用途 |
|---|---|---|
| Taichi 版（本仓库） | https://github.com/chunleili/nonNewton-taichi ，分支 `main` | `SPH/constraint_solver.py` 求解器，`SPH/compare_cpp.py` 对比脚本 |
| C++ 原版 | https://github.com/chunleili/nonNewtonCode ，基准 commit `47d6a53e1eb34a0017ffc383d16981aadce65f41` | 论文 TVCG 2023 的 SPlisHSPlasH fork |

代码通过 GitHub 同步：Mac 端改完 push，Windows 端 `git pull`；Windows 端的改动也请提交到 `main` 并 push，
提交前先 `git pull --rebase`，因为 Mac 端可能同时在改 `constraint_solver.py`（bicep 肌肉场景代码也在其中，勿动）。
**仿真输出（`SPH/output/`、C++ 的 `bin/output/`）不要提交**，体积很大；对比结果只提交 `compare.csv`、`compare.png` 与结论。

## 1. 环境

### C++（Windows）
```
git clone https://github.com/chunleili/nonNewtonCode && cd nonNewtonCode
git checkout 47d6a53e1eb34a0017ffc383d16981aadce65f41
cmake --preset vs2022
cmake --build --preset vs2022-Rel
```
要求 Visual Studio 2022、CMake ≥ 3.22。可执行文件在 `bin/SPHSimulator.exe`。

### Taichi（Windows 上同样能跑）
Python 3.9–3.11，`pip install taichi==1.7.2 numpy usd-core matplotlib`（Mac 端版本：Python 3.9.13、taichi 1.7.2、
numpy 1.26.4、usd-core 26.3）。有 NVIDIA GPU 时用 `--arch cuda`，否则 `--arch cpu`。

## 2. 跑 C++ 并导出

C++ 导出器是 `Simulator/Exporter/ParticleExporter_MyPartio.cpp`：开关 `enableMyPartioExport`，
写到 `<output-dir>/mypartio/ParticleData_<fluidId>_<frame>.bgeo.gz`，frame 从 1 开始，第 k 帧对应 t=(k-1)/fps，
fps 由 `dataExportFPS` 决定（默认 25）。导出哪些属性由 `particleAttributes` 决定（分号分隔，position 总会导出）。

C++ 的 `--param` **每次运行只接受一个**（`SimulatorBase.cpp:297-311` 只读取一个值），所以导出设置要写进 json。
**不要改原 json**，复制为 `data/MyScenes/align_<场景>.json`，在 `Configuration` 里加 / 改：
```json
"enableMyPartioExport": true,
"dataExportFPS": 25,
"particleAttributes": "velocity;nonNewtonViscosity;strainRateNorm",
"stopAt": 5.0
```
然后在 `bin/` 下：
```
SPHSimulator.exe --no-gui --no-initial-pause --output-dir out/ramp1 ../data/MyScenes/align_ramp1.json
```
需要跑的场景：

| 场景 json | fluid id（`--ids` 参数，顺序即 body 编号） | stopAt | particleAttributes |
|---|---|---|---|
| `ramp1.json` | `Newtonain,PowerLaw1,PowerLaw2`（注意原文拼写 Newtonain） | 5 | `velocity;nonNewtonViscosity;strainRateNorm` |
| `ramp2.json` | `Cross,Casson,Carreau` | 5 | 同上 |
| `ramp3.json` | `Bingham,HerschelBulkley` | 5 | 同上 |
| `ice-cream.json` | `Fluid` | 10 | `velocity;nonNewtonViscosity;temperature` |
| `hotcut.json` | `Fluid` | 20 | `velocity;nonNewtonViscosity;temperature` |

注意：
- 副本里引用模型的相对路径（`../models/...`）不变，副本放在 `data/MyScenes/` 下即可。
- `ice-cream.json` 的 Configuration 里有 `"pauseAt": 5.0`，在副本里删掉，否则 5 s 处暂停。
- `strainRateNorm` 是 NonNewton 模块注册的字段（`NonNewton.cpp:29`），只有 ramp 场景有；
  `nonNewtonViscosity`、`temperature` 是 FluidModel 字段，所有场景都有。不认识的字段导出器只打警告并跳过。
- `hotcut.json` 没写 `rSource`，C++ 用默认值 0.1（`Coagulation.cpp:33`），论文写的是 R=1。按 json 原样跑，不要改。

## 3. 跑 Taichi 并导出（同样 25 fps）

在 `SPH/` 下：
```
python constraint_solver.py --scene ramp --group ramp1 --noslip --cpp_compat --export usd --fps 25 --out output/align --arch cuda
python constraint_solver.py --scene ramp --group ramp2 --noslip --cpp_compat --export usd --fps 25 --out output/align --arch cuda
python constraint_solver.py --scene ramp --group ramp3 --noslip --cpp_compat --export usd --fps 25 --out output/align --arch cuda
python constraint_solver.py --scene icecream --cpp_compat --export usd --fps 25 --out output/align --arch cuda
python constraint_solver.py --scene hotcut   --cpp_compat --export usd --fps 25 --out output/align --arch cuda
```
- 输出 `output/align/<tag>.usdc`（tag 如 `ramp_ramp1_noslip_cpp`、`icecream_cpp`、`hotcut_cpp`）。
- **坐标已与 C++ 一致**（导出时减掉了仿真内部的平移），可直接与 C++ 的 bgeo 叠加比较。
- ramp 默认是 C++ 原分辨率（间距 0.05，每只犰狳 51564 粒子，ramp1 共 154692 粒子；Mac M 系列 GPU 上约 0.3 s/step，
  5 s 共 2500 步）。Mac 上为了速度用过 `--coarsen 2`（间距 0.1），对齐时**不要**加 `--coarsen`。
- usdc 里 `/World/particles` 的 primvars：`body`、`mu`（动力黏度 = C++ 运动黏度 × 1000）、`T`、`strainRateNorm`；
  `/World/boundary/*` 是边界网格与边界粒子。

## 4. 对比

```
python compare_cpp.py --cpp_dir <C++输出>/out/ramp1/mypartio --ids Newtonain,PowerLaw1,PowerLaw2 ^
  --taichi_usd output/align/ramp_ramp1_noslip_cpp.usdc --out output/align/cmp_ramp1
```
输出 `compare.csv` 与 `compare.png`：每个 body 每帧的质心 `com_x/y/z`、包围盒 `x/y/z_min/max`、平均速率 `v_mean`、
平均运动黏度 `nu_mean`（Taichi 的 mu 已 /1000 换算）、平均温度 `T_mean`、平均应变率 `strainRateNorm_mean`。
脚本已在 Mac 上用合成的 C++ 格式 bgeo.gz（与 BGEO.cpp 同布局）做过自洽测试：同一数据两边读出完全一致。

建议的对齐判据（先定性后定量）：
1. 第 0 帧两边粒子数与包围盒应一致（ramp 一致；hotcut 不一致，见 5.3）。
2. ramp：8 个模型沿坡位移（看 `com_z`）的**排序**一致，再看数值差。
3. ice-cream / hotcut：高度 `y_max` 随时间的曲线、`nu_mean` 与 `T_mean` 的下降曲线。

## 5. 已知差异（Mac 端已确认，需要逐项在 C++ 数据上验证或修正）

### 5.1 黏性求解器（已解决）
C++ 用 `NonNewton_Weiler2018`（viscosityMethod 13，ramp）或 `Viscosity_Casson`（8，热场景），都是隐式 CG。
Taichi 默认的 XPBD 黏性约束在 μ 很大时不够硬。现在两类场景都有 C++ 兼容的隐式求解（`--cpp_visc_cg --cpp_dfsph`）：
- 热场景：Casson 算子用邻居黏度 μ_j，以 q = √μ·v 对称化。
- ramp：Weiler2018 标量分支用本粒子黏度 μ_i 乘整行，以 q = v/√μ 对称化；边界黏度每步取 μ_i（`:921`）。
  容差取 ramp json：`maxError` 0.1%、`viscoMaxError` 1e-3。

### 5.2 应变率与黏度截断（`--cpp_compat` 已复刻）
- C++ `NonNewton::calcStrainRate`（`NonNewton.cpp:224-257`）在邻居循环里用 `=` 而不是 `+=`，只保留最后一个邻居；
  `--cpp_compat` 复刻了这一点（只取最后一个**流体**邻居；C++ 的邻居顺序由 CompactNSearch 决定，无法逐粒子一致）。
- C++ 不截断黏度；Taichi 默认把黏度截断在 [μ∞, μ0]，`--cpp_compat` 时不截断。
- Carreau：C++ 写的是 `μ∞ + (μ0-μ∞)/(1 + (m·ε̇²)^((1-n)/2))`，与论文式不同，`--cpp_compat` 用 C++ 写法。
- 结论：**论文 Fig.14 的排序依赖这个应变率 bug**。不开 `--cpp_compat` 时排序与论文不符。

### 5.3 边界
- **更正**：此前写"C++ ramp 用 Akinci 边界黏性 `viscosityBoundary: 0.1`（`:341`）"是错的。ramp 的
  `boundaryHandlingMethod` 是 2（Bender2019 体积图），`:341` 所在的 Akinci 分支不执行；而且
  `NonNewton_Weiler2018::step`（`:921`）每步令边界黏度等于本粒子黏度，json 的 0.1 在 ramp 中不起作用。
- ramp 现用 `--bender_table`：`data/models/cpp/ramp_bender_table.npz` 是用 Discregrid 桥接器（`cpp_map_bridge/`）
  从 C++ 的 `Cache/ramp_sb_vm_0.025_s3_1_1_r40_40_40_i0_t0.cdm` 采样的 (y,z) 表，间隔 0.01。ramp.obj 沿 x 拉伸，
  抽查不同 x 处差异小于 1e-7。密度、DFSPH 因子与压力、Weiler 边界黏性（切向 4 点，每点 0.25 V_j）、
  进入边界后推回，都照 `TimeStep::computeVolumeAndBoundaryX` 与 `NonNewton_Weiler2018.cpp:396-440` 实现。
  旧路径（XPBD + `--noslip` 斜面粒子）的无滑移不够强，粒子会从坡底端滑出掉落（y_min 达 −60 ～ −95 m），不要再用于对比。
- ice-cream / hotcut 的 boundaryHandlingMethod 是 0（Akinci 粒子），Taichi 读 C++ 的边界采样缓存。
- hotcut：Taichi 删掉了与切割板距离 < 0.75d 的兔子粒子（24872 → 22202），C++ 靠边界压力推开。第 0 帧粒子数会不同。

### 5.4 质量与静息密度
C++ 粒子质量按 `particleRadius` 0.025（间距 0.05）算。冰淇淋模型点间距 0.07、兔子 0.06，所以 C++ 中这两个物体
初始只有约 36% / 58% 的静息密度，形状靠黏性撑住。Taichi 按模型实际间距设 d（0.07 / 0.06）和质量，初始即为静息密度。
这会影响塌落行为，**是冰淇淋 / 热切割不对齐的另一个主要嫌疑**。如要对齐，需要在 Taichi 里按 C++ 的方式设质量。

### 5.5 温度与热源（`Coagulation.cpp:180-300`）
- 冰淇淋：表面粒子（C++ 邻居数 < 10，h=0.1）每步加 `surfaceSource0`=1。Taichi 的 h=0.14，阈值改为 15
  使表面粒子比例接近（C++ 约 6%）。
- 热切割：box `[-1,0,-1]-[1,0.05,1]` 内加 `rSource`。Taichi 当前热源判定是 `y < 0.05`，但流体底层停在约 0.07，
  **实际没被加热**（Mac 端已发现，未修）。
- C++ 的源项写在邻居循环内，被累加“邻居数”次；`--cpp_compat` 复刻了这一点，并复刻了 C++ 的速度限幅 4 m/s。
  C++ 还有 `y < 0.01` 的地板摩擦（速度 xz 分量 ×0.1），Taichi 未复刻。
- 黏度：μ = μ0 · exp(−decay · T)，decay 默认 0.1（json 未写，取 `Coagulation.h:35` 默认值）。

### 5.6 时间步
C++：`timeStepSize` 0.001（ramp）/ 0.005（ice-cream、hotcut），`cflMethod: 1` 自适应，`cflMaxTimeStepSize` 0.005。
Taichi：固定步长，ramp 2e-3、ice-cream / hotcut 5e-3。对比时按时间 t 对齐，不按步数。

## 6. ramp 对齐结果（2026-10-05，Mac，原分辨率间距 0.05）
命令（每组在 M1 CPU 上约 40 分钟；M1 的 Metal 不支持 f64，须用 `--arch cpu`）：
```
python constraint_solver.py --scene ramp --group ramp1 --cpp_compat --cpp_visc_cg --cpp_dfsph \
  --bender_table data/models/cpp/ramp_bender_table.npz --arch cpu --export usd --fps 25 --out output/align
```
t=4.96 s 质心沿坡位移（com_z 变化，m），C++ / Taichi（括号内为首轮 XPBD Taichi）：

| 模型 | C++ / Taichi | 首轮 | 全程质心距离 RMSE |
|---|---|---|---|
| Newtonian | 0.822 / 0.817 | 1.133 | 0.027 |
| Cross | 0.817 / 0.852 | 1.177 | 0.030 |
| Carreau | 0.799 / 0.850 | 1.143 | 0.043 |
| Bingham | 0.798 / 0.849 | 1.129 | 0.042 |
| HerschelBulkley | 0.798 / 0.849 | 1.123 | 0.042 |
| PowerLaw2 | 5.727 / 5.757 | 5.446 | — |
| PowerLaw1 / Casson | C++ 无有效基准 / 0.329, 0.191 | 0.803, 0.628 | — |

- 五个近牛顿模型的 y_max、y_min、v_mean 曲线与 C++ 基本重合，末帧 y_min 1.54–1.57，不再漏粒子。
- C++ 的 PowerLaw1 / Casson：应变率为 0 时 `NonNewton.cpp:378`、`:403` 给出 ν=inf，黏性 CG 返回 NaN
  （日志 `Visco iterations: 0 error: -nan`），末帧全部粒子以 g·dt 匀速竖直下落。这不是物理行为，不能作为对比基准。
  Taichi 把应变率下限设为 1e-6，所以有有限黏度。
- 仍有的差异：PowerLaw2 约 3 s 后 Taichi 从坡端掉落的粒子比 C++ 多（末帧 y_min −27.9 / −7.8），末段 v_mean 偏高；
  平均应变率约为 C++ 的一半（末邻居应变率依赖邻居顺序）。
- 对比时 C++ 侧用 `align_results/ramp*/compare.csv` 中的 `*_cpp` 列，Taichi 帧按 C++ 真实导出时间线性插值。

## 7. 交付物（请提交到 `SPH/align_results/`）
- 每个场景的 `compare.csv`、`compare.png`
- `REPORT.md`：每个场景的对齐情况、第 5 节各条差异的核实结论（尤其 5.3 边界黏性量纲、5.4 质量、5.5 热源）、
  对 Taichi 版所做的修改及理由
- C++ 端如需改动（例如加导出字段），改动放在 nonNewtonCode 的单独分支上，并在 REPORT 中说明
