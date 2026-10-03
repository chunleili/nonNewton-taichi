# Taichi 版与 nonNewtonCode（C++）数值对齐：交接说明

> 2026-10-03 Windows 验证更新：远程实际含求解器和本文的分支是 `v2`，不是下表历史记载的 `main`。
> 当前可复现入口为 `python run_alignment.py`，依赖见 `requirements-align.in`。
> 热场景兼容模式须先运行 C++，随后读取同一份 `data/MyScenes/Cache` 边界采样；也可用 `--cpp_boundary_cache` 指定。
> 已核实的质量、边界黏度、热源、C++ Casson 预条件器错误及最终逐帧结果，见 `align_results/REPORT.md`。
> 第 5 节保留交接时的问题清单；已修复项与被源码/实验否定的旧推断，以报告为准。

给在 Windows 机器上接手的 agent。目标：在 Windows 上跑 C++ 原版，导出逐帧粒子数据，与本仓库 Taichi 版
（`SPH/constraint_solver.py`）逐帧比较，定位并缩小差异。

## 0. 两个仓库

| | 仓库 | 用途 |
|---|---|---|
| Taichi 版（本仓库） | https://github.com/chunleili/nonNewton-taichi ，分支 `main` | `SPH/constraint_solver.py` 求解器，`SPH/compare_cpp.py` 对比脚本 |
| C++ 原版 | https://github.com/chunleili/nonNewtonCode ，基准 commit `47d6a53e1eb34a0017ffc383d16981aadce65f41` | 论文 TVCG 2023 的 SPlisHSPlasH fork |

代码通过 GitHub 同步：Mac 端改完 push，Windows 端 `git pull`；Windows 端的改动也请提交到 `main` 并 push，
提交前先 `git pull --rebase`，因为 Mac 端可能同时在改 `constraint_solver.py`（另有 bicep 肌肉场景在开发中，勿动）。
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

### 5.1 黏性求解器不同（根本差异，不必改成一样）
C++ 用 `NonNewton_Weiler2018`（viscosityMethod 13，隐式 CG）或 `Viscosity_Casson`（8）；Taichi 用 XPBD 黏性约束 +
Jacobi，柔度 α = 1/(2 μ V dt)。μ 很大（冰淇淋 10⁶ Pa·s）时 Taichi 在有限迭代内“不够硬”：
冰淇淋前 2 s 塌成锥形，论文里保形。这是主要待解决问题，可尝试加 `--iters`、减小 `--dt`、调 `--omega_s`。

### 5.2 应变率与黏度截断（`--cpp_compat` 已复刻）
- C++ `NonNewton::calcStrainRate`（`NonNewton.cpp:224-257`）在邻居循环里用 `=` 而不是 `+=`，只保留最后一个邻居；
  `--cpp_compat` 复刻了这一点（只取最后一个**流体**邻居；C++ 的邻居顺序由 CompactNSearch 决定，无法逐粒子一致）。
- C++ 不截断黏度；Taichi 默认把黏度截断在 [μ∞, μ0]，`--cpp_compat` 时不截断。
- Carreau：C++ 写的是 `μ∞ + (μ0-μ∞)/(1 + (m·ε̇²)^((1-n)/2))`，与论文式不同，`--cpp_compat` 用 C++ 写法。
- 结论：**论文 Fig.14 的排序依赖这个应变率 bug**。不开 `--cpp_compat` 时排序与论文不符。

### 5.3 边界
- ramp：`--noslip` 让斜面粒子参与黏性约束（无滑移）。C++ 用 Akinci 边界黏性 `viscosityBoundary: 0.1`
  （`NonNewton_Weiler2018.cpp:341`），Taichi 不加 `--noslip` 时用同一离散的显式版本，但排序与论文不符，
  量纲换算未与 C++ 逐项核对 —— **请用 C++ 数据核对**。
- C++ boundaryHandlingMethod：ramp 是 2（Bender2019 体积图），ice-cream / hotcut 是 0（Akinci 粒子）。Taichi 一律用
  边界粒子（网格表面按间距 d 撒点，单层）。ramp 的坡底另加了一块水平地面（C++ 由 domain 下边界承接）。
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

## 6. Mac 端已有结果（供参考，粒子间距 0.1）
ramp 8 模型（`--group all`，一次跑 8 只，C++ 是分 3 个 json 跑），`--noslip --cpp_compat --coarsen 2`，t=5 s 时沿坡位移（m）：
PowerLaw2 7.23 ≫ Cross 1.47 ≈ Carreau 1.44 ≈ Bingham 1.44 ≈ Newtonian 1.44 ≈ HerschelBulkley 1.43 > PowerLaw1 0.92 > Casson 0.69。
与论文 Fig.14 定性一致（PowerLaw2 摊开最远，Casson / PowerLaw1 保持形状）。

## 7. 交付物（请提交到 `SPH/align_results/`）
- 每个场景的 `compare.csv`、`compare.png`
- `REPORT.md`：每个场景的对齐情况、第 5 节各条差异的核实结论（尤其 5.3 边界黏性量纲、5.4 质量、5.5 热源）、
  对 Taichi 版所做的修改及理由
- C++ 端如需改动（例如加导出字段），改动放在 nonNewtonCode 的单独分支上，并在 REPORT 中说明
