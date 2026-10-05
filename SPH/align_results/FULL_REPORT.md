# 完整对齐报告（两轮合并）

本文件包含两轮完整报告，先列第二轮当前结果，再列首轮历史结果。会话交接见 [SESSION_HANDOFF.md](SESSION_HANDOFF.md)，审查要求见 [CLAUDE_REVIEW_PROMPT.md](CLAUDE_REVIEW_PROMPT.md)。统计数据仍以各轮 CSV 和 summary.json 为准。

---

# C++ / Taichi 第二轮定量对齐

2026-10-04。本轮针对两个热场景继续量化修复，C++ 参考仍是 `nonNewtonCode/align-windows` 的 `83a4acb`，
没有为贴合曲线改变 C++ 热场景、重力、粒子分辨率、热源或材料参数。
首轮 Taichi 基线是 `11dfc47`，其数据和结论保留在 [REPORT.md](REPORT.md)。

## 完整验证结果

冰淇淋跑满 10 秒、251 帧、6250 粒子；热切割跑满 20 秒、501 帧、24872 粒子。
所有比较帧粒子数一致，位置均有限；两个场景采样日志的哈希桶溢出、邻居容量截断均为 0。
相对误差以 C++ 末帧为分母。RMSE 与最大误差覆盖全部比较帧，ν 排除尚未计算黏度的 C++ t=0 帧。

|场景 / 指标|C++ 末帧|Taichi 末帧|首轮末帧绝对误差|本轮末帧绝对误差|本轮相对误差|全程 RMSE|全程最大绝对误差|
|---|---:|---:|---:|---:|---:|---:|---:|
|ice-cream / 最大高度 (m)|0.280411|0.280003|0.023465|0.000408|0.145%|0.001208|0.007095|
|ice-cream / 平均温度|33.924285|33.822790|1.474182|0.101494|0.299%|0.051637|0.101494|
|ice-cream / 平均运动黏度|145.814590|146.756287|9.818037|0.941697|0.646%|1.135712|1.344999|
|hotcut / 最大高度 (m)|1.128894|1.127177|0.807646|0.001717|0.152%|0.001738|0.002681|
|hotcut / 平均温度|2.034380|2.005209|0.489667|0.029170|1.434%|0.014077|0.029170|
|hotcut / 平均运动黏度|16.602076|16.643858|0.974170|0.041783|0.252%|0.021654|0.041783|

**ice-cream**：末帧质心距离 0.000194 m，全程质心距离 RMSE 0.000981 m；真实导出时间与所匹配 Taichi 帧的最大偏差 0.005000 s。

[逐帧 CSV](quantitative/ice-cream/compare.csv) · [全部指标曲线](quantitative/ice-cream/compare.png)

**hotcut**：末帧质心距离 0.000830 m，全程质心距离 RMSE 0.001392 m；真实导出时间与所匹配 Taichi 帧的最大偏差 0.005000 s。

[逐帧 CSV](quantitative/hotcut/compare.csv) · [全部指标曲线](quantitative/hotcut/compare.png)

[本轮与首轮曲线总览](quantitative/overview.png) · [机器可读统计](quantitative/summary.json)

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
.\.venv311\Scripts\python.exe SPH/run_alignment.py --case ice-cream --only taichi
.\.venv311\Scripts\python.exe SPH/run_alignment.py --case ice-cream --only compare
.\.venv311\Scripts\python.exe SPH/run_alignment.py --case hotcut --only taichi
.\.venv311\Scripts\python.exe SPH/run_alignment.py --case hotcut --only compare
.\.venv311\Scripts\python.exe SPH/build_quantitative_report.py
.\.venv311\Scripts\python.exe -m unittest discover -s SPH -p test_alignment.py -v
```

默认热场景使用 `cpp_cg + cpp_dfsph`。可用 `--pressure_solver xpbd` 保留位置约束，
或者直接调用求解器并用 `--cpp_visc_boundary 0` 隔离边界项。新输出分别在 `SPH/output/quant/full/` 与
`SPH/align_results/quantitative/`；原始模拟输出不提交。

## ramp 的新基准问题与尚存限制

> **2026-10-05 更正（Mac 端审查）**：本节"AVX 分支用固定 `m_viscosity`"的判断不成立。
> `NonNewton_Weiler2018.cpp:20` 有 `#undef USE_AVX`，ramp 实际运行标量分支，矩阵、预条件器、RHS 都用逐粒子黏度。
> 另外 ramp 是 Bender2019 边界，`viscosityBoundary: 0.1` 不起作用（`:921` 每步覆盖为本粒子黏度）。
> ramp 已改用 DFSPH + Weiler2018 + C++ 体积图边界完成对齐，结果见 [../ALIGN_CPP.md](../ALIGN_CPP.md) 第 6 节。以下原文保留。


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

---

# Windows C++ / Taichi 数值对齐报告

> 本报告是提交 `11dfc47` 的首轮结果。2026-10-04 的修复和新数据见 [QUANTITATIVE_REPORT.md](QUANTITATIVE_REPORT.md)。
> 以下复现命令须使用旧提交；当前 runner 的热场景分步顺序、输出目录已更新。

2026-10-03。两个仓库位于 `D:/Dev/nonNewton/`。Taichi 从远程 `v2` 的 `346e478` 开始；
交接说明位于 `SPH/ALIGN_CPP.md`，旧说明中的 `main` 与实际远程分支不一致。
C++ 从指定基准 `47d6a53e1eb34a0017ffc383d16981aadce65f41` 开始，修改在独立分支 `align-windows`，
正式参考修复提交为 `83a4acbfdeadaa05298abf5814cdd43b01719302`。

五个场景均使用原粒子分辨率，25 fps；ramp1/2/3 跑满 5 秒，冰淇淋 10 秒，热切割 20 秒。
首帧几何与粒子数已对齐，兼容参数和导出协议已修正。下表直接报告剩余数值误差，不能把这批结果理解为逐粒子或逐帧完全等价。
三个 ramp 的比较使用原 XPBD 黏性路径；两个热场景使用显式选择的 C++ 黏性离散兼容路径 `--cpp_visc_cg`。

## 运行环境与复现

Windows 10；Ryzen 9 7950X、RTX 4090；Visual Studio 2022 / MSVC 19.44、CMake 3.31.3。
独立 Python 3.11.17，Taichi 1.7.2、NumPy 1.26.4、usd-core 26.3。原 Conda Python 的 USD DLL 初始化失败，改用独立解释器解决。

```powershell
# C++：在基准 commit 上应用 cpp_alignment.patch，或使用 align-windows 本地分支
cmake --preset vs2022
cmake --build --preset vs2022-Rel
# Taichi：仓库根目录
.\.venv311\Scripts\python.exe -m pip install -r SPH/requirements-align.in
.\.venv311\Scripts\python.exe SPH/run_alignment.py
.\.venv311\Scripts\python.exe SPH/build_alignment_report.py
.\.venv311\Scripts\python.exe -m unittest discover -s SPH -p test_alignment.py -v
```

`run_alignment.py` 默认寻找同级 `nonNewtonCode`；也可用 `--cpp_repo`。它仅生成 `align_*.json` 副本，不改原场景。
副本删除 ice-cream 的 pauseAt、禁用热切割按 object 拆分导出，写入 stopAt/FPS/属性配置。
`--only cpp|taichi|compare` 可单独重跑阶段；`--visc_solver xpbd|cg|cpp_cg` 可选择局部 XPBD、同一应变能的全局 CG、C++ 离散兼容 CG。
`auto` 在 ramp 选 XPBD，在热场景选 cpp_cg。固定 Taichi dt：ramp 0.002 s，热场景 0.005 s；位置约束 10 次、无散约束 5 次，兼容模式密度松弛 0.5。

## C++ 基准发现与修复

未改数值算法的原版 hotcut 在约 0.20 秒出现全体粒子的 NaN：frame 5 (t≈0.165 s) 尚有限，frame 6 (t≈0.205 s) 的位置、速度、温度、黏度全为 NaN。
原始失败输出保存在 C++ `bin/output/align_failed/hotcut/`。正式热场景参考使用下面的预条件器修复，不能标为“未经修改的 47d6a53”。

`SPlisHSPlasH/My/Viscosity_Casson/Viscosity_Casson.cpp` 的 block diagonal preconditioner 漏乘 matrixVecProd 使用的邻居非牛顿黏度，
并返回了 L 的负半定块，缺少实际矩阵 `I − dt/ρ_i × L` 的单位矩阵和缩放。孤立粒子的块会变成零矩阵。
修复仅令预条件器匹配已有线性算子，不改 matrixVecProd 的黏性离散。先通过 0.5 秒探针，再完成 hotcut 20 秒、ice-cream 10 秒。
两个热场景的全部正式比较帧位置均有限；ramp 的数值算法没有 C++ 端改动。

另两项 C++ 改动为诊断：MyPartio 导出 `frame_times.csv` 记录自适应步长下的真实时间；无 GUI 运行先导出 t=0 的未推进初始状态。
原版首帧发生在第一次推进之后，交接说明的 `(k−1)/fps` 只是目标采样时刻。
完整源码补丁见 [cpp_alignment.patch](cpp_alignment.patch)。原始大体积仿真数据不提交。

## 已核实差异与 Taichi 改动

1. **质量与核半径（交接 5.4）**：`FluidModel::initMasses` 实际使用 `m=0.8×ρ0×(2r)^3`。r=0.025、ρ0=1000 时所有场景 m=0.1，h=0.1。
   兼容模式固定这些值，不再按冰淇淋/兔子采样间距设置质量与核半径。初始稀疏密度保留为参考的物理状态。
2. **边界（交接 5.3）**：热场景直接读 C++ 的 Poisson 边界采样缓存：glass 41073 个点，hotcut 两个边界共 99439 个点。
   同时按 Akinci `V_b=1/Σ_b W` 计算边界体积，密度/压力/无散约束使用 `ρ0 V_b`，不再把边界粒子视为等质量流体。
   热切割保留全部 24872 个流体粒子，不再删掉切割板附近的点；不再添加额外盒子边界。
   ramp 仍是两层斜面粒子，而 C++ 是 Bender2019 体积图。去掉了原 Taichi 人造坡底地面和位置限幅；原 json 中没有承接地面。
3. **边界黏度量纲（交接 5.3）**：Weiler 的实际路径在每步设置 `ν_b,i=ν_i`，覆盖 JSON 的 viscosityBoundary=0.1。
   Akinci 边界项展开后系数是 `10 ν_b ρ0² V_b/ρ_i²`。Taichi 可选显式边界项据此修正；正式 ramp 仍按说明使用 noslip。
   这完成了源码/量纲核查，不等于逐粒子重现 Bender 体积图边界。
4. **温度（交接 5.5）**：hotcut 严格使用世界坐标盒 `[-1,0,-1]–[1,0.05,1]`、默认 rSource=0.1，源项按同相流体邻居数累加。
   ice-cream 的阈值改回 h=0.1 下的邻居数 <10；复刻 C++ 持久保留已标记表面热源的行为。
   复刻 y<0.01 的 xz 摩擦和 4 m/s 限幅；每次温度更新后更新用于导出的黏度，避免 T 与 μ 相差一步。
   C++ 的原地并行温度更新存在顺序/线程依赖；Taichi 保留双缓冲扩散，不复制数据竞争。
5. **热切割实际黏度**：Casson matrixVecProd 还乘 `Materials.viscosity`。ice-cream 的系数为 1，hotcut 为 20。
   所以 hotcut 初始力学运动黏度实际为 20×20=400，而导出的 nonNewtonViscosity 字段仍为 20。
   兼容求解按该力学系数计算；比较 ν_mean 保持字段值 / Taichi μ÷1000，不能将导出字段误当全部力学系数。
6. **应变率（交接 5.2）**：保留末邻居覆盖 bug、Carreau 分母、无黏度上限；末邻居仅取同一 phase。
   CompactNSearch 与 GPU 哈希的邻居顺序不同，逐粒子的应变率仍不能一致。Taichi 对零应变率仍用 1e-6 正则化，避免无穷黏度。
   原 C++ 的 PowerLaw1 和 Casson 在零应变率产生 inf，导致 ν_mean=inf；CSV 额外给出 finite_fraction，不能用这些黏度均值计算相对误差。
7. **稳定性与全局黏性**：兼容模式以 f64 保存位置，避免平移坐标后的 f32 差分生成虚假速度。无界空间哈希保留域外粒子的邻域；
   同一哈希桶用真实 cell 坐标过滤，避免碰撞和重复计数。邻居容量为 128，保留容量/桶溢出诊断。
   增加同一 XPBD 应变能的矩阵无关 SPD CG，并增加热场景的 C++ Casson 离散兼容求解。
   后者用 `q_i=√μ_i v_i` 对角相似变换将参考的 μ_j 算子对称化；预条件器为真实 3×3 对角块。
   C++ 兼容 CG 的相对残差目标 0.01、上限 1000，与 hotcut 配置一致。标量递推在 GPU 执行，每 10 次迭代检查一次，避免逐次 CPU/GPU 同步。
   ramp 保留局部 XPBD；原 bicep 场景和参数没有修改。
8. **对比与导出（交接 5.6）**：对比器流式读取实际 C++ 时间、匹配最接近的 USD 帧，输出实际时间误差、完整六个包围盒边界、非有限黏度比例。
   几何非有限即报错；Taichi 不再在 NaN 提前终止后仍报告完成全部请求步数。

## 完整逐帧结果

### ramp1

[逐帧统计](ramp1/compare.csv) · [全部指标曲线](ramp1/compare.png)

|物体|粒子数 C++ / Taichi|初始包围盒最大差 (m)|末帧 COM 距离 (m)|末帧 y_max C++ / Taichi (m)|末帧平均 ν C++ / Taichi|末帧平均 T C++ / Taichi|
|---|---:|---:|---:|---:|---:|---:|
|Newtonain|51564 / 51564|0|0.4752|4.2985 / 4.3024|10 / 10|—|
|PowerLaw1|51564 / 51564|0|2.1827|6.2861 / 4.3317|inf / 56.985|—|
|PowerLaw2|51564 / 51564|0|3.2250|4.0626 / 4.1963|0.21477 / 0.27254|—|

实际 C++ 导出时间与所配 Taichi 帧的最大偏差：0.004683 s。

### ramp2

[逐帧统计](ramp2/compare.csv) · [全部指标曲线](ramp2/compare.png)

|物体|粒子数 C++ / Taichi|初始包围盒最大差 (m)|末帧 COM 距离 (m)|末帧 y_max C++ / Taichi (m)|末帧平均 ν C++ / Taichi|末帧平均 T C++ / Taichi|
|---|---:|---:|---:|---:|---:|---:|
|Cross|51564 / 51564|0|0.5357|4.3117 / 4.2961|9.6816 / 9.642|—|
|Casson|51564 / 51564|0|1.9551|6.2586 / 4.5158|inf / 1.0029e+05|—|
|Carreau|51564 / 51564|0|0.5099|4.3022 / 4.3090|9.9261 / 9.9104|—|

实际 C++ 导出时间与所配 Taichi 帧的最大偏差：0.003484 s。

### ramp3

[逐帧统计](ramp3/compare.csv) · [全部指标曲线](ramp3/compare.png)

|物体|粒子数 C++ / Taichi|初始包围盒最大差 (m)|末帧 COM 距离 (m)|末帧 y_max C++ / Taichi (m)|末帧平均 ν C++ / Taichi|末帧平均 T C++ / Taichi|
|---|---:|---:|---:|---:|---:|---:|
|Bingham|51564 / 51564|0|0.4941|4.2985 / 4.3114|10 / 10|—|
|HerschelBulkley|51564 / 51564|0|0.4796|4.2964 / 4.3026|10 / 10|—|

实际 C++ 导出时间与所配 Taichi 帧的最大偏差：0.004789 s。

### ice-cream

[逐帧统计](ice-cream/compare.csv) · [全部指标曲线](ice-cream/compare.png)

|物体|粒子数 C++ / Taichi|初始包围盒最大差 (m)|末帧 COM 距离 (m)|末帧 y_max C++ / Taichi (m)|末帧平均 ν C++ / Taichi|末帧平均 T C++ / Taichi|
|---|---:|---:|---:|---:|---:|---:|
|Fluid|6250 / 6250|0|0.0203|0.2804 / 0.2569|145.81 / 155.63|33.9243 / 32.4501|

实际 C++ 导出时间与所配 Taichi 帧的最大偏差：0.005000 s。

### hotcut

[逐帧统计](hotcut/compare.csv) · [全部指标曲线](hotcut/compare.png)

|物体|粒子数 C++ / Taichi|初始包围盒最大差 (m)|末帧 COM 距离 (m)|末帧 y_max C++ / Taichi (m)|末帧平均 ν C++ / Taichi|末帧平均 T C++ / Taichi|
|---|---:|---:|---:|---:|---:|---:|
|Fluid|24872 / 24872|0|0.1761|1.1289 / 0.3212|16.602 / 17.576|2.0344 / 1.5447|

实际 C++ 导出时间与所配 Taichi 帧的最大偏差：0.005000 s。

## 判读与剩余限制

首帧粒子数和包围盒可直接比较；t=0 的 C++ nonNewtonViscosity 尚未计算，零值是初始化状态，不用于评判物理黏度。
ramp 的主要定性排序应看沿坡质心位移，八个模型的具体位移见 summary.json；相近模型的细小排序受邻居顺序和边界离散影响。
两个热场景同时核查 y_max、温度和黏度曲线，表格保留实际误差，不能只看最终静止外观判断对齐。

冰淇淋在 10 秒的最大高度差为 0.02346 m，平均温度相对差为 4.35%，平均运动黏度相对差为 6.73%。
热切割在 20 秒的最大高度为 C++ 1.12889 m、Taichi 0.32125 m，保形仍明显不一致；
平均温度相对差为 24.07%，平均运动黏度相对差为 5.87%。这些相对差均以 C++ 末帧为分母。
三个 ramp 的 PowerLaw2 最远、PowerLaw1/Casson 最短的主要趋势相符，但后两者在 C++ 几乎没有位移，
Taichi 仍有约 0.8 / 0.6 m 的位移，不能视为定量通过。上述差异需继续研究压力与黏性求解的耦合、边界离散和邻居顺序；
仅凭这批曲线不能把热切割差异归因于某一个因素。

剩余差异包括：ramp 的 Bender 体积图与 Taichi 粒子边界、XPBD/PBF 压力与 C++ DFSPH 的不同误差终止规则、固定/自适应时间步，
C++ 热更新原地并行数据竞争，以及末邻居应变率顺序和零应变率正则化。不能用额外热源、改模型分辨率或删除粒子掩盖这些差异。

## 验证

VS2022 Release 构建成功；五个 C++/Taichi 正式场景跑满时长；比较器逐帧检查位置有限并记录真实采样时间。
七项 CPU 数值测试通过：热源世界坐标与全盒子边界、扩散总温度守恒、Akinci 体积归一化、摩擦/温度黏度同步、
域外哈希邻居与微小运动、应变能 CG 的 SPD/稠密解/动量/耗散、C++ 兼容 CG 与独立有限差分核导数构造的稠密解一致。
局部 XPBD 增至 100 次的热切割探针发生严重位移发散，未用作正式结果；原始失败/探针数据均留在忽略的 output 目录。
报告及统计在本地仓库保存；远程同步须注意旧说明的 main 与当前 v2 分支差异。
