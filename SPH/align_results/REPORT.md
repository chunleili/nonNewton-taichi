# Windows C++ / Taichi 数值对齐报告

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
