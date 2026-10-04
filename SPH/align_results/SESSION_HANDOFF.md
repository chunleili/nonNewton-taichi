# 会话关键信息与工程交接

整理日期：2026-10-04（Asia/Shanghai）。这是关键决策和工作状态的摘要，不是会话逐字记录。

## 用户任务和本次交付

用户要求在 `D:/Dev/nonNewton` 克隆 `https://github.com/chunleili/nonNewton-taichi`，按照 SPH 的 ALIGN 指示对齐 C++ 和 Taichi，随后要求继续定量对齐，并把完整报告与关键信息整理成一个可直接复制给 Claude 的目录/压缩包。

第一轮完成五个场景的全时长运行、基准修复及逐帧统计；第二轮修复两个热场景的边界黏性与压力分步，重新完成 10 秒/20 秒验证。本次只整理交接资料，不改变求解器或重新定义 C++ 基准。

## 仓库和确定版本

|项目|本机目录|分支|用于结果的版本|
|---|---|---|---|
|Taichi 初始代码|`D:/Dev/nonNewton/nonNewton-taichi`|`v2`|`346e478b64ff7db8ae2a94a8b332ac962f706fc0`|
|Taichi 首轮|同上|`v2`|`11dfc4709606390898eb41b2ae4f2787281ece44`|
|Taichi 第二轮代码|同上|`v2`|`8791c37144560c521d1f8bc6c47c94d5ca6f0dc1`|
|C++ 原基准|`D:/Dev/nonNewton/nonNewtonCode`|原基准|`47d6a53e1eb34a0017ffc383d16981aadce65f41`|
|C++ 正式参考|同上|`align-windows`|`83a4acbfdeadaa05298abf5814cdd43b01719302`|

Taichi 地址为 `https://github.com/chunleili/nonNewton-taichi`，C++ 地址为 `https://github.com/chunleili/nonNewtonCode`。
远程 Taichi `main` 为 MPM；本任务的 SPH 求解器实际在 `v2`。旧 `ALIGN_CPP.md` 的历史文字需结合新 `ALIGN.md` 阅读。
第二轮源码和 C++ 参考均已推送；交接文档可能有后续提交，数值结果仍对应上表版本。
包内两边都是源码快照，没有 `.git`、虚拟环境或编译产物；补丁用于审查历史差异，不能再应用到已经修改的快照。

## 当前结论及边界

|场景|验证时长 / 帧 / 粒子|末帧高度差|末帧平均温度相对差|末帧平均运动黏度相对差|
|---|---|---:|---:|---:|
|ice-cream|10 s / 251 / 6250|0.408 mm|0.299%|0.646%|
|hotcut|20 s / 501 / 24872|1.717 mm|1.434%|0.252%|
|ramp1/2/3|各 5 s / 各 126 帧 / 每物体 51564|仍有明显差异|无温度比较|部分 C++ 模型为 inf|

CSV 每物体每帧一行，因此 ramp1/2 各 378 行、ramp3 为 252 行。
热场景所有比较帧粒子数一致、位置有限，采样诊断未发现桶溢出、邻居截断或域外计数问题；这不是逐步记录的穷尽证明。
比较按真实 C++ 时间匹配最接近的 USD 帧，最大时间偏差约 0.005 s。黏度 RMSE 排除 C++ 尚未更新黏度的 t=0。
全部指标、全程 RMSE 和最大误差见完整报告，不能只凭末帧高度判定求解器完全等价。

## 实际修改与源码入口

1. C++ 原 hotcut 在约 0.20 s 变成全 NaN。正式参考修复 Casson 块预条件器，使其包含矩阵原有的邻居黏度、单位矩阵及时间/密度缩放；没有改变热场景矩阵离散、材料或热源。另增加真实帧时间和无 GUI t=0 导出，生成五个 `align_*.json` 副本。详见 `cpp_alignment.patch`。
2. 首轮 Taichi 修正质量/核半径、Akinci 边界体积与缓存、热源坐标、表面热源、摩擦和黏度更新时机、哈希域与容量、非有限检测、时间匹配，并加入独立的 CG 模式。兼容位置用 f64，其余主要字段及 USD 导出仍为 f32。
3. 第二轮为 Casson 算子和 3×3 预条件器同时补齐静态边界黏性（参考默认 ν_b=0.1），加入可选 `--cpp_dfsph`，匹配步初邻域、散度、温度/黏度、CG、重力、线性化压力、积分顺序和 warmstart。ice-cream CG 上限为 100，hotcut 为 1000。
4. 热切割 `nonNewtonViscosity` 导出值与力学系数有区别：流体矩阵另乘材料 `viscosity=20`。Taichi 保留该力学倍数，CSV 的 ν 仍比较导出字段。
5. 已准备 Discregrid 只读 `.cdm` 桥接器，在 20 个真实近边界点检查 SDF 梯度与有限差分一致；仅支持调用方处理平移，尚未接入 ramp 的时间积分。

|检查内容|源码入口|
|---|---|
|Taichi 求解及场景|`SPH/constraint_solver.py`|
|C++ Casson 矩阵/预条件器|`SPlisHSPlasH/My/Viscosity_Casson/Viscosity_Casson.cpp`|
|C++ ramp AVX/标量|`SPlisHSPlasH/My/NonNewton/NonNewton_Weiler2018.cpp`|
|非牛顿本构/应变率|`SPlisHSPlasH/My/NonNewton/NonNewton.cpp` 及相关实现|
|C++ 压力和分步|`SPlisHSPlasH/DFSPH/TimeStepDFSPH.cpp`|
|C++ 温度|`SPlisHSPlasH/My/Diffusion/Coagulation.cpp`|
|导出|`Simulator/Exporter/ParticleExporter_MyPartio.cpp`、`Simulator/SimulatorBase.cpp`|
|对比/报告|`SPH/bgeo_io.py`、`compare_cpp.py`、`build_quantitative_report.py`|
|复现与测试|`SPH/run_alignment.py`、`SPH/test_alignment.py`|
|体积图桥接|`SPH/cpp_volume_map.py`、`cpp_map_bridge/`、`check_cpp_volume_map.py`|

## 已执行验证

- Windows 10，Ryzen 9 7950X / RTX 4090；VS2022 Release / MSVC 19.44 / CMake 3.31.3。
- C++ `USE_AVX=ON`、`USE_PERFORMANCE_OPTIMIZATION=ON`、`USE_DOUBLE_PRECISION=OFF`。
- 独立 Python 3.11.17、Taichi 1.7.2、NumPy 1.26.4、usd-core 26.3、Matplotlib 3.11.2。原 Conda Python 的 USD DLL 初始化失败，使用独立解释器解决。
- 10 项 CPU 数值测试通过：先运行 8 项（141.363 s），新增 DFSPH 后运行 2 项（44.209 s），不是在最终版本一次运行完整 10 项。包内保留两份原始日志；复核可以重新跑完整套件。
- 两个热场景完整 CUDA 运行完成：ice-cream 2000 步，215.0 s；hotcut 4000 步，587.8 s。两次运行同时使用 GPU，耗时不是独占性能测试。
- ice-cream 部分 CG 步触及 100 次上限，采样 q 残差最高约 0.2473；不能宣称所有步骤都达到 0.01。hotcut 采样最高 CG 次数 248、残差约 0.009997。
- 第二轮代码做过 Python 编译检查与 diff 空白检查。桥接梯度检查确实有 20 个有效近边界查询，不是空样本测试。

## 未解决事项与已询问的问题

C++ ramp 实际启用 AVX。其矩阵使用固定 `m_viscosity` / `m_boundaryViscosity`，标量分支使用逐粒子非牛顿黏度；预条件器/RHS 又使用逐粒子黏度。ramp JSON 只设 `viscosity0`，基础 `viscosity` 默认 0.01。PowerLaw1/Casson 零应变率产生 inf，CG 日志出现 NaN 残差。

已询问用户：修复 C++ 算子与零应变率后建立有限的新基准，还是保留现有行为进行复刻。尚未收到选择；用户最新要求是先整理资料给 Claude 检查。未擅自修改这些 C++ 物理/数值行为，也未重新运行第二轮 ramp。

其他限制：ramp 两层粒子边界与 C++ Bender 体积图仍不同；应变率末邻居覆盖 bug 与邻居顺序有关；Taichi 零应变率用 1e-6 正则化；C++ 温度原地并行更新有数据竞争，Taichi 双缓冲；固定/自适应步长、浮点求和、位置/导出精度不同；CG 在 v/q 空间的停止量不同。压力与积分实际采用的 dt 是否在 C++ CFL 更新前后完全对应，也应由审查者核实。

## 本机完整原始数据位置

- C++ 五个正式场景：`D:/Dev/nonNewton/nonNewtonCode/bin/output/align/{ramp1,ramp2,ramp3,ice-cream,hotcut}/mypartio/`，含全部 `.bgeo.gz` 和 `frame_times.csv`，约 2.0 GB。
- C++ 原版失败热切割：`bin/output/align_failed/hotcut/`；单步诊断：`bin/output/quant/step/mypartio/`。
- Taichi 首轮：`D:/Dev/nonNewton/nonNewton-taichi/SPH/output/align/`。
- Taichi 第二轮：`SPH/output/quant/full/{icecream_cpp,hotcut_cpp}.usdc` 和同名 JSON，两个 USD 共约 449 MB。
- 热切割短测：`SPH/output/quant/{boundary-only,dfsph,step}/`。
- C++ 日志在 `nonNewtonCode/bin/output-*.log`；Taichi 第二轮日志在 `SPH/output/quant-*-full.log`，测试日志在 `SPH/tests-quant-*.log`。

交接包带完整源码、全部已提交统计/曲线、上述关键日志、热场景运行 JSON、边界缓存、五个 C++ 场景各物体首末帧样本与完整时间表。完整原始时间序列不在包内；仅凭包内样本不能重新计算全部逐帧统计。需要时在本机读上列目录，或按复现文档重跑。
