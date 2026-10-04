# 检查与复现

命令为 PowerShell。包内源码目录为同级 `nonNewton-taichi/`、`nonNewtonCode/`；以下用 `$reviewRoot` 指向包根目录。源码快照已包含修改，不要再次应用补丁。首次构建 C++ 可能需要联网下载其依赖。

## 只读检查优先

先读 CSV、summary.json 和 `evidence/` 日志。执行包根 `VERIFY_PACKAGE.ps1` 可以校验文件 SHA-256 与关键 CSV 行数，不需要安装仿真环境。
`evidence/cpp_samples/` 每个物体仅带首末帧，附完整 `frame_times.csv`。这不是完整导出目录，不应直接传给 `compare_cpp.py` 生成全程结论。

## 环境和测试

```powershell
$reviewRoot = 'D:/Dev/nonNewton/CLAUDE_REVIEW_2026-10-04'
Set-Location "$reviewRoot/nonNewton-taichi"
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r SPH/requirements-align.in
$env:TI_OFFLINE_CACHE_FILE_PATH = "$reviewRoot/nonNewton-taichi/SPH/output/ti-cache"
.\.venv311\Scripts\python.exe -m unittest discover -s SPH -p test_alignment.py -v
```

本机原环境为 `D:/Dev/nonNewton/nonNewton-taichi/.venv311/Scripts/python.exe`；独立解释器为 `D:/Dev/nonNewton/runtimes/cpython-3.11.17-windows-x86_64-none/python.exe`。
完整重跑需 GPU/CUDA；CPU 数值测试不需要 CUDA。Matplotlib 原运行版本为 3.11.2，requirements 中未固定它；必要时单独固定。

## 构建并生成完整 C++ 热场景参考

```powershell
Set-Location "$reviewRoot/nonNewtonCode"
cmake --preset vs2022 -DUSE_AVX=ON -DUSE_PERFORMANCE_OPTIMIZATION=ON -DUSE_DOUBLE_PRECISION=OFF
cmake --build --preset vs2022-Rel
Set-Location "$reviewRoot/nonNewton-taichi"
.\.venv311\Scripts\python.exe SPH/run_alignment.py --case ice-cream --only cpp --threads 12
.\.venv311\Scripts\python.exe SPH/run_alignment.py --case hotcut --only cpp --threads 12
```

runner 默认寻找同级 C++ 仓库；可以加 `--cpp_repo`。C++ 热更新具有线程/顺序依赖，新运行未必逐位复现旧 CSV。请记录编译配置、线程数、代码版本和真实帧时间。
包内已经提供原运行的边界缓存供读取。完整 C++ 输出会新生成到包内 `nonNewtonCode/bin/output/align/`，可能占用较多空间。

## 重跑第二轮 Taichi 和统计

```powershell
Set-Location "$reviewRoot/nonNewton-taichi"
.\.venv311\Scripts\python.exe SPH/run_alignment.py --case ice-cream --only taichi
.\.venv311\Scripts\python.exe SPH/run_alignment.py --case hotcut --only taichi
.\.venv311\Scripts\python.exe SPH/run_alignment.py --case ice-cream --only compare
.\.venv311\Scripts\python.exe SPH/run_alignment.py --case hotcut --only compare
.\.venv311\Scripts\python.exe SPH/build_quantitative_report.py
```

默认使用 `cpp_cg + cpp_dfsph`、原粒子分辨率、25 fps、10/20 秒、固定 dt=0.005 s。输出 `SPH/output/quant/full/`；统计/图在 `SPH/align_results/quantitative/`。
以上命令会覆盖同名第二轮结果/报告。请先复制一份待审包，或通过 `--output_dir`、`--result_dir` 将新结果放到其他目录并手动对照。报告生成器默认读原统计目录，不会自动读取自定义目录。
首轮报告属于 `11dfc47`。重现首轮需另外克隆/切换该提交；用当前 runner 不会重现旧热场景结果。

## 体积图桥接

```powershell
Set-Location "$reviewRoot/nonNewton-taichi"
cmake -S SPH/cpp_map_bridge -B SPH/output/quant/map-build -G 'Visual Studio 17 2022' -A x64 "-DCPP_REPO=$reviewRoot/nonNewtonCode"
cmake --build SPH/output/quant/map-build --config Release
.\.venv311\Scripts\python.exe SPH/check_cpp_volume_map.py --particles "$reviewRoot/evidence/cpp_samples/ramp1/ParticleData_Newtonain_126.bgeo.gz"
```

需要先完成 C++ 构建。包内带同一 ramp `.cdm` 缓存。上述末帧有接触粒子，t=0 不适合作为近边界梯度检查样本。

## 后续 ramp

本轮没有重跑 ramp；当前 runner 的 ramp 默认仍为 XPBD，不代表已经解决 AVX、inf 或体积图边界问题。先审查并确定 C++ 基准路线，再决定修改和重跑，保留原基准与数据供比较。
