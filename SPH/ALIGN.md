# C++ / Taichi 对齐

交接说明在 [ALIGN_CPP.md](ALIGN_CPP.md)。Windows 验证、代码修复和逐帧结果在
[align_results/REPORT.md](align_results/REPORT.md)。远程求解器位于 `v2` 分支。

在仓库根目录使用 Python 3.11：

```powershell
python -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r SPH/requirements-align.in
.\.venv311\Scripts\python.exe SPH/run_alignment.py
.\.venv311\Scripts\python.exe SPH/build_alignment_report.py
```

默认 C++ 仓库为同级 `nonNewtonCode`，须先在基准 `47d6a53` 上应用报告列出的
导出诊断和 Casson 预条件器修复，然后运行 `cmake --preset vs2022` 与
`cmake --build --preset vs2022-Rel`。可用 `--cpp_repo` 指定其他位置。

`run_alignment.py` 为原场景生成 `align_*.json` 副本，依次运行五个场景并生成对比。
`--only cpp|taichi|compare` 和 `--case ramp1|ramp2|ramp3|ice-cream|hotcut` 可单独重跑阶段/场景。
所有正式结果使用原分辨率、25 fps、ramp 5 秒、冰淇淋 10 秒、热切割 20 秒。
默认 ramp 使用 XPBD，热场景使用 `--cpp_visc_cg`，按 C++ Casson 离散进行全局黏性求解。
`--visc_solver xpbd|cg|cpp_cg` 可切换比较局部约束、同一应变能的全局求解与 C++ 兼容求解。
原始仿真输出位于忽略目录，仅提交 `align_results/` 的统计、曲线与报告。
