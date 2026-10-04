# Windows C++ 体积图只读桥接

从已经构建的 nonNewtonCode 链接同一 Discregrid 库，读取 Bender2019 的 `.cdm` 缓存。
返回距离、体积和 SDF 梯度；查询不改变 C++ 模拟状态。当前用于核查 ramp 的边界基准，尚未接入 Taichi 的时间积分。

在 Taichi 仓库根目录构建：

```powershell
cmake -S SPH/cpp_map_bridge -B SPH/output/quant/map-build -G "Visual Studio 17 2022" -A x64 -DCPP_REPO=D:/Dev/nonNewton/nonNewtonCode
cmake --build SPH/output/quant/map-build --config Release
.\.venv311\Scripts\python.exe SPH/check_cpp_volume_map.py --particles D:/Dev/nonNewton/nonNewtonCode/bin/output/align/ramp1/mypartio/ParticleData_Newtonain_126.bgeo.gz
```

检查必须有至少 20 个近边界粒子，比较插值函数梯度与中央有限差分。
`--map` 可指定其他缓存，`--translation` 指定世界坐标下的刚体平移，查询会减掉该平移。
当前桥接只处理平移；一般刚体旋转需要调用方先转换坐标。
DLL 和构建产物在忽略的 `SPH/output/` 中，不提交二进制文件。
