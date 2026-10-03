## bicep 肌肉场景
```
  python3 constraint_solver.py --scene bicep --E 1e6 --mu_belly 5 --k_fiber 1e7 --act_start 0.3 --steps 1300 --export usd --out
  output/bicep
```
  --k_fiber、--eps_fiber、--omega_f、--pin both|top、--act_start/--act_ramp/--act_hold。激活曲线是：静置一段时间，线性升到1，保持一段，再线性降回 0。

