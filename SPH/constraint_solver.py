# -*- coding: utf-8 -*-
"""基于约束的非牛顿粒子求解器（PBF/XPBD，DFSPH 式双层不可压缩约束）。

对应论文第 4 章“基于约束的非牛顿流体仿真”：
  - 密度约束（位置层） C = rho/rho0 - 1，单边 lambda <= 0
  - 无散约束（速度层） C_dot = (1/rho0) Drho/Dt，与密度约束共享系统矩阵
  - 黏性约束 C = ||dev(d_eps)||，柔度 alpha = dt / (2 mu V)，mu 由六种应变率黏度模型给出
  - 共旋弹性约束（偏量 + 体积），von Mises 塑性回映更新静息状态
  - 温度扩散，通过 mu(T)、G(T) 调制柔度实现热致相变
所有约束均以 XPBD 乘子更新求解：同类约束 Jacobi 并行，不同类约束 Gauss-Seidel 先后。
"""
import argparse
import json
import math
import os
import time

import numpy as np
import taichi as ti

MAX_NB = 64
MAX_CELL = 48
MAX_BODY = 16

NEWTONIAN, POWER_LAW, CROSS, CASSON, CARREAU, BINGHAM, HERSCHEL_BULKLEY = range(7)
MODEL_NAMES = ["Newtonian", "PowerLaw", "Cross", "Casson", "Carreau", "Bingham", "HerschelBulkley"]


def default_material(**kw):
    mat = dict(
        model=NEWTONIAN,
        mu0=0.01,  # 零剪切黏度 / 上限
        mu_inf=0.0,  # 无穷剪切黏度 / 下限
        m=1.0,  # 稠度系数 / 松弛时间
        n=1.0,  # 幂指数
        crit=1.0,  # 临界应变率
        muC=1.0,  # Casson 黏度
        tau0=0.0,  # Casson 屈服应力
        E=0.0,  # 杨氏模量，0 表示无弹性
        nu=0.3,  # 泊松比
        gamma1=1e30,  # 弹性极限（von Mises）
        gamma2=1e30,  # 塑性极限
        decay_mu=0.0,  # mu = mu_model * exp(-decay_mu * T)
        decay_G=0.0,  # G = G0 * exp(-decay_G * T)
        k_fiber=0.0,  # 主动纤维约束刚度，0 表示无主动收缩
        eps_fiber=0.3,  # 满激活时的目标纤维缩短应变
    )
    mat.update(kw)
    return mat


@ti.data_oriented
class ConstraintSolver:
    def __init__(
        self,
        pos,
        body,
        materials,
        spacing,
        domain,
        dt,
        gravity=(0.0, -9.81, 0.0),
        iters=10,
        div_iters=5,
        omega=1.0,
        omega_s=1.0,
        use_div=True,
        friction=1.0,
        heat_y=-1.0,
        heat_R=0.0,
        diffusion=0.0,
        density_compliance=0.0,
        mu_scale=1.0,
        cpp_compat=False,
        box_walls=True,
        boundary=None,
        surf_source=0.0,
        surf_thresh=0,
        visc_boundary=None,
        v_clamp=0.0,
        pinned=None,
        obj=None,
        rest_kernel_sum=None,
        fiber_dir=None,
        fiber_w=None,
        omega_f=0.25,
        particle_mass=None,
        rest_density=None,
        heat_box=None,
        cpp_offset=(0.0, 0.0, 0.0),
        boundary_akinci=False,
        clip_domain=True,
        viscosity_force_scale=1.0,
        visc_cg=False,
        cpp_visc_cg=False,
        cpp_visc_boundary=0.1,
        cpp_dfsph=False,
        cpp_visc_max_iters=1000,
    ):
        self.d = spacing
        self.h = 2.0 * spacing  # 支撑半径 = 4 倍粒子半径
        self.dt = dt
        self.iters = iters
        self.div_iters = div_iters
        self.omega = omega
        self.omega_s = omega_s  # 黏性/弹性约束的 Jacobi 松弛
        self.omega_f = omega_f  # 主动纤维约束的 Jacobi 松弛；纤维约束的雅可比在邻域内高度重叠，需更小的松弛才不振荡
        self.use_div = use_div
        self.friction = friction
        self.heat_y = heat_y
        self.heat_R = heat_R
        self.heat_box = heat_box
        self.has_heat_box = heat_box is not None
        self.cpp_offset = ti.Vector(list(cpp_offset))
        self.boundary_akinci = boundary_akinci
        self.clip_domain = clip_domain
        self.viscosity_force_scale = viscosity_force_scale
        self.visc_cg = visc_cg
        self.cpp_visc_cg = cpp_visc_cg
        self.cpp_visc_boundary = cpp_visc_boundary
        self.cpp_dfsph = cpp_dfsph
        self.cpp_visc_max_iters = cpp_visc_max_iters
        self.diffusion = diffusion
        self.mu_scale = mu_scale  # 黏度模型输出乘子；nonNewtonCode 的黏度为运动黏度，取 rho0 换算为动力黏度
        self.cpp_compat = cpp_compat  # True：复刻 nonNewtonCode 的应变率（仅末邻居）、Carreau 写法、不截断黏度
        self.max_nb = 128 if cpp_compat else MAX_NB
        self.surf_source = surf_source  # 表面热源（邻居数 < surf_thresh 的粒子），对应论文冰淇淋融化
        self.surf_thresh = surf_thresh
        # None：边界粒子作为位移为 0 的邻居参与黏性约束（无滑移，黏度同流体）
        # 数值：边界不参与黏性约束，改用 nonNewtonCode 的 Akinci 显式边界黏性，visc_boundary 为运动黏度 nu_b
        self.visc_bnd = visc_boundary
        self.skip_bnd_visc = visc_boundary is not None  # Taichi 作用域内不支持 'is not'
        self.v_clamp = v_clamp  # 速度上限，0 表示不限制（nonNewtonCode Coagulation 中为 4 m/s）
        self.domain = np.array(domain, dtype=np.float64)
        self.pad = 0.5 * spacing
        self.gravity = ti.Vector(list(gravity))
        # 非立方点阵采样（如 Houdini 的六方密排）时由场景给出初始构型的核函数和，使静息密度为 1000
        ksum = rest_kernel_sum or self._lattice_kernel_sum()
        self.mass = 1000.0 * spacing**3 if rest_kernel_sum is None else 1000.0 / ksum
        self.rho0 = self.mass * ksum
        if particle_mass is not None:
            self.mass = particle_mass
        if rest_density is not None:
            self.rho0 = rest_density
        self.V = self.mass / self.rho0
        self.kappa_inv = density_compliance  # alpha_rho * V，0 即严格不可压
        self.has_elastic = any(mt["E"] > 0 for mt in materials)
        self.has_fiber = fiber_dir is not None and any(mt["k_fiber"] > 0 for mt in materials)
        self.has_heat = heat_R != 0.0 or diffusion != 0.0 or surf_source != 0.0
        self.materials = materials
        # 静态边界粒子（逆质量 w=0）：两层，覆盖地面与四周侧壁
        bpos = self._boundary_points() if box_walls else np.zeros((0, 3))
        if boundary is not None:
            bpos = np.concatenate([bpos, boundary])
        self.nf = len(pos)
        self.n = n = self.nf + len(bpos)
        pos = np.concatenate([pos, bpos])
        body = np.concatenate([body, -np.ones(len(bpos), np.int32)])
        self.origin = -3.0 * spacing

        # 粒子状态
        position_type = ti.f64 if cpp_compat else ti.f32
        self.x = ti.Vector.field(3, position_type, n)
        self.x_old = ti.Vector.field(3, position_type, n)
        self.x0 = ti.Vector.field(3, position_type, n)
        self.v = ti.Vector.field(3, ti.f32, n)
        self.dx = ti.Vector.field(3, ti.f32, n)
        self.body = ti.field(ti.i32, n)
        self.pin = ti.field(ti.i32, n)  # 1：固定在参考位置（如肌腱端点）
        self.fext = ti.Vector.field(3, ti.f32, n)  # 逐粒子外加速度（力 / 质量），用于 Neumann / Robin 边界
        self.obj = ti.field(ti.i32, n)  # 弹性参考邻域按 obj 划分，默认同 body；同一物体内可含多种材料
        self.rho = ti.field(ti.f32, n)
        self.pmass = ti.field(ti.f32, n)
        self.T = ti.field(ti.f32, n)
        self.T_new = ti.field(ti.f32, n)
        self.surface_source = ti.field(ti.f32, n)
        self.mu = ti.field(ti.f32, n)
        self.sr = ti.field(ti.f32, n)
        self.Vt = ti.field(ti.f32, n)  # 参考体积
        self.R = ti.Matrix.field(3, 3, ti.f32, n)
        self.epsP = ti.Matrix.field(3, 3, ti.f32, n)
        self.Nmat = ti.Matrix.field(3, 3, ti.f32, n)
        # 乘子
        self.lam_rho = ti.field(ti.f32, n)
        self.lam_div = ti.field(ti.f32, n)
        self.lam_vis = ti.field(ti.f32, n)
        self.lam_dev = ti.field(ti.f32, n)
        self.lam_vol = ti.field(ti.f32, n)
        self.lam_fib = ti.field(ti.f32, n)
        # 主动纤维：参考构型下的单位纤维方向、逐粒子激活权重（肌腱为 0）、全局激活 a(t)
        self.fdir = ti.Vector.field(3, ti.f32, n)
        self.fw = ti.field(ti.f32, n)
        self.activation = ti.field(ti.f32, ())
        self.dlam = ti.field(ti.f32, n)
        # 邻域
        self.nb = ti.field(ti.i32, (n, self.max_nb))
        self.nb_cnt = ti.field(ti.i32, n)
        self.ref_nb = ti.field(ti.i32, (n, self.max_nb))
        self.ref_cnt = ti.field(ti.i32, n)
        self.g0 = ti.Vector.field(3, ti.f32, (n, self.max_nb))
        self.gdim = [int(math.ceil((L + 6.0 * spacing) / self.h)) + 1 for L in self.domain]
        self.hash_size = 1 << int(math.ceil(math.log2(max(1024, 4 * n))))
        cell_shape = [self.hash_size] if cpp_compat else self.gdim
        self.cell_count = ti.field(ti.i32, cell_shape)
        self.cell_list = ti.field(ti.i32, cell_shape + [MAX_CELL])
        # 材料表
        self.b_model = ti.field(ti.i32, MAX_BODY)
        self.b_elastic = ti.field(ti.i32, MAX_BODY)
        self.b_par = ti.field(ti.f32, (MAX_BODY, 16))
        # 诊断
        self.stat = ti.field(ti.f32, 4)
        self.cg_x = ti.Vector.field(3, ti.f32, n)
        self.cg_r = ti.Vector.field(3, ti.f32, n)
        self.cg_p = ti.Vector.field(3, ti.f32, n)
        self.cg_Ap = ti.Vector.field(3, ti.f32, n)
        self.cg_D = ti.Matrix.field(3, 3, ti.f32, n)
        self.cg_diag = ti.field(ti.f32, n)
        self.cg_rhs = ti.Vector.field(3, ti.f32, n)
        self.cg_scale = ti.field(ti.f32, n)
        self.cg_Minv = ti.Matrix.field(3, 3, ti.f32, n)
        self.cg_state = ti.field(ti.f64, 8)
        self.cg_vdiff = ti.Vector.field(3, ti.f32, n)
        self.df_factor = ti.field(ti.f32, n)
        self.df_adv = ti.field(ti.f32, n)
        self.df_kappa = ti.field(ti.f32, n)
        self.df_kappa_v = ti.field(ti.f32, n)
        self.df_stat = ti.field(ti.f32, 2)

        position_dtype = np.float64 if cpp_compat else np.float32
        self.x.from_numpy(pos.astype(position_dtype))
        self.x0.from_numpy(pos.astype(position_dtype))
        self.x_old.from_numpy(pos.astype(position_dtype))
        self.body.from_numpy(body.astype(np.int32))
        pin = np.zeros(n, np.int32)
        if pinned is not None:
            pin[: self.nf] = pinned
        self.pin.from_numpy(pin)
        self.obj.from_numpy((body if obj is None else np.concatenate([obj, body[self.nf :]])).astype(np.int32))
        fd = np.zeros((n, 3), np.float32)
        fw = np.zeros(n, np.float32)
        if fiber_dir is not None:
            fd[: self.nf] = fiber_dir / np.linalg.norm(fiber_dir, axis=1, keepdims=True)
            fw[: self.nf] = 1.0 if fiber_w is None else fiber_w
        self.fdir.from_numpy(fd)
        self.fw.from_numpy(fw)
        self._load_materials(materials)
        self._init_state()
        self.build_grid(self.x)
        self.init_boundary_mass()
        self.find_neighbors(self.x)
        self.init_reference()

    # ------------------------------------------------------------------ setup
    def _boundary_points(self):
        d = self.d
        L = self.domain
        g = [np.arange(-1.5 * d, L[a] + 1.6 * d, d) for a in range(3)]
        P = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
        inside = np.ones(len(P), bool)
        for a in range(3):
            if a == 1:
                inside &= P[:, 1] > 0.0  # 顶部开口
            else:
                inside &= (P[:, a] > 0.0) & (P[:, a] < L[a])
        return P[~inside].astype(np.float64)

    def _lattice_kernel_sum(self):
        d, h = self.d, self.h
        r = int(math.ceil(h / d))
        s = 0.0
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                for k in range(-r, r + 1):
                    s += self._W_np(d * math.sqrt(i * i + j * j + k * k))
        return s

    def _W_np(self, r):
        h = self.h
        k = 8.0 / (math.pi * h**3)
        q = r / h
        if q > 1.0:
            return 0.0
        if q <= 0.5:
            return k * (6 * q**3 - 6 * q**2 + 1)
        return k * 2 * (1 - q) ** 3

    def _load_materials(self, materials):
        par = np.zeros((MAX_BODY, 16), np.float32)
        model = np.zeros(MAX_BODY, np.int32)
        elastic = np.zeros(MAX_BODY, np.int32)
        for b, mt in enumerate(materials):
            model[b] = mt["model"]
            elastic[b] = 1 if mt["E"] > 0 else 0
            E, nu = mt["E"], mt["nu"]
            G = E / (2 * (1 + nu)) if E > 0 else 0.0
            K = E / (3 * (1 - 2 * nu)) if (E > 0 and nu < 0.49) else 0.0  # 近不可压时交给密度约束
            par[b, :15] = [
                mt["mu0"], mt["mu_inf"], mt["m"], mt["n"], mt["crit"], mt["muC"], mt["tau0"],
                G, K, mt["gamma1"], mt["gamma2"], mt["decay_mu"], mt["decay_G"], mt["k_fiber"], mt["eps_fiber"],
            ]
        self.b_par.from_numpy(par)
        self.b_model.from_numpy(model)
        self.b_elastic.from_numpy(elastic)

    @ti.kernel
    def _init_state(self):
        for i in range(self.n):
            self.v[i] = ti.Vector([0.0, 0.0, 0.0])
            self.R[i] = ti.Matrix.identity(ti.f32, 3)
            self.epsP[i] = ti.Matrix.zero(ti.f32, 3, 3)
            self.T[i] = 0.0

    # ------------------------------------------------------------------ kernels
    @ti.func
    def W(self, r):
        h = self.h
        k = 8.0 / (math.pi * h**3)
        q = r / h
        res = 0.0
        if q <= 1.0:
            if q <= 0.5:
                res = k * (6.0 * q * q * q - 6.0 * q * q + 1.0)
            else:
                res = k * 2.0 * (1.0 - q) ** 3
        return res

    @ti.func
    def gradW(self, r):  # r = x_i - x_j，返回 \nabla_{x_i} W
        h = self.h
        k = 48.0 / (math.pi * h**3)
        rn = r.norm()
        q = rn / h
        res = ti.Vector([0.0, 0.0, 0.0])
        if rn > 1e-9 and q <= 1.0:
            gq = r / (rn * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * gq
            else:
                res = -k * (1.0 - q) ** 2 * gq
        return res

    @ti.func
    def dev(self, A):
        return A - A.trace() / 3.0 * ti.Matrix.identity(ti.f32, 3)

    # ------------------------------------------------------------------ neighbor search
    @ti.kernel
    def init_boundary_mass(self):
        for i in range(self.n):
            self.pmass[i] = self.mass
            if ti.static(self.boundary_akinci):
                if i >= self.nf:
                    c = self.cell_of(self.x[i])
                    delta = self.W(0.0)
                    for off in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                        cc = c + off
                        if self.in_grid(cc):
                            ci = self.cell_index(cc)
                            for k in range(ti.min(self.cell_count[ci], MAX_CELL)):
                                j = self.cell_list[ci, k]
                                if j >= self.nf and j != i and (self.cell_of(self.x[j]) == cc).all():
                                    delta += self.W((self.x[i] - self.x[j]).norm())
                    self.pmass[i] = self.rho0 / delta

    @ti.func
    def cell_of(self, p):
        q = (p - self.origin) / self.h
        return ti.Vector([int(ti.floor(q[0])), int(ti.floor(q[1])), int(ti.floor(q[2]))])

    @ti.func
    def in_grid(self, c):
        valid = True
        if ti.static(not self.cpp_compat):
            valid = 0 <= c[0] < self.gdim[0] and 0 <= c[1] < self.gdim[1] and 0 <= c[2] < self.gdim[2]
        return valid

    @ti.func
    def cell_index(self, c):
        if ti.static(self.cpp_compat):
            return ((c[0] * 73856093) ^ (c[1] * 19349663) ^ (c[2] * 83492791)) & (self.hash_size - 1)
        else:
            return c

    @ti.kernel
    def build_grid(self, xs: ti.template()):
        for I in ti.grouped(self.cell_count):
            self.cell_count[I] = 0
        for i in range(self.n):
            c = self.cell_of(xs[i])
            if self.in_grid(c):
                ci = self.cell_index(c)
                k = ti.atomic_add(self.cell_count[ci], 1)
                if k < MAX_CELL:
                    self.cell_list[ci, k] = i

    @ti.kernel
    def find_neighbors(self, xs: ti.template()):
        for i in range(self.nf):
            c = self.cell_of(xs[i])
            cnt = 0
            for off in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cc = c + off
                if self.in_grid(cc):
                    ci = self.cell_index(cc)
                    for k in range(ti.min(self.cell_count[ci], MAX_CELL)):
                        j = self.cell_list[ci, k]
                        if j != i and (self.cell_of(xs[j]) == cc).all() and (xs[i] - xs[j]).norm() < self.h and cnt < self.max_nb:
                            self.nb[i, cnt] = j
                            cnt += 1
            self.nb_cnt[i] = cnt

    @ti.kernel
    def init_reference(self):
        # 弹性约束的参考构型：固定邻域（仅同一物体）、参考体积、参考核梯度
        for i in range(self.nf):
            rho = self.mass * self.W(0.0)
            cnt = 0
            for jj in range(self.nb_cnt[i]):
                j = self.nb[i, jj]
                if self.obj[j] == self.obj[i]:
                    rho += self.mass * self.W((self.x0[i] - self.x0[j]).norm())
                    self.ref_nb[i, cnt] = j
                    self.g0[i, cnt] = self.gradW(self.x0[i] - self.x0[j])
                    cnt += 1
            self.ref_cnt[i] = cnt
            self.Vt[i] = self.mass / ti.max(rho, 0.5 * self.rho0)

    # ------------------------------------------------------------------ viscosity models
    @ti.func
    def viscosity(self, b, sr):
        mu0 = self.b_par[b, 0]
        mui = self.b_par[b, 1]
        m = self.b_par[b, 2]
        n = self.b_par[b, 3]
        crit = self.b_par[b, 4]
        muC = self.b_par[b, 5]
        tau0 = self.b_par[b, 6]
        s = ti.max(sr, 1e-6)
        model = self.b_model[b]
        mu = mu0
        if model == POWER_LAW:
            mu = m * ti.pow(s, n - 1.0)
        elif model == CROSS:
            mu = mui + (mu0 - mui) / (1.0 + ti.pow(m * s, n))
        elif model == CASSON:
            r = ti.sqrt(muC) + ti.sqrt(tau0 / s)
            mu = r * r
        elif model == CARREAU:
            if ti.static(self.cpp_compat):  # nonNewtonCode 的写法，与论文式不同
                mu = mui + (mu0 - mui) / (1.0 + ti.pow(m * s * s, (1.0 - n) / 2.0))
            else:
                mu = mui + (mu0 - mui) / ti.pow(1.0 + (m * s) ** 2, (1.0 - n) / 2.0)
        elif model == BINGHAM:
            if s > crit:
                mu = mui + crit * (mu0 - mui) / s
        elif model == HERSCHEL_BULKLEY:
            if s > crit:
                t0 = mu0 * crit - m * ti.pow(crit, n)
                mu = t0 / s + m * ti.pow(s, n - 1.0)
        if ti.static(not self.cpp_compat):
            if model != NEWTONIAN:
                mu = ti.min(ti.max(mu, mui), mu0)  # 以 mu_inf、mu0 为上下界截断
        return mu

    # ------------------------------------------------------------------ step parts
    @ti.kernel
    def compute_density(self, xs: ti.template()):
        for i in range(self.nf):
            rho = self.mass * self.W(0.0)
            for jj in range(self.nb_cnt[i]):
                j = self.nb[i, jj]
                rho += self.pmass[j] * self.W((xs[i] - xs[j]).norm())
            self.rho[i] = rho

    @ti.kernel
    def compute_strain_rate_and_mu(self):
        for i in range(self.nf):
            gv = ti.Matrix.zero(ti.f32, 3, 3)
            for jj in range(self.nb_cnt[i]):
                j = self.nb[i, jj]
                gv += self.mass * (self.v[j] - self.v[i]).outer_product(self.gradW(self.x[i] - self.x[j]))
            gv /= self.rho[i]
            D = 0.5 * (gv + gv.transpose())
            sr = D.norm()
            if ti.static(self.cpp_compat):
                # nonNewtonCode NonNewton::calcStrainRate：循环内用 '=' 覆盖，只保留最后一个（流体）邻居
                last = ti.Matrix.zero(ti.f32, 3, 3)
                for jj in range(self.nb_cnt[i]):
                    j = self.nb[i, jj]
                    if j < self.nf and self.body[j] == self.body[i]:
                        gl = (self.v[j] - self.v[i]).outer_product(self.gradW(self.x[i] - self.x[j]))
                        last = self.mass * (gl + gl.transpose())
                sr = (0.5 / self.rho[i] * last).norm()
            self.sr[i] = sr
            b = self.body[i]
            self.mu[i] = self.mu_scale * self.viscosity(b, sr) * ti.exp(-self.b_par[b, 11] * self.T[i])

    @ti.kernel
    def predict(self):
        for i in range(self.nf):
            self.x_old[i] = self.x[i]
            self.v[i] += self.dt * (self.gravity + self.fext[i])
            self.x[i] += self.dt * self.v[i]
            self.lam_rho[i] = 0.0
            self.lam_div[i] = 0.0
            self.lam_vis[i] = 0.0
            self.lam_dev[i] = 0.0
            self.lam_vol[i] = 0.0
            self.lam_fib[i] = 0.0

    @ti.kernel
    def apply_dx(self, w: ti.f32):
        for i in range(self.nf):
            self.x[i] += w * self.dx[i]

    # ---- 密度约束（位置层）
    @ti.func
    def density_diag(self, i, xs: ti.template()):
        gi = ti.Vector([0.0, 0.0, 0.0])
        s2 = 0.0
        for jj in range(self.nb_cnt[i]):
            j = self.nb[i, jj]
            g = self.pmass[j] / self.rho0 * self.gradW(xs[i] - xs[j])
            gi += g
            if j < self.nf:
                s2 += g.norm_sqr() / self.mass
        return gi.norm_sqr() / self.mass + s2

    @ti.kernel
    def density_k1(self):
        a_t = self.kappa_inv / (self.V * self.dt * self.dt)
        for i in range(self.nf):
            rho = self.mass * self.W(0.0)
            for jj in range(self.nb_cnt[i]):
                j = self.nb[i, jj]
                rho += self.pmass[j] * self.W((self.x[i] - self.x[j]).norm())
            self.rho[i] = rho
            C = rho / self.rho0 - 1.0
            A = self.density_diag(i, self.x) + a_t
            dl = (-C - a_t * self.lam_rho[i]) / (A + 1e-12)
            lnew = ti.min(self.lam_rho[i] + dl, 0.0)  # 单边：p >= 0
            self.dlam[i] = lnew - self.lam_rho[i]
            self.lam_rho[i] = lnew

    @ti.kernel
    def density_k2(self, xs: ti.template(), out: ti.template()):
        for i in range(self.nf):
            d = ti.Vector([0.0, 0.0, 0.0])
            for jj in range(self.nb_cnt[i]):
                j = self.nb[i, jj]
                d += (self.pmass[j] / self.mass) * (self.dlam[i] + self.dlam[j]) * self.gradW(xs[i] - xs[j])
            out[i] = d / self.rho0  # 等质量：w_i m_j = 1

    # ---- 无散约束（速度层）
    @ti.kernel
    def div_k1(self):
        a_t = self.kappa_inv / (self.V * self.dt * self.dt)
        for i in range(self.nf):
            cd = 0.0
            for jj in range(self.nb_cnt[i]):
                j = self.nb[i, jj]
                cd += self.pmass[j] / self.rho0 * (self.v[i] - self.v[j]).dot(self.gradW(self.x[i] - self.x[j]))
            A = self.density_diag(i, self.x) + a_t
            dl = (-cd - a_t * self.lam_div[i]) / (A + 1e-12)
            lnew = ti.min(self.lam_div[i] + dl, 0.0)
            self.dlam[i] = lnew - self.lam_div[i]
            self.lam_div[i] = lnew

    @ti.kernel
    def apply_dv(self):
        for i in range(self.nf):
            self.v[i] += self.omega * self.dx[i]

    # ---- C++ reference pressure splitting (thermal Akinci scenes only)
    @ti.kernel
    def dfsph_factor(self):
        for i in range(self.nf):
            denom = self.mass * self.density_diag(i, self.x)
            self.df_factor[i] = 0.0
            if denom > 1e-5:
                self.df_factor[i] = -1.0 / denom

    @ti.kernel
    def dfsph_rhs(self, pressure: ti.template()):
        self.df_stat[0] = 0.0
        self.df_stat[1] = 0.0
        for i in range(self.nf):
            delta = 0.0
            for jj in range(self.nb_cnt[i]):
                j = self.nb[i, jj]
                delta += self.pmass[j] / self.rho0 * (self.v[i] - self.v[j]).dot(self.gradW(self.x[i] - self.x[j]))
            error = ti.max(delta, 0.0)
            if ti.static(pressure):
                error = ti.max(self.rho[i] / self.rho0 + self.dt * delta - 1.0, 0.0)
            elif self.nb_cnt[i] < 20:
                error = 0.0
            self.df_adv[i] = error
            self.df_stat[0] += error / self.nf
            ti.atomic_max(self.df_stat[1], error)

    @ti.kernel
    def dfsph_warmstart(self, pressure: ti.template()):
        for i in range(self.nf):
            value = 0.0
            if self.df_adv[i] > 0:
                if ti.static(pressure):
                    value = 0.5 * ti.max(self.df_kappa[i], -0.00025) / self.dt**2
                else:
                    value = 0.5 * ti.max(self.df_kappa_v[i], -0.5) / self.dt
            self.dlam[i] = value
            # C++ retains the warmstarted multiplier when accumulating ki.
            if ti.static(pressure):
                self.df_kappa[i] = value
            else:
                self.df_kappa_v[i] = value

    @ti.kernel
    def dfsph_multiplier(self, pressure: ti.template()):
        for i in range(self.nf):
            factor = self.df_factor[i] / self.dt
            if ti.static(pressure):
                factor /= self.dt
            ki = self.df_adv[i] * factor
            self.dlam[i] = ki
            if ti.static(pressure):
                self.df_kappa[i] += ki
            else:
                self.df_kappa_v[i] += ki

    @ti.kernel
    def dfsph_apply(self):
        for i in range(self.nf):
            dv = ti.Vector([0.0, 0.0, 0.0])
            ki = self.dlam[i]
            for jj in range(self.nb_cnt[i]):
                j = self.nb[i, jj]
                ks = ki
                if j < self.nf:
                    ks += self.dlam[j]
                if j < self.nf or ti.abs(ki) > 1e-5:
                    dv += self.dt * ks * self.pmass[j] / self.rho0 * self.gradW(self.x[i] - self.x[j])
            self.dx[i] = dv
        for i in range(self.nf):
            self.v[i] += self.dx[i]

    @ti.kernel
    def dfsph_finish(self, pressure: ti.template()):
        for i in range(self.nf):
            if ti.static(pressure):
                self.df_kappa[i] *= self.dt**2
            else:
                self.df_kappa_v[i] *= self.dt

    def solve_dfsph(self, pressure):
        self.dfsph_rhs(pressure)
        self.dfsph_warmstart(pressure)
        self.dfsph_apply()
        self.dfsph_rhs(pressure)
        minimum = 2 if pressure else 1
        tolerance = 0.0005 if pressure else 0.001 / self.dt
        for iteration in range(100):
            self.dfsph_multiplier(pressure)
            self.dfsph_apply()
            self.dfsph_rhs(pressure)
            error = float(self.df_stat[0])
            if iteration + 1 >= minimum and error <= tolerance:
                break
        self.dfsph_finish(pressure)
        if pressure:
            self.df_pressure_iters = iteration + 1
        else:
            self.df_div_iters = iteration + 1

    @ti.kernel
    def dfsph_external_velocity(self):
        for i in range(self.nf):
            self.v[i] += self.dt * (self.gravity + self.fext[i])

    @ti.kernel
    def dfsph_advect(self):
        for i in range(self.nf):
            self.x_old[i] = self.x[i]
            self.x[i] += self.dt * self.v[i]

    # ---- 黏性约束
    @ti.kernel
    def viscosity_operator(self, src: ti.template(), out: ti.template()):
        # A = I + B^T diag(2 mu V dt / mass) B, B = dev(sym(grad)).
        # This is the same quadratic strain energy underlying the XPBD viscosity.
        for i in range(self.nf):
            G = ti.Matrix.zero(ti.f32, 3, 3)
            for jj in range(self.nb_cnt[i]):
                j = self.nb[i, jj]
                if j < self.nf and self.body[j] == self.body[i]:
                    G += self.V * (src[j] - src[i]).outer_product(self.gradW(self.x[i] - self.x[j]))
            self.cg_D[i] = self.mu[i] * self.viscosity_force_scale * self.dev(0.5 * (G + G.transpose()))
        for i in range(self.nf):
            force = ti.Vector([0.0, 0.0, 0.0])
            for jj in range(self.nb_cnt[i]):
                j = self.nb[i, jj]
                if j < self.nf and self.body[j] == self.body[i]:
                    force -= self.V * ((self.cg_D[i] + self.cg_D[j]) @ self.gradW(self.x[i] - self.x[j]))
            out[i] = src[i] + 2.0 * self.V * self.dt / self.mass * force

    @ti.kernel
    def cpp_viscosity_operator(self, src: ti.template(), out: ti.template()):
        # Casson.cpp uses mu_j, producing a diagonally symmetrizable matrix.
        # q_i=sqrt(mu_i)*v_i makes the reference operator symmetric positive definite.
        for i in range(self.nf):
            value = src[i]
            mui = self.cg_scale[i]**2
            for jj in range(self.nb_cnt[i]):
                j = self.nb[i, jj]
                if j < self.nf and self.body[j] == self.body[i]:
                    r = (self.x[i] - self.x[j]).cast(ti.f32)
                    muj = self.cg_scale[j]**2
                    dv = src[i] / self.cg_scale[i] - src[j] / self.cg_scale[j]
                    value -= 10.0 * self.dt * self.mass * muj * self.cg_scale[i] / (self.rho[i] * self.rho[j]) * dv.dot(r) / (r.norm_sqr() + 0.01 * self.h**2) * self.gradW(r)
                elif j >= self.nf:
                    # Casson::matrixVecProd retains the static Akinci boundary
                    # term even though computeRHS/applyForces remove theirs.
                    r = (self.x[i] - self.x[j]).cast(ti.f32)
                    value -= 10.0 * self.dt * self.cpp_visc_boundary * self.rho0 * self.pmass[j] / self.rho[i]**2 * src[i].dot(r) / (r.norm_sqr() + 0.01 * self.h**2) * self.gradW(r)
            out[i] = value

    @ti.kernel
    def viscosity_cg_init(self):
        for i in range(self.nf):
            self.cg_scale[i] = 1.0
            if ti.static(self.cpp_visc_cg):
                self.cg_scale[i] = ti.sqrt(ti.max(self.mu[i] * self.viscosity_force_scale, 1e-8))
            self.cg_rhs[i] = self.cg_scale[i] * self.v[i]
            self.cg_x[i] = self.cg_rhs[i]
            if ti.static(self.cpp_dfsph):
                self.cg_x[i] += self.cg_scale[i] * self.cg_vdiff[i]
            gsum = ti.Vector([0.0, 0.0, 0.0])
            diag = ti.Matrix.zero(ti.f32, 3, 3)
            for jj in range(self.nb_cnt[i]):
                j = self.nb[i, jj]
                if j < self.nf and self.body[j] == self.body[i]:
                    g = self.gradW(self.x[i] - self.x[j])
                    gsum += g
                    diag += self.mu[j] * g.outer_product(g)
            diag += self.mu[i] * gsum.outer_product(gsum)
            # Positive scalar bound for the three-component diagonal block.
            self.cg_diag[i] = 1.0 + 2.0 * self.V**3 * self.dt / self.mass * self.viscosity_force_scale * diag.trace()
            self.cg_Minv[i] = ti.Matrix.identity(ti.f32, 3) / self.cg_diag[i]
            if ti.static(self.cpp_visc_cg):
                block = ti.Matrix.identity(ti.f32, 3)
                for jj in range(self.nb_cnt[i]):
                    j = self.nb[i, jj]
                    if j < self.nf and self.body[j] == self.body[i]:
                        r = (self.x[i] - self.x[j]).cast(ti.f32)
                        block -= 10.0 * self.dt * self.mass * ti.max(self.mu[j] * self.viscosity_force_scale, 1e-8) / (self.rho[i] * self.rho[j]) * self.gradW(r).outer_product(r) / (r.norm_sqr() + 0.01 * self.h**2)
                    elif j >= self.nf:
                        r = (self.x[i] - self.x[j]).cast(ti.f32)
                        block -= 10.0 * self.dt * self.cpp_visc_boundary * self.rho0 * self.pmass[j] / self.rho[i]**2 * self.gradW(r).outer_product(r) / (r.norm_sqr() + 0.01 * self.h**2)
                self.cg_Minv[i] = block.inverse()

    @ti.kernel
    def viscosity_cg_residual(self):
        for i in range(self.nf):
            self.cg_r[i] = self.cg_rhs[i] - self.cg_Ap[i]
            self.cg_p[i] = self.cg_Minv[i] @ self.cg_r[i]

    @ti.kernel
    def cg_dot(self, a: ti.template(), b: ti.template()) -> ti.f64:
        value = ti.cast(0.0, ti.f64)
        for i in range(self.nf):
            value += ti.cast(a[i].dot(b[i]), ti.f64)
        return value

    @ti.kernel
    def cg_preconditioned_norm(self) -> ti.f64:
        value = ti.cast(0.0, ti.f64)
        for i in range(self.nf):
            value += ti.cast(self.cg_r[i].dot(self.cg_Minv[i] @ self.cg_r[i]), ti.f64)
        return value

    @ti.kernel
    def viscosity_cg_update(self, alpha: ti.f32):
        for i in range(self.nf):
            self.cg_x[i] += alpha * self.cg_p[i]
            self.cg_r[i] -= alpha * self.cg_Ap[i]

    @ti.kernel
    def viscosity_cg_direction(self, beta: ti.f32):
        for i in range(self.nf):
            self.cg_p[i] = self.cg_Minv[i] @ self.cg_r[i] + beta * self.cg_p[i]

    @ti.kernel
    def viscosity_cg_finish(self):
        for i in range(self.nf):
            new_v = self.cg_x[i] / self.cg_scale[i]
            self.cg_vdiff[i] = new_v - self.v[i]
            self.v[i] = new_v

    @ti.kernel
    def cg_prepare_batch(self):
        for i in range(self.nf):
            if self.cg_state[5] > 0:
                self.cg_p[i] = self.cg_Minv[i] @ self.cg_r[i] + ti.cast(self.cg_state[1], ti.f32) * self.cg_p[i]

    @ti.kernel
    def cg_dot_batch(self):
        self.cg_state[2] = 0.0
        for i in range(self.nf):
            self.cg_state[2] += ti.cast(self.cg_p[i].dot(self.cg_Ap[i]), ti.f64)

    @ti.kernel
    def cg_update_batch(self):
        alpha = ti.cast(0.0, ti.f32)
        if self.cg_state[5] > 0:
            if self.cg_state[2] > 0:
                alpha = ti.cast(self.cg_state[0] / self.cg_state[2], ti.f32)
                self.cg_state[6] += 1.0
            else:
                self.cg_state[7] = 1.0
                self.cg_state[5] = 0.0
        for i in range(self.nf):
            self.cg_x[i] += alpha * self.cg_p[i]
            self.cg_r[i] -= alpha * self.cg_Ap[i]
        self.cg_state[2] = 0.0
        self.cg_state[3] = 0.0
        for i in range(self.nf):
            self.cg_state[2] += ti.cast(self.cg_r[i].dot(self.cg_Minv[i] @ self.cg_r[i]), ti.f64)
            self.cg_state[3] += ti.cast(self.cg_r[i].norm_sqr(), ti.f64)
        self.cg_state[1] = self.cg_state[2] / ti.max(self.cg_state[0], 1e-30)
        self.cg_state[0] = self.cg_state[2]
        if self.cg_state[3] <= self.cg_state[4]:
            self.cg_state[5] = 0.0

    def solve_viscosity_cg(self, tol=None, max_iters=None):
        tol = tol if tol is not None else (1e-2 if self.cpp_visc_cg else 1e-3)
        max_iters = max_iters if max_iters is not None else (self.cpp_visc_max_iters if self.cpp_visc_cg else 300)
        self.viscosity_cg_init()
        operator = self.cpp_viscosity_operator if self.cpp_visc_cg else self.viscosity_operator
        operator(self.cg_x, self.cg_Ap)
        self.viscosity_cg_residual()
        rz = float(self.cg_preconditioned_norm())
        rhs2 = float(self.cg_dot(self.cg_rhs, self.cg_rhs))
        r2 = float(self.cg_dot(self.cg_r, self.cg_r))
        threshold = tol**2 * max(rhs2, 1e-20)
        state = np.array([rz, 0, 0, r2, threshold, float(r2 > threshold), 0, 0], np.float64)
        self.cg_state.from_numpy(state)
        # Keep scalar recurrences on-device; synchronize once per ten iterations.
        for first in range(0, max_iters, 10):
            if state[5] == 0:
                break
            for _ in range(min(10, max_iters - first)):
                self.cg_prepare_batch()
                operator(self.cg_p, self.cg_Ap)
                self.cg_dot_batch()
                self.cg_update_batch()
            state = self.cg_state.to_numpy()
            if not np.isfinite(state).all() or state[7] != 0:
                raise FloatingPointError("Viscosity CG lost positive definiteness or became non-finite")
        self.viscosity_cg_finish()
        self.cg_iterations = int(state[6])
        self.cg_error = math.sqrt(float(state[3]) / max(rhs2, 1e-20))

    @ti.kernel
    def viscous_k1(self):
        for i in range(self.nf):
            G = ti.Matrix.zero(ti.f32, 3, 3)
            dxi = self.x[i] - self.x_old[i]
            for jj in range(self.nb_cnt[i]):
                j = self.nb[i, jj]
                if ti.static(self.skip_bnd_visc):
                    if j >= self.nf:
                        continue
                g = self.gradW(self.x_old[i] - self.x_old[j])
                G += self.V * ((self.x[j] - self.x_old[j]) - dxi).outer_product(g)
            De = self.dev(0.5 * (G + G.transpose()))
            C = De.norm()
            self.dlam[i] = 0.0
            self.Nmat[i] = ti.Matrix.zero(ti.f32, 3, 3)
            if C > 1e-9 and self.mu[i] > 1e-12:  # 过小的 mu 视为无黏，否则柔度倒数溢出 f32 导致发散
                N = De / C
                gi = ti.Vector([0.0, 0.0, 0.0])
                s2 = 0.0
                for jj in range(self.nb_cnt[i]):
                    j = self.nb[i, jj]
                    if ti.static(self.skip_bnd_visc):
                        if j >= self.nf:
                            continue
                    a = self.V * (N @ self.gradW(self.x_old[i] - self.x_old[j]))
                    gi += a
                    if j < self.nf:
                        s2 += a.norm_sqr()
                A = (gi.norm_sqr() + s2) / self.mass
                a_t = 1.0 / (2.0 * self.mu[i] * self.viscosity_force_scale * self.V * self.dt)
                dl = (-C - a_t * self.lam_vis[i]) / (A + a_t)
                self.lam_vis[i] += dl
                self.dlam[i] = dl
                self.Nmat[i] = N

    @ti.kernel
    def viscous_k2(self):
        for k in range(self.nf):
            d = ti.Vector([0.0, 0.0, 0.0])
            for jj in range(self.nb_cnt[k]):
                j = self.nb[k, jj]
                if ti.static(self.skip_bnd_visc):
                    if j >= self.nf:
                        continue
                g = self.gradW(self.x_old[k] - self.x_old[j])
                d -= self.V * (self.Nmat[k] @ g) * self.dlam[k]
                d -= self.V * (self.Nmat[j] @ g) * self.dlam[j]
            self.dx[k] = d / self.mass

    # ---- 弹性约束
    @ti.kernel
    def compute_rotation(self):
        for i in range(self.nf):
            if self.b_elastic[self.body[i]] == 1:
                Apq = ti.Matrix.zero(ti.f32, 3, 3)
                for jj in range(self.ref_cnt[i]):
                    j = self.ref_nb[i, jj]
                    r0 = self.x0[j] - self.x0[i]
                    Apq += self.mass * self.W(r0.norm()) * (self.x[j] - self.x[i]).outer_product(r0)
                U, S, Vm = ti.svd(Apq)
                if U.determinant() < 0:
                    U[0, 2] *= -1.0
                    U[1, 2] *= -1.0
                    U[2, 2] *= -1.0
                if Vm.determinant() < 0:
                    Vm[0, 2] *= -1.0
                    Vm[1, 2] *= -1.0
                    Vm[2, 2] *= -1.0
                self.R[i] = U @ Vm.transpose()

    @ti.func
    def elastic_strain(self, i):
        gu = ti.Matrix.zero(ti.f32, 3, 3)
        for jj in range(self.ref_cnt[i]):
            j = self.ref_nb[i, jj]
            u = self.R[i].transpose() @ (self.x[j] - self.x[i]) - (self.x0[j] - self.x0[i])
            gu += self.Vt[j] * u.outer_product(self.g0[i, jj])
        return 0.5 * (gu + gu.transpose())

    @ti.kernel
    def elastic_k1(self, mode: ti.template()):
        for i in range(self.nf):
            self.dlam[i] = 0.0
            self.Nmat[i] = ti.Matrix.zero(ti.f32, 3, 3)
            b = self.body[i]
            if self.b_elastic[b] == 1:
                ee = self.elastic_strain(i) - self.epsP[i]
                C = 0.0
                P = ti.Matrix.identity(ti.f32, 3)
                stiff = 0.0
                if ti.static(mode == 0):
                    De = self.dev(ee)
                    C = De.norm()
                    P = De / ti.max(C, 1e-12)
                    stiff = 2.0 * self.b_par[b, 7] * ti.exp(-self.b_par[b, 12] * self.T[i])  # 2G
                elif ti.static(mode == 1):
                    C = ee.trace()
                    stiff = self.b_par[b, 8] * ti.exp(-self.b_par[b, 12] * self.T[i])  # K
                else:
                    # 主动纤维约束 C = d^T eps_e d + a eps_f：纤维应变大于目标缩短量时收缩，单边
                    a = self.activation[None] * self.fw[i]
                    d = self.fdir[i]
                    C = d.dot(ee @ d) + a * self.b_par[b, 14]
                    P = d.outer_product(d)
                    stiff = self.b_par[b, 13] if a > 0.0 else 0.0
                if stiff > 0.0 and (ti.static(mode == 1) or C > 1e-9):
                    gi = ti.Vector([0.0, 0.0, 0.0])
                    s2 = 0.0
                    for jj in range(self.ref_cnt[i]):
                        j = self.ref_nb[i, jj]
                        a = self.Vt[j] * (P @ self.g0[i, jj])
                        gi += a
                        s2 += a.norm_sqr()
                    A = (gi.norm_sqr() + s2) / self.mass
                    a_t = 1.0 / (stiff * self.Vt[i] * self.dt * self.dt)
                    lam = 0.0
                    if ti.static(mode == 0):
                        lam = self.lam_dev[i]
                    elif ti.static(mode == 1):
                        lam = self.lam_vol[i]
                    else:
                        lam = self.lam_fib[i]
                    dl = (-C - a_t * lam) / (A + a_t)
                    if ti.static(mode == 0):
                        self.lam_dev[i] += dl
                    elif ti.static(mode == 1):
                        self.lam_vol[i] += dl
                    else:
                        self.lam_fib[i] += dl
                    self.dlam[i] = dl
                    self.Nmat[i] = P

    @ti.kernel
    def elastic_k2(self):
        for k in range(self.nf):
            d = ti.Vector([0.0, 0.0, 0.0])
            for jj in range(self.ref_cnt[k]):
                i = self.ref_nb[k, jj]
                g = self.g0[k, jj]  # gradW(x0_k - x0_i)
                d -= self.Vt[i] * (self.R[k] @ (self.Nmat[k] @ g)) * self.dlam[k]
                d -= self.Vt[k] * (self.R[i] @ (self.Nmat[i] @ g)) * self.dlam[i]
            self.dx[k] = d / self.mass

    @ti.kernel
    def plasticity(self):
        for i in range(self.nf):
            b = self.body[i]
            if self.b_elastic[b] == 1:
                g1 = self.b_par[b, 9]
                g2 = self.b_par[b, 10]
                eps = self.elastic_strain(i)
                ed = self.dev(eps - self.epsP[i])
                en = ed.norm()
                if en > g1:
                    ep = self.epsP[i] + (en - g1) / en * ed
                    self.epsP[i] = ep * ti.min(1.0, g2 / ti.max(ep.norm(), 1e-12))

    # ---- 边界
    @ti.kernel
    def boundary_pos(self):
        for i in range(self.nf):
            p = self.x[i]
            if ti.static(self.clip_domain):
                for a in ti.static(range(3)):
                    lo = self.pad
                    hi = self.domain[a] - self.pad
                    p[a] = ti.min(ti.max(p[a], lo), hi)
                if p[1] <= self.pad + 1e-7:
                    p[0] = self.x_old[i][0] + (1.0 - self.friction) * (p[0] - self.x_old[i][0])
                    p[2] = self.x_old[i][2] + (1.0 - self.friction) * (p[2] - self.x_old[i][2])
            if self.pin[i] == 1:
                p = self.x0[i]
            self.x[i] = p

    @ti.kernel
    def update_velocity(self):
        for i in range(self.nf):
            v = (self.x[i] - self.x_old[i]) / self.dt
            if ti.static(self.v_clamp > 0.0):
                vn = v.norm()
                if vn > self.v_clamp:
                    v *= self.v_clamp / vn
            self.v[i] = v

    @ti.kernel
    def boundary_viscosity(self):
        # Akinci2012 边界黏性（与 nonNewtonCode NonNewton_Weiler2018 边界项相同的离散），显式
        nu = ti.static(float(self.visc_bnd or 0.0))
        h2 = self.h * self.h
        for i in range(self.nf):
            a = ti.Vector([0.0, 0.0, 0.0])
            for jj in range(self.nb_cnt[i]):
                j = self.nb[i, jj]
                if j >= self.nf:
                    xij = self.x[i] - self.x[j]
                    # C++ matrixVecProd includes both mub=nu*rho0 and the outer 1/rho_i.
                    weight = self.mass / self.rho[i]
                    if ti.static(self.cpp_compat):
                        weight = self.rho0 * self.pmass[j] / (self.rho[i] * self.rho[i])
                    a += 10.0 * nu * weight * self.v[i].dot(xij) / (xij.norm_sqr() + 0.01 * h2) * self.gradW(xij)
            self.dx[i] = a
        for i in range(self.nf):
            self.v[i] += self.dt * self.dx[i]

    @ti.kernel
    def boundary_vel(self):
        for i in range(self.nf):
            if ti.static(self.clip_domain):
                for a in ti.static(range(3)):
                    if self.x[i][a] <= self.pad + 1e-7 and self.v[i][a] < 0:
                        self.v[i][a] = 0.0
                    if self.x[i][a] >= self.domain[a] - self.pad - 1e-7 and self.v[i][a] > 0:
                        self.v[i][a] = 0.0
            if self.pin[i] == 1:
                self.v[i] = ti.Vector([0.0, 0.0, 0.0])

    # ---- 温度扩散
    @ti.kernel
    def diffuse(self):
        for i in range(self.nf):
            lap = 0.0
            for jj in range(self.nb_cnt[i]):
                j = self.nb[i, jj]
                if j >= self.nf:
                    continue
                lap += self.mass / (self.rho[j] * self.rho[i]) * (self.T[j] - self.T[i]) * self.gradW(self.x[i] - self.x[j]).norm()
            src = 0.0
            if ti.static(self.has_heat_box):
                p = self.x[i] - self.cpp_offset
                inside = True
                for a in ti.static(range(3)):
                    inside = inside and self.heat_box[0][a] < p[a] < self.heat_box[1][a]
                if inside:
                    src = self.heat_R
            else:
                if self.x[i][1] < self.heat_y:
                    src = self.heat_R
            nf_cnt = 0
            for jj in range(self.nb_cnt[i]):
                if self.nb[i, jj] < self.nf:
                    nf_cnt += 1
            if nf_cnt < self.surf_thresh:
                self.surface_source[i] = self.surf_source
            if ti.static(self.cpp_compat):
                src += self.surface_source[i]  # C++ retains the source after a surface particle moves inward.
            else:
                if nf_cnt < self.surf_thresh:
                    src += self.surf_source
            if ti.static(self.cpp_compat):
                src *= nf_cnt  # nonNewtonCode Coagulation：源项写在邻居循环内，被累加 nf_cnt 次
            self.T_new[i] = self.T[i] + self.dt * (self.diffusion * lap + src)
        for i in range(self.nf):
            self.T[i] = self.T_new[i]

    # ---- 诊断
    @ti.kernel
    def density_error(self) -> ti.types.vector(2, ti.f32):
        s = 0.0
        mx = 0.0
        for i in range(self.nf):
            e = ti.max(self.rho[i] / self.rho0 - 1.0, 0.0)
            s += e
            ti.atomic_max(mx, e)
        return ti.Vector([s / self.n, mx])

    @ti.kernel
    def neighbor_diagnostics(self) -> ti.types.vector(3, ti.i32):
        cell_overflow = 0
        capped = 0
        outside = 0
        for I in ti.grouped(self.cell_count):
            if self.cell_count[I] > MAX_CELL:
                cell_overflow += 1
        for i in range(self.nf):
            if self.nb_cnt[i] == self.max_nb:
                capped += 1
            if not self.in_grid(self.cell_of(self.x[i])):
                outside += 1
        return ti.Vector([cell_overflow, capped, outside])

    # ------------------------------------------------------------------ main step
    def step(self):
        if self.cpp_dfsph:
            return self.step_cpp_dfsph()
        # 1. 步初：邻域、应变率、黏度（柔度）
        self.build_grid(self.x)
        self.find_neighbors(self.x)
        self.compute_density(self.x)
        self.compute_strain_rate_and_mu()
        if self.visc_cg:
            self.solve_viscosity_cg()
        # 2. 预测
        self.predict()
        self.boundary_pos()
        self.build_grid(self.x)
        self.find_neighbors(self.x)
        self.compute_density(self.x)
        err_pred = self.density_error()
        # 3. 位置层约束
        for _ in range(self.iters):
            self.density_k1()
            self.density_k2(self.x, self.dx)
            self.apply_dx(self.omega)
            if not self.visc_cg:
                self.viscous_k1()
                self.viscous_k2()
                self.apply_dx(self.omega_s)
            if self.has_elastic:
                self.compute_rotation()
                for mode in (0, 1, 2) if self.has_fiber else (0, 1):
                    self.elastic_k1(mode)
                    self.elastic_k2()
                    self.apply_dx(self.omega_f if mode == 2 else self.omega_s)
            self.boundary_pos()
        self.compute_density(self.x)
        err_post = self.density_error()
        # 4. 速度层：无散约束
        self.update_velocity()
        if self.use_div:
            for _ in range(self.div_iters):
                self.div_k1()
                self.density_k2(self.x, self.dx)
                self.apply_dv()
        if self.visc_bnd is not None and self.visc_bnd > 0.0:
            self.boundary_viscosity()
        self.boundary_vel()
        # 5. 塑性回映、温度扩散
        if self.has_elastic:
            self.compute_rotation()
            self.plasticity()
        if self.has_heat:
            self.diffuse()
            if self.cpp_compat:
                self.thermal_compat_state()
        return err_pred, err_post

    def step_cpp_dfsph(self):
        # Match TimeStepDFSPH::step: divergence -> thermal/friction ->
        # implicit viscosity -> gravity -> linearized pressure -> advection.
        # All forces and pressure gradients use the same step-start positions.
        self.build_grid(self.x)
        self.find_neighbors(self.x)
        self.compute_density(self.x)
        self.dfsph_factor()
        if self.use_div:
            self.solve_dfsph(False)
        if self.has_heat:
            self.diffuse()
            self.thermal_compat_state()
        else:
            self.compute_strain_rate_and_mu()
        self.solve_viscosity_cg()
        self.dfsph_external_velocity()
        self.dfsph_rhs(True)
        err_pred = self.df_stat.to_numpy()
        self.solve_dfsph(True)
        err_post = self.df_stat.to_numpy()
        self.dfsph_advect()
        return err_pred, err_post

    @ti.kernel
    def thermal_compat_state(self):
        for i in range(self.nf):
            if self.x[i][1] - self.cpp_offset[1] < 0.01:
                self.v[i][0] *= 0.1
                self.v[i][2] *= 0.1
            vn = self.v[i].norm()
            if vn > 4.0:
                self.v[i] *= 4.0 / vn
            b = self.body[i]
            self.mu[i] = self.mu_scale * self.viscosity(b, self.sr[i]) * ti.exp(-self.b_par[b, 11] * self.T[i])


# ====================================================================== scenes
def box_points(lo, hi, d):
    axes = [np.arange(lo[a] + 0.5 * d, hi[a] - 0.25 * d, d) for a in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return g


def sphere_points(c, r, d):
    g = box_points(np.array(c) - r, np.array(c) + r, d)
    return g[np.linalg.norm(g - np.array(c), axis=1) <= r]


def scene_dambreak(args):
    d = 0.02
    pos = box_points([0.0, 0.0, 0.0], [0.4, 0.6, 0.4], d) + d * 0.5
    body = np.zeros(len(pos), np.int32)
    mats = [default_material(mu0=1e-3)]
    return dict(pos=pos, body=body, mats=mats, d=d, domain=[1.2, 1.0, 0.44], dt=2e-3, steps=args.steps or 400)


def scene_models(args):
    # 八个立方块沿斜坡（倾斜重力）下滑，仅黏度模型不同（参数同论文表 tab:fluid-armadillo）
    d = 0.01
    mats = [
        default_material(model=NEWTONIAN, mu0=10.0),
        default_material(model=POWER_LAW, n=0.667, m=4.5, mu_inf=0.1, mu0=10.0),
        default_material(model=POWER_LAW, n=1.5, m=1.0, mu_inf=0.1, mu0=10.0),
        default_material(model=CROSS, n=0.667, m=1.0, mu_inf=0.1, mu0=10.0),
        default_material(model=CASSON, muC=1.0, tau0=10.0, mu_inf=0.1, mu0=10.0),
        default_material(model=CARREAU, n=0.1, m=0.2, mu_inf=0.1, mu0=10.0),
        default_material(model=BINGHAM, crit=1.0, mu_inf=0.1, mu0=10.0),
        default_material(model=HERSCHEL_BULKLEY, n=0.667, m=10.0, crit=10.0, mu_inf=0.1, mu0=10.0),
    ]
    names = ["Newtonian", "PowerLaw1", "PowerLaw2", "Cross", "Casson", "Carreau", "Bingham", "HerschelBulkley"]
    if args.mu_sweep:  # 牛顿黏度扫描：验证黏性约束的柔度随 mu 单调
        mus = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
        mats = [default_material(model=NEWTONIAN, mu0=m) for m in mus]
        names = [f"mu={m:g}" for m in mus]
    pos, body = [], []
    s = 0.1
    for b in range(len(mats)):
        z0 = 0.03 + b * (s + 0.04)
        p = box_points([0.05, 0.0, z0], [0.05 + s, s, z0 + s], d) + np.array([0, 0.5 * d, 0])
        pos.append(p)
        body.append(np.full(len(p), b))
    th = math.radians(30)
    g = (9.81 * math.sin(th), -9.81 * math.cos(th), 0.0)
    return dict(pos=np.concatenate(pos), body=np.concatenate(body), mats=mats, names=names, d=d,
                domain=[2.5, 0.3, 0.03 + len(mats) * (s + 0.04)], dt=1e-3, steps=args.steps or 600, gravity=g)


def scene_tomato(args):
    # 四个弹塑性球体落地，仅弹性极限 gamma1 不同
    d = 0.01
    g1s = [0.001, 0.05, 0.1, 1e30]
    names = ["gamma1=0.001", "gamma1=0.05", "gamma1=0.1", "elastic"]
    mats = [default_material(mu0=0.01, E=args.E, nu=0.42, gamma1=g, gamma2=1.0) for g in g1s]
    pos, body = [], []
    r = 0.08
    for b in range(4):
        c = [0.12 + b * 0.24, 0.35, 0.12]
        p = sphere_points(c, r, d)
        pos.append(p)
        body.append(np.full(len(p), b))
    return dict(pos=np.concatenate(pos), body=np.concatenate(body), mats=mats, names=names, d=d,
                domain=[1.0, 0.6, 0.24], dt=1e-3, steps=args.steps or 1500)


def scene_melt(args):
    # 高黏度块：左侧地面加热（融化），右侧不加热（对照）
    d = 0.01
    mats = [default_material(mu0=200.0, decay_mu=2.0), default_material(mu0=200.0, decay_mu=2.0)]
    p0 = box_points([0.05, 0.0, 0.05], [0.2, 0.25, 0.2], d) + np.array([0, 0.5 * d, 0])
    p1 = p0 + np.array([0.4, 0, 0])
    pos = np.concatenate([p0, p1])
    body = np.concatenate([np.zeros(len(p0)), np.ones(len(p1))]).astype(np.int32)
    return dict(pos=pos, body=body, mats=mats, names=["heated", "unheated"], d=d, domain=[0.65, 0.4, 0.25],
                dt=1e-3, steps=args.steps or 2000, heat=True)


CPP_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "models", "cpp")

# nonNewtonCode/data/MyScenes/ramp{1,2,3}.json 的材料参数（黏度为运动黏度，单位 m^2/s）
RAMP_GROUPS = dict(
    ramp1=[
        ("Newtonian", dict(model=NEWTONIAN, mu0=10.0)),
        ("PowerLaw1", dict(model=POWER_LAW, mu0=10.0, mu_inf=0.01, n=0.667, m=4.5)),
        ("PowerLaw2", dict(model=POWER_LAW, mu0=10.0, mu_inf=0.01, n=1.5, m=1.0)),
    ],
    ramp2=[
        ("Cross", dict(model=CROSS, mu0=10.0, mu_inf=0.01, n=0.667, m=1.0)),
        ("Casson", dict(model=CASSON, mu0=10.0, mu_inf=0.01, muC=1.0, tau0=10.0)),
        ("Carreau", dict(model=CARREAU, mu0=10.0, mu_inf=0.01, n=0.2, m=0.1)),
    ],
    ramp3=[
        ("Bingham", dict(model=BINGHAM, mu0=10.0, mu_inf=0.01, crit=1.0)),
        ("HerschelBulkley", dict(model=HERSCHEL_BULKLEY, mu0=10.0, mu_inf=0.01, n=0.667, m=10.0, crit=10.0)),
    ],
)
RAMP_GROUPS["all"] = RAMP_GROUPS["ramp1"] + RAMP_GROUPS["ramp2"] + RAMP_GROUPS["ramp3"]


def lattice_subsample(p, d0, k):
    # 规则点阵上每 k 个取一个，得到间距 k*d0 的粗化点阵
    idx = np.round((p - p.min(0)) / d0).astype(np.int64)
    return p[(idx % k == 0).all(1)]


def scene_ramp(args):
    # 复刻 nonNewtonCode ramp{1,2,3}.json：armadillo.bhclassic 粒子，ramp.obj（scale [3,1,1]，平移 [-8,0,0]）为 30 度斜面
    from bgeo_io import read_bhclassic

    group = RAMP_GROUPS[args.group]
    d0 = 0.05  # particleRadius 0.025
    k = max(1, args.coarsen)
    d = d0 * k
    arm = read_bhclassic(os.path.join(CPP_MODELS, "armadillo.bhclassic"))["position"].astype(np.float64)
    if k > 1:
        arm = lattice_subsample(arm, d0, k)
    pos, body = [], []
    for b in range(len(group)):
        pos.append(arm + np.array([-5.0 * b, 0.0, 0.0]))  # json 中 translation 依次为 0, -5, -10
        body.append(np.full(len(arm), b, np.int32))
    pos = np.concatenate(pos)
    body = np.concatenate(body)
    # 斜面：由 ramp.obj 拟合得法向 (0, cos30, sin30)，过点 (-8, 2.272, 1.312)，沿 z 方向 [-5.18, 7.81]
    V, F = load_obj(os.path.join(CPP_MODELS, "ramp.obj"))
    V = V * np.array([3.0, 1.0, 1.0]) + np.array([-8.0, 0.0, 0.0])
    c = V.mean(0)
    nrm = np.array([0.0, math.cos(math.radians(30)), math.sin(math.radians(30))])
    t1 = np.array([1.0, 0.0, 0.0])
    t2 = np.cross(t1, nrm)  # 沿坡向下（+z, -y）
    s2 = (V - c) @ t2
    u = np.arange(pos[:, 0].min() - 1.0, pos[:, 0].max() + 1.0, d)
    w = np.arange(s2.min(), s2.max() + 0.5 * d, d)
    U, W = np.meshgrid(u, w, indexing="ij")
    plane = c + np.outer(U.ravel(), t1) + np.outer(W.ravel(), t2)
    plane[:, 0] = U.ravel()
    bnd = [plane - (0.5 + l) * d * nrm for l in range(2)]
    # 坡底水平地面（C++ 场景中由 domain 下边界承接），延伸 4 m
    y_floor = (c + s2.max() * t2)[1]
    zs = np.arange((c + s2.max() * t2)[2] - 1.0, (c + s2.max() * t2)[2] + 4.0, d)
    FU, FZ = np.meshgrid(u, zs, indexing="ij")
    if not args.cpp_compat:
        for l in range(2):
            bnd.append(np.stack([FU.ravel(), np.full(FU.size, y_floor - (0.5 + l) * d), FZ.ravel()], 1))
    bnd = np.concatenate(bnd)
    allp = np.concatenate([pos, bnd])
    lo = allp.min(0)
    off = -lo + 3.0 * d
    dom = allp.max(0) + off + 3.0 * d
    mats = [default_material(**m) for _, m in group]
    names = [nm for nm, _ in group]
    dt = args.dt or 2e-3
    return dict(pos=pos + off, body=body, mats=mats, names=names, d=d, domain=dom.tolist(), dt=dt,
                steps=args.steps or int(round(5.0 / dt)), axis=2, view=(2, 1),
                cpp_offset=off, meshes=[("ramp", V, F)],
                solver_kw=dict(box_walls=False, boundary=bnd + off, clip_domain=not args.cpp_compat, mu_scale=1000.0,
                               visc_boundary=None if args.noslip else 0.1))


def sample_mesh(V, F, d):
    """在三角网格表面按间距约 d 均匀撒点（重心坐标网格），用作静态边界粒子。"""
    out = []
    for f in F:
        for k in range(1, len(f) - 1):
            a, b, c = V[f[0]], V[f[k]], V[f[k + 1]]
            n = max(1, int(math.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)) / d)))
            for i in range(n + 1):
                for j in range(n + 1 - i):
                    out.append(a + (b - a) * (i / n) + (c - a) * (j / n))
    P = np.array(out)
    key = np.round(P / (0.5 * d)).astype(np.int64)  # 合并共享边上的重复点
    _, idx = np.unique(key, axis=0, return_index=True)
    return P[np.sort(idx)]


def min_dist(src, query, r):
    """query 中每个点到 src 的最近距离（只在 r 内精确，超出返回 r）。哈希网格实现，避免依赖 scipy。"""
    key = np.floor(src / r).astype(np.int64)
    order = np.lexsort(key.T)
    key, src = key[order], src[order]
    cells = {}
    uk, start, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    for k, a, c in zip(map(tuple, uk), start, cnt):
        cells[k] = src[a:a + c]
    out = np.full(len(query), r)
    qk = np.floor(query / r).astype(np.int64)
    for n, (q, k) in enumerate(zip(query, qk)):
        best = r
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    P = cells.get((k[0] + dx, k[1] + dy, k[2] + dz))
                    if P is not None:
                        best = min(best, float(np.sqrt(((P - q) ** 2).sum(1)).min()))
        out[n] = best
    return out


def cpp_boundary_points(args, mesh, scale):
    from pathlib import Path
    from bgeo_io import read_bhclassic

    default = Path(__file__).resolve().parents[2] / "nonNewtonCode/data/MyScenes/Cache"
    cache = Path(args.cpp_boundary_cache) if args.cpp_boundary_cache else default
    path = cache / f"{mesh}_sb_0.025_{scale}_m0.bgeo"
    if not path.exists():
        raise FileNotFoundError(f"C++ boundary cache missing: {path}. Run the C++ scene first, or set --cpp_boundary_cache.")
    return read_bhclassic(path)["position"].astype(np.float64)


def scene_icecream(args):
    # 复刻 nonNewtonCode ice-cream.json：ice-cream.bhclassic（间距 0.07），glass.obj（scale 0.3，平移 y-1.5）
    # 论文 Table 3：mu_i = mu0 exp(-d T)，mu0=1000, D=30, R=1（仅表面，邻居数<10）, d=0.1
    from bgeo_io import read_bhclassic

    d = 0.05 if args.cpp_compat else 0.07
    pos = read_bhclassic(os.path.join(CPP_MODELS, "ice-cream.bhclassic"))["position"].astype(np.float64)
    V, F = load_obj(os.path.join(CPP_MODELS, "glass.obj"))
    V = V * 0.3 + np.array([0.0, -1.5, 0.0])
    if args.cpp_compat:
        bnd = cpp_boundary_points(args, "glass", "0.3_0.3_0.3") + np.array([0.0, -1.5, 0.0])
    else:
        bnd = sample_mesh(V, F, d)
        bnd = bnd[min_dist(pos, bnd, d) > 0.6 * d]
    allp = np.concatenate([pos, bnd])
    off = -allp.min(0) + 3.0 * d
    dom = allp.max(0) + off + 3.0 * d
    dom[1] += 0.5
    mats = [default_material(mu0=1000.0, decay_mu=0.1)]
    dt = args.dt or 5e-3
    return dict(pos=pos + off, body=np.zeros(len(pos), np.int32), mats=mats, names=["ice-cream"], d=d,
                domain=dom.tolist(), dt=dt, steps=args.steps or int(round(10.0 / dt)), heat=True, omega_s=0.2,
                cpp_offset=off, meshes=[("glass", V, F)],
                solver_kw=dict(box_walls=False, boundary=bnd + off, clip_domain=not args.cpp_compat, boundary_akinci=args.cpp_compat, mu_scale=1000.0, visc_boundary=0.0,
                               heat_y=-1.0, heat_R=0.0, diffusion=30.0, surf_source=1.0,
                               surf_thresh=10 if args.cpp_compat else 15))


def scene_hotcut(args):
    # 复刻 nonNewtonCode hotcut.json：hot_cut_bunny_sampled.bhclassic（平移 y+0.1），十字切割板 + 圆柱容器
    # 论文：地面热源 box [-1,0,-1]-[1,0.05,1]，R=1，D=100，mu0=20，d=0.1
    from bgeo_io import read_bhclassic

    pos = read_bhclassic(os.path.join(CPP_MODELS, "hot_cut_bunny_sampled.bhclassic"))["position"].astype(np.float64)
    pos += np.array([0.0, 0.1, 0.0])
    d = 0.05 if args.cpp_compat else 0.06
    bnd, meshes = [], []
    for name in ("hot_cut_cross_plane", "hot_cut_bcylinder"):
        V, F = load_obj(os.path.join(CPP_MODELS, name + ".obj"))
        bnd.append(cpp_boundary_points(args, name, "1_1_1") if args.cpp_compat else sample_mesh(V, F, d))
        meshes.append((name, V, F))
    bnd = np.concatenate(bnd)
    if not args.cpp_compat:
        pos = pos[min_dist(bnd, pos, d) > 0.75 * d]
    allp = np.concatenate([pos, bnd])
    off = -allp.min(0) + 3.0 * d
    dom = allp.max(0) + off + 3.0 * d
    mats = [default_material(mu0=20.0, decay_mu=0.1)]
    dt = args.dt or 5e-3
    hb_y = 0.05 + off[1]
    return dict(pos=pos + off, body=np.zeros(len(pos), np.int32), mats=mats, names=["bunny"], d=d,
                domain=dom.tolist(), dt=dt, steps=args.steps or int(round(20.0 / dt)), heat=True,
                cpp_offset=off, meshes=meshes,
                solver_kw=dict(box_walls=not args.cpp_compat, boundary=bnd + off, clip_domain=not args.cpp_compat, boundary_akinci=args.cpp_compat, mu_scale=1000.0, visc_boundary=0.0,
                               viscosity_force_scale=20.0 if args.cpp_compat else 1.0,
                               heat_y=hb_y, heat_R=0.1 if args.cpp_compat else 1.0, diffusion=100.0,
                               **(dict(heat_box=((-1.0, 0.0, -1.0), (1.0, 0.05, 1.0))) if args.cpp_compat else {})))


def load_obj(path):
    V, F = [], []
    with open(path) as f:
        for line in f:
            s = line.split()
            if not s:
                continue
            if s[0] == "v":
                V.append([float(t) for t in s[1:4]])
            elif s[0] == "f":
                F.append([int(t.split("/")[0]) - 1 for t in s[1:]])
    return np.array(V), F


BICEP_PLY = os.path.expanduser("~/CG/nonNewton-hou/geo/pointsOri.ply")


def read_ply_points(path):
    # Houdini gply 导出的二进制 PLY（仅顶点、float 属性），返回 {属性名: 数组}
    with open(path, "rb") as f:
        raw = f.read()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode("ascii").splitlines()
    endian = ">" if "binary_big_endian" in head[1] else "<"
    n, props = 0, []
    for line in head:
        t = line.split()
        if t[:2] == ["element", "vertex"]:
            n = int(t[2])
        elif t[:2] == ["property", "float"]:
            props.append(t[2])
        elif t[:1] == ["element"] and props:
            break
    a = np.frombuffer(raw, endian + "f4", n * len(props), end).reshape(n, len(props))
    return {k: a[:, c].astype(np.float64) for c, k in enumerate(props)}


def scene_bicep(args):
    # 二头肌：近端（两个头的肌腱端）Dirichlet 固定；远端肌腱端为 Neumann（给定力）或 Robin（弹簧）边界。
    # 肌腱与肌腹为同一弹性体内的两种材料（tendonmask > 0.5 为肌腱）
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    d = 0.005
    pts = read_ply_points(args.ply)
    pos = np.stack([pts["x"], pts["y"], pts["z"]], 1)
    tendon = pts["tendonmask"] > 0.5
    c = pos.mean(0)
    axis = np.linalg.svd(pos - c, full_matrices=False)[2][0]
    if axis[1] > 0:
        axis = -axis  # 主轴从近端（肩，上）指向远端（肘，下）
    s = (pos - c) @ axis
    # 肌腱按连通性分成若干束：近端为长头、短头两束，远端一束
    ti_ = np.where(tendon)[0]
    pr = cKDTree(pos[ti_]).query_pairs(1.5 * d, output_type="ndarray")
    _, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(ti_),) * 2), directed=False)
    pinned = np.zeros(len(pos), np.int32)
    loaded = np.zeros(len(pos), bool)
    for k in np.unique(lab):
        m = ti_[lab == k]
        if len(m) < 10:
            continue
        if s[m].mean() < 0:  # 近端各头：从该头末端起 args.end_len 内固定
            pinned[m[s[m] < s[m].min() + args.end_len]] = 1
        else:  # 远端：从末端起 args.end_len 内施加载荷
            loaded[m[s[m] > s[m].max() - args.end_len]] = True
    # Houdini 撒点为六方密排，邻居数约为立方点阵的两倍；取初始构型核函数和的最大值作为静息密度基准，
    # 使所有粒子初始都不超过静息密度（单边密度约束在静止时不激活）。取中位数时约一半粒子超出 0.67%，静置会抖动
    probe = type("P", (), {"h": 2.0 * d})()
    nb = cKDTree(pos).query_ball_point(pos, probe.h)
    ksum = float(np.max([sum(ConstraintSolver._W_np(probe, r) for r in np.linalg.norm(pos[j] - pos[i], axis=1))
                         for i, j in enumerate(nb)]))
    margin = 0.1
    shift = margin - pos.min(0)
    pos = pos + shift
    domain = (pos.max(0) + margin).tolist()
    body = tendon.astype(np.int32)  # 0 肌腹，1 肌腱
    mats = [
        default_material(mu0=args.mu_belly, E=args.E, nu=0.42, k_fiber=args.k_fiber, eps_fiber=args.eps_fiber),
        default_material(mu0=10.0 * args.mu_belly, E=args.E_tendon, nu=0.42),
    ]
    fdir = np.stack([pts["materialW1"], pts["materialW2"], pts["materialW3"]], 1)
    fw = np.clip(1.0 - pts["tendonmask"], 0.0, 1.0)  # 肌腱不收缩，交界处按 tendonmask 平滑过渡

    def activation(t):  # 先静置 t0，再线性升到 1、保持、线性降回 0
        t0, ramp, hold = args.act_start, args.act_ramp, args.act_hold
        u = t - t0
        if u <= 0:
            return 0.0
        if u < ramp:
            return u / ramp
        if u < ramp + hold:
            return 1.0
        return max(0.0, 1.0 - (u - ramp - hold) / ramp)

    # 远端载荷（总力，牛顿），由载荷粒子均分：
    #   neumann：F = F0 * axis，给定力
    #   robin：  F = F0 * axis - k (c - c0)，c 为载荷粒子质心，c0 为其静息位置（弹簧另一端固定于此）
    # F0 在 [0, act_start] 内线性加载，避免突加载荷引起的冲击
    c0 = pos[loaded].mean(0)
    nl = int(loaded.sum())
    k_spring = args.load_k if args.load == "robin" else 0.0
    state = dict(F=np.zeros(3), u=0.0)

    def load(t, x, mass):
        F0 = args.load_F * min(1.0, t / max(args.act_start, 1e-6))
        cl = x[loaded].mean(0)
        F = F0 * axis - k_spring * (cl - c0)
        state["F"], state["u"] = F, float((cl - c0) @ axis)
        acc = np.zeros((len(x), 3), np.float32)
        acc[loaded] = F / (nl * mass)
        return acc

    def monitor():  # 远端载荷点沿主轴的位移（正为向远端伸长）与载荷力沿主轴的分量
        return dict(distal_u=round(state["u"], 5), load_F=round(float(state["F"] @ axis), 4))

    print(f"[bicep] pinned={int(pinned.sum())} (proximal heads)  loaded={nl} (distal, {args.load}, "
          f"F0={args.load_F} N, k={k_spring} N/m)  axis={axis.round(3).tolist()}")
    return dict(pos=pos, body=body, mats=mats, names=["belly", "tendon"], d=d, domain=domain, dt=1e-3,
                steps=args.steps or 1500, activation=activation, load=load, monitor=monitor, axis=1,
                gravity=(0.0, -args.gravity, 0.0), cpp_offset=shift,
                primvars=dict(pinned=pinned, loaded=loaded.astype(np.int32)),
                solver_kw=dict(box_walls=False, pinned=pinned, obj=np.zeros(len(pos), np.int32), rest_kernel_sum=ksum,
                               fiber_dir=fdir, fiber_w=fw, omega_f=args.omega_f))


SCENES = dict(dambreak=scene_dambreak, models=scene_models, tomato=scene_tomato, melt=scene_melt, ramp=scene_ramp, icecream=scene_icecream,
              hotcut=scene_hotcut, bicep=scene_bicep)


def body_stats(x, body, nb):
    out = []
    for b in range(nb):
        p = x[body == b]
        out.append(dict(height=float(p[:, 1].max()), front=float(p[:, 0].max()), com=p.mean(0).round(4).tolist(),
                        extent=(p.max(0) - p.min(0)).round(4).tolist()))
    return out


class UsdPointsWriter:
    """把粒子序列写成一个 USD 文件：UsdGeom.Points，points/velocities/primvars 按帧时间采样。"""

    def __init__(self, path, fps, radius, body):
        from pxr import Sdf, Usd, UsdGeom, Vt

        self.Vt = Vt
        self.stage = Usd.Stage.CreateNew(path)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.stage.SetTimeCodesPerSecond(fps)
        self.stage.SetFramesPerSecond(fps)
        self.stage.SetStartTimeCode(0)
        root = UsdGeom.Xform.Define(self.stage, "/World")
        self.stage.SetDefaultPrim(root.GetPrim())
        self.pts = UsdGeom.Points.Define(self.stage, "/World/particles")
        n = len(body)
        self.pts.CreateWidthsAttr(Vt.FloatArray.FromNumpy(np.full(n, 2.0 * radius, np.float32)))
        self.pts.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        pv = UsdGeom.PrimvarsAPI(self.pts)
        pv.CreatePrimvar("body", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.vertex).Set(
            Vt.IntArray.FromNumpy(body.astype(np.int32)))
        self.mu = pv.CreatePrimvar("mu", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
        self.T = pv.CreatePrimvar("T", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
        self.sr = pv.CreatePrimvar("strainRateNorm", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
        self.last = 0

    def add_static_primvar(self, name, arr):
        from pxr import Sdf, UsdGeom

        UsdGeom.PrimvarsAPI(self.pts).CreatePrimvar(name, Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.vertex).Set(
            self.Vt.IntArray.FromNumpy(np.asarray(arr, np.int32)))

    def set_scalar(self, name, value, frame):
        # 粒子 prim 上的逐帧标量（如激活 a(t)），Houdini 中为 detail 属性
        from pxr import Sdf

        attr = self.pts.GetPrim().GetAttribute(name) or self.pts.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Float)
        attr.Set(value, frame)

    def add_boundary(self, meshes, points, radius):
        # 静态边界：原始三角网格（/World/boundary/<name>）+ 求解器实际使用的边界粒子（/World/boundary/particles）
        from pxr import UsdGeom

        Vt = self.Vt
        UsdGeom.Scope.Define(self.stage, "/World/boundary")
        for name, V, F in meshes:
            m = UsdGeom.Mesh.Define(self.stage, f"/World/boundary/{name}")
            m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(V, np.float32)))
            m.CreateFaceVertexCountsAttr(Vt.IntArray([len(f) for f in F]))
            m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for f in F for i in f]))
            m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        if len(points):
            bp = UsdGeom.Points.Define(self.stage, "/World/boundary/particles")
            bp.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, np.float32)))
            bp.CreateWidthsAttr(Vt.FloatArray([2.0 * radius]))
            bp.SetWidthsInterpolation(UsdGeom.Tokens.constant)

    def add(self, frame, x, v, mu, T, sr):
        Vt = self.Vt
        self.sr.Set(Vt.FloatArray.FromNumpy(sr.astype(np.float32)), frame)
        self.pts.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(x.astype(np.float32)), frame)
        self.pts.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)), frame)
        self.mu.Set(Vt.FloatArray.FromNumpy(mu.astype(np.float32)), frame)
        self.T.Set(Vt.FloatArray.FromNumpy(T.astype(np.float32)), frame)
        self.last = frame

    def close(self):
        self.stage.SetEndTimeCode(self.last)
        self.stage.GetRootLayer().Save()


def snapshot(x, body, path, title, view=(0, 1)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(x[:, view[0]], x[:, view[1]], c=body, s=1, cmap="tab10", vmin=0, vmax=9)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="dambreak", choices=list(SCENES))
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--div_iters", type=int, default=5)
    ap.add_argument("--no_div", action="store_true")
    ap.add_argument("--E", type=float, default=2e5)
    ap.add_argument("--arch", default="cpu")
    ap.add_argument("--out", default="output")
    ap.add_argument("--snap_every", type=int, default=0)
    ap.add_argument("--mu_sweep", action="store_true")
    ap.add_argument("--omega_s", type=float, default=0.0, help="黏性/弹性约束 Jacobi 松弛，0 表示用场景默认（0.5）")
    ap.add_argument("--omega_rho", type=float, default=0.0, help="Density/divergence relaxation; compatibility default 0.5")
    ap.add_argument("--visc_cg", action="store_true", help="Solve the quadratic viscous strain energy globally by matrix-free CG")
    ap.add_argument("--cpp_visc_cg", action="store_true", help="Use the C++ Casson fluid viscosity discretization and symmetrized CG")
    ap.add_argument("--cpp_visc_boundary", type=float, default=0.1, help="Casson boundary viscosity (C++ default 0.1); zero disables for ablation")
    ap.add_argument("--cpp_dfsph", action="store_true", help="Match C++ thermal-scene DFSPH pressure, warmstart, stopping criteria and force order")
    ap.add_argument("--group", default="ramp1", choices=list(RAMP_GROUPS), help="ramp 场景：对应 nonNewtonCode 的 ramp1/2/3.json")
    ap.add_argument("--coarsen", type=int, default=1, help="ramp 场景：粒子点阵粗化倍数，2 即间距 0.1、粒子数 1/8")
    ap.add_argument("--dt", type=float, default=0.0, help="覆盖场景默认时间步")
    ap.add_argument("--noslip", action="store_true", help="ramp 场景：斜面参与黏性约束（无滑移），默认用 C++ 的 Akinci 边界黏性 0.1")
    ap.add_argument("--cpp_compat", action="store_true", help="复刻 nonNewtonCode 的应变率（仅末邻居）与 Carreau 写法，且不截断黏度")
    ap.add_argument("--cpp_boundary_cache", default="", help="C++ data/MyScenes/Cache directory for matching thermal-scene boundary samples")
    ap.add_argument("--export", default="none", choices=["none", "usd"],
                    help="导出 usdc 序列；含边界网格与边界粒子，C++ 场景（ramp/icecream/hotcut）坐标与 nonNewtonCode 一致")
    ap.add_argument("--fps", type=float, default=30.0, help="导出帧率，每 round(1/(fps*dt)) 步导出一帧")
    ap.add_argument("--ply", default=BICEP_PLY, help="bicep 场景：带 tendonmask 的粒子 PLY")
    ap.add_argument("--E_tendon", type=float, default=2e7, help="bicep 场景：肌腱杨氏模量（肌腹用 --E）")
    ap.add_argument("--mu_belly", type=float, default=1.0, help="bicep 场景：肌腹黏度，肌腱取其 10 倍")
    ap.add_argument("--load", default="robin", choices=["neumann", "robin"], help="bicep 场景：远端边界，给定力或弹簧")
    ap.add_argument("--load_F", type=float, default=5.0, help="bicep 场景：远端沿主轴向远端的力（N）；robin 时为弹簧预紧力")
    ap.add_argument("--load_k", type=float, default=200.0, help="bicep 场景：robin 弹簧刚度（N/m）")
    ap.add_argument("--end_len", type=float, default=0.015, help="bicep 场景：近端固定段 / 远端载荷段沿主轴的长度（m）")
    ap.add_argument("--gravity", type=float, default=0.0, help="bicep 场景：重力加速度大小，默认 0（载荷由远端边界给出）")
    ap.add_argument("--k_fiber", type=float, default=0.0, help="bicep 场景：主动纤维约束刚度，0 表示不收缩（1e7 可见收缩）")
    ap.add_argument("--omega_f", type=float, default=0.1, help="bicep 场景：主动纤维约束的 Jacobi 松弛，大于 0.25 会明显抖动")
    ap.add_argument("--eps_fiber", type=float, default=0.3, help="bicep 场景：满激活时的目标纤维缩短应变")
    ap.add_argument("--act_start", type=float, default=0.5, help="bicep 场景：开始激活的时刻（s），之前静置")
    ap.add_argument("--act_ramp", type=float, default=0.2, help="bicep 场景：激活升降时长（s）")
    ap.add_argument("--act_hold", type=float, default=0.5, help="bicep 场景：满激活保持时长（s）")
    args = ap.parse_args()
    if args.cpp_visc_cg:
        args.visc_cg = True
        if not args.cpp_compat:
            ap.error("--cpp_visc_cg requires --cpp_compat")
    if args.cpp_dfsph and not (args.cpp_visc_cg and args.scene in ("icecream", "hotcut")):
        ap.error("--cpp_dfsph requires --cpp_visc_cg and icecream/hotcut")
    if args.visc_cg and args.scene not in ("icecream", "hotcut"):
        ap.error("--visc_cg currently supports the fluid-only viscosity of icecream/hotcut")

    ti.init(arch=getattr(ti, args.arch), default_fp=ti.f32, random_seed=0)
    sc = SCENES[args.scene](args)
    if args.cpp_compat and args.scene in ("ramp", "icecream", "hotcut"):
        sc.setdefault("solver_kw", {}).update(particle_mass=0.8 * 1000.0 * 0.05**3,
                                              rest_density=1000.0, cpp_offset=sc["cpp_offset"])
    omega_s = args.omega_s or sc.get("omega_s", 0.5)
    if args.cpp_compat and sc.get("heat", False):
        sc.setdefault("solver_kw", {})["v_clamp"] = 4.0
    heat = sc.get("heat", False)
    solver = ConstraintSolver(
        sc["pos"], sc["body"], sc["mats"], sc["d"], sc["domain"], sc["dt"],
        gravity=sc.get("gravity", (0.0, -9.81, 0.0)), iters=args.iters, div_iters=args.div_iters, omega_s=omega_s,
        omega=args.omega_rho or (0.5 if args.cpp_compat else 1.0),
        visc_cg=args.visc_cg,
        cpp_visc_cg=args.cpp_visc_cg,
        cpp_visc_boundary=args.cpp_visc_boundary,
        cpp_dfsph=args.cpp_dfsph,
        cpp_visc_max_iters=100 if args.cpp_dfsph and args.scene == "icecream" else 1000,
        use_div=not args.no_div, cpp_compat=args.cpp_compat,
        **{**dict(heat_y=0.03 if heat else -1.0, heat_R=30.0 if heat else 0.0, diffusion=100.0 if heat else 0.0),
           **sc.get("solver_kw", {})},
    )
    solver.compute_density(solver.x)
    solver.compute_strain_rate_and_mu()
    # 加热场景：只加热左块（body 0），右块每步温度清零作为对照
    tag = args.scene + ("_musweep" if args.mu_sweep else "") + ("_nodiv" if args.no_div else "")
    if args.scene == "ramp":
        tag += f"_{args.group}" + (f"_c{args.coarsen}" if args.coarsen > 1 else "") + ("_noslip" if args.noslip else "")
    if args.cpp_compat:
        tag += "_cpp"
    axis = sc.get("axis", 0)
    os.makedirs(args.out, exist_ok=True)
    print(f"[{tag}] particles={solver.nf} boundary={solver.n - solver.nf} rho0={solver.rho0:.2f} mass={solver.mass:.3e} h={solver.h}")
    nb = len(sc["mats"])
    body = sc["body"]
    log = []
    every = max(1, round(1.0 / (args.fps * sc["dt"])))
    cpp_off = np.asarray(sc.get("cpp_offset", np.zeros(3)), np.float64)  # 求解器坐标 = C++ 坐标 + cpp_off
    usd = None
    if args.export == "usd":
        usd = UsdPointsWriter(f"{args.out}/{tag}.usdc", 1.0 / (every * sc["dt"]), 0.5 * sc["d"], body)
        usd.add_boundary(sc.get("meshes", []), solver.x.to_numpy()[solver.nf:] - cpp_off, 0.5 * sc["d"])
        for name, arr in sc.get("primvars", {}).items():
            usd.add_static_primvar(name, arr)

    def export(frame):
        nf = solver.nf
        xs, vs = solver.x.to_numpy()[:nf] - cpp_off, solver.v.to_numpy()[:nf]
        mus, Ts, srs = solver.mu.to_numpy()[:nf], solver.T.to_numpy()[:nf], solver.sr.to_numpy()[:nf]
        usd.add(frame, xs, vs, mus, Ts, srs)
        if "activation" in sc:
            usd.set_scalar("activation", float(solver.activation[None]), frame)

    if args.export != "none":
        print(f"[{tag}] export={args.export} every {every} steps ({1.0 / (every * sc['dt']):.1f} fps)")
        export(0)
    t0 = time.time()
    for s in range(sc["steps"]):
        if heat and args.scene == "melt":  # melt 对照组：右块不加热
            T = solver.T.to_numpy()
            T[:solver.nf][body == 1] = 0.0
            solver.T.from_numpy(T)
        if "activation" in sc:
            solver.activation[None] = sc["activation"](s * sc["dt"])
        if "load" in sc:
            solver.fext.from_numpy(sc["load"](s * sc["dt"], solver.x.to_numpy(), solver.mass))
        ep, eo = solver.step()
        x = solver.x.to_numpy()[:solver.nf]
        if not np.isfinite(x).all():
            if usd is not None:
                usd.close()
            raise FloatingPointError(f"Non-finite positions at step {s}")
        if s % 50 == 0 or s == sc["steps"] - 1:
            vmax = float(np.linalg.norm(solver.v.to_numpy()[:solver.nf], axis=1).max())
            st = body_stats(x, body, nb)
            rec = dict(step=s, t=round((s + 1) * sc["dt"], 4), err_pred_avg=float(ep[0]), err_pred_max=float(ep[1]),
                       err_post_avg=float(eo[0]), err_post_max=float(eo[1]), vmax=vmax,
                       heights=[round(b["height"], 4) for b in st])
            nd = solver.neighbor_diagnostics()
            rec.update(cell_overflow=int(nd[0]), neighbor_cap=int(nd[1]), outside_grid=int(nd[2]))
            if args.visc_cg:
                rec.update(visc_cg_iters=solver.cg_iterations, visc_cg_error=solver.cg_error)
            if args.cpp_dfsph:
                rec.update(pressure_iters=solver.df_pressure_iters, divergence_iters=getattr(solver, "df_div_iters", 0))
            rec["com_" + "xyz"[axis]] = [round(b["com"][axis], 4) for b in st]
            if "activation" in sc:
                rec["activation"] = round(float(solver.activation[None]), 3)
            if "monitor" in sc:
                rec.update(sc["monitor"]())
            mu = solver.mu.to_numpy()[:solver.nf]
            rec["mu_mean"] = [round(float(mu[body == b].mean()), 3) for b in range(nb)]
            if heat:
                T = solver.T.to_numpy()[:solver.nf]
                rec["T_mean"] = [round(float(T[body == b].mean()), 3) for b in range(nb)]
            log.append(rec)
            print(json.dumps(rec, ensure_ascii=False))
        if args.export != "none" and (s + 1) % every == 0:
            export((s + 1) // every)
        if args.snap_every and s % args.snap_every == 0:
            view = sc.get("view", (2, 1) if args.scene == "models" else (0, 1))
            snapshot(x, body, f"{args.out}/{tag}_{s:05d}.png", f"{tag} t={(s+1)*sc['dt']:.3f}s", view)
    el = time.time() - t0
    if usd is not None:
        usd.close()
    x = solver.x.to_numpy()[:solver.nf]
    st = body_stats(x, body, nb)
    view = sc.get("view", (2, 1) if args.scene == "models" else (0, 1))
    snapshot(x, body, f"{args.out}/{tag}_final.png", f"{tag} final", view)
    if args.scene == "models":
        snapshot(x, body, f"{args.out}/{tag}_final_top.png", f"{tag} final (top view, x=downslope)", (2, 0))
    summary = dict(scene=tag, particles=solver.nf, steps=sc["steps"], seconds=round(el, 1),
                   ms_per_step=round(1000 * el / max(1, sc["steps"]), 1), names=sc.get("names"), bodies=st, log=log)
    summary["parameters"] = dict(dt=sc["dt"], iters=args.iters, div_iters=args.div_iters,
                                 omega_rho=solver.omega, omega_s=omega_s, visc_cg=args.visc_cg,
                                 cpp_visc_cg=args.cpp_visc_cg,
                                 cpp_visc_boundary=args.cpp_visc_boundary, cpp_dfsph=args.cpp_dfsph,
                                 cpp_visc_max_iters=solver.cpp_visc_max_iters,
                                 mass=solver.mass, rho0=solver.rho0, h=solver.h,
                                 viscosity_force_scale=solver.viscosity_force_scale,
                                 position_precision="f64" if args.cpp_compat else "f32",
                                 unbounded_hash=args.cpp_compat, max_neighbors=solver.max_nb)
    with open(f"{args.out}/{tag}.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"[{tag}] done in {el:.1f}s ({summary['ms_per_step']} ms/step)")


if __name__ == "__main__":
    main()
