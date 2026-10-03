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
    ):
        self.d = spacing
        self.h = 2.0 * spacing  # 支撑半径 = 4 倍粒子半径
        self.dt = dt
        self.iters = iters
        self.div_iters = div_iters
        self.omega = omega
        self.omega_s = omega_s  # 黏性/弹性约束的 Jacobi 松弛
        self.use_div = use_div
        self.friction = friction
        self.heat_y = heat_y
        self.heat_R = heat_R
        self.diffusion = diffusion
        self.domain = np.array(domain, dtype=np.float64)
        self.pad = 0.5 * spacing
        self.gravity = ti.Vector(list(gravity))
        self.mass = 1000.0 * spacing**3
        self.rho0 = self.mass * self._lattice_kernel_sum()
        self.V = self.mass / self.rho0
        self.kappa_inv = density_compliance  # alpha_rho * V，0 即严格不可压
        self.has_elastic = any(mt["E"] > 0 for mt in materials)
        self.has_heat = heat_R != 0.0 or diffusion != 0.0
        self.materials = materials
        # 静态边界粒子（逆质量 w=0）：两层，覆盖地面与四周侧壁
        bpos = self._boundary_points()
        self.nf = len(pos)
        self.n = n = self.nf + len(bpos)
        pos = np.concatenate([pos, bpos])
        body = np.concatenate([body, -np.ones(len(bpos), np.int32)])
        self.origin = -3.0 * spacing

        # 粒子状态
        self.x = ti.Vector.field(3, ti.f32, n)
        self.x_old = ti.Vector.field(3, ti.f32, n)
        self.x0 = ti.Vector.field(3, ti.f32, n)
        self.v = ti.Vector.field(3, ti.f32, n)
        self.dx = ti.Vector.field(3, ti.f32, n)
        self.body = ti.field(ti.i32, n)
        self.rho = ti.field(ti.f32, n)
        self.T = ti.field(ti.f32, n)
        self.T_new = ti.field(ti.f32, n)
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
        self.dlam = ti.field(ti.f32, n)
        # 邻域
        self.nb = ti.field(ti.i32, (n, MAX_NB))
        self.nb_cnt = ti.field(ti.i32, n)
        self.ref_nb = ti.field(ti.i32, (n, MAX_NB))
        self.ref_cnt = ti.field(ti.i32, n)
        self.g0 = ti.Vector.field(3, ti.f32, (n, MAX_NB))
        self.gdim = [int(math.ceil((L + 6.0 * spacing) / self.h)) + 1 for L in self.domain]
        self.cell_count = ti.field(ti.i32, self.gdim)
        self.cell_list = ti.field(ti.i32, self.gdim + [MAX_CELL])
        # 材料表
        self.b_model = ti.field(ti.i32, MAX_BODY)
        self.b_elastic = ti.field(ti.i32, MAX_BODY)
        self.b_par = ti.field(ti.f32, (MAX_BODY, 16))
        # 诊断
        self.stat = ti.field(ti.f32, 4)

        self.x.from_numpy(pos.astype(np.float32))
        self.x0.from_numpy(pos.astype(np.float32))
        self.x_old.from_numpy(pos.astype(np.float32))  # 边界粒子的 x_old 必须等于其位置，否则黏性约束会看到虚假位移
        self.body.from_numpy(body.astype(np.int32))
        self._load_materials(materials)
        self._init_state()
        self.build_grid(self.x)
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
            par[b, :13] = [
                mt["mu0"], mt["mu_inf"], mt["m"], mt["n"], mt["crit"], mt["muC"], mt["tau0"],
                G, K, mt["gamma1"], mt["gamma2"], mt["decay_mu"], mt["decay_G"],
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
    @ti.func
    def cell_of(self, p):
        q = (p - self.origin) / self.h
        return ti.Vector([int(ti.floor(q[0])), int(ti.floor(q[1])), int(ti.floor(q[2]))])

    @ti.func
    def in_grid(self, c):
        return 0 <= c[0] < self.gdim[0] and 0 <= c[1] < self.gdim[1] and 0 <= c[2] < self.gdim[2]

    @ti.kernel
    def build_grid(self, xs: ti.template()):
        for I in ti.grouped(self.cell_count):
            self.cell_count[I] = 0
        for i in range(self.n):
            c = self.cell_of(xs[i])
            k = ti.atomic_add(self.cell_count[c], 1)
            if k < MAX_CELL:
                self.cell_list[c, k] = i

    @ti.kernel
    def find_neighbors(self, xs: ti.template()):
        for i in range(self.nf):
            c = self.cell_of(xs[i])
            cnt = 0
            for off in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cc = c + off
                if self.in_grid(cc):
                    for k in range(ti.min(self.cell_count[cc], MAX_CELL)):
                        j = self.cell_list[cc, k]
                        if j != i and (xs[i] - xs[j]).norm() < self.h and cnt < MAX_NB:
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
                if self.body[j] == self.body[i]:
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
            mu = mui + (mu0 - mui) / ti.pow(1.0 + (m * s) ** 2, (1.0 - n) / 2.0)
        elif model == BINGHAM:
            if s > crit:
                mu = mui + crit * (mu0 - mui) / s
        elif model == HERSCHEL_BULKLEY:
            if s > crit:
                t0 = mu0 * crit - m * ti.pow(crit, n)
                mu = t0 / s + m * ti.pow(s, n - 1.0)
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
                rho += self.mass * self.W((xs[i] - xs[j]).norm())
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
            self.sr[i] = sr
            b = self.body[i]
            self.mu[i] = self.viscosity(b, sr) * ti.exp(-self.b_par[b, 11] * self.T[i])

    @ti.kernel
    def predict(self):
        for i in range(self.nf):
            self.x_old[i] = self.x[i]
            self.v[i] += self.dt * self.gravity
            self.x[i] += self.dt * self.v[i]
            self.lam_rho[i] = 0.0
            self.lam_div[i] = 0.0
            self.lam_vis[i] = 0.0
            self.lam_dev[i] = 0.0
            self.lam_vol[i] = 0.0

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
            g = self.mass / self.rho0 * self.gradW(xs[i] - xs[j])
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
                rho += self.mass * self.W((self.x[i] - self.x[j]).norm())
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
                d += (self.dlam[i] + self.dlam[j]) * self.gradW(xs[i] - xs[j])
            out[i] = d / self.rho0  # 等质量：w_i m_j = 1

    # ---- 无散约束（速度层）
    @ti.kernel
    def div_k1(self):
        a_t = self.kappa_inv / (self.V * self.dt * self.dt)
        for i in range(self.nf):
            cd = 0.0
            for jj in range(self.nb_cnt[i]):
                j = self.nb[i, jj]
                cd += self.mass / self.rho0 * (self.v[i] - self.v[j]).dot(self.gradW(self.x[i] - self.x[j]))
            A = self.density_diag(i, self.x) + a_t
            dl = (-cd - a_t * self.lam_div[i]) / (A + 1e-12)
            lnew = ti.min(self.lam_div[i] + dl, 0.0)
            self.dlam[i] = lnew - self.lam_div[i]
            self.lam_div[i] = lnew

    @ti.kernel
    def apply_dv(self):
        for i in range(self.nf):
            self.v[i] += self.omega * self.dx[i]

    # ---- 黏性约束
    @ti.kernel
    def viscous_k1(self):
        for i in range(self.nf):
            G = ti.Matrix.zero(ti.f32, 3, 3)
            dxi = self.x[i] - self.x_old[i]
            for jj in range(self.nb_cnt[i]):
                j = self.nb[i, jj]
                g = self.gradW(self.x_old[i] - self.x_old[j])
                G += self.V * ((self.x[j] - self.x_old[j]) - dxi).outer_product(g)
            De = self.dev(0.5 * (G + G.transpose()))
            C = De.norm()
            self.dlam[i] = 0.0
            self.Nmat[i] = ti.Matrix.zero(ti.f32, 3, 3)
            if C > 1e-9 and self.mu[i] > 0.0:
                N = De / C
                gi = ti.Vector([0.0, 0.0, 0.0])
                s2 = 0.0
                for jj in range(self.nb_cnt[i]):
                    j = self.nb[i, jj]
                    a = self.V * (N @ self.gradW(self.x_old[i] - self.x_old[j]))
                    gi += a
                    if j < self.nf:
                        s2 += a.norm_sqr()
                A = (gi.norm_sqr() + s2) / self.mass
                a_t = 1.0 / (2.0 * self.mu[i] * self.V * self.dt)
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
                else:
                    C = ee.trace()
                    stiff = self.b_par[b, 8] * ti.exp(-self.b_par[b, 12] * self.T[i])  # K
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
                    lam = self.lam_dev[i] if ti.static(mode == 0) else self.lam_vol[i]
                    dl = (-C - a_t * lam) / (A + a_t)
                    if ti.static(mode == 0):
                        self.lam_dev[i] += dl
                    else:
                        self.lam_vol[i] += dl
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
            for a in ti.static(range(3)):
                lo = self.pad
                hi = self.domain[a] - self.pad
                p[a] = ti.min(ti.max(p[a], lo), hi)
            if p[1] <= self.pad + 1e-7:  # 地面摩擦
                p[0] = self.x_old[i][0] + (1.0 - self.friction) * (p[0] - self.x_old[i][0])
                p[2] = self.x_old[i][2] + (1.0 - self.friction) * (p[2] - self.x_old[i][2])
            self.x[i] = p

    @ti.kernel
    def update_velocity(self):
        for i in range(self.nf):
            self.v[i] = (self.x[i] - self.x_old[i]) / self.dt

    @ti.kernel
    def boundary_vel(self):
        for i in range(self.nf):
            for a in ti.static(range(3)):
                if self.x[i][a] <= self.pad + 1e-7 and self.v[i][a] < 0:
                    self.v[i][a] = 0.0
                if self.x[i][a] >= self.domain[a] - self.pad - 1e-7 and self.v[i][a] > 0:
                    self.v[i][a] = 0.0

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
            if self.x[i][1] < self.heat_y:
                src = self.heat_R
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

    # ------------------------------------------------------------------ main step
    def step(self):
        # 1. 步初：邻域、应变率、黏度（柔度）
        self.build_grid(self.x)
        self.find_neighbors(self.x)
        self.compute_density(self.x)
        self.compute_strain_rate_and_mu()
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
            self.viscous_k1()
            self.viscous_k2()
            self.apply_dx(self.omega_s)
            if self.has_elastic:
                self.compute_rotation()
                for mode in (0, 1):
                    self.elastic_k1(mode)
                    self.elastic_k2()
                    self.apply_dx(self.omega_s)
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
        self.boundary_vel()
        # 5. 塑性回映、温度扩散
        if self.has_elastic:
            self.compute_rotation()
            self.plasticity()
        if self.has_heat:
            self.diffuse()
        return err_pred, err_post


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


SCENES = dict(dambreak=scene_dambreak, models=scene_models, tomato=scene_tomato, melt=scene_melt)


def body_stats(x, body, nb):
    out = []
    for b in range(nb):
        p = x[body == b]
        out.append(dict(height=float(p[:, 1].max()), front=float(p[:, 0].max()), com=p.mean(0).round(4).tolist(),
                        extent=(p.max(0) - p.min(0)).round(4).tolist()))
    return out


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
    ap.add_argument("--omega_s", type=float, default=0.5)
    args = ap.parse_args()

    ti.init(arch=getattr(ti, args.arch), default_fp=ti.f32, random_seed=0)
    sc = SCENES[args.scene](args)
    heat = sc.get("heat", False)
    solver = ConstraintSolver(
        sc["pos"], sc["body"], sc["mats"], sc["d"], sc["domain"], sc["dt"],
        gravity=sc.get("gravity", (0.0, -9.81, 0.0)), iters=args.iters, div_iters=args.div_iters, omega_s=args.omega_s,
        use_div=not args.no_div, heat_y=0.03 if heat else -1.0, heat_R=30.0 if heat else 0.0,
        diffusion=100.0 if heat else 0.0,
    )
    # 加热场景：只加热左块（body 0），右块每步温度清零作为对照
    tag = args.scene + ("_musweep" if args.mu_sweep else "") + ("_nodiv" if args.no_div else "")
    os.makedirs(args.out, exist_ok=True)
    print(f"[{tag}] particles={solver.nf} boundary={solver.n - solver.nf} rho0={solver.rho0:.2f} mass={solver.mass:.3e} h={solver.h}")
    nb = len(sc["mats"])
    body = sc["body"]
    log = []
    t0 = time.time()
    for s in range(sc["steps"]):
        if heat:
            T = solver.T.to_numpy()
            T[:solver.nf][body == 1] = 0.0
            solver.T.from_numpy(T)
        ep, eo = solver.step()
        x = solver.x.to_numpy()[:solver.nf]
        if not np.isfinite(x).all():
            print(f"NaN at step {s}")
            break
        if s % 50 == 0 or s == sc["steps"] - 1:
            vmax = float(np.linalg.norm(solver.v.to_numpy()[:solver.nf], axis=1).max())
            st = body_stats(x, body, nb)
            rec = dict(step=s, t=round((s + 1) * sc["dt"], 4), err_pred_avg=float(ep[0]), err_pred_max=float(ep[1]),
                       err_post_avg=float(eo[0]), err_post_max=float(eo[1]), vmax=vmax,
                       heights=[round(b["height"], 4) for b in st], com_x=[round(b["com"][0], 4) for b in st])
            if heat:
                T = solver.T.to_numpy()[:solver.nf]
                rec["T_mean"] = [round(float(T[body == b].mean()), 3) for b in range(nb)]
                rec["mu_mean"] = [round(float(solver.mu.to_numpy()[:solver.nf][body == b].mean()), 3) for b in range(nb)]
            log.append(rec)
            print(json.dumps(rec, ensure_ascii=False))
        if args.snap_every and s % args.snap_every == 0:
            view = (2, 1) if args.scene == "models" else (0, 1)
            snapshot(x, body, f"{args.out}/{tag}_{s:05d}.png", f"{tag} t={(s+1)*sc['dt']:.3f}s", view)
    el = time.time() - t0
    x = solver.x.to_numpy()[:solver.nf]
    st = body_stats(x, body, nb)
    view = (2, 1) if args.scene == "models" else (0, 1)
    snapshot(x, body, f"{args.out}/{tag}_final.png", f"{tag} final", view)
    if args.scene == "models":
        snapshot(x, body, f"{args.out}/{tag}_final_top.png", f"{tag} final (top view, x=downslope)", (2, 0))
    summary = dict(scene=tag, particles=solver.nf, steps=sc["steps"], seconds=round(el, 1),
                   ms_per_step=round(1000 * el / max(1, sc["steps"]), 1), names=sc.get("names"), bodies=st, log=log)
    with open(f"{args.out}/{tag}.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"[{tag}] done in {el:.1f}s ({summary['ms_per_step']} ms/step)")


if __name__ == "__main__":
    main()
