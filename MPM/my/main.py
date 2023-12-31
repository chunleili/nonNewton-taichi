import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
from time import time
import os
import argparse
import taichi as ti
import logging


ti.init(arch=ti.cpu)

from global_constants import *
from InitialParticle import InitialParticle
from FEMmatrix3D import FEMmatrix3D
from natcoords import natcoords
from P2G import P2G
from Hstar import Hstar
from massA2 import massA2
from full_scale_operators import full_scale_operators
from Matrix_reduction import Matrix_reduction
from SolveAssemble import SolveAssemble
from SolveEuler import SolveEuler
from maptopointsPC import maptopointsPC


def main():
    # problem inputs
    ## ====plotting and saving====
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_results", type=int, default=1)
    parser.add_argument("--enable_plot", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--record_time", type=int, default=1)
    parser.add_argument("--record_detail_time", type=int, default=0)
    parser.add_argument("--save_Tp", type=int, default=0)
    parser.add_argument("--save_vp", type=int, default=0)
    parser.add_argument("--save_npy", type=int, default=0)
    parser.add_argument("--save_ply", type=int, default=1)
    args = parser.parse_args()
    if args.save_results or args.record_time:
        if not os.path.exists("results"):
            os.makedirs("results")
    program_start_time = time()
    step_time_list = []
    # RK4_time_list = []
    # P2G_time_list = []
    # massA2_time_list = []
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    VBC = np.zeros((Ng, 3))

    icon, NBCv, G10, Mt0, Gx, Gy, Gz, M, xp, vp, pp, Tp = initialize(dx)

    Np = xp.shape[0]
    vp[:, 2] = v0

    step_num = 0
    loop_start_time = time()
    print(f"Initialization done.\nInitialzation time used {loop_start_time-program_start_time:.2f}s")
    while step_num < args.num_steps:
        step_num, xp, vp, pp, Tp = step(
            args,
            xp,
            vp,
            pp,
            Tp,
            icon,
            Np,
            NBCv,
            VBC,
            G10,
            Mt0,
            Gx,
            Gy,
            Gz,
            M,
            step_time_list,
            step_num,
        )

    program_end_time = time()
    if args.record_time:
        np.savetxt("results/step_time.txt", np.array(step_time_list))
    avg_step = np.array(step_time_list).mean()
    print("Initialization time used: ", loop_start_time - program_start_time)
    print("loop time used: ", program_end_time - loop_start_time)
    print("total time used: ", program_end_time - program_start_time)
    print("average step time used: ", avg_step)

    # if args.record_time and args.record_detail_time:
    #     avg_P2g = np.array(P2G_time_list).mean()
    #     avg_RK4 = np.array(RK4_time_list).mean()
    #     avg_massA2 = np.array(massA2_time_list).mean()
    #     np.savetxt("results/P2G_time.txt", np.array(P2G_time_list))
    #     np.savetxt("results/RK4_time.txt", np.array(RK4_time_list))
    #     np.savetxt("results/massA2_time.txt", np.array(massA2_time_list))
    #     print("average P2G time used: ", avg_P2g)
    #     print("average RK4 time used: ", avg_RK4)
    #     print("average massA2 time used: ", avg_massA2)


def plot_step(xp, plot=True, step_num=0):
    if not plot:
        return
    plt.ion()
    ax = plt.axes(projection="3d")
    ax.scatter(xp[:, 0], xp[:, 1], xp[:, 2], c="b")
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.2, 0.8)
    ax.set_zlim(0, 0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([2, 2, 1])
    ax.text2D(0.05, 0.95, f"frame: {step_num}", transform=ax.transAxes)
    plt.show()
    plt.pause(0.01)
    plt.clf()


def plot_init(Xg, icon, Ne, plot=True):
    if not plot:
        return
    plt.figure(num="Initial mesh grid and mp")
    ax = plt.axes(projection="3d")
    for j in range(Ne):
        id = icon[j, :]
        id = np.append(id, id[0])
        ax.scatter(Xg[id - 1, 0], Xg[id - 1, 1], Xg[id - 1, 2], c="r", s=1)
    plt.show()


def initialize(dx):
    ## ======Ordering: (1) x y loop (2) z loop
    xe = np.arange(xa, xb + Lx, Lx).T  # xe: x position sampling
    ye = np.arange(ya, yb + Ly, Ly).T  # ye: y position sampling
    ze = np.arange(za, zb + Lz, Lz).T  # ze: z position sampling
    Ix = np.ones(nx + 1)
    Iy = np.ones(ny + 1)
    Iz = np.ones(nz + 1)
    xg = np.kron(Iz, np.kron(Iy, xe))
    # xg: all nodes x position
    yg = np.kron(Iz, np.kron(ye, Ix))
    # yg: all nodes x position
    zg = np.kron(ze, np.kron(Iy, Ix))
    # yg: all nodes x position
    ## ==================

    Xg = np.vstack((xg, yg, zg)).T  # all nodes position
    # Vg = np.zeros((Ng, 3))
    # velocity on the grids- vector [Vx; Vy; Vz];
    # BC value
    ## connectivity matrix
    icon = np.zeros((Ne, 8), dtype=int)
    # ====first four nodes label on x-y layer======
    for k in range(0, nz):
        for j in range(0, ny):
            for i in range(0, nx):
                e = (k) * ny * nx + (j) * nx + i
                icon[e, 0] = (k) * (nx + 1) * (ny + 1) + (j) * (nx + 1) + i + 1
                icon[e, 1] = icon[e, 0] + 1
                icon[e, 2] = icon[e, 1] + nx + 1
                icon[e, 3] = icon[e, 2] - 1
    # =====adding one x-y face total nodes to get other four nodes
    icon[:, 4:8] = icon[:, 0:4] + (nx + 1) * (ny + 1)

    # iniitalize the material point
    nep = np.array([8, 8, 8])
    [Np, xp, vp] = InitialParticle(nep, dx, icon, Xg, Ne)
    Tp = np.zeros((Np, 6))
    pp = np.zeros((Np))
    ## =========plot initial================
    plot_init(Xg, icon, Ne, plot=False)
    ## === boundary setup======================
    # ntop = np.where(Xg[:, 2] >= 1)[0]
    nbot = np.where(Xg[:, 2] <= 0)[0]
    # nright = np.where(Xg[:, 0] >= 1)[0]
    # nleft = np.where(Xg[:, 0] <= 0)[0]
    # nfront = np.where(Xg[:, 1] <= 0)[0]
    # nback = np.where(Xg[:, 1] >= 1)[0]

    NBCv = nbot

    ## =============Assemble Matrix============
    [M, K, Gx, Gy, Gz] = FEMmatrix3D(Ne, Ng, icon, Lx, Ly, Lz)

    # zer = np.zeros(shape=Gx.shape)
    zer = scipy.sparse.csr_matrix(Gx.shape)
    # Gradient of Velocity
    G10 = scipy.sparse.bmat(
        [
            [Gx, zer, zer],
            [zer, Gy, zer],
            [zer, zer, Gz],
            [0.5 * Gy, 0.5 * Gx, zer],
            [0.5 * Gz, zer, 0.5 * Gx],
            [zer, 0.5 * Gz, 0.5 * Gy],
        ]
    )
    # # Gradient of P
    G20 = scipy.sparse.bmat([[Gx, Gy, Gz]])
    # # Divergence of stress
    G30 = scipy.sparse.bmat([[Gx, zer, zer, Gy, Gz, zer], [zer, Gy, zer, Gz, zer, Gx], [zer, zer, Gz, zer, Gx, Gy]])

    # mass matrix
    Mu0 = scipy.sparse.block_diag([M, M, M])
    Mt0 = scipy.sparse.block_diag([M, M, M, M, M, M])
    Mu = np.sum(Mu0, axis=1)
    Mu = np.squeeze(np.asarray(Mu))
    inv_Mu = 1.0 / Mu[:]
    dMu = scipy.sparse.diags(inv_Mu)
    Mt0 = scipy.sparse.diags(np.squeeze(np.asarray(np.sum(Mt0, axis=1))))

    return icon, NBCv, G10, Mt0, Gx, Gy, Gz, M, xp, vp, pp, Tp


def step(args, xp, vp, pp, Tp, icon, Np, NBCv, VBC, G10, Mt0, Gx, Gy, Gz, M, step_time_list, step_num):
    if args.record_time:
        step_start_time = time()
    # -------------------------------- Map to grid ------------------------------- #
    [xpn, nep] = natcoords(xp, dx, xmin, nexyz)
    if args.record_time and args.record_detail_time:
        logging.info(f"natcoords: {time()-step_start_time}")
        last_time = time()

    # ---------------------------------- P2G ---------------------------------- #
    [mv, vnew, Tnew, p, nn] = P2G(Ng, icon, xpn, nep, Np, vp, pp, Tp)
    if args.record_time and args.record_detail_time:
        logging.info(f"P2G: {time()-last_time}")
        # P2G_time_list.append(time() - last_time)
        last_time = time()

    for i in range(len(NBCv)):
        boundary_index = NBCv[i].item()
        if vnew[boundary_index][2] < 0:
            vnew[boundary_index] = VBC[boundary_index]
    v = vnew
    T = Tnew

    # ---------------------------------- Hstar ---------------------------------- #
    [Hg, f] = Hstar(T, alph, gamma, Q0, xi, We, Ng, G10, vnew, Mt0)
    if args.record_time and args.record_detail_time:
        logging.info(f"Hstar: {time()-last_time}")
        last_time = time()

    # ---------------------------------- massA2 ---------------------------------- #
    Mf = massA2(dx, Ng, Ne, icon, f, We, dt)
    if args.record_time and args.record_detail_time:
        logging.info(f"massA2: {time()-last_time}")
        # massA2_time_list.append(time() - last_time)
        last_time = time()

    # ----------------------------- Matrix_reduction ----------------------------- #
    NBC = np.where(nn <= 0)[0]  # NBC: non particle nodes, NBS: free surface nodes
    Phi = scipy.sparse.eye(Ng)
    keep_columns = np.ones(Phi.shape[1], dtype=bool)
    keep_columns[NBC] = False
    Phi_subset = Phi.tocsr()
    Phi_subset = Phi_subset[:, keep_columns]
    Phi = Phi_subset
    dxm = Phi.T @ Gx @ Phi
    dym = Phi.T @ Gy @ Phi
    dzm = Phi.T @ Gz @ Phi
    nk = dxm.shape[0]
    Ck = Phi.T @ M @ Phi
    zk = np.zeros((nk, nk))
    [G1, D, G, Mu, Md, Mt, dMu, A22d] = full_scale_operators(dxm, dym, dzm, Ck, Mf, Phi, Re, beta, zk)
    [Nk, vk, Hk, Tk, pk, znk] = Matrix_reduction(nn, nk, Ng, Phi, vnew, Hg, Tnew, p)
    if args.record_time and args.record_detail_time:
        logging.info(f"Matrix_reduction: {time()-last_time}")
        last_time = time()

    ## ==========Free Surface================
    Gnx = dxm @ Nk
    Gny = dym @ Nk
    Gnz = dzm @ Nk
    Gn = np.sqrt(Gnx * Gnx + Gny * Gny + Gnz * Gnz)
    nnx = Gnx / (Gn + 1e-8)
    nny = Gny / (Gn + 1e-8)
    nnz = Gnz / (Gn + 1e-8)
    NBS = np.where(Gn > 1e-8)[0]
    nd = np.hstack([nnx, nny, nnz])
    X10 = beta / Re * G1 @ vk
    X20 = Tk
    X30 = pk.flatten()
    if args.record_time and args.record_detail_time:
        logging.info(f"Free Surface: {time()-last_time}")
        last_time = time()

    ## ===========Solve ============
    [A11, A12, A13, A21, A22, A23, A31, A32, A33, B1, B2, B3] = SolveAssemble(
        X20, Hk, dMu, Mt, A22d, NBS, beta, Re, G1, D, G, xi, nk, vk, dt
    )
    [X1, X2, X3] = SolveEuler(
        X10, X20, X30, B1, B2, B3, A11, A12, A13, A21, A22, A23, A31, A32, A33, beta, Re, NBS, nd, nk, yita
    )
    if args.record_time and args.record_detail_time:
        logging.info(f"Solve: {time()-last_time}")
        last_time = time()

    ## ==========Results Reterive====
    zn = np.zeros((Ng, nk))
    Tg = scipy.sparse.block_diag([Phi, Phi, Phi, Phi, Phi, Phi]) @ X2
    p = Phi @ X3
    dV = dMu @ (beta / Re * D @ X1 + D @ X2 - G @ X3) * dt
    deltV = scipy.sparse.block_diag([Phi, Phi, Phi]) @ dV
    vtilda = vnew.flatten() + deltV
    vnew = vtilda.reshape((Ng, 3))
    if args.record_time and args.record_detail_time:
        logging.info(f"Results Reterive: {time()-last_time}")
        last_time = time()

    ## ==========Boundary Condition==========
    for i in range(len(NBCv)):
        boundary_index = NBCv[i]
        if vnew[boundary_index][2] < 0:
            vnew[boundary_index] = VBC[boundary_index]

    if args.record_time and args.record_detail_time:
        logging.info(f"Boundary Condition: {time()-last_time}")
        last_time = time()

    ## ===== Grids to Particles=============
    [xp, vp, Tp, pp] = maptopointsPC(Ng, Tg, xmin, nexyz, Np, xp, vp, icon, vnew, v, dx, dt, p, Fr, g)
    v = vnew
    if args.record_time and args.record_detail_time:
        logging.info(f"Grids to Particles(RK4): {time()-last_time}")
        # RK4_time_list.append(time() - last_time)
        last_time = time()

    step_num += 1

    ## ====Plot======
    if args.record_time:
        step_end_time = time()
        step_time = step_end_time - step_start_time
        step_time_list.append(step_time)
        print(f"---\nstep: {step_num}, time: {(step_time)}")
    else:
        print(f"---\nstep: {step_num}")
    if args.enable_plot:
        plot_step(xp, True, step_num)
    if args.save_results:
        # swap y z for output
        xp[:, [1, 2]] = xp[:, [2, 1]]
        if args.save_npy:
            np.save(f"results/xp_{step_num}.npy", xp)
            if args.save_Tp:
                np.save(f"results/Tp_{step_num}.npy", Tp)
            if args.save_vp:
                np.save(f"results/vp_{step_num}.npy", vp)
        elif args.save_ply:
            import plyfile

            vertices = np.zeros(xp.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
            vertices["x"] = xp[:, 0]
            vertices["y"] = xp[:, 1]
            vertices["z"] = xp[:, 2]
            el = plyfile.PlyElement.describe(vertices, "vertex")
            plyfile.PlyData([el]).write(f"results/xp_{step_num}.ply")
        else:
            np.savetxt(f"results/xp_{step_num}.txt", xp)
            if args.save_Tp:
                np.savetxt(f"results/Tp_{step_num}.txt", Tp)
            if args.save_vp:
                np.savetxt(f"results/vp_{step_num}.txt", vp)
        # swap back
        xp[:, [1, 2]] = xp[:, [2, 1]]

    return step_num, xp, vp, pp, Tp


if __name__ == "__main__":
    main()
