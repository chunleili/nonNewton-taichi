"""Physical invariants for C++ compatibility fixes; run with Python unittest."""
import unittest
import numpy as np
import taichi as ti
from constraint_solver import ConstraintSolver, default_material


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        ti.reset()
        ti.init(arch=ti.cpu, offline_cache=False)

    def solver(self, p, **kw):
        return ConstraintSolver(np.array(p), np.zeros(len(p), np.int32),
                                [default_material(mu0=20, decay_mu=0.1)],
                                0.05, (4, 4, 4), 0.001, box_walls=False,
                                cpp_compat=True, particle_mass=0.1,
                                rest_density=1000, mu_scale=1000, **kw)

    def test_heat_box_uses_scene_coordinates_and_all_axes(self):
        p = np.array([[0, 0.025, 0], [0, 0.065, 0],
                      [0.99, 0.025, 0], [1.01, 0.025, 0]])
        off = np.array([1, 1, 1])
        s = self.solver(p + off, cpp_offset=off,
                        heat_box=((-1, 0, -1), (1, 0.05, 1)), heat_R=0.1)
        s.compute_density(s.x)
        s.diffuse()
        np.testing.assert_allclose(s.T.to_numpy(), [0.0001, 0, 0.0001, 0], atol=1e-8)
        self.assertEqual(s.mass, 0.1)
        self.assertEqual(s.rho0, 1000)

    def test_diffusion_conserves_total_temperature_without_sources(self):
        s = self.solver([[1, 1, 1], [1.05, 1, 1], [1.09, 1, 1]], diffusion=30)
        s.compute_density(s.x)
        s.T.from_numpy(np.array([2, 3, 7], np.float32))
        s.diffuse()
        self.assertAlmostEqual(float(s.T.to_numpy().sum()), 12, places=5)
        self.assertGreater(float(s.T[0]), 2)
        self.assertLess(float(s.T[2]), 7)

    def test_boundary_volume_normalizes_kernel_density(self):
        b = np.array([[1, 1, 1], [1.05, 1, 1], [1.10, 1, 1]])
        s = self.solver([[2, 2, 2]], boundary=b, boundary_akinci=True)
        mass = s.pmass.to_numpy()[1:]
        for i in range(len(b)):
            kernel_sum = sum(s._W_np(np.linalg.norm(b[i] - x)) for x in b)
            self.assertAlmostEqual(float(mass[i] * kernel_sum), 1000, delta=0.005)

    def test_floor_friction_and_exported_viscosity_match_temperature(self):
        off = np.array([1, 1, 1])
        s = self.solver([[1, 1.005, 1], [1, 1.02, 1]], cpp_offset=off)
        s.v.from_numpy(np.array([[1, 0, 2], [1, 0, 2]], np.float32))
        s.T.from_numpy(np.array([10, 10], np.float32))
        s.thermal_compat_state()
        np.testing.assert_allclose(s.v.to_numpy(), [[0.1, 0, 0.2], [1, 0, 2]], atol=1e-6)
        np.testing.assert_allclose(s.mu.to_numpy(), 20000 * np.exp(-1), rtol=1e-6)

    def test_unbounded_hash_preserves_neighbors_and_small_motion(self):
        p = np.array([[1000, -1000, 1000], [1000.05, -1000, 1000]])
        s = self.solver(p, clip_domain=False, iters=0, use_div=False)
        for _ in range(10):
            s.step()
        np.testing.assert_allclose(s.x.to_numpy()[:, 1] - p[:, 1], -9.81 * 0.001**2 * 55, atol=1e-9)
        np.testing.assert_array_equal(s.nb_cnt.to_numpy(), [1, 1])
        np.testing.assert_array_equal(np.array(s.neighbor_diagnostics()), [0, 0, 0])

    def test_global_viscosity_is_spd_matches_dense_solve_and_conserves_momentum(self):
        s = self.solver([[1, 1, 1], [1.05, 1, 1], [1.025, 1.04, 1]], visc_cg=True)
        s.mu.from_numpy(np.array([5000, 10000, 3000], np.float32))
        A = np.empty((9, 9))
        for col in range(9):
            e = np.zeros((3, 3), np.float32)
            e.flat[col] = 1
            s.cg_p.from_numpy(e)
            s.viscosity_operator(s.cg_p, s.cg_Ap)
            A[:, col] = s.cg_Ap.to_numpy().ravel()
        np.testing.assert_allclose(A, A.T, atol=2e-5)
        self.assertGreater(np.linalg.eigvalsh(A).min(), 0.999)
        rhs = np.array([[1, 0, 0], [0, 2, 0], [-1, -2, 1]], np.float32)
        s.v.from_numpy(rhs)
        s.solve_viscosity_cg(tol=1e-6)
        answer = s.v.to_numpy()
        np.testing.assert_allclose(answer.ravel(), np.linalg.solve(A, rhs.ravel()), atol=2e-5)
        np.testing.assert_allclose(answer.sum(0), rhs.sum(0), atol=2e-5)
        self.assertLess(np.square(answer).sum(), np.square(rhs).sum())

    def test_cpp_viscosity_matches_reference_laplacian_dense_solve(self):
        p = np.array([[1, 1, 1], [1.05, 1, 1], [1.025, 1.04, 1]])
        s = self.solver(p, visc_cg=True, cpp_visc_cg=True)
        mu = np.array([5000, 10000, 3000], np.float32)
        s.mu.from_numpy(mu)
        s.rho.from_numpy(np.full(3, 1000, np.float32))
        A = np.eye(9)
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                r = p[i] - p[j]
                rn = np.linalg.norm(r)
                eps = 1e-7
                grad = (s._W_np(rn + eps) - s._W_np(rn - eps)) / (2 * eps) * r / rn
                block = -10 * s.dt * s.mass * mu[j] / 1000**2 * np.outer(grad, r) / (rn**2 + 0.01 * s.h**2)
                A[3*i:3*i+3, 3*i:3*i+3] += block
                A[3*i:3*i+3, 3*j:3*j+3] -= block
        scale = np.repeat(np.sqrt(mu), 3)
        sym = scale[:, None] * A / scale[None, :]
        np.testing.assert_allclose(sym, sym.T, atol=1e-6)
        self.assertGreater(np.linalg.eigvalsh(sym).min(), 0.999)
        rhs = np.array([[1, 0, 0], [0, 2, 0], [-1, -2, 1]], np.float32)
        s.v.from_numpy(rhs)
        s.solve_viscosity_cg(tol=1e-6)
        np.testing.assert_allclose(s.v.to_numpy().ravel(), np.linalg.solve(A, rhs.ravel()), atol=3e-5)

    def test_casson_static_boundary_matches_independent_dense_matrix(self):
        p = np.array([[1, 1, 1], [1.05, 1, 1]])
        b = np.array([[1, 0.96, 1], [1.05, 0.96, 1]])
        s = self.solver(p, boundary=b, boundary_akinci=True,
                        visc_cg=True, cpp_visc_cg=True)
        mu = np.array([5000, 10000, 0, 0], np.float32)
        rho = np.array([700, 800, 0, 0], np.float32)
        s.mu.from_numpy(mu)
        s.rho.from_numpy(rho)
        A = np.eye(6)
        for i in range(2):
            for j, xj in enumerate(np.concatenate([p, b])):
                if i == j:
                    continue
                r = p[i] - xj
                rn = np.linalg.norm(r)
                grad = (s._W_np(rn + 1e-7) - s._W_np(rn - 1e-7)) / 2e-7 * r / rn
                coeff = (s.mass * mu[j] / (rho[i] * rho[j]) if j < 2 else
                         0.1 * s.rho0 * s.pmass[j] / rho[i]**2)
                block = -10 * s.dt * coeff * np.outer(grad, r) / (rn**2 + 0.01*s.h**2)
                A[3*i:3*i+3, 3*i:3*i+3] += block
                if j < 2:
                    A[3*i:3*i+3, 3*j:3*j+3] -= block
        rhs = np.zeros((4, 3), np.float32)
        rhs[:2] = [[1, -1, 0], [-1, -1, 1]]
        s.v.from_numpy(rhs)
        s.solve_viscosity_cg(tol=1e-6)
        np.testing.assert_allclose(s.v.to_numpy()[:2].ravel(), np.linalg.solve(A, rhs[:2].ravel()), atol=3e-5)
        s.cpp_viscosity_operator(s.cg_x, s.cg_Ap)
        np.testing.assert_allclose(s.cg_Ap.to_numpy()[:2], s.cg_rhs.to_numpy()[:2], atol=0.003)

    def test_dfsph_pressure_iteration_matches_constraint_jacobian(self):
        p = np.array([[1, 1, 1], [1.05, 1, 1], [1.025, 1.04, 1]])
        b = np.array([[1.02, 0.97, 1]])
        s = self.solver(p, boundary=b, boundary_akinci=True,
                        visc_cg=True, cpp_visc_cg=True, cpp_dfsph=True)
        v = np.array([[1, -1, 0], [-1, -2, 0], [0, -1, 1], [0, 0, 0]], np.float32)
        s.v.from_numpy(v)
        s.rho.from_numpy(np.array([1100, 1050, 1150, 0], np.float32))
        # Independently build J for C_i=rho_i/rho0-1; boundaries are fixed.
        J = np.zeros((3, 9))
        for i in range(3):
            for j, xj in enumerate(np.concatenate([p, b])):
                if i == j:
                    continue
                r = p[i] - xj
                rn = np.linalg.norm(r)
                grad = (s._W_np(rn+1e-7)-s._W_np(rn-1e-7))/2e-7 * r/rn
                volume = s.pmass[j] / s.rho0
                J[i, 3*i:3*i+3] += volume * grad
                if j < 3:
                    J[i, 3*j:3*j+3] -= volume * grad
        rhs = np.maximum(np.array([0.1, 0.05, 0.15]) + s.dt * J @ v[:3].ravel(), 0)
        multipliers = -rhs / np.diag(J @ J.T) / s.dt**2
        expected = v[:3].ravel() + s.dt * J.T @ multipliers
        s.dfsph_factor()
        s.dfsph_rhs(True)
        np.testing.assert_allclose(s.df_adv.to_numpy()[:3], rhs, rtol=1e-5)
        s.dfsph_multiplier(True)
        s.dfsph_apply()
        np.testing.assert_allclose(s.v.to_numpy()[:3].ravel(), expected, rtol=2e-5, atol=2e-5)

    def test_dfsph_deficient_neighbors_preserve_rigid_freefall(self):
        p = np.array([[1, 1, 1], [1.05, 1, 1]])
        s = self.solver(p, visc_cg=True, cpp_visc_cg=True,
                        cpp_dfsph=True, clip_domain=False)
        # C++ disables divergence for fewer than 20 total neighbors.
        s.v.from_numpy(np.array([[1, 0, 0], [-1, 0, 0]], np.float32))
        s.dfsph_rhs(False)
        np.testing.assert_array_equal(s.df_adv.to_numpy(), 0)
        s.v.fill(0)
        for _ in range(5):
            s.step()
        np.testing.assert_allclose(s.x.to_numpy()[:, 1] - p[:, 1], -9.81 * s.dt**2 * 15, atol=1e-9)
        np.testing.assert_allclose(s.v.to_numpy()[:, 1], -9.81 * s.dt * 5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
