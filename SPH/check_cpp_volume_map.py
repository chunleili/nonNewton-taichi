"""Validate the Windows map bridge on real near-boundary particle positions."""
import argparse
from pathlib import Path
import json
import numpy as np
from bgeo_io import read_bhclassic
from cpp_volume_map import CppVolumeMap


def main():
    ap = argparse.ArgumentParser(__doc__)
    default = Path(__file__).resolve().parents[2] / "nonNewtonCode/data/MyScenes/Cache/ramp_sb_vm_0.025_s3_1_1_r40_40_40_i0_t0.cdm"
    ap.add_argument("--map", type=Path, default=default)
    ap.add_argument("--particles", type=Path, required=True)
    ap.add_argument("--translation", nargs=3, type=float, default=[-8, 0, 0])
    args = ap.parse_args()
    volume_map = CppVolumeMap(args.map)
    try:
        p = read_bhclassic(args.particles)["position"].astype(np.float64) - np.array(args.translation)
        q = volume_map.query(p)
        near = (q[:, 0] > 0) & (q[:, 0] < 0.1) & (q[:, 1] > 0)
        if near.sum() < 20:
            raise ValueError("Need at least 20 near-boundary points; use a frame after contact")
        x, eps = p[near][:20], 1e-6
        fd = np.stack([(volume_map.query(x + np.eye(3)[a]*eps)[:, 0] -
                        volume_map.query(x - np.eye(3)[a]*eps)[:, 0]) / (2*eps) for a in range(3)], 1)
        np.testing.assert_allclose(fd, volume_map.query(x)[:, 2:], rtol=1e-5, atol=1e-6)
        print(json.dumps(dict(particles=len(p), near=int(near.sum()), inside=int((q[:, 0] <= 0).sum()),
                              mean_volume=float(q[near, 1].mean()), gradient_queries_passed=20), indent=2))
    finally:
        volume_map.close()


if __name__ == "__main__":
    main()
