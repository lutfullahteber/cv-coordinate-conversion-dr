#!/usr/bin/env python3
"""
coordinate.py - convert source (.ply + traj.txt) into the Unity viewer frame.

Viewer evaluates `world = P * p`. Source is OpenCV-style (RH, +Y down, +Z forward)
and the viewer is Unity (LH, +Y up). One change-of-basis M (source -> viewer) is
applied to both files so the invariant `P_v * p_v = M * (P * p)` holds:

    p_v = M . B . p              (baked into each .ply)
    P_v = M . P . M^-1           (baked into traj.txt — similarity transform)

Initial hypothesis (first attempt):
    M = diag(1, -1, 1)    (flipY)   OpenCV -> Unity: up-axis + handedness in one
                                    matrix (det -1 = single reflection = RH -> LH).
    B = diag(1, 1, -1)    (flipZ)   .ply camera frame (+Z forward, OpenCV) ->
                                    pose camera frame (-Z forward, OpenGL-style).
    M.B = diag(1, -1, -1)           net transform applied to ply points.

Pipeline:
    python coordinate.py --convert
"""
import argparse
import os

import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
POINTS_IN = os.path.join(HERE, "data", "Points")
TRAJ_IN = os.path.join(HERE, "data", "traj.txt")
OUT = os.path.join(HERE, "output")
POINTS_OUT = os.path.join(OUT, "Points")
TRAJ_OUT = os.path.join(OUT, "traj.txt")
IMAGES = ["image1.ply", "image2.ply", "image3.ply"]

M_DEFAULT = np.diag([1.0, -1.0, 1.0])   # flipY: OpenCV -> Unity (up-axis + handedness)
B_DEFAULT = np.diag([1.0, 1.0, -1.0])   # flipZ: ply-cam (+Z fwd) -> pose-cam (-Z fwd)


def to4(R3, t=None):
    T = np.eye(4)
    T[:3, :3] = R3
    if t is not None:
        T[:3, 3] = t
    return T


def load_traj(path):
    return [np.array([float(v) for v in line.split()]).reshape(4, 4)
            for line in open(path) if line.strip()]


def write_traj(path, poses):
    with open(path, "w") as f:
        for P in poses:
            f.write(" ".join("%.18e" % v for v in P.reshape(-1)) + "\n")


def write_ply(path, pts, colors_u8):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n")
        for (x, y, z), (r, g, b) in zip(pts, colors_u8):
            f.write(f"{x:.8g} {y:.8g} {z:.8g} {int(r)} {int(g)} {int(b)}\n")


def convert():
    """Bake M into both .ply (p_v = M.B.p) and traj.txt (P_v = M.P.M^-1)."""
    M, B = M_DEFAULT, B_DEFAULT
    L3 = M @ B
    M4 = to4(M)
    M4inv = M4.T                                # orthogonal: inv = transpose

    os.makedirs(POINTS_OUT, exist_ok=True)
    print(f"convert: M=diag(1,-1,1) flipY  B=diag(1,1,-1) flipZ  "
          f"(det M = {int(round(np.linalg.det(M)))}, det M.B = {int(round(np.linalg.det(L3)))})")

    for name in IMAGES:
        pcd = o3d.io.read_point_cloud(os.path.join(POINTS_IN, name))
        pts = np.asarray(pcd.points) @ L3.T     # row form of (M.B).p
        cols = (np.asarray(pcd.colors) * 255.0).round().astype(np.uint8) \
            if pcd.has_colors() else np.zeros((len(pts), 3), np.uint8)
        write_ply(os.path.join(POINTS_OUT, name), pts, cols)
        print(f"  {name}: {len(pts)} pts")

    poses = [M4 @ P @ M4inv for P in load_traj(TRAJ_IN)]
    write_traj(TRAJ_OUT, poses)
    print(f"  traj.txt: {len(poses)} poses")
    print(f"output written to {OUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--convert", action="store_true",
                    help="bake M into .ply (p_v = M.B.p), conjugate M into poses (P_v = M.P.M^-1)")
    args = ap.parse_args()

    if args.convert:
        convert()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
