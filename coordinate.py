#!/usr/bin/env python3
"""
coordinate.py - convert source (.ply + traj.txt) into the Unity viewer frame.

Viewer evaluates `world = P * p`. Source is OpenCV-style (RH, +Y down, +Z forward)
and the viewer is Unity (LH, +Y up). One change-of-basis M (source -> viewer) is
applied to both files so the invariant `P_v * p_v = M * (P * p)` holds:

    p_v = M . B . p              (baked into each .ply)
    P_v = M . P . M^-1           (baked into traj.txt — similarity transform)

Working M (verified in the .exe viewer):
    M_Z = diag( 1, 1, -1)           step 1: negate Z (fix front/back)
    M_X = diag(-1, 1,  1)           step 2: negate X (180 deg yaw about Y, with Z)
    M   = M_X . M_Z = diag(-1, 1, -1)
    det M = +1                       proper rotation (two reflections cancel).

Geometric reading: source world and Unity world share the same handedness for
this dataset; only the X and Z axis directions differ. Y is left untouched
because flipping it inverts the vertical (scene renders upside down).

B = identity. No extra ply-cam fix needed.

Pipeline:
    python coordinate.py --convert
    python coordinate.py --flatten      # align camera X axes to anchor (image2)
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

M_Z = np.diag([1.0, 1.0, -1.0])         # step 1: negate Z
M_X = np.diag([-1.0, 1.0, 1.0])         # step 2: negate X (180 deg yaw about Y, together with Z)
M_DEFAULT = M_X @ M_Z                   # = diag(-1, 1, -1), Z-then-X composition
B_DEFAULT = np.eye(3)                   # no extra ply-cam fix needed


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
    print(f"convert: M=M_X.M_Z=diag(-1,1,-1)  B=identity  "
          f"(det M = {int(round(np.linalg.det(M)))})")

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


def rotation_between(a, b):
    """Minimal 3x3 rotation R such that R . (a / ||a||) = (b / ||b||) (Rodrigues).

    For unit vectors a, b:
        v = a x b           (rotation axis, magnitude = sin(theta))
        c = a . b           (= cos(theta))
        s = ||v||           (= sin(theta))
        K = skew(v)
        R = I + K + K^2 * (1 - c) / s^2

    Edge cases:
        s = 0, c > 0   ->  R = I              (already aligned)
        s = 0, c < 0   ->  180 deg flip about any axis perpendicular to a
    """
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * K @ K                     # Rodrigues at theta = pi
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def flatten(anchor_idx=1):
    """Align each non-anchor pose's camera X axis with the anchor's X axis.

    Each pose P_i = [R_i | t_i]. The columns of R_i are the camera axes in
    world coordinates: R_i[:, 0] = local X, [:, 1] = local Y, [:, 2] = local Z.
    For a panorama captured by yawing the camera, image1 / image3 X axes
    diverge from image2's X by the capture yaw angle (~30-40 deg).

    Math (per non-anchor pose i):
        a = R_i[:, 0]                   # current world-space X of camera i
        b = R_anchor[:, 0]              # target world-space X (anchor's)
        R_align = rotation_between(a, b)    # minimal SO(3) rotation a -> b
        R_i'  = R_align . R_i           # left-multiply: rotates the whole frame
        t_i'  = t_i                     # translation column kept as-is

    Verification:  R_i'[:, 0] = R_align . R_i[:, 0] = R_align . a = b   OK

    Important: this is NOT diag(R_align, 1) . P_i. That would also rotate t_i
    (since [[R,0],[0,1]] . [[R_i, t_i],[0,1]] = [[R.R_i, R.t_i],[0,1]]). The
    function composes the rotation block manually and copies t_i untouched, so
    the camera position is provably unchanged.

    Acts on traj.txt only; .ply files stay rigid. For i == anchor_idx,
    R_align = I -> pose unchanged.
    """
    if not os.path.exists(TRAJ_OUT):
        raise SystemExit("no output/traj.txt; run --convert first")
    poses = load_traj(TRAJ_OUT)
    R_anchor = poses[anchor_idx][:3, :3]
    b = R_anchor[:, 0]
    print(f"flatten: anchor=image{anchor_idx+1}  (align R_i[:,0] -> R_anchor[:,0], traj.txt only)")

    new_poses = []
    for i, P in enumerate(poses):
        if i == anchor_idx:
            new_poses.append(P.copy())
            print(f"  image{i+1}: anchor, pose unchanged")
            continue
        R_i = P[:3, :3]
        a = R_i[:, 0]
        R_align = rotation_between(a, b)
        P_new = np.eye(4)
        P_new[:3, :3] = R_align @ R_i                    # rotation block only
        P_new[:3, 3]  = P[:3, 3]                         # translation preserved
        new_poses.append(P_new)
        ang = np.degrees(np.arccos(np.clip(
            np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)), -1.0, 1.0)))
        print(f"  image{i+1}: rotated pose rotation block by {ang:.2f} deg")
    write_traj(TRAJ_OUT, new_poses)
    print(f"done. ply files untouched -> {TRAJ_OUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--convert", action="store_true",
                    help="bake M into .ply (p_v = M.B.p), conjugate M into poses (P_v = M.P.M^-1)")
    ap.add_argument("--flatten", action="store_true",
                    help="align non-anchor camera X axes with the anchor "
                         "(R_align = Rodrigues(R_i[:,0] -> R_anchor[:,0]); "
                         "R_i' = R_align . R_i, t_i unchanged; ply untouched)")
    ap.add_argument("--anchor", type=int, default=2,
                    help="1-based anchor index for --flatten (default 2)")
    args = ap.parse_args()

    if args.convert:
        convert()
    elif args.flatten:
        flatten(anchor_idx=args.anchor - 1)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
