#!/usr/bin/env python3
"""
one-click.py - run the full coordinate.py pipeline in one shot.

Reads data/ (raw source) -> writes output/ (final corrected state for the
Unity viewer). Same four stages as before:

    1. convert    (M = diag(-1, 1, -1))
    2. flatten    (align camera X axes to the anchor)
    3. yaw        (own Y at own t)        - angles derived from data, no manual values
    4. shift      (along anchor X axis)   - deltas derived from data, no manual values

The yaw angle and the shift delta were previously hand-typed in this script
(YAW1, YAW3, SHIFT1, SHIFT3). They are now computed from the input poses and
clouds via the formulas documented in MATH.md. Swap in different .ply files
or a different traj.txt and the pipeline still produces the right values
without any code edits.

Run:
    python one-click.py
"""
import itertools
import os

import numpy as np
import open3d as o3d

from coordinate import (
    convert,
    flatten,
    yaw_about_image2,
    shift_along_image2_X,
    strip_shift_deltas,
    load_traj,
    IMAGES,
    POINTS_IN,
    TRAJ_IN,
    POINTS_OUT,
    TRAJ_OUT,
)

ANCHOR_IDX = len(IMAGES) // 2        # 0-based middle image (anchor); N=3 -> image2
PROJ_STRIDE = 100                    # downsample for shift projection
SAMPLE = 4000                        # chamfer sub-sample size
TOPK = 8                             # how many M candidates to display


# ============================================================================
# Step 1 helper: 48-permutation brute-force analysis (display only).
# The actual bake uses the hard-coded working M from coordinate.py.convert().
# ============================================================================
def signed_perms():
    """48 signed permutation 3x3 matrices, keyed by stable string ids."""
    axes = "xyz"
    out = {}
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            label = ""
            for src_axis in range(3):
                dst = perm[src_axis]
                s = signs[src_axis]
                M[dst, src_axis] = s
                label += ("+" if s > 0 else "-") + axes[dst]
            out[label] = M
    return out


PERMS = signed_perms()


def _pc(arr):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(arr)
    return pc


def chamfer(a, b, sample=SAMPLE, seed=0):
    rng = np.random.default_rng(seed)
    if len(a) > sample:
        a = a[rng.choice(len(a), sample, replace=False)]
    if len(b) > sample:
        b = b[rng.choice(len(b), sample, replace=False)]
    pa, pb = _pc(a), _pc(b)
    da = np.asarray(pa.compute_point_cloud_distance(pb)).mean()
    db = np.asarray(pb.compute_point_cloud_distance(pa)).mean()
    return 0.5 * (da + db)


def pairwise_chamfer(worlds, sample=SAMPLE):
    if len(worlds) < 2:
        return 0.0
    pairs = list(itertools.combinations(range(len(worlds)), 2))
    return float(np.mean([chamfer(worlds[i], worlds[j], sample) for i, j in pairs]))


def camera_forward_consistency(worlds, poses_v):
    """Mean |cos(v_i, R_v_i[:, 2])| over cameras; higher = better."""
    cs = []
    for w, P_v in zip(worlds, poses_v):
        c = w.mean(axis=0)
        t = P_v[:3, 3]
        v = c - t
        n = np.linalg.norm(v)
        if n < 1e-9:
            continue
        v = v / n
        z = P_v[:3, 2]
        cs.append(abs(float(np.dot(v, z))))
    return float(np.mean(cs)) if cs else 0.0


def camera_height_spread(poses_v):
    """range(t_v_i[Y]) — smaller = cameras at similar heights."""
    ys = [P[1, 3] for P in poses_v]
    return float(max(ys) - min(ys))


def show_48_perm_analysis():
    """Brute-force score the 48 signed-permutation matrices and print the top-K
    finalists. Informational only — the bake itself is handled by convert()
    using the verified working M = diag(-1, 1, -1)."""
    print("48-permutation brute force (scored from data, display only):")
    poses = load_traj(TRAJ_IN)
    clouds_local = [np.asarray(o3d.io.read_point_cloud(os.path.join(POINTS_IN, n))
                               .uniform_down_sample(PROJ_STRIDE).points)
                    for n in IMAGES]

    scored = []
    for pid, M3 in PERMS.items():
        M4 = np.eye(4)
        M4[:3, :3] = M3
        M4inv = M4.T
        worlds, poses_v = [], []
        for c_local, P in zip(clouds_local, poses):
            pts_v = c_local @ M3.T
            P_v = M4 @ P @ M4inv
            h = np.c_[pts_v, np.ones(len(pts_v))]
            worlds.append((h @ P_v.T)[:, :3])
            poses_v.append(P_v)
        cons = camera_forward_consistency(worlds, poses_v)
        hs = camera_height_spread(poses_v)
        cham = pairwise_chamfer(worlds)
        scored.append((-cons, hs, cham, pid))

    scored.sort(key=lambda t: (t[0], t[1], t[2]))

    print(f"\n  top {TOPK} candidates (higher consistency, lower height-spread = better):")
    print(f"    {'rank':>4}  {'M_id':<10}  {'consistency':>12}  {'height-spread':>14}  {'chamfer':>10}")
    for i, (neg_c, hs, ch, pid) in enumerate(scored[:TOPK]):
        print(f"    {i+1:>4}  {pid:<10}  {-neg_c:>12.4f}  {hs:>14.4f}  {ch:>10.4f}")

    top_neg_c, top_hs = scored[0][0], scored[0][1]
    finalists = [t for t in scored
                 if abs(t[0] - top_neg_c) < 1e-4 and abs(t[1] - top_hs) < 1e-4]
    if len(finalists) > 1:
        print(f"\n  NOTE: {len(finalists)} M's tie on consistency + height-spread "
              f"(intrinsic sign ambiguity).")
        print(f"    finalists: " + ", ".join(t[3] for t in finalists))
        print(f"    verified working M for this dataset: '-x+y-z' (= diag(-1, 1, -1))")


# ============================================================================
# Formula 1: signed panorama capture yaw of camera i about anchor's local Y.
# (MATH.md Step 3 / Step 4)
# ============================================================================
def signed_capture_yaw(R_i, R_anchor):
    """Return the signed yaw (degrees) of camera i about the anchor's local Y.

        a       = R_i[:, 0],   b = R_anchor[:, 0]
        axis_X  = a x b                                            # before normalisation
        angle_X = atan2(||axis_X||, a . b)                         # in degrees
        sign    = sign(axis_X . R_anchor[:, 1])                    # +Y projection
        result  = sign * angle_X

    For a panorama captured by rotating about the photographer's head/up axis,
    this returns the exact rotation angle that took camera i away from the
    anchor — what step 4 needs to rotate it back to a clean baseline.
    """
    a = R_i[:, 0] / np.linalg.norm(R_i[:, 0])
    b = R_anchor[:, 0] / np.linalg.norm(R_anchor[:, 0])
    axis_X = np.cross(a, b)
    angle_X = float(np.degrees(np.arctan2(np.linalg.norm(axis_X), float(np.dot(a, b)))))
    sign_y = 1.0 if float(np.dot(axis_X, R_anchor[:, 1])) >= 0 else -1.0
    return sign_y * angle_X


# ============================================================================
# Formula 2: edge-alignment shift delta along anchor X axis. (MATH.md Step 5)
# ============================================================================
def edge_align_delta(R_i, t_i, ply_path, R_anchor, t_anchor, anchor_ply_path,
                     side, stride=PROJ_STRIDE):
    """Return the signed shift (along anchor's X axis) that puts the edge of
    cloud i in contact with the adjacent edge of the anchor cloud.

        d        = R_anchor[:, 0]                                  # anchor X in world
        proj_i   = (world_pts_i - t_anchor) . d
        proj_a   = (world_pts_anchor - t_anchor) . d
        if side == 'left':   delta = min(proj_a) - max(proj_i)
        if side == 'right':  delta = max(proj_a) - min(proj_i)

    Pure function of cloud points and the current poses; no hand-typed numbers.
    """
    d = R_anchor[:, 0]

    pa = np.asarray(o3d.io.read_point_cloud(anchor_ply_path)
                    .uniform_down_sample(stride).points)
    wa = pa @ R_anchor.T + t_anchor
    proj_a = (wa - t_anchor) @ d
    min_a, max_a = float(proj_a.min()), float(proj_a.max())

    pi = np.asarray(o3d.io.read_point_cloud(ply_path)
                    .uniform_down_sample(stride).points)
    wi = pi @ R_i.T + t_i
    proj_i = (wi - t_anchor) @ d
    min_i, max_i = float(proj_i.min()), float(proj_i.max())

    if side == "left":
        return float(min_a - max_i)
    return float(max_a - min_i)


def banner(step, total, name):
    print(f"\n=== [{step}/{total}] {name} ===")


def main():
    print("=== one-click pipeline: data/ -> output/ (all angles/shifts formula-derived) ===")

    banner(1, 4, "convert  (M = diag(-1, 1, -1), bake into ply + conjugate into poses)")
    convert()

    # -------- Formula 1: capture yaw of every non-anchor cloud from poses ----
    poses = load_traj(TRAJ_OUT)
    R_anchor = poses[ANCHOR_IDX][:3, :3]
    yaws = {i: signed_capture_yaw(poses[i][:3, :3], R_anchor)
            for i in range(len(poses)) if i != ANCHOR_IDX}
    print("\n[formula 1]  yaw = sign(axis_X . R_anchor[:,1]) * atan2(||axis_X||, R_i[:,0] . R_anchor[:,0])"
          "\n             where axis_X = R_i[:,0] x R_anchor[:,0]")
    for i in sorted(yaws):
        print(f"             -> yaw_image{i+1} = {yaws[i]:+.4f} deg")

    banner(2, 4, "flatten  (align camera X axes to anchor; traj.txt only)")
    flatten(anchor_idx=ANCHOR_IDX)

    banner(3, 4, "yaw      (per-image capture yaw, formula-derived)")
    yaw_about_image2(yaws=yaws, anchor_idx=ANCHOR_IDX)

    # -------- Formula 2: cumulative edge-to-edge strip tiling, post-yaw ------
    poses = load_traj(TRAJ_OUT)
    shifts = strip_shift_deltas(poses, ANCHOR_IDX, POINTS_OUT)
    print("\n[formula 2]  cumulative strip tiling along anchor X = R_anchor[:,0]"
          "\n             right of anchor: delta_i = right_edge - min(proj_i);  right_edge += width_i"
          "\n             left  of anchor: delta_i = left_edge  - max(proj_i);  left_edge  -= width_i"
          "\n             where proj = (world_points - t_anchor) . R_anchor[:,0]")
    for i in sorted(shifts):
        print(f"             -> shift_image{i+1} = {shifts[i]:+.4f}")

    banner(4, 4, "shift    (cumulative edge-to-edge tiling, formula-derived)")
    shift_along_image2_X(shifts=shifts, anchor_idx=ANCHOR_IDX)

    print("\n=== done. output/ is ready for ComputerVisionAssignment.exe ===")


if __name__ == "__main__":
    main()
