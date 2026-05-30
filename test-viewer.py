#!/usr/bin/env python3
"""
test-viewer.py - local stand-in for the Unity viewer.

View the raw data/ clouds or the exported output/ clouds WITHOUT relaunching
the .exe.

The .exe (Unity) is left-handed; Open3D is right-handed. The display correction
depends on what is already baked into the files being viewed:
    D  = negate Z   handedness bridge (RH Open3D -> LH Unity)
    OF = negate Y   Unity up-axis orientation

  - --optimize output (orientation baked into the POSE as A=flipy, recorded as
    an "A" key in output/_chosen.json): apply only D. Display = D . Pose . local,
    i.e. exactly D times what the .exe evaluates (Pose * local).
  - raw data/ and --align output (orientation in the .ply / identity pose):
    apply OF . D. Display = OF . D . Pose . local.

If it looks right here it should look right in the .exe.

Usage:
    python test-viewer.py                  # view data/ (raw source)
    python test-viewer.py --from output    # view output/ (exported)
    python test-viewer.py --tint           # solid color per cloud (check overlap)
"""
import argparse
import json
import os

import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_POINTS = os.path.join(HERE, "data", "Points")
DATA_TRAJ = os.path.join(HERE, "data", "traj.txt")
OUT_POINTS = os.path.join(HERE, "output", "Points")
OUT_TRAJ = os.path.join(HERE, "output", "traj.txt")
OUT_CHOSEN = os.path.join(HERE, "output", "_chosen.json")
IMAGES = ["image1.ply", "image2.ply", "image3.ply"]
TINTS = [[0.90, 0.20, 0.20], [0.20, 0.80, 0.30], [0.25, 0.45, 0.95]]

# display-only corrections (not baked into output/)
D = np.diag([1.0, 1.0, -1.0, 1.0])    # RH Open3D -> LH Unity (negate Z)
OF = np.diag([1.0, -1.0, 1.0, 1.0])   # Unity up-axis (negate Y)


def load_traj(path):
    poses = []
    for line in open(path):
        line = line.strip()
        if line:
            poses.append(np.array([float(v) for v in line.split()]).reshape(4, 4))
    return poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", choices=["data", "output"], default="data",
                    help="which folder to view (default data)")
    ap.add_argument("--tint", action="store_true", help="solid color per cloud")
    args = ap.parse_args()

    pts_dir, traj_path = (OUT_POINTS, OUT_TRAJ) if args.src == "output" \
        else (DATA_POINTS, DATA_TRAJ)
    if not os.path.exists(traj_path):
        raise SystemExit(f"{args.src}/ not found"
                         + ("; run coordinate.py --optimize or --align first"
                            if args.src == "output" else ""))
    poses = load_traj(traj_path)

    # Pick the display correction by what is baked. If output/ carries an A key
    # in _chosen.json, orientation is already in the pose -> only the handedness
    # bridge D is needed; otherwise apply the full OF . D.
    orient_in_pose = False
    vfix_baked = False
    if args.src == "output" and os.path.exists(OUT_CHOSEN):
        ch = json.load(open(OUT_CHOSEN))
        # --convert/--export bake M into BOTH ply and pose (orientation in pose);
        # legacy --optimize used an "A" key. Either means only D is needed.
        orient_in_pose = "M" in ch or "A" in ch
        # if the handedness bridge (vfix) is baked into the pose, the .exe shows
        # it directly -> the preview must add NO correction to match the .exe.
        vfix_baked = ch.get("vfix", "identity") not in ("identity", "+x+y+z", None)

    if vfix_baked:
        corr = np.eye(4)
        label = "identity (vfix bridge baked in pose)"
    elif orient_in_pose:
        corr = D
        label = "D (orientation baked in pose)"
    else:
        corr = OF @ D
        label = "OF . D"
    print(f"display correction: {label}")

    geoms = []
    for i, name in enumerate(IMAGES):
        pcd = o3d.io.read_point_cloud(os.path.join(pts_dir, name))
        # T = corr . Pose   (output local -> display world)
        T = corr @ poses[i]
        pcd.transform(T)
        if args.tint:
            pcd.paint_uniform_color(TINTS[i])
        geoms.append(pcd)
        print(f"{name}: {len(pcd.points)} pts")
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
        frame.transform(T)
        geoms.append(frame)

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    title = f"test-viewer  src={args.src}  view=exe-LH"
    print("\n" + title)
    print("Controls: left-drag rotate, scroll zoom, right-drag pan.")
    o3d.visualization.draw_geometries(geoms, window_name=title)


if __name__ == "__main__":
    main()
