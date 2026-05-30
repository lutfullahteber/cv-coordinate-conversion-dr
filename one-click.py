#!/usr/bin/env python3
"""
one-click.py - run the full coordinate.py pipeline in one shot.

Reads data/ (raw source) -> writes output/ (final corrected state for the
Unity viewer). Equivalent to running these four commands in order:

    python coordinate.py --convert
    python coordinate.py --flatten
    python coordinate.py --yaw          # defaults: yaw1=+35, yaw3=-39
    python coordinate.py --shift        # defaults: shift1=-1.0, shift3=+0.5

Run:
    python one-click.py
"""
from coordinate import (
    convert,
    flatten,
    yaw_about_image2,
    shift_along_image2_X,
)

# CLI-default values (match argparse defaults in coordinate.py main())
ANCHOR_IDX = 1          # 0-based index of image2 (anchor)
YAW1 = 35.0             # deg, panorama capture yaw for image1
YAW3 = -39.0            # deg, panorama capture yaw for image3
SHIFT1 = -1.0            # units along anchor's X axis, image1
SHIFT3 = 0.5            # units along anchor's X axis, image3


def banner(step, total, name):
    print(f"\n=== [{step}/{total}] {name} ===")


def main():
    print("=== one-click pipeline: data/ -> output/ ===")

    banner(1, 4, "convert  (M=diag(-1,1,-1), bake into ply + conjugate into poses)")
    convert()

    banner(2, 4, "flatten  (align camera X axes to anchor; traj.txt only)")
    flatten(anchor_idx=ANCHOR_IDX)

    banner(3, 4, f"yaw      (own Y at own t; yaw1={YAW1:+}, yaw3={YAW3:+})")
    yaw_about_image2(yaw1=YAW1, yaw3=YAW3, anchor_idx=ANCHOR_IDX)

    banner(4, 4, f"shift    (along anchor X; shift1={SHIFT1:+}, shift3={SHIFT3:+})")
    shift_along_image2_X(shift1=SHIFT1, shift3=SHIFT3, anchor_idx=ANCHOR_IDX)

    print("\n=== done. output/ is ready for ComputerVisionAssignment.exe ===")


if __name__ == "__main__":
    main()
