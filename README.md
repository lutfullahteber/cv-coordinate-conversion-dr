### May 30, 2026 - 3:00 PM

## Coordinate System Conversion — Computer Vision Assignment

- To see the result with your own test, create `data/` folder, add `Points/` folder and `traj.txt` file and run `one-click.py`.


# Approach

While solving this problem, I first worked out the mathematical logic on paper; coordinate conventions, handedness (RH/LH), the change-of-basis matrix M, and etc. Once the math made sense, I started the Python implementation.

I'd like to inform you this is my second attempt. Thought I understood the assignment but turned out everything messed up.

While writing the Python code, I used Claude Code (Opus 4.7) as a sparring partner for the math derivations and to debug a few non-obvious failures (the world-Y vs own-Y tilt artefact in particular).

As is says, I only touch `.ply` files while converting from OepnCv to Unity viewer. Otherwise I always configurate `traj.txt` file!

- coordinate.py
    - --convert
        - Converting coordinations from OpenCV to Unity needs an up-axis + handedness fix. 
            - It was Y/Z. 
            - Viewer refected it. 
            - Tried Z flip alone. 
            - Closer, but the scene faced away. Added X on top: `M = M_X · M_Z = diag(-1, 1, -1)`, `det = +1 (proper 180° rotation about Y)`. 
            - Worked.  
    - --flatten
        - The three camera X axes were ~35°/40° off (panorama capture yaw). 
            - Baked the rotation into the `traj.txt` and rotated around it's X axis.
    - --yaw 
        - Needed to spread `image1` left and `image3` right. 
            - First attempt: world-Y rotation at image2's position. 
            - Bug: cameras aren't upright `(local Y ≠ world Y)`, so world-Y tilted clouds vertically. 
            - Fixed by rotating about each camera's own Y at its own position
    - --shift 
        - Last gap: cameras were captured from nearly the same spot, but the panorama needs lateral spread. 

Each step writes only what it owns (poses for `--flatten`/`--yaw`/`--shift`, both for `--convert`), so each can be re-run cleanly from --convert.


## Steps

### Initial - inspect the input

- I unzipped the project and first ran the `.exe` to understand what we were working with. 
- Then I opened `image1.ply`, `image2.ply`, `image3.ply` and `traj.txt` in VSCode to inspect the format, the layout of the coordinate planes, and the colors.

## Result!

Thanks to the `_rays.png` files, I saw that the `.ply` clouds are in the OpenCV coordinate system. To verify, I compared with `traj.txt`.

`traj.txt` is a matrix file. 3 rows, each one image. Each row is a 4x4 matrix (bottom row `(0,0,0,1)` is the standard homogeneous filler). Image1 example:

```
    R =
    [  0.62356   0.77515  -0.10151 ]      t = [  3.96954 ]
    [  0.78171  -0.61988   0.06838 ]          [ -3.21007 ]
    [ -0.00992  -0.12199  -0.99248 ]          [ -1.44542 ]

    R: rotation (camera axes in world)
    t: camera position in world

    Camera positions read directly from t:
      Image1 -> ( 3.97, -3.21, -1.45)
      Image2 -> ( 4.52, -3.15, -1.40)
      Image3 -> ( 4.71, -3.12, -1.40)

```

- `.ply` is OpenCV (`+Z forward, +Y down, RH`), 
- `traj.txt` is OpenGL-style (`+Y up, -Z forward`, RH), 
- Unity is LH. 

### Initial guess for the solution

I wrote the `coordinate.py` python code to start converting from right-handed to left-handed.

Added a mode named `--convert`. 

Initial coordinate.py:

```

M = diag(1, -1, 1)    (flipY)
B = diag(1, 1, -1)    (flipZ)
M·B = diag(1, -1, -1)  applied to ply

```
Reasoning: OpenCV → Unity assumed up-axis + handedness flip.

### It was wrong but I solved it!

Check the original file and converted file and noticed I must drop Y and flip Z and X. 

```
M_X · M_Z = diag(-1, 1, -1)    (180° about Y axis)
det = +1                        (two reflections cancel)
```

- And this worked! Source and Unity share handedness; only X+Z directions differ; Y left alone because flipping it inverted the scene.

## Alignment

As I see in the original file, `image1.ply`, `image2.ply` and `image3.ply` files are mixed and rotated around **their camera's coordination.

Aligning image1 and image3's X axes with image2's X. 

Rodrigues rotation_between(a, b) formula, minimal SO(3) rotation:

```
a = R_i[:, 0]               (current X)
b = R_anchor[:, 0]          (target X)
R_align = Rodrigues(a → b)
R_i' = R_align · R_i        (rotation block only)
t_i' = t_i                  (translation preserved)
```

Worked:

```
image1: 34.92°
image3: 39.30°
These angles = the panorama capture yaw (how much the camera rotated between shots).
```

Tested in .exe and in:

```
python test-viewer.py --from output
```

## Rotate around the y-axis to bring them into a straight line.

Rotating image1 and image3 about Y, image1 left, image3 right. image2 fixed.

```
world Y axis, pivot = image2's position, full SE(3) yaw (R and t rotated together).

Auto angles: image1 = -99°, image3 = +85°.
```

### Tested with test-viewer.py and it's wrong.

Noticed we do rotation around `World Y` which is not correct. It must be around own Y rotation!

### Solved : Rotation own Y Axis works

Everything works so far. 

### X axis translation

Next step: push image1 to image2's `-X` side, image3 to `+X` side for a clean fit.

Translation:
```
t_i' = t_i + delta · R_anchor[:, 0]
R_i' = R_i                   (rotation untouched)
```

Auto delta (edge alignment):

- image1: -2.397 (right edge → image2's left edge)
- image3: +1.556 (left edge → image2's right edge)

This numbers turned out wrong. I manually found the perfect distance.
- Image1 : -1.0
- Image3 : +0.5

## TESTED! IT WORKS

Tested in both `.exe` and:

```
python test-viever.py --from output
```

## Requirements

```
pip install numpy open3d
```

## Folder Structure

```
delta-reality-assignment/
├── coordinate.py            # main tool: --convert / --flatten / --yaw / --shift
├── test-viewer.py           # local Open3D stand-in for the Unity viewer
├── README.md                # step-by-step work log
├── data/                    # INPUT (source data) 
│   ├── traj.txt             # camera poses (4x4 matrices, one per image)
│   └── Points/
│       ├── image1-3.ply     # source point clouds (OpenCV frame)
│       ├── image1-3.png     # RGB images
│       ├── image1-3_depth.png
│       └── image1-3_rays.png  # ray dirs (used to confirm OpenCV frame)
└── output/                  # OUTPUT (generated by coordinate.py)
    ├── traj.txt             # corrected/identity poses
    └── Points/
        └── image1-3.ply     # corrected clouds (Unity frame)
```
