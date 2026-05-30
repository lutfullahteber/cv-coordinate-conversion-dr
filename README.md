### May 30, 2026 - 3:00 PM

## Coordinate System Conversion — Computer Vision Assignment

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

image1: 34.92°
image3: 39.30°
These angles = the panorama capture yaw (how much the camera rotated between shots).

Tested in .exe and in:

```
python test-viewer.py --from output
```