# Assets

`assets/` contains calibration scripts, ArUco helper scripts, camera SDK examples, and one-off transform-generation scripts. This folder is best understood as the experimental toolbox used to produce or inspect the runtime artifacts in `parameters/` and the virtual-object transform files expected by the top-level localization pipeline.

## Contents

| File | Purpose |
| --- | --- |
| `calibration.py` | Calibrates camera intrinsics from checkerboard images and estimates board-to-camera poses. |
| `HandEyeCalibrate.py` | Builds gripper-to-base poses from hard-coded robot measurements and runs `cv2.calibrateHandEye`. |
| `cuding2virtual1.py` | Computes a coarse-stage virtual-target transform from object, camera, gripper, and base transforms. |
| `xidingwei2virtual.py` | Computes a fine-stage virtual-target transform from object, camera, gripper, and base transforms. |
| `eyetest.py` | Sanity-check script for composing board, camera, gripper, and base transforms. |
| `OpenCV_Demo.py` | DVP industrial-camera preview and camera-parameter example. |
| `generate_aruco_tags.py` | Creates ArUco marker images. |
| `detect_aruco_images.py` | Detects ArUco markers in a single image. |
| `detect_aruco_video.py` | Detects ArUco markers from a webcam or video file. |
| `utils.py` | ArUco dictionary and display helper used by the ArUco scripts in this folder. |
| `dvp.pyd`, `DVPCamera64.dll` | Windows DVP camera SDK binaries. |
| `MvCameraControl_class.py`, `CameraParams_*.py`, `PixelType_header.py`, `MvErrorDefine_const.py` | MVS/Hikvision-style camera control bindings and constants. |

Several vendor/helper files also exist at the repository root. The ArUco scripts in this folder are usually safer to run because their `from utils import ...` import resolves to `assets/utils.py`.

## Camera Calibration

`calibration.py` expects a directory of checkerboard images. It writes camera intrinsics and board-to-camera pose arrays to the current working directory.

Example:

```bash
python assets/calibration.py --dir 1212/calibration --width 11 --height 8 --square_size 0.01 --visualize False
```

Outputs created by the script:

- `calibration_matrix_4k_1212.npy`
- `distortion_coefficients_4k_1212.npy`
- `R_board2camera_4k_1212.npy`
- `T_board2camera_4k_1212.npy`

The committed calibration files currently live in `parameters/`. If you rerun this script from the repository root, move or copy the generated files into `parameters/` only after confirming they belong to the same camera and checkerboard setup.

## Hand-Eye Calibration

`HandEyeCalibrate.py` loads:

- `R_board2camera_4k_1212.npy`
- `T_board2camera_4k_1212.npy`

It contains a hard-coded `robot_poses` list, converts those poses into `R_gripper2base` and `T_gripper2base`, then calls:

```python
cv2.calibrateHandEye(R_gripper2base, T_gripper2base, R_board2camera, T_board2camera, method=None)
```

Outputs created by the script:

- `R_gripper2base_4k_1212.npy`
- `T_gripper2base_4k_1212.npy`
- `R_camera2gripper_4k_1212.npy`
- `T_camera2gripper_4k_1212.npy`
- `M_camera2gripper_4k_1212.npy`

The script loads `.npy` files from the current working directory, not from `parameters/`. Run it from a folder containing those files or update the paths before use.

## Virtual-Target Transform Scripts

`cuding2virtual1.py` and `xidingwei2virtual.py` compose transforms of the form:

```text
object -> camera -> gripper -> base
virtual -> base
virtual -> object
```

They save:

- `M_object2base_cu_1018.npy`
- `M_virtual2object_cu_1018.npy`
- `M_object2base_xi_1018.npy`
- `M_virtual2object_xi_1018.npy`

The top-level runtime functions expect these final virtual-object matrices under:

- `1018/parameters/M_virtual2object_cu_1018.npy`
- `1018/parameters/M_virtual2object_xi_1018.npy`

Those files are not committed in the current checkout. Recreate them only with the correct robot poses and target measurements for the experiment being reproduced.

## ArUco Helper Commands

Create an output folder before generating a marker:

```bash
mkdir -p tags
python assets/generate_aruco_tags.py --id 24 --type DICT_5X5_100 --output tags --size 400
```

Detect a marker in an image:

```bash
python assets/detect_aruco_images.py --image tags/DICT_5X5_100_id_24.png --type DICT_5X5_100
```

Detect markers from a webcam:

```bash
python assets/detect_aruco_video.py --camera True --type DICT_5X5_100
```

Detect markers from a video:

```bash
python assets/detect_aruco_video.py --camera False --video test_video.mp4 --type DICT_5X5_100
```

These scripts use OpenCV's `cv2.aruco` module, so install `opencv-contrib-python`, not only `opencv-python`.

## Hardware Notes

`OpenCV_Demo.py` and the vendor camera files require Windows camera SDK support. On a machine without the DVP/MVS runtime, imports such as `from dvp import *` or `WinDLL(...)` will fail before any image-processing code runs.

Before using hardware scripts:

1. Confirm the camera appears in the vendor camera tool.
2. Confirm `dvp.pyd` and `DVPCamera64.dll` match the Python architecture.
3. Run `OpenCV_Demo.py` and verify live frames before running the full control loop.
4. Avoid overwriting committed calibration files until the camera, checkerboard, and robot pose set are confirmed.
