# Parameters

`parameters/` stores calibration and transform arrays used by the localization and pose-conversion scripts. These files are NumPy `.npy` arrays and are part of the experiment state for the 4K camera calibration dated `1212` in the filenames.

## File Inventory

| File | Shape | Meaning in code |
| --- | --- | --- |
| `calibration_matrix_4k_1212.npy` | `(3, 3)` | Camera intrinsic matrix loaded by `cu_dingwei_2.py`, `xi_dingwei.py`, and `xi_jiaozheng_1.py`. |
| `distortion_coefficients_4k_1212.npy` | `(1, 5)` | Camera distortion coefficients loaded with the intrinsic matrix. |
| `R_board2camera_4k_1212.npy` | `(15, 3, 3)` | Rotation matrices from checkerboard/board frame to camera frame. |
| `T_board2camera_4k_1212.npy` | `(15, 3, 1)` | Translation vectors from checkerboard/board frame to camera frame. |
| `R_gripper2base_4k_1212.npy` | `(15, 3, 3)` | Robot gripper-to-base rotations used for hand-eye calibration. |
| `T_gripper2base_4k_1212.npy` | `(15, 3, 1)` | Robot gripper-to-base translations used for hand-eye calibration. |
| `R_camera2gripper_4k_1212.npy` | `(3, 3)` | Hand-eye rotation from camera frame to gripper frame. |
| `T_camera2gripper_4k_1212.npy` | `(3, 1)` | Hand-eye translation from camera frame to gripper frame. |
| `M_camera2gripper_4k_1212.npy` | `(4, 4)` | Homogeneous camera-to-gripper transform used by `cu_predict.py` and `xi_predict.py`. |
| `R_board2camera_4k_1212_test.npy` | `(2, 3, 3)` | Test board-to-camera rotations used by `assets/eyetest.py`. |
| `T_board2camera_4k_1212_test.npy` | `(2, 3, 1)` | Test board-to-camera translations used by `assets/eyetest.py`. |

## How These Files Are Produced

The expected generation flow is:

1. `assets/calibration.py` reads checkerboard images and writes camera intrinsics, distortion coefficients, and board-to-camera arrays.
2. `assets/HandEyeCalibrate.py` combines board-to-camera arrays with hard-coded robot poses and writes gripper-to-base and camera-to-gripper transforms.
3. `assets/eyetest.py` can be used as a transform-composition sanity check.

The scripts in `assets/` currently read and write files in the current working directory. If you regenerate parameters, keep the new outputs separate until you verify they belong to the intended experiment.

## Runtime Use

The committed localization scripts load these files directly:

```text
cu_dingwei_2.py
  parameters/calibration_matrix_4k_1212.npy

xi_dingwei.py
  parameters/calibration_matrix_4k_1212.npy
  parameters/distortion_coefficients_4k_1212.npy

xi_jiaozheng_1.py
  parameters/calibration_matrix_4k_1212.npy
  parameters/distortion_coefficients_4k_1212.npy

cu_predict.py
  parameters/M_camera2gripper_4k_1212.npy

xi_predict.py
  parameters/M_camera2gripper_4k_1212.npy
```

`cu_predict.py` and `xi_predict.py` also load virtual-object transforms from `1018/parameters/`, which are not included in this repository:

```text
1018/parameters/M_virtual2object_cu_1018.npy
1018/parameters/M_virtual2object_xi_1018.npy
```

## Units and Conventions

The code uses a mixture of units:

- Camera intrinsics are pixel-space calibration values.
- Robot poses in `assets/HandEyeCalibrate.py` are written in meters for translation and degrees for Euler angles.
- Several image-geometry scripts use millimeter-scale feature layouts, then convert by `1e-3` when composing homogeneous transforms.
- `udp_client.py` receives robot translation values in millimeters, divides by `1000` before transform composition, then sends output translations in millimeters.

When changing any parameter file, check both the numeric unit and the frame direction. A valid matrix shape is not enough to guarantee that the runtime command will move in the intended direction.

## Safe Handling

- Do not overwrite these `.npy` files during exploratory runs.
- Save regenerated parameters in a dated folder first.
- Record the checkerboard size, square size, camera resolution, robot pose list, and date.
- Update code paths or README notes if you intentionally switch to a new calibration set.
