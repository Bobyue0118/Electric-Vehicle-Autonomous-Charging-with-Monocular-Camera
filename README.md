
# Research on EV Autonomous Charging with Monocular Camera

<div align="center">
  <table cellspacing="0" cellpadding="0">
    <tr>
      <td width="38%" align="center" valign="top" rowspan="2">
        <img
          src="demo/autonomous-charging-demo.gif"
          alt="Autonomous charging demo"
          width="100%">
      </td>
      <td width="62%" align="center" valign="top">
        <img
          src="https://github.com/user-attachments/assets/be41e4f6-a282-4c31-a8df-2022e126e0fc"
          alt="project1"
          width="100%">
      </td>
    </tr>
    <tr>
      <td width="60%" align="center" valign="bottom">
        <a href="https://www.starcharge.com/zdcdb/detail">
          <img
            src="https://web-suite-backend-oss.wbstar.com/web/1758272013124preImage.png?x-oss-process=image%2Fresize%2Cl_1920%2Ch_500%2Fformat%2Cwebp"
            alt="Star Charge Armstrong charging robot thumbnail"
            width="100%">
        </a>
      </td>
    </tr>
  </table>
</div>

- Qilong Wu\*, **Bo Yue**\*, Xin Liu, Shibei Xue

- *School Corporate Cooperation Project with [Star Charge](https://www.wbstar.com/)*, 2023.03-2023.09

- Video demo: [[Real Car]](https://www.bilibili.com/video/BV1ax4y1C7n3/?spm_id_from=333.880.my_history.page.click&vd_source=8debf3b3fb5f9dca46569bbb6cfa839c); [[Lab Env]](https://www.bilibili.com/video/BV1aN4y187i4/?spm_id_from=autoNext&vd_source=8debf3b3fb5f9dca46569bbb6cfa839c); [[Application]](https://www.starcharge.com/zdcdb/detail).

## What This Repository Contains

This repository contains research code for monocular-vision autonomous EV charging. The code combines object detection, geometric feature localization, camera calibration, hand-eye calibration, and a TCP control loop that returns robot motion commands to an external charging controller.

At a high level, the pipeline is:

1. Capture an image from an industrial camera.
2. Detect charging-port features with YOLOv3 models.
3. Use coarse detection to center the target and estimate depth.
4. Use fine detection plus circle/PnP geometry to estimate the plug or socket pose.
5. Transform camera-frame pose estimates into robot/base-frame motion commands.
6. Send formatted command strings back through `udp_client.py`.

The repository is research-oriented rather than packaged as a library. Many scripts are experiment entry points with local paths, saved calibration dates, and hardware assumptions. The README files in each folder describe what each part does and what must be prepared before rerunning it.

## Repository Map

| Part | Purpose | Documentation |
| --- | --- | --- |
| Top-level scripts | Main detection, localization, camera demo, ArUco demo, and TCP control scripts. | See [Top-Level Script Guide](#top-level-script-guide). |
| `assets/` | Calibration utilities, vendor camera examples, ArUco helpers, and one-off transform scripts. | [assets/README.md](assets/README.md) |
| `parameters/` | Saved camera intrinsics, distortion coefficients, hand-eye matrices, and board/robot pose arrays. | [parameters/README.md](parameters/README.md) |
| `model_data/` | YOLO class lists, anchors, and font assets. Trained weights are referenced by code but are not included. | [model_data/README.md](model_data/README.md) |
| `nets/` | YOLOv3/DarkNet network definitions and loss/scheduler utilities. | [nets/README.md](nets/README.md) |
| `utils/` | Data loading, preprocessing, decoding, mAP, callbacks, and training-loop helpers. | [utils/README.md](utils/README.md) |
| `demo/` | Example high-resolution images for inspection and detector experiments. | [demo/README.md](demo/README.md) |

## Environment Notes

The code mixes pure Python vision components with Windows-only industrial-camera components.

For detector and geometry scripts, expect these Python packages:

```bash
pip install numpy opencv-contrib-python scipy pillow scikit-image matplotlib tqdm torch torchvision
```

Choose the `torch` and `torchvision` builds that match your machine and CUDA setup. The repository does not currently include `requirements.txt` or `environment.yml`, so treat the command above as a practical starting point, not a pinned reproduction environment.

Hardware-facing scripts require additional vendor runtime support:

- `dvp.pyd` and `DVPCamera64.dll` are Windows binary camera SDK files.
- `OpenCV_Demo.py`, `udp_client.py`, and `pose_estimation_newnew.py` import `dvp`.
- `MvCameraControl_class.py` loads `C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64\MvCameraControl.dll`.
- `pose_estimation_newnew.py` also imports `pyzed.sl` for an older ZED-camera workflow, although that path is mostly commented out.

On macOS or Linux, documentation inspection and pure utility imports may work, but hardware capture scripts should be expected to fail unless the corresponding camera SDK and binaries are available.

## Required External Artifacts

The committed repo includes calibration `.npy` files under `parameters/`, but it does not include all runtime artifacts referenced by the scripts.

Expected but not committed:

- `model_data/best_epoch_weights.pth`, used by `yolo.py` and `predict_cu.py`.
- `model_data/best_epoch_weights_nine.pth`, used by `yolo_nine.py` and `predict_xi.py`.
- `1018/parameters/M_virtual2object_cu_1018.npy`, used by `cu_predict.py`.
- `1018/parameters/M_virtual2object_xi_1018.npy`, used by `xi_predict.py`.
- Runtime image folders such as `0822/cudingwei/`, `0822/xidingwei/`, and related capture folders used by `udp_client.py`.

Without the detector weight files, YOLO inference will not run. Without the virtual-object transform files, the final camera-to-base pose conversion functions cannot complete.

## Top-Level Script Guide

### Main Control Loop

| File | Role | Notes |
| --- | --- | --- |
| `udp_client.py` | TCP server that opens the camera, captures images, calls localization functions, and returns command strings. | Binds to `192.168.1.85:5`. This IP is local to the original experimental setup and should be changed before reuse. |
| `OpenCV_Demo.py` | Vendor DVP camera preview and camera-parameter demo. | Requires `dvp.pyd` and the DVP camera runtime. It runs immediately because `main()` is called at the bottom. |
| `pose_estimation_newnew.py` | ArUco/camera pose-estimation and image-capture experiment script. | Contains old ZED, DVP, and MVS code paths. Current active path opens `Camera(0)` through DVP and saves calibration images on key press. |

`udp_client.py` expects controller messages of the form `101,...` to start camera capture and `102,...` to process a captured image. The second field selects the stage:

| Stage code | Script path called | Meaning in the current code |
| --- | --- | --- |
| `3` | `cu_jiaozheng_qian_2.py` | Coarse pre-correction in image x/y using the large charging-port box. |
| `1` | `cu_dingwei_2.py` | Coarse depth correction using two small detected circles. |
| `5` | `xi_jiaozheng_0.py` | Fine-stage x/y pre-correction using two large fine markers. |
| `4` | `xi_jiaozheng_1.py` plus `cu_predict.py` | Fine correction pose solve, then transform to the coarse virtual target. |
| `2` | `xi_dingwei.py` plus `xi_predict.py` | Fine pose solve, then transform to the fine virtual target. |

### Detection Wrappers

| File | Role | Model/class file |
| --- | --- | --- |
| `predict_cu.py` | Runs coarse detector on a single PIL image and saves annotated output to `img_out/`. | `yolo.py`, `model_data/lens.txt` with classes `Big` and `small`. |
| `predict_xi.py` | Runs fine detector on a single PIL image and saves annotated output to `img_out/`. | `yolo_nine.py`, `model_data/lens_nine.txt` with classes `a`, `b`, and `c`. |
| `yolo.py` | YOLOv3 inference wrapper for the coarse two-class detector. | Expects `model_data/best_epoch_weights.pth`. |
| `yolo_nine.py` | YOLOv3 inference wrapper for the fine marker detector. | Expects `model_data/best_epoch_weights_nine.pth`. |

The `predict_*` files are mainly used as imported functions:

```python
from PIL import Image
from predict_cu import predict

image = Image.open("demo/5_5_5.png.png")
detections = predict(image, "5_5_5.png")
```

The `if __name__ == "__main__"` blocks call `predict()` without required arguments, so direct CLI execution of these two wrapper files is not currently a reliable entry point.

### Localization and Pose Conversion

| File | Role | Output |
| --- | --- | --- |
| `cu_jiaozheng_qian_2.py` | Uses the coarse detector's `Big` box to compute x/y image-centering correction. | Command string beginning with `102,3,...`. |
| `cu_dingwei_2.py` | Uses two `small` detections and camera intrinsics to estimate z correction. | Command string beginning with `102,3,...`. |
| `xi_jiaozheng_0.py` | Uses two fine `a` detections to compute x/y correction. | Command string beginning with `102,5,...`. |
| `xi_jiaozheng_1.py` | Fine pose-estimation variant used during correction. | `(tvec, rvec)` from `cv2.solvePnPRansac`. |
| `xi_dingwei.py` | Fine pose-estimation variant used for final localization. | `(tvec, rvec)` from `cv2.solvePnPRansac`. |
| `cu_predict.py` | Converts a detected pose to a base-frame virtual target using coarse virtual-object calibration. | `(T_virtual2base, euler_virtual2base)`. |
| `xi_predict.py` | Converts a detected pose to a base-frame virtual target using fine virtual-object calibration. | `(T_virtual2base, euler_virtual2base)`. |

`xi_dingwei.py` and `xi_jiaozheng_1.py` first use YOLO detections as anchors, then refine circular feature centers with OpenCV thresholding, Hough circle detection, and ellipse fitting. The final pose is solved with `cv2.solvePnPRansac` against a hard-coded 3D feature layout.

### ArUco Utilities

| File | Role | Preferred copy |
| --- | --- | --- |
| `generate_aruco_tags.py` | Generates ArUco marker images. | Prefer `assets/generate_aruco_tags.py`. |
| `detect_aruco_images.py` | Detects ArUco tags in one image. | Prefer `assets/detect_aruco_images.py`. |
| `detect_aruco_video.py` | Detects ArUco tags from a webcam or video. | Prefer `assets/detect_aruco_video.py`. |
| `utils_aruco.py` | Root-level ArUco dictionary/display helper. | Used by `pose_estimation_newnew.py`. |

The root ArUco demo scripts import `utils`, while the root helper is named `utils_aruco.py` and there is also a `utils/` package. To avoid import ambiguity, use the `assets/` copies or update imports before relying on the root copies.

Example:

```bash
mkdir -p tags
python assets/generate_aruco_tags.py --id 24 --type DICT_5X5_100 --output tags --size 400
python assets/detect_aruco_images.py --image tags/DICT_5X5_100_id_24.png --type DICT_5X5_100
```

## Typical Reproduction Workflow

For a full hardware rerun, prepare the pieces in this order:

1. Install Python dependencies and the required Windows camera SDK/runtime.
2. Place detector weights in `model_data/`.
3. Confirm the calibration matrices in `parameters/` match the camera and robot setup.
4. Generate or restore `1018/parameters/M_virtual2object_cu_1018.npy` and `1018/parameters/M_virtual2object_xi_1018.npy`.
5. Update local paths and the TCP bind address in `udp_client.py`.
6. Run the camera preview with `OpenCV_Demo.py` before starting the control loop.
7. Start `udp_client.py` and send controller messages from the external robot/charging controller.

For a vision-only review, inspect `demo/`, `model_data/`, `yolo.py`, `yolo_nine.py`, `predict_cu.py`, `predict_xi.py`, and the `xi_*`/`cu_*` localization scripts. This is enough to understand the algorithmic flow without the camera hardware.

## Known Research-Code Assumptions

- Several paths are hard-coded to experiment folders such as `0822/`, `1018/`, and `1212/`.
- Several scripts assume Windows path separators.
- The camera and controller IP/port are fixed in `udp_client.py`.
- The repository does not include detector weights or all virtual-target calibration files.
- Some scripts are duplicated between the root and `assets/`; treat `assets/` as the safer place for calibration and ArUco helper scripts.
- There is no formal test suite in the current checkout.

## Validation Guidance

After changing code or calibration data, use the smallest check that matches the risk:

- For docs-only edits, check Markdown links and file references.
- For detector edits, run one image through `predict_cu.predict(...)` or `predict_xi.predict(...)` after restoring weights.
- For geometry edits, run one saved image through the relevant `cu_*` or `xi_*` function and check output shape and units.
- For hardware changes, first run `OpenCV_Demo.py`, then run one controller request through `udp_client.py` before attempting a full charging sequence.
