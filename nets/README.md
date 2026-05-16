# Networks

`nets/` defines the YOLOv3 network and training loss helpers used by the detector wrappers at the repository root.

## Files

| File | Purpose |
| --- | --- |
| `darknet.py` | DarkNet-53 backbone with residual blocks and feature outputs at 52x52, 26x26, and 13x13 for 416x416 inputs. |
| `yolo.py` | YOLOv3 detection head built on the DarkNet backbone. Returns three output tensors for the three feature scales. |
| `yolo_training.py` | YOLO loss, target assignment, ignore-mask computation, weight initialization, learning-rate scheduling, and optimizer LR updates. |
| `__init__.py` | Package marker. |

## How It Connects to Inference

The inference path is:

```text
predict_cu.py or predict_xi.py
  -> yolo.py or yolo_nine.py
    -> nets.yolo.YoloBody
      -> nets.darknet.darknet53
    -> utils.utils_bbox.DecodeBox
```

`YoloBody` builds three detection branches. The output channel count for each branch is:

```text
len(anchors_for_scale) * (num_classes + 5)
```

For the current coarse detector, `num_classes = 2`. For the current fine detector, `num_classes = 3`.

## Training Utilities

`nets/yolo_training.py` is a helper module, not a complete training script. It provides:

- `YOLOLoss` for localization, objectness, and classification loss.
- GIoU box loss support.
- Anchor matching and ignore-mask logic.
- `weights_init(...)` for model initialization.
- `get_lr_scheduler(...)` for cosine or step learning-rate schedules.
- `set_optimizer_lr(...)` for per-epoch LR updates.

A full training run would also need a training script, annotation files, dataset folders, optimizer setup, dataloaders from `utils/dataloader.py`, callbacks from `utils/callbacks.py`, and output directories for logs and weights.

## Dependency Notes

This folder depends on:

- `torch`
- `torch.nn`
- `numpy`

Inference also depends on `torchvision.ops.nms` through `utils/utils_bbox.py`.

## Reproducibility Notes

- The backbone can optionally load `model_data/darknet53_backbone_weights.pth` when `pretrained=True`, but that file is not present in the repository.
- Current detector wrappers instantiate `YoloBody(...)` with `pretrained=False` and then load the final detector weights from `model_data/`.
- If the class list changes, update the detector weights and the corresponding `model_data/*.txt` file together.
