# Utilities

`utils/` contains reusable helpers for YOLO data loading, preprocessing, decoding, training callbacks, training loops, and mAP evaluation. These modules support the detector code, but there is no standalone training script in the current repository.

## Files

| File | Purpose |
| --- | --- |
| `utils.py` | Image color conversion, resizing, class/anchor loading, seed setup, worker seed setup, input normalization, and config printing. |
| `utils_bbox.py` | YOLO output decoding, coordinate correction, confidence filtering, and non-maximum suppression. |
| `dataloader.py` | `YoloDataset` and `yolo_dataset_collate` for training/validation batches. |
| `callbacks.py` | Loss logging, TensorBoard writing, mAP evaluation callback, and temporary map-output generation. |
| `utils_fit.py` | One-epoch training/validation loop with optional mixed precision and checkpoint saving. |
| `utils_map.py` | VOC-style mAP and COCO-style mAP helper functions. |
| `__init__.py` | Package marker. |

## Annotation Format

`YoloDataset` expects each annotation line to be:

```text
path/to/image.jpg x_min,y_min,x_max,y_max,class_id x_min,y_min,x_max,y_max,class_id
```

Example:

```text
images/sample.jpg 120,80,240,210,0 300,190,360,260,1
```

During training, `dataloader.py` converts boxes from corner format into normalized center-width-height format:

```text
x_center, y_center, width, height, class_id
```

## Inference Helper Flow

The detector wrappers call utility functions in this order:

```text
get_classes(...)
get_anchors(...)
resize_image(...)
preprocess_input(...)
DecodeBox.decode_box(...)
DecodeBox.non_max_suppression(...)
```

`DecodeBox.non_max_suppression(...)` uses `torchvision.ops.nms`, so `torchvision` must be installed with a version compatible with the local PyTorch installation.

## Training Helper Flow

A complete training script would normally compose these utilities as:

1. Load class names and anchors with `utils.py`.
2. Build `YoloDataset` objects from annotation lines.
3. Build `YoloBody` and `YOLOLoss`.
4. Use `LossHistory` and `EvalCallback` from `callbacks.py`.
5. Call `fit_one_epoch(...)` from `utils_fit.py`.
6. Save checkpoints into a configured output directory.

`fit_one_epoch(...)` writes:

- periodic `epXXX-loss...-val_loss....pth` checkpoints;
- `best_epoch_weights.pth` when validation loss improves;
- `last_epoch_weights.pth` after each epoch.

## Reproducibility Notes

`utils.seed_everything(seed=11)` sets Python, NumPy, and PyTorch seeds and disables CuDNN benchmark mode. Use it in any new training script before creating datasets, dataloaders, or models.

For evaluation, keep class names and confidence thresholds aligned with the detector weights. Otherwise mAP results will not be comparable to the original experiments.
