# Model Data

`model_data/` contains small YOLO metadata files and display assets. It does not currently contain the trained PyTorch weight files required for inference.

## Files

| File | Purpose |
| --- | --- |
| `lens.txt` | Class list for the coarse detector in `yolo.py`: `Big`, `small`. |
| `lens_nine.txt` | Class list for the fine detector in `yolo_nine.py`: `a`, `b`, `c`. |
| `yolo_anchors.txt` | YOLO anchor boxes used by both detector wrappers. |
| `simhei.ttf` | Font used by `yolo.py` and `yolo_nine.py` when drawing labels. |
| `coco_classes.txt` | Standard COCO class list, retained from the YOLO codebase. Not used by the current detector defaults. |
| `voc_classes.txt` | Standard VOC class list, retained from the YOLO codebase. Not used by the current detector defaults. |

## Expected Weight Files

The detector wrappers reference these files:

| Expected file | Used by | Class file |
| --- | --- | --- |
| `best_epoch_weights.pth` | `yolo.py`, `predict_cu.py` | `lens.txt` |
| `best_epoch_weights_nine.pth` | `yolo_nine.py`, `predict_xi.py` | `lens_nine.txt` |

These files are not present in the repository. Restore them from the experiment artifact store before running inference.

## Detector Defaults

Both `yolo.py` and `yolo_nine.py` use:

- Input shape: `416 x 416`
- Anchor file: `model_data/yolo_anchors.txt`
- Anchor masks: `[[6, 7, 8], [3, 4, 5], [0, 1, 2]]`
- Confidence threshold: `0.5`
- NMS IoU threshold: `0.3`
- `letterbox_image = False`
- `cuda = False`

If you change a class file, the model head shape must match the number of classes. A class-count mismatch will cause `torch.load(...).load_state_dict(...)` to fail.

## Artifact Policy

Model weights are usually large and should not be committed directly unless the repository explicitly decides to store them, for example with Git LFS. Keep the small metadata files here, and document the external source for trained weights when publishing reproduction instructions.
