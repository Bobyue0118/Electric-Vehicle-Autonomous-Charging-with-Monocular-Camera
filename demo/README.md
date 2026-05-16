# Demo Images

`demo/` contains sample high-resolution RGB images from the project. The files are useful for visual inspection, detector smoke tests, and documentation examples.

## Files

| File | Image size | Notes |
| --- | --- | --- |
| `1023_rgb_16_21.png` | `5480 x 3648` | Sample capture. |
| `1023_rgb_16_28.png` | `5480 x 3648` | Sample capture. |
| `1023_rgb_16_33.png` | `5480 x 3648` | Sample capture. |
| `1023_rgb_16_36.png` | `5480 x 3648` | Sample capture. |
| `5_5_5.png.png` | `5480 x 3648` | Sample image used by several script-level examples. |

## Example Use

After restoring the missing detector weights in `model_data/`, a coarse detector smoke test can use a demo image:

```python
from PIL import Image
from predict_cu import predict

image_path = "demo/5_5_5.png.png"
image = Image.open(image_path)
result = predict(image, "5_5_5.png")
print(result)
```

The detector wrapper saves annotated images into `img_out/`.

## Notes

- These are large images, so inference may be slower than on resized inputs.
- The images are not a full benchmark dataset.
- Do not overwrite them when collecting new hardware data; save new captures into a dated experiment folder instead.
