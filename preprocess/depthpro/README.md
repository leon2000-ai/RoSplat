# DepthPro — Monocular Depth Estimation

Use [Apple Depth Pro](https://github.com/apple/ml-depth-pro) to generate metric depth maps
for multi-view plant images.

> **Why DepthPro?** RoSplat uses per-view depth as a geometry prior for depth regularization
> and contour extraction. DepthPro outputs sharp metric depth in <1 second with zero-shot
> generalization — no per-scene calibration needed.

---

## Installation

```bash
# Clone DepthPro repo
git clone https://github.com/apple/ml-depth-pro
cd ml-depth-pro
pip install -e .

# Download pre-trained model weights
source get_pretrained_models.sh
```

Alternatively, use the HuggingFace Transformers version (simpler, no local clone):

```bash
pip install transformers torch
# Model auto-downloaded: apple/DepthPro-hf
```

---

## Single Image Depth Estimation

### Option A: Official DepthPro API

```python
import depth_pro
from PIL import Image

# Load model
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess
image, _, f_px = depth_pro.load_rgb("scene/image_0001.jpg")
image = transform(image)

# Inference
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # metric depth in meters (numpy array)

# Save as 16-bit PNG (depth in millimeters)
import numpy as np
import cv2
depth_mm = (depth * 1000).astype(np.uint16)
cv2.imwrite("scene/depth/image_0001.png", depth_mm)
```

### Option B: HuggingFace Transformers

```python
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
import torch
import numpy as np
import cv2
from PIL import Image

model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to("cuda")
processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")

image = Image.open("scene/image_0001.jpg")
inputs = processor(images=image, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)

depth = outputs.predicted_depth.squeeze().cpu().numpy()  # metric depth in meters
depth_mm = (depth * 1000).astype(np.uint16)
cv2.imwrite("scene/depth/image_0001.png", depth_mm)
```

---

## Batch Processing All Views

```python
import os
import depth_pro
from PIL import Image
import numpy as np
import cv2

model, transform = depth_pro.create_model_and_transforms()
model.eval()

data_root = "/path/to/your/data"
image_dir = os.path.join(data_root, "images")
depth_dir = os.path.join(data_root, "depth")
os.makedirs(depth_dir, exist_ok=True)

for fname in sorted(os.listdir(image_dir)):
    if not fname.lower().endswith(('.jpg', '.png')):
        continue

    image, _, f_px = depth_pro.load_rgb(os.path.join(image_dir, fname))
    image_t = transform(image)

    prediction = model.infer(image_t, f_px=f_px)
    depth = prediction["depth"]

    # Save as 16-bit PNG (depth in millimeters, preserves precision)
    depth_mm = (depth * 1000).astype(np.uint16)
    cv2.imwrite(os.path.join(depth_dir, fname), depth_mm)
    print(f"  {fname} → depth/{fname}")
```

---

## Output Specification

| Item | Format | Value Range | Description |
|------|--------|-------------|-------------|
| File | `depth/*.png` | — | Same filename as source image |
| Pixels | 16-bit grayscale PNG | 0–65535 | Depth in **millimeters** |
| Resolution | Same as input | — | No downscaling |

> **Note:** DepthPro outputs metric depth (meters). We save as `uint16` millimeters
> to preserve precision. RoSplat's data loader reads these as float32 and normalizes
> during training.

---

## References

- [DepthPro GitHub](https://github.com/apple/ml-depth-pro)
- [DepthPro Paper (ICLR 2025)](https://arxiv.org/abs/2410.02073)
- [HuggingFace Model](https://huggingface.co/apple/DepthPro-hf)
