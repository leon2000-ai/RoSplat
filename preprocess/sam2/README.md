# SAM 2 — Object Mask Generation

Use [Meta Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/sam2) to generate
foreground segmentation masks for multi-view plant images.

> **Why SAM 2?** RoSplat requires per-view binary masks (foreground=1, background=0) for contour-aware
> training. SAM 2 provides SOTA zero-shot segmentation quality with minimal prompting.

---

## Installation

```bash
# Clone SAM2 repo
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .

# Download model checkpoints
cd checkpoints && ./download_ckpts.sh && cd ..
```

> Recommended checkpoint: `sam2.1_hiera_large.pt` (best accuracy).

---

## Single Image Mask Generation (Prompt-based)

Use a **bounding box** or **single point** to segment the plant from its background.

```python
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load model
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Load image
image = Image.open("scene/image_0001.jpg")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)

    # Option A: Bounding box prompt [x1, y1, x2, y2]
    masks, _, _ = predictor.predict(box=[[[0, 0, 1599, 1199]]])

    # Option B: Point prompt (click the center of the plant)
    # masks, _, _ = predictor.predict(
    #     point_coords=[[[[800, 600]]]],
    #     point_labels=[[[1]]]        # 1 = foreground
    # )

# Save mask (0=background, 255=foreground)
mask = (masks[0, 0] * 255).astype("uint8")
Image.fromarray(mask).save("scene/mask/image_0001.png")
```

---

## Batch Processing All Views

```python
import os
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

data_root = "/path/to/your/data"
image_dir = os.path.join(data_root, "images")
mask_dir  = os.path.join(data_root, "mask")
os.makedirs(mask_dir, exist_ok=True)

# Use the SAME bounding box for all views (or adjust per view)
bbox = [[[0, 0, 1599, 1199]]]

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue

        image = Image.open(os.path.join(image_dir, fname))
        predictor.set_image(image)
        masks, _, _ = predictor.predict(box=bbox)

        mask = (masks[0, 0] * 255).astype("uint8")
        Image.fromarray(mask).save(os.path.join(mask_dir, fname))
        print(f"  {fname} → mask/{fname}")
```

---

## Output Specification

| Item | Format | Value Range | Description |
|------|--------|-------------|-------------|
| File | `mask/*.png` | — | Same filename as source image |
| Pixels | Grayscale PNG | 0–255 | 0 = background, 255 = plant foreground |
| Resolution | Same as input | — | No downscaling |

> The `mask/` directory is fed into `extract_contours.py` along with `depth/`.

---

## References

- [SAM 2 GitHub](https://github.com/facebookresearch/sam2)
- [SAM 2 Paper](https://arxiv.org/abs/2408.00714)
- [HuggingFace Model Hub](https://huggingface.co/facebook/sam2-hiera-large)
