# RomaSfm — Camera Pose Estimation

Wrapper around [VGGSfM](https://github.com/facebookresearch/vggsfm) (CVPR 2024).

Runs Structure-from-Motion on a folder of images and outputs COLMAP-format camera
poses directly usable by RoSplat.

---

## Installation

```bash
cd preprocess/RomaSfm
source install.sh
python -m pip install -e .
```

The pre-trained model is auto-downloaded from HuggingFace on first run.

---

## Usage

```bash
python demo.py SCENE_DIR=/path/to/your/scene
```

**Input:** a folder containing `images/` with multi-view RGB images.

**Output:** `sparse/0/` in COLMAP format (`cameras.bin`, `images.bin`, `points3D.bin`).

---

## Common Options

Override any option from `cfgs/demo.yaml`:

```bash
# Fast mode (no fine tracking)
python demo.py SCENE_DIR=/your/scene fine_tracking=False

# For turntable plant scenes (shared intrinsics, radial distortion)
python demo.py SCENE_DIR=/your/scene shared_camera=True camera_type=SIMPLE_RADIAL

# Use masks to filter background (put masks in SCENE_DIR/masks/)
python demo.py SCENE_DIR=/your/scene
```

> **Note:** RoSplat's `mask/` outputs from SAM 2 can be placed as `SCENE_DIR/masks/`
> to help RomaSfm filter out background during reconstruction. Masks should be binary:
> 1 = pixel to filter out, 0 = keep.

---

## References

- [VGGSfM GitHub](https://github.com/facebookresearch/vggsfm)
- [VGGSfM Paper](https://arxiv.org/abs/2312.04563)
