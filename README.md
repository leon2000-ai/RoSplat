## 📖 Lightweight Contour-Aware 2D Gaussian Splatting under Plant-to-Camera

🔥 Segmentation-guided pose estimation enables robust Plant-to-Camera reconstruction. ⭐ Lightweight 2DGS representation improves geometric efficiency and fidelity. 🔥 2D-to-3D semantic lifting enables direct phenotypic analysis.

> Paper(Update after receiving)

Liang Zhao, Hanwen Tong, Hangyu Liu, Zhanwang Zhu, Weibing Jin, Yuping Zhong, Bo Wu, Weifu Li, Lin Li

> Huazhong Agricultural University
> Food Crops Institute, Hubei Academy of Agricultural Sciences
> National Key Laboratory of Crop Genetic Improvement
> Hubei Hongshan Laboratory
> Hubei Huichuzhi Biological Technolog

🚩 **Updates**

☑ A lightweight, semantic-aware framework for Plant-to-Camera 3D plant phenotyping.

☑ Code released. Environment setup and demo instructions are provided below.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Quick Demo](#quick-demo)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Plant-to-Camera setting](#plant-to-camera-setting)
- [RoSplat pipeline](#rosplat-pipeline)
- [Experimental results](#experimental-results)
- [Contact](#contact)

---

## Environment Setup

### Prerequisites

- CUDA 11.8 or higher
- Python 3.8
- Conda (recommended) or pip

### Create environment

```bash
# Clone the repository
git clone https://github.com/your-org/RoSplat.git --recursive
cd RoSplat

# Create and activate conda environment
conda env create -f environment.yml
conda activate rosplat

# Install submodules (editable)
pip install -e submodules/diff-surfel-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/lanczos-resampling
```

> **Note:** If the submodule directories are empty, run `git submodule update --init --recursive` to pull them.

---

## Quick Demo

### 1. Run training on your own data

```bash
python train.py \
  --source_path /path/to/your/data \
  --model_path ./output/rosplat_demo \
  --iterations 30000
```

Your data should follow the structure described in [Data Preparation](#data-preparation).

### 2. Render from a trained model

```bash
python render.py \
  --model_path ./output/rosplat_demo \
  --iteration 30000
```

### 3. Export semantic point cloud and mesh

```bash
# Export RGB point cloud & mesh + semantic point cloud
python render.py \
  --model_path ./output/rosplat_demo \
  --iteration 30000 \
  --gs_id
```

Output files:
- `point_cloud/iteration_30000/point_cloud.ply` — RGB point cloud
- `train/point_cloud_object.ply` — Semantic point cloud (each Gaussian assigned an object label)
- `train/fuse_object.ply` — Semantic mesh

---

## Data Preparation

RoSplat requires per-view **masks**, **depth maps**, and **contour maps** in addition to RGB images.
The full preprocessing pipeline uses three external tools:

<p align="center">
  <b>RGB images</b>
  &nbsp;→&nbsp; <b>SAM 2</b> → <code>mask/</code>
  &nbsp;│&nbsp;
  <b>DepthPro</b> → <code>depth/</code>
  &nbsp;│&nbsp;
  <b>extract_contours.py</b> → <code>objects_contours/</code>
  &nbsp;│&nbsp;
  <b>RomaSfm</b> → <code>sparse/</code>
</p>

### Step 1: Generate masks with SAM 2

See [`preprocess/sam2/README.md`](preprocess/sam2/README.md) for detailed instructions.

<div align="center">

| Input | Tool | Output |
|-------|------|--------|
| `images/*.jpg` | [SAM 2](https://github.com/facebookresearch/sam2) | `mask/*.png` (0=bg, 255=plant) |

</div>

### Step 2: Generate depth maps with DepthPro

See [`preprocess/depthpro/README.md`](preprocess/depthpro/README.md) for detailed instructions.

<div align="center">

| Input | Tool | Output |
|-------|------|--------|
| `images/*.jpg` | [DepthPro](https://github.com/apple/ml-depth-pro) | `depth/*.png` (16-bit, mm) |

</div>

### Step 3: Extract contours from masks + depth

```bash
python preprocess/extract_contours.py --root /path/to/your/data
```

| Input | Tool | Output |
|-------|------|--------|
| `mask/*.png` + `depth/*.png` | `extract_contours.py` | `objects_contours/*.png` |

This fuses mask edges (Canny) with depth edges (Sobel) into a single contour map.

### Step 4: Estimate camera poses with RomaSfm

See [`preprocess/RomaSfm/README.md`](preprocess/RomaSfm/README.md) for installation and detailed options.

```bash
cd preprocess/RomaSfm
source install.sh && python -m pip install -e .

# Run SfM on your scene (turn off fine tracking for faster results)
python demo.py SCENE_DIR=/path/to/your/scene fine_tracking=False

# For turntable plant data with masks:
python demo.py SCENE_DIR=/path/to/your/scene shared_camera=True camera_type=SIMPLE_RADIAL
```

| Input | Tool | Output |
|-------|------|--------|
| `images/*.jpg` (+ optional `masks/*.png`) | RomaSfm (VGGSfM) | `sparse/0/` (cameras.bin, images.bin, points3D.bin) |

> **Tip:** Copy SAM 2 masks from `mask/` to `masks/` (with suffix `s`) in the SCENE_DIR to help RomaSfm
> filter background during reconstruction. RomaSfm expects masks where 1 = pixel to ignore, 0 = keep.

### Expected directory structure

After preprocessing, your dataset should look like this:

```
data/your_scene/
├── images/              # Multi-view RGB images
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
├── mask/                # SAM 2 foreground masks
│   ├── 0001.png
│   └── ...
├── depth/               # DepthPro metric depth maps
│   ├── 0001.png
│   └── ...
├── objects_contours/    # Fused contour maps (extract_contours.py)
│   ├── 0001.png
│   └── ...
├── sparse/              # RomaSfm (VGGSfM) camera poses
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── (optional) objects/  # Semantic labels (for gs_id 2D-to-3D lifting)
```

> **Note:** If you don't have `objects_contours/` or `depth/`, the training script falls back
> to standard reconstruction (RGB + masks only).

---

## Training

```bash
python train.py \
  --source_path ./data/your_dataset \
  --model_path ./output/your_experiment \
  --iterations 30000 \
  --test_iterations 15000 30000 \
  --save_iterations 15000 30000
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--source_path` | (required) | Path to dataset |
| `--model_path` | `./output/<uuid>` | Output directory |
| `--iterations` | `30000` | Total training iterations |
| `--test_iterations` | `15000 30000` | When to evaluate |
| `--save_iterations` | `15000 30000` | When to save checkpoint |
| `--start_checkpoint` | `None` | Resume from checkpoint |

**Training stages:**

1. **Warmup (< 10,000 iters):** Standard L1 + SSIM loss. Contour-aware edge loss is active from the start.
2. **Fine-tuning (≥ 10,000 iters):** Smooth transition loss added, enforcing depth-consistent boundaries.
3. **Densification:** DashGaussian scheduler controls primitive growth rate based on resolution curriculum.

---


## Plant-to-Camera setting

<p align="center">
  <img src="figures1.png" alt="Acquisition modes" width="80%">
</p>

<p align="center">
<b>Fig. 1.</b> Acquisition modes for 3D plant modeling.
(A) Plant-to-Camera: multiple views are captured by moving the camera around the target plant.
(B) Camera-to-Plant: multiple views are captured by fixed-pose camera systems during plant rotation.
</p>

<p align="center">
  <img src="figures3.png" alt="Feature matching" width="80%">
</p>

<p align="center">
<b>Fig. 2.</b>
(A) Images are captured using a smartphone or a camera fixed on a tripod, with plants placed on a rotating turntable in front of a backdrop.
(B) Cross-view feature matching. Top: corresponding feature points matched from original image views.
Bottom: corresponding feature points matched from background-removed views.
</p>

---

## RoSplat pipeline

<p align="center">
  <img src="figures2.png" alt="RoSplat pipeline overview" width="90%">
</p>

<p align="center">
<b>Fig. 3.</b> Overview of the proposed 3D plant phenotyping pipeline.
(A) Multi-view image acquisition of maize and wheat.
(B) Semantic depth estimation combining foreground segmentation and depth priors to suppress background interference.
(C) Lightweight contour-aware 2D Gaussian Splatting for efficient and faithful reconstruction.
(D) 2D-to-3D semantic lifting enables direct extraction of phenotypic traits.
</p>

<p align="center">
  <img src="figures4.png" alt="2D-to-3D semantic lifting" width="90%">
</p>

<p align="center">
<b>Fig. 4.</b> 2D-to-3D semantic lifting for plant phenotype extraction.
(A) Pixel-level segmentation masks from multiple views are lifted to 2D Gaussian primitives and aggregated across views.
(B) Phenotypic traits derived from semantically labeled Gaussians, including plant height, leaf angle, and spike length.
</p>

---

## Experimental results

<p align="center">
  <img src="figures5.png" alt="Pose estimation comparison" width="90%">
</p>

<p align="center">
<b>Fig. 5.</b> Qualitative comparison of camera pose estimation under the plant-to-camera setting.
Results are shown for maize at seedling (A) and jointing (B) stages, and wheat at the waxing stage (C).
SfM and VGGSfM produce scattered trajectories, whereas the proposed method consistently yields closed-loop trajectories.
</p>

<p align="center">
  <img src="figures6.png" alt="Rendering comparison" width="95%">
</p>

<p align="center">
<b>Fig. 6.</b> Qualitative rendering comparison and ablation study.
From left to right: ground-truth images, 3DGS, 2DGS, the proposed method, and ablation results.
Green dashed circles indicate typical structural artifacts that are alleviated by the proposed method.
</p>

<p align="center">
  <img src="figures7.png" alt="Point cloud comparison" width="95%">
</p>

<p align="center">
<b>Fig. 7.</b> Qualitative point cloud comparison and ablation study.
The proposed method yields denser and more coherent geometric reconstructions,
while removing the lightweight design or depth regularization introduces noise and structural degradation.
</p>

<p align="center">
  <img src="figures8.png" alt="Quantitative evaluation" width="90%">
</p>

<p align="center">
<b>Fig. 8.</b> Quantitative evaluation of phenotypic trait extraction.
Scatter plots compare predicted and manually measured values for leaf angle, plant height, and spike length.
</p>

---

### Contact
If you have any question or collaboration needs, please email zhao_liang@webmail.hzau.edu.cn.

## Acknowledgement
This study is based on [2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). We appreciate their great codes.
