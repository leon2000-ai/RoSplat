#!/bin/bash
# RoSplat demo: train + render + export semantic point cloud
#
# Usage:
#   bash scripts/train.sh /path/to/your/dataset [output_name]

DATA=${1:?"Usage: bash scripts/train.sh /path/to/your/dataset [output_name]"}
NAME=${2:-demo}
OUTPUT=./output/${NAME}
ITER=30000

set -e

echo "============================================"
echo " RoSplat Demo"
echo " Data:   ${DATA}"
echo " Output: ${OUTPUT}"
echo " Iters:  ${ITER}"
echo "============================================"

# Step 1: Training
echo ""
echo "[1/3] Training..."
python train.py \
    -s "${DATA}" \
    -m "${OUTPUT}" \
    --iterations ${ITER} \
    --test_iterations 15000 ${ITER} \
    --save_iterations 15000 ${ITER}

# Step 2: Render novel views
echo ""
echo "[2/3] Rendering..."
python render.py \
    --model_path "${OUTPUT}" \
    --iteration ${ITER} \
    --skip_train

# Step 3: Export semantic point cloud + mesh
echo ""
echo "[3/3] Exporting semantic point cloud and mesh..."
python render.py \
    --model_path "${OUTPUT}" \
    --iteration ${ITER} \
    --gs_id

echo ""
echo "============================================"
echo " Demo complete!"
echo " Results saved to: ${OUTPUT}"
echo "   - point_cloud/iteration_${ITER}/point_cloud.ply"
echo "   - train/point_cloud_object.ply"
echo "   - train/fuse_object.ply"
echo "============================================"
