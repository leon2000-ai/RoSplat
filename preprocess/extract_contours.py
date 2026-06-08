"""
Extract object contours from binary masks and depth maps.

For each data folder, reads mask/*.png and depth/*.png,
produces fused contour maps saved to objects_contours/*.png.

Usage:
    python preprocess/extract_contours.py --root /path/to/data_root
"""

import os
import argparse
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor


def sobel_edges(depth):
    """Sobel edge detection on a depth tensor."""
    if depth.ndim == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=depth.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=depth.device).view(1, 1, 3, 3)
    grad_x = F.conv2d(depth, sobel_x, padding=1)
    grad_y = F.conv2d(depth, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze()
    grad_norm = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
    return grad_norm


def get_combined_edge_mask(mask_tensor, depth_tensor, w_img=1.0, w_depth=0.0, threshold=0.05):
    """
    Fuse mask edges (Canny) and depth edges (Sobel) into a combined contour mask.

    Args:
        mask_tensor: binary mask [1, H, W] or [H, W], foreground=1, background=0
        depth_tensor: depth map [H, W] or [1, H, W]
        w_img: weight for mask Canny edges
        w_depth: weight for depth Sobel edges
        threshold: binarization threshold for fused result
    """
    if mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.squeeze(0)
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    edges_img = cv2.Canny(mask_np, 50, 150)
    img_edge_mask = torch.from_numpy(edges_img / 255.0).float().to(depth_tensor.device)

    depth_edge_mask = sobel_edges(depth_tensor).clamp(0, 1)

    fused = w_img * img_edge_mask + w_depth * depth_edge_mask
    final_mask = (fused > threshold).float()
    return final_mask


def load_mask_binary(path):
    """Load semantic mask as binary (foreground=1, background=0)."""
    img = Image.open(path).convert('L')
    mask = to_tensor(img)
    mask = (mask > 0.0).float()
    return mask


def load_depth_as_tensor(path):
    """Load depth map as [H, W] float tensor."""
    img = Image.open(path).convert("L")
    depth = to_tensor(img).squeeze(0)
    return depth


def process_depth_mask_pairs(depth_dir, mask_dir, output_dir):
    """Process all depth-mask pairs in given directories."""
    os.makedirs(output_dir, exist_ok=True)

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.lower().endswith(('.png', '.jpg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.png')])
    if not mask_files:
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.jpg')])

    assert len(depth_files) == len(mask_files), \
        f"File count mismatch: depth={len(depth_files)}, mask={len(mask_files)}"

    for dname, mname in zip(depth_files, mask_files):
        depth_path = os.path.join(depth_dir, dname)
        mask_path = os.path.join(mask_dir, mname)

        depth_tensor = load_depth_as_tensor(depth_path)
        mask_tensor = load_mask_binary(mask_path)

        # Align depth to mask size
        if depth_tensor.shape != mask_tensor.shape[1:]:
            depth_tensor = F.interpolate(
                depth_tensor.unsqueeze(0).unsqueeze(0),
                size=mask_tensor.shape[1:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

        edge_mask = get_combined_edge_mask(mask_tensor, depth_tensor)

        edge_mask_img = (edge_mask * 255).byte().cpu().numpy()
        # Zero out border pixels
        edge_mask_img[0, :] = 0
        edge_mask_img[-1, :] = 0
        edge_mask_img[:, 0] = 0
        edge_mask_img[:, -1] = 0

        save_name = os.path.splitext(mname)[0] + ".png"
        cv2.imwrite(os.path.join(output_dir, save_name), edge_mask_img)
        print(f"  Saved: {save_name}")

    print(f"Done: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract contours from masks and depth maps")
    parser.add_argument("--root", "-r", type=str, required=True,
                        help="Root directory containing subfolders with mask/ and depth/")
    args = parser.parse_args()

    root_dir = args.root

    for subfolder in sorted(os.listdir(root_dir)):
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        depth_dir = os.path.join(subfolder_path, "depth")
        mask_dir = os.path.join(subfolder_path, "mask")
        output_dir = os.path.join(subfolder_path, "objects_contours")

        if not os.path.isdir(depth_dir) or not os.path.isdir(mask_dir):
            print(f"Skip {subfolder}: missing depth/ or mask/")
            continue

        print(f"Processing: {subfolder}")
        process_depth_mask_pairs(depth_dir, mask_dir, output_dir)

    print("All done!")


if __name__ == "__main__":
    main()
