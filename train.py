#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, smooth_transition_loss, l1_loss_edge
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
import os
import torchvision
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from utils.schedule_utils import TrainingScheduler
from FastLanczos import lanczos_resample
import cv2
import torch.nn.functional as F

def save_image_and_gsid(gt_object, gs_id, step, save_dir):
    """
    Save GT object image and GS ID visualization side by side.

    gt_object: ground truth object, grayscale or color image
               shape CxHxW or HxWxC, value range [0,1] or [0,255]
    gs_id: corresponding GS ID map, shape HxW
    step: step number
    save_dir: save directory
    """
    import os
    import numpy as np
    import torch
    import torchvision
    from PIL import Image

    os.makedirs(save_dir, exist_ok=True)

    # Ensure gt_object is tensor on CPU
    if isinstance(gt_object, torch.Tensor):
        gt_object_cpu = gt_object.detach().cpu()
    else:
        gt_object_cpu = gt_object

    # Save image directly, no color change
    img_path = os.path.join(save_dir, f"image_{step:05d}.png")

    if isinstance(gt_object_cpu, torch.Tensor):
        # Handle torch tensor
        if gt_object_cpu.ndim == 3:
            if gt_object_cpu.shape[0] == 3:  # 3xHxW color
                if gt_object_cpu.max() <= 1.0:  # range [0,1]
                    gt_object_to_save = (gt_object_cpu * 255).byte()
                else:  # range [0,255]
                    gt_object_to_save = gt_object_cpu.byte()
                torchvision.utils.save_image(gt_object_to_save.float() / 255.0, img_path)
            elif gt_object_cpu.shape[0] == 1:  # 1xHxW grayscale
                if gt_object_cpu.max() <= 1.0:  # range [0,1]
                    gt_object_to_save = (gt_object_cpu * 255).byte()
                else:  # range [0,255]
                    gt_object_to_save = gt_object_cpu.byte()
                torchvision.utils.save_image(gt_object_to_save.float() / 255.0, img_path)
            else:
                raise ValueError(f"Unexpected tensor shape: {gt_object_cpu.shape}")
        elif gt_object_cpu.ndim == 2:  # HxW grayscale
            if gt_object_cpu.max() <= 1.0:  # range [0,1]
                gt_object_to_save = (gt_object_cpu * 255).byte()
            else:  # range [0,255]
                gt_object_to_save = gt_object_cpu.byte()
            torchvision.utils.save_image(gt_object_to_save.float() / 255.0, img_path)
        else:
            raise ValueError(f"Unexpected tensor shape: {gt_object_cpu.shape}")
    else:
        # Handle numpy array
        if isinstance(gt_object_cpu, np.ndarray):
            if gt_object_cpu.ndim == 3:
                if gt_object_cpu.shape[2] == 3:  # HxWx3 color
                    # Ensure correct value range
                    if gt_object_cpu.max() <= 1.0:  # range [0,1]
                        gt_object_to_save = (gt_object_cpu * 255).astype(np.uint8)
                    else:  # range [0,255]
                        gt_object_to_save = gt_object_cpu.astype(np.uint8)
                    Image.fromarray(gt_object_to_save).save(img_path)
                elif gt_object_cpu.shape[2] == 1:  # HxWx1 grayscale
                    if gt_object_cpu.max() <= 1.0:  # range [0,1]
                        gt_object_to_save = (gt_object_cpu * 255).astype(np.uint8)
                    else:  # range [0,255]
                        gt_object_to_save = gt_object_cpu.astype(np.uint8)
                    Image.fromarray(gt_object_to_save.squeeze(), mode='L').save(img_path)
                else:
                    raise ValueError(f"Unexpected array shape: {gt_object_cpu.shape}")
            elif gt_object_cpu.ndim == 2:  # HxW grayscale
                # Ensure correct value range
                if gt_object_cpu.max() <= 1.0:  # range [0,1]
                    gt_object_to_save = (gt_object_cpu * 255).astype(np.uint8)
                else:  # range [0,255]
                    gt_object_to_save = gt_object_cpu.astype(np.uint8)
                Image.fromarray(gt_object_to_save, mode='L').save(img_path)
            else:
                raise ValueError(f"Unexpected array shape: {gt_object_cpu.shape}")
        else:
            raise TypeError(f"Unsupported gt_object type: {type(gt_object_cpu)}")

    # Ensure gs_id is numpy array
    if isinstance(gs_id, torch.Tensor):
        gs_id = gs_id.detach().cpu()
        gs_id_np = gs_id.numpy()
    else:
        gs_id_np = gs_id

    # Get gt_object dimensions
    if isinstance(gt_object_cpu, torch.Tensor):
        if gt_object_cpu.ndim == 3:
            # CxHxW format
            gt_h, gt_w = gt_object_cpu.shape[1], gt_object_cpu.shape[2]
        elif gt_object_cpu.ndim == 2:
            # HxW format
            gt_h, gt_w = gt_object_cpu.shape
        else:
            raise ValueError(f"Unsupported gt_object shape: {gt_object_cpu.shape}")
    else:  # numpy array
        if gt_object_cpu.ndim == 3:
            if gt_object_cpu.shape[2] in [1, 3]:  # HxWxC
                gt_h, gt_w = gt_object_cpu.shape[0], gt_object_cpu.shape[1]
            elif gt_object_cpu.shape[0] in [1, 3]:  # CxHxW
                gt_h, gt_w = gt_object_cpu.shape[1], gt_object_cpu.shape[2]
            else:
                raise ValueError(f"Unsupported gt_object shape: {gt_object_cpu.shape}")
        elif gt_object_cpu.ndim == 2:
            gt_h, gt_w = gt_object_cpu.shape
        else:
            raise ValueError(f"Unsupported gt_object shape: {gt_object_cpu.shape}")

    # Get gs_id dimensions
    gs_h, gs_w = gs_id_np.shape

    print(f"gt_object shape: {gt_object_cpu.shape}, size: {gt_h}x{gt_w}")
    print(f"gs_id shape: {gs_id_np.shape}, size: {gs_h}x{gs_w}")

    # Check size mismatch, resize gs_id if needed
    if (gt_h != gs_h) or (gt_w != gs_w):
        print(f"Warning: Size mismatch! Resizing gs_id from {gs_h}x{gs_w} to {gt_h}x{gt_w}")

        import torch.nn.functional as F

        if isinstance(gs_id, torch.Tensor):
            # If gs_id is tensor, use torch interpolation
            gs_id_resized = F.interpolate(
                gs_id.unsqueeze(0).unsqueeze(0).float(),  # add batch and channel dims
                size=(gt_h, gt_w),
                mode='nearest'
            ).squeeze().long()
            gs_id_np = gs_id_resized.numpy()
        else:
            # If gs_id is numpy, use PIL interpolation
            gs_id_pil = Image.fromarray(gs_id_np.astype(np.uint32))
            gs_id_pil = gs_id_pil.resize((gt_w, gt_h), Image.NEAREST)
            gs_id_np = np.array(gs_id_pil)

    # Create mask from gt_object: find non-zero semantic regions
    # Use raw gt_object values to create mask
    if isinstance(gt_object_cpu, torch.Tensor):
        # Convert to numpy for processing
        if gt_object_cpu.ndim == 3 and gt_object_cpu.shape[0] == 3:  # 3xHxW
            gt_object_np = gt_object_cpu.permute(1, 2, 0).numpy()  # HxWx3
            mask = np.any(gt_object_np > 0, axis=2)
        elif gt_object_cpu.ndim == 3 and gt_object_cpu.shape[0] == 1:  # 1xHxW
            gt_object_np = gt_object_cpu.squeeze(0).numpy()  # HxW
            mask = gt_object_np > 0
        elif gt_object_cpu.ndim == 2:  # HxW
            gt_object_np = gt_object_cpu.numpy()
            mask = gt_object_np > 0
        else:
            raise ValueError(f"Unsupported tensor shape for mask: {gt_object_cpu.shape}")
    else:
        gt_object_np = gt_object_cpu

        # Create mask
        if gt_object_np.ndim == 3:  # HxWx3 or HxWx1
            if gt_object_np.shape[2] == 3:  # RGB
                # Check if all pixel channels are zero
                mask = np.any(gt_object_np > 0, axis=2)
            else:  # single channel
                mask = gt_object_np.squeeze() > 0
        elif gt_object_np.ndim == 2:  # HxW
            mask = gt_object_np > 0
        else:
            raise ValueError(f"Unsupported array shape for mask: {gt_object_np.shape}")

    # Double-check dimensions
    if mask.shape != gs_id_np.shape:
        raise ValueError(f"Size mismatch after resizing: mask {mask.shape}, gs_id {gs_id_np.shape}")

    # Visualize gs_id: assign colors to each gs_id, only in masked region
    masked_gs_ids = gs_id_np[mask]
    unique_ids = np.unique(masked_gs_ids)

    # Generate color for each unique gs_id
    def id_to_tggb_color(id):
        np.random.seed(int(id))
        return np.random.randint(0, 256, size=3, dtype=np.uint8)

    color_map = {i: id_to_tggb_color(i) for i in unique_ids}

    # Set background color to #1D2D5B
    background_color = np.array([0x1D, 0x2D, 0x5B], dtype=np.uint8)  # RGB: 29, 45, 91

    # Create color image with background
    gs_color = np.full((gs_id_np.shape[0], gs_id_np.shape[1], 3),
                      background_color,
                      dtype=np.uint8)

    # Fill mask region with random colors
    for i in unique_ids:
        # Find positions matching this gs_id within mask
        pos_mask = (gs_id_np == i) & mask
        if np.any(pos_mask):
            gs_color[pos_mask] = color_map[i]

    # Save gs_id visualization
    gsid_img_path = os.path.join(save_dir, f"gsid_{step:05d}.png")

    # Use PIL for direct color control
    Image.fromarray(gs_color, mode='RGB').save(gsid_img_path)

    print(f"Saved to {save_dir}:")
    print(f"  - Image: {img_path}")
    print(f"  - GS ID visualization: {gsid_img_path}")
    print(f"  - Unique GS IDs in masked area: {len(unique_ids)}")

    return gs_color, mask

def normalize_depth(depth_tensor: torch.Tensor, eps=1e-8) -> torch.Tensor:
    depth = depth_tensor.squeeze(0)  # (H, W)
    depth_min = depth.amin()
    depth_max = depth.amax()
    norm_depth = (depth - depth_min) / (depth_max - depth_min + eps)
    return norm_depth.clamp(0.0, 1.0)  # shape: (H, W)

def save_tensor_as_image(tensor: torch.Tensor, path: str):
    """
    Save CUDA Tensor as image (supports 1-channel or 3-channel, float [0,1] format).

    Args:
        tensor: (C, H, W) shape, CUDA or CPU Tensor
        path: save path, e.g. 'output.png'
    """
    tensor = tensor.detach().cpu()

    if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
        tensor = (tensor.clamp(0, 1) * 255).to(torch.uint8)

    img_np = tensor.numpy()

    if img_np.ndim == 3:
        img_np = np.transpose(img_np, (1, 2, 0))  # C,H,W -> H,W,C

        # If 3-channel (RGB), convert to BGR for OpenCV
        if img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if img_np.shape[-1] == 1:
        img_np = img_np[:, :, 0]

    cv2.imwrite(path, img_np)

def fast_dilate_contour(gt_contour, kernel_size=3, iterations=1):
    """
    gt_contour: [C,H,W] or [H,W] 0/1 mask, torch.Tensor
    """
    # Add batch dimension
    if gt_contour.ndim == 3:  # [C,H,W]
        x = gt_contour.unsqueeze(0).float()  # [1,C,H,W]
    elif gt_contour.ndim == 2:  # [H,W]
        x = gt_contour.unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
    else:
        raise ValueError("gt_contour must be [H,W] or [C,H,W]")

    device = gt_contour.device
    x = x.to(device)

    for _ in range(iterations):
        x = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    return x.squeeze(0)  # [C,H,W] or [1,H,W], preserves channel count

def detect_concave_regions(render_depth: torch.Tensor,
                           gt_depth: torch.Tensor,
                           window_size: int = 15,
                           threshold: float = 0.02,
                           dilation_kernel_size: int = 5):
    """
    Detect concave regions in render_depth, remove concave regions in gt_depth.
    Dilate + fill gt_mask to prevent render_mask from leaking into GT regions.

    Args:
        render_depth: [H,W] torch.Tensor
        gt_depth: [H,W] torch.Tensor
        window_size: local mean window size
        threshold: concavity depth threshold
        dilation_kernel_size: dilation kernel size (odd)

    Returns:
        concave_mask: [1,1,H,W] torch.bool
        gt_mask: [1,1,H,W] torch.bool, dilated and filled
    """
    def ensure_4d(depth: torch.Tensor) -> torch.Tensor:
        """Convert depth map to [1,1,H,W] 4D shape"""
        if depth.ndim == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.ndim == 3:
            depth = depth.unsqueeze(0)
        elif depth.ndim == 4:
            pass
        else:
            raise ValueError(f"Unsupported depth shape: {depth.shape}")
        return depth

    render_depth = ensure_4d(render_depth)
    gt_depth = ensure_4d(gt_depth)

    # Local mean convolution
    def local_mean_mask(depth):
        kernel = torch.ones((1,1,window_size,window_size), device=depth.device) / (window_size**2)
        padding = window_size // 2
        depth_padded = F.pad(depth, [padding]*4, mode='replicate')
        mean = F.conv2d(depth_padded, kernel)
        mask = (mean - depth) > threshold
        return mask

    # Compute concave masks for render and gt
    render_mask = local_mean_mask(render_depth)
    gt_mask = local_mean_mask(gt_depth)

    # --------------------
    # Dilate gt_mask
    # --------------------
    dilation_kernel = torch.ones((1, 1, dilation_kernel_size, dilation_kernel_size), device=gt_mask.device)
    padding = dilation_kernel_size // 2
    gt_mask_float = gt_mask.float()
    gt_mask_dilated = F.conv2d(F.pad(gt_mask_float, [padding]*4, mode='replicate'), dilation_kernel)
    gt_mask_dilated = gt_mask_dilated > 0  # dilated gt_mask

    # --------------------
    # Fill internal holes (approximate, using max pooling)
    # --------------------
    gt_mask_filled = F.max_pool2d(gt_mask_dilated.float(), kernel_size=dilation_kernel_size, stride=1, padding=padding)
    gt_mask_filled = gt_mask_filled > 0

    # Remove gt_mask regions from render_mask
    concave_mask = render_mask & (~gt_mask)

    return concave_mask, gt_mask


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_depth_for_log = 0.0
    depth_loss = 0.0

    # Init DashGaussian scheduler
    scheduler = TrainingScheduler(opt, pipe, gaussians,
                                    [cam.original_image for cam in scene.getTrainCameras()])
    render_scale = scheduler.get_res_scale(1)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    train_loss_l1_edge = []

    def plot_loss_curve(loss_values, save_path=None, title="L1 Edge Loss Curve"):
        """
        Plot and save loss curve.

        Args:
            loss_values: list of loss values per epoch
            save_path: save path (e.g. "results/loss_curve.png"), None = don't save
            title: plot title
        """
        plt.figure(figsize=(10, 5))
        plt.plot(loss_values, label="L1 Edge Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.grid(True)
        plt.legend()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # auto-create dir
            plt.savefig(save_path, bbox_inches='tight', dpi=300)  # high-res save
            print(f"Loss curve saved to: {save_path}")

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image = viewpoint_cam.original_image.cuda()

        # Depth
        gt_object = viewpoint_cam.objects.cuda()
        gt_depth = normalize_depth(viewpoint_cam.depths.cuda())
        gt_contour = viewpoint_cam.contours.cuda() > 0
        gt_contour_dilated = fast_dilate_contour(gt_contour, kernel_size=3, iterations=3)

        # Down-sample
        if render_scale > 1:
            gt_image = lanczos_resample(gt_image.permute(1, 2, 0), scale_factor=render_scale).permute(2, 0, 1)
            gt_contour_dilated = lanczos_resample(gt_contour_dilated.repeat(3,1,1).permute(1,2,0), scale_factor=render_scale).permute(2,0,1)[0:1]
            gt_depth = lanczos_resample(gt_depth.unsqueeze(0).repeat(3,1,1).permute(1,2,0), scale_factor=render_scale).permute(2,0,1)[0:1]

        # Render at resized resolution
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, render_size=gt_image.shape[-2:])
        image, viewspace_point_tensor, visibility_filter, radii, gs_id = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["gs_id"]

        depth = render_pkg["surf_depth"]
        depth_normal = normalize_depth(depth)
        gt_depth_normal = normalize_depth(gt_depth)

        contour_mask = gt_contour_dilated.squeeze(0).bool()
        contour_ids = gs_id[contour_mask].unique().long()

        # edge_mask, gt_edge_mask = detect_concave_regions(depth_normal, gt_depth_normal)

        # Save as PNG
        if iteration >1000 and iteration % 1000 == 0:
            # save_image_and_gsid(gt_object, gs_id, iteration, r"/extdatashare/dir_liang0/3DGS_refine_project/2d-gaussian-splatting_gsid_seg_down_depth_plus_contour/output/gsid_small_corn")
            save_image_and_gsid(gt_object, gs_id, iteration, r"/extdatashare/dir_liang0/3DGS_refine_project/2d-gaussian-splatting_gsid_seg_down_depth_plus_contour/output/gsid_big_wheat")
        #     cv2.imwrite(f"/extdatashare/dir_liang0/3DGS_refine_project/2d-gaussian-splatting_gsid_seg_down_depth_plus/vis/depth_image_{iteration}.png", (depth_normal*255).detach().cpu().numpy())
        #     save_tensor_as_image(edge_mask.squeeze().float(),f"/extdatashare/dir_liang0/3DGS_refine_project/2d-gaussian-splatting_gsid_seg_down_depth_plus/vis/edge_{iteration}.png")
        #     save_tensor_as_image(gt_edge_mask.squeeze().float(),f"/extdatashare/dir_liang0/3DGS_refine_project/2d-gaussian-splatting_gsid_seg_down_depth_plus/vis/gt_edge_{iteration}.png")

        # Ll1 = l1_loss(image, gt_image)
        Ll1 = l1_loss_edge(image, gt_image,contour_mask)

        if iteration < 10000:

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        else:
            depth_loss, logs = smooth_transition_loss(depth_normal, gt_contour)
            train_loss_l1_edge.append(logs["smooth_transition_loss"])
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + (1.0 - opt.lambda_dssim) * depth_loss

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss

        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_depth_for_log = 0.4 * depth_loss + 0.6 * ema_depth_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "depth": f"{ema_depth_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:

                    # contour_ids_opt = gs_id[edge_mask.squeeze().squeeze()].unique()
                    # gaussians.attenuate_opacity_by_contours(contour_ids_opt)

                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)

                    # Apply DashGaussian primitive scheduler to control densification.
                    densify_rate = scheduler.get_densify_rate(iteration, gaussians.get_xyz.shape[0], render_scale)
                    momentum_add = gaussians.prune_and_densify_edge(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                               size_threshold, radii, contour_ids, densify_rate=densify_rate)
                    # Update max_n_gaussian
                    scheduler.update_momentum(momentum_add)
                    # Update render scale based on the DashGaussian resolution scheduler.
                    render_scale = scheduler.get_res_scale(iteration)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTrainCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=60168)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
