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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

from PIL import Image
import os
import torchvision.transforms.functional as TF
import numpy as np
import cv2

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def laplacian(depth):
    """2D Laplacian kernel operation"""
    return (
        -4 * depth +
        torch.roll(depth, 1, dims=0) +
        torch.roll(depth, -1, dims=0) +
        torch.roll(depth, 1, dims=1) +
        torch.roll(depth, -1, dims=1)
    )

def extract_contour_line_mask(contour_rgb):
    """
    Input:
        - [3, H, W] or [H, W, 3] RGB contour map (white = boundary)
        - [1, H, W] or [H, W] grayscale contour map (255 = boundary)
    Output: [H, W] float mask
    """
    if contour_rgb.ndim == 3:
        if contour_rgb.shape[0] == 3:  # CHW format RGB
            is_white = (contour_rgb >= 0.5).float()
            return (is_white.sum(dim=0) == 3).float()
        elif contour_rgb.shape[2] == 3:  # HWC format RGB
            contour_rgb = contour_rgb.permute(2, 0, 1)
            is_white = (contour_rgb >= 0.5).float()
            return (is_white.sum(dim=0) == 3).float()
        elif contour_rgb.shape[0] == 1:  # [1, H, W] grayscale
            return (contour_rgb.squeeze(0) >= 0.5).float()
    elif contour_rgb.ndim == 2:  # [H, W] grayscale
        return (contour_rgb >= 0.5).float()

    raise ValueError(f"Unsupported contour shape: {contour_rgb.shape}")

def dilate_contour_mask(mask, kernel_size=3, iterations=1):
    """Morphological dilation (for narrow band regions)"""
    for _ in range(iterations):
        mask = F.max_pool2d(
            mask.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1, padding=kernel_size // 2
        ).squeeze()
    return mask

def smooth_transition_loss(D_pred, contour_mask_rgb, weight=1.0):
    """
    Self-supervised boundary smoothness loss (no GT required)
    D_pred: [H, W] float
    contour_mask_rgb: [3, H, W] or [H, W, 3]
    """
    with torch.no_grad():
        contour_mask = extract_contour_line_mask(contour_mask_rgb)  # [H, W]
        band_mask = dilate_contour_mask(contour_mask, kernel_size=3, iterations=1)

    lap = laplacian(D_pred)
    loss = (lap ** 2 * band_mask).sum() / (band_mask.sum() + 1e-8)
    return weight * loss, {
        "smooth_transition_loss": loss.item()
    }

def l1_loss_edge(network_output, gt, contour_bool=None, edge_weight=3.0):
    """
    network_output: predicted (B,C,H,W)
    gt: ground truth (B,C,H,W)
    contour_bool: binary contour map (B,1,H,W), 1 = boundary
    """
    diff = torch.abs(network_output - gt)

    if contour_bool is not None:
        # Convert to float mask
        contour_mask = contour_bool.float()

        edge_loss = (diff * contour_mask * edge_weight).sum()
        non_edge_loss = (diff * (1.0 - contour_mask)).sum()

        num_pixels = diff.numel()
        return (edge_loss + non_edge_loss) / num_pixels
    else:
        return diff.mean()
