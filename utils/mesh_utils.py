#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
import trimesh

###
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from scipy import stats

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(),
            fx = intrins[0,0].item(),
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj

def update_semantic_per_id(
    id_tensor: torch.Tensor,
    semantic_mask: torch.Tensor,
    full_semantic_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Update the semantic tensor of given length, assigning dominant semantic label
    to each ID that appears.

    Args:
        id_tensor: (H, W) tensor of integer IDs, on CUDA
        semantic_mask: (H, W) tensor of integer semantic labels, on CUDA
        full_semantic_tensor: (total_ids,) tensor, pre-initialized semantic labels, on CUDA

    Returns:
        Updated full_semantic_tensor (in-place also OK).
    """
    assert id_tensor.shape == semantic_mask.shape, "Shape mismatch"
    assert id_tensor.is_cuda and semantic_mask.is_cuda and full_semantic_tensor.is_cuda, "Tensors must be on CUDA"
    assert full_semantic_tensor.dim() == 1, "full_semantic_tensor must be 1D"

    flat_ids = id_tensor.view(-1).long()
    flat_sem = semantic_mask.view(-1).long()

    num_ids = int(flat_ids.max().item()) + 1
    num_classes = int(flat_sem.max().item()) + 1

    # 2D histogram: pixel count per (ID, class)
    indices = flat_ids * num_classes + flat_sem
    bincount = torch.bincount(indices, minlength=num_ids * num_classes)
    counts = bincount.view(num_ids, num_classes)

    # Dominant class per ID
    semantics_per_id = counts.argmax(dim=1)

    # Update existing IDs in full_semantic_tensor
    full_semantic_tensor[:semantics_per_id.shape[0]] = semantics_per_id
    return full_semantic_tensor

# def accumulate_semantic_counts(
#     id_tensor: torch.Tensor,
#     semantic_mask: torch.Tensor,
#     counts_total: torch.Tensor,i
# ) -> torch.Tensor:
#     """
#     Accumulate pixel counts per (ID, semantic class) into counts_total.
#
#     Args:
#         id_tensor: (H, W) Gaussian ID per pixel
#         semantic_mask: (H, W) or (3, H, W) or (H, W, 3), semantic label map
#         counts_total: (num_gaussians, num_classes) statistics matrix
#
#     Returns:
#         Updated counts_total (in-place)
#     """
#
#     # --- Auto-detect semantic_mask shape ---
#     if semantic_mask.ndim == 3:
#         if semantic_mask.shape[0] == 3:
#             # (3, H, W) -> (H, W, 3), then grayscale
#             semantic_mask = semantic_mask.permute(1, 2, 0)
#             semantic_mask = (
#                 0.2989 * semantic_mask[..., 0] +
#                 0.5870 * semantic_mask[..., 1] +
#                 0.1140 * semantic_mask[..., 2]
#             ).round().long()
#
#         elif semantic_mask.shape[0] == 1:
#             # (1, H, W) -> (H, W), squeeze
#             semantic_mask = semantic_mask.squeeze(0).long()
#
#         else:
#             # (H, W, 3) -> grayscale label
#             semantic_mask = (
#                 0.2989 * semantic_mask[..., 0] +
#                 0.5870 * semantic_mask[..., 1] +
#                 0.1140 * semantic_mask[..., 2]
#             ).round().long()
#
#     flat_ids = id_tensor.view(-1).long()
#     flat_sem = semantic_mask.view(-1).long()
#
#     num_classes = counts_total.shape[1]
#     indices = flat_ids * num_classes + flat_sem
#     bincount = torch.bincount(indices, minlength=counts_total.shape[0] * num_classes)
#     counts = bincount.view(-1, num_classes)
#
#     counts_total[:counts.shape[0]] += counts
#     return counts_total


def accumulate_semantic_counts(
    id_tensor: torch.Tensor,
    semantic_mask: torch.Tensor,
    counts_total: torch.Tensor,
    step: int = 0,
    save_dir: str = r"/extdatashare/dir_liang0/3DGS_refine_project/2d-gaussian-splatting_gsid_seg_down_depth_plus_contour/output/a302a7e3-8/point_cloud",
) -> torch.Tensor:
    """
    Accumulate pixel counts per (ID, semantic class) into counts_total,
    and optionally visualize semantic distribution.

    Args:
        id_tensor: (H, W) Gaussian ID per pixel
        semantic_mask: (H, W) or (3, H, W) or (H, W, 3), semantic label map
        counts_total: (num_gaussians, num_classes) statistics matrix
        save_dir: directory to save visualization (optional)
        step: current step number (optional)

    Returns:
        Updated counts_total (in-place)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.ticker import MaxNLocator

    # --- Auto-detect semantic_mask shape ---
    if semantic_mask.ndim == 3:
        if semantic_mask.shape[0] == 3:
            # (3, H, W) -> (H, W, 3), then grayscale
            semantic_mask = semantic_mask.permute(1, 2, 0)
            semantic_mask = (
                0.2989 * semantic_mask[..., 0] +
                0.5870 * semantic_mask[..., 1] +
                0.1140 * semantic_mask[..., 2]
            ).round().long()

        elif semantic_mask.shape[0] == 1:
            # (1, H, W) -> (H, W), squeeze
            semantic_mask = semantic_mask.squeeze(0).long()

        else:
            # (H, W, 3) -> grayscale label
            semantic_mask = (
                0.2989 * semantic_mask[..., 0] +
                0.5870 * semantic_mask[..., 1] +
                0.1140 * semantic_mask[..., 2]
            ).round().long()

    flat_ids = id_tensor.view(-1).long()
    flat_sem = semantic_mask.view(-1).long()

    num_classes = counts_total.shape[1]
    indices = flat_ids * num_classes + flat_sem
    bincount = torch.bincount(indices, minlength=counts_total.shape[0] * num_classes)
    counts = bincount.view(-1, num_classes)

    counts_total[:counts.shape[0]] += counts

    # --- Visualization ---
    save_dir = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # Create visualization figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Semantic Distribution Analysis - Step {step}', fontsize=16)

        # 1. Semantic distribution per Gaussian (top 50)
        num_gaussians_to_show = min(50, counts_total.shape[0])
        ax1 = axes[0, 0]
        x = np.arange(num_gaussians_to_show)
        width = 0.8 / num_classes

        for class_idx in range(num_classes):
            ax1.bar(x + width * class_idx,
                   counts_total[:num_gaussians_to_show, class_idx].cpu().numpy(),
                   width=width,
                   label=f'Class {class_idx}')

        ax1.set_xlabel('Gaussian ID')
        ax1.set_ylabel('Pixel Count')
        ax1.set_title('Semantic Distribution per Gaussian (Top 50)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. Total pixel count per semantic class
        ax2 = axes[0, 1]
        class_totals = counts_total.sum(dim=0).cpu().numpy()
        bars = ax2.bar(range(num_classes), class_totals)
        ax2.set_xlabel('Semantic Class')
        ax2.set_ylabel('Total Pixel Count')
        ax2.set_title('Total Pixels per Semantic Class')
        ax2.set_xticks(range(num_classes))
        ax2.grid(True, alpha=0.3)

        # Add values on bars
        for i, v in enumerate(class_totals):
            ax2.text(i, v, str(int(v)), ha='center', va='bottom')

        # 3. Dominant semantic class per Gaussian
        ax3 = axes[1, 0]
        dominant_classes = counts_total.argmax(dim=1).cpu().numpy()
        unique_classes, class_counts = np.unique(dominant_classes, return_counts=True)

        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
        wedges, texts, autotexts = ax3.pie(class_counts,
                                          labels=[f'Class {c}' for c in unique_classes],
                                          colors=colors,
                                          autopct='%1.1f%%',
                                          startangle=90)
        ax3.set_title('Dominant Semantic Class per Gaussian')

        # 4. Gaussian count distribution by total pixels
        ax4 = axes[1, 1]
        gaussian_totals = counts_total.sum(dim=1).cpu().numpy()

        # Histogram by pixel count ranges
        bins = [0, 10, 50, 100, 500, 1000, 5000, float('inf')]
        bin_labels = ['0-10', '11-50', '51-100', '101-500', '501-1000', '1001-5000', '5000+']

        hist_data = np.digitize(gaussian_totals, bins)
        hist_counts = np.bincount(hist_data, minlength=len(bins)+1)[1:-1]

        bars = ax4.bar(range(len(bin_labels)), hist_counts)
        ax4.set_xlabel('Pixel Count Range')
        ax4.set_ylabel('Number of Gaussians')
        ax4.set_title('Gaussian Distribution by Pixel Count')
        ax4.set_xticks(range(len(bin_labels)))
        ax4.set_xticklabels(bin_labels, rotation=45)
        ax4.grid(True, alpha=0.3)

        # Add values on bars
        for i, v in enumerate(hist_counts):
            ax4.text(i, v, str(int(v)), ha='center', va='bottom')

        plt.tight_layout()

        # Save figure
        save_path = os.path.join(save_dir, f'semantic_distribution_step_{step:05d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved semantic distribution visualization to: {save_path}")

        # Print statistics
        print(f"\n=== Semantic Distribution Statistics (Step {step}) ===")
        print(f"Total Gaussians: {counts_total.shape[0]}")
        print(f"Number of classes: {num_classes}")
        print(f"Total pixels processed: {int(counts_total.sum().item())}")
        print(f"Average pixels per Gaussian: {counts_total.sum().item() / counts_total.shape[0]:.1f}")

        for class_idx in range(num_classes):
            class_total = class_totals[class_idx]
            if class_total > 0:
                print(f"  Class {class_idx}: {class_total} pixels ({class_total/class_totals.sum()*100:.1f}%)")

    return counts_total


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtrator.export_mesh_bounded(...)
        """
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        # self.alphamaps = []
        self.rgbmaps = []
        # self.normals = []
        # self.depth_normals = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def remove_outlier_semantics(self, gaussians, eps=0.1, min_samples=10):
        """
        Set semantic outliers' object label to 0 instead of deleting them.

        Args:
            gaussians: object with _xyz [N, 3] and _object [N]
            eps: DBSCAN spatial distance threshold
            min_samples: DBSCAN min samples per cluster
        """
        xyz = gaussians._xyz.detach().cpu().numpy()         # [N, 3]
        labels = gaussians._object.detach().cpu().numpy()   # [N]
        N = xyz.shape[0]

        # Final updated semantic labels
        new_labels = labels.copy()

        for sem_id in np.unique(labels):
            if sem_id == 0:
                continue  # skip background points

            sem_mask = (labels == sem_id)
            sem_xyz = xyz[sem_mask]
            sem_indices = np.where(sem_mask)[0]

            if len(sem_indices) < min_samples:
                new_labels[sem_indices] = 0
                continue

            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(sem_xyz)
            cluster_labels = clustering.labels_

            # -1 indicates noise (outlier)
            if np.all(cluster_labels == -1):
                new_labels[sem_indices] = 0
                continue

            # Keep the main cluster (largest by sample count)
            major_cluster_id = np.argmax(np.bincount(cluster_labels[cluster_labels >= 0]))
            keep_local = (cluster_labels == major_cluster_id)
            drop_local = ~keep_local

            new_labels[sem_indices[drop_local]] = 0  # non-main cluster -> set to background

        # Update semantic labels
        gaussians._object = torch.from_numpy(new_labels).to(gaussians._object.device).long()
        return gaussians

    @torch.no_grad()
    def fill_semantic_with_knn_and_loose_hull(self, gaussians, k=5, loosen_eps=0.01):
        """
        Semantic propagation using KNN + relaxed convex hull constraints,
        avoiding semantic leakage while allowing edge completion.

        Args:
            gaussians: object with _xyz [N, 3] and _object [N]
            k: number of nearest neighbors
            loosen_eps: convex hull relaxation (meters), suggested 0.01~0.05
        """
        xyz = gaussians._xyz.detach().cpu().numpy()         # [N, 3]
        labels = gaussians._object.detach().cpu().numpy()   # [N]

        mask_known = labels != 0
        mask_unknown = labels == 0

        if mask_unknown.sum() == 0:
            print("All points already have semantics, no filling needed")
            return gaussians

        xyz_unknown = xyz[mask_unknown]
        filled_labels = np.zeros(len(xyz_unknown), dtype=np.int32)

        unique_sem_ids = np.unique(labels[mask_known])
        for sem_id in unique_sem_ids:
            class_mask = (labels == sem_id)
            xyz_class = xyz[class_mask]

            if len(xyz_class) < 4:
                continue

            # ---- Relaxation: add random perturbation to semantic points ----
            noise = np.random.normal(scale=loosen_eps, size=xyz_class.shape)
            xyz_loosen = xyz_class + noise

            try:
                hull = Delaunay(xyz_loosen)
            except:
                continue

            inside_mask = hull.find_simplex(xyz_unknown) >= 0
            idxs_inside = np.where(inside_mask)[0]

            if len(idxs_inside) == 0:
                continue

            knn = NearestNeighbors(n_neighbors=min(k, len(xyz_class)))
            knn.fit(xyz_class)
            sub_xyz_unknown = xyz_unknown[inside_mask]
            _, indices = knn.kneighbors(sub_xyz_unknown)

            neighbor_labels_all = labels[class_mask]
            neighbor_labels_batch = neighbor_labels_all[indices]  # shape (N, k)
            modes, counts = stats.mode(neighbor_labels_batch, axis=1)
            filled_labels[idxs_inside] = modes.flatten()

        # Update gaussians
        new_labels = labels.copy()
        new_labels[mask_unknown] = filled_labels
        gaussians._object = torch.from_numpy(new_labels).to(gaussians._object.device).long()

        print(f"KNN + relaxed hull fill complete, filled: {(filled_labels > 0).sum()} points")
        return gaussians


    # @torch.no_grad()
    # def gs_id(self, viewpoint_stack):
    #     """
    #     Assign dominant semantic class to each Gaussian from multi-view accumulation.
    #     """
    #     self.clean()
    #     self.viewpoint_stack = viewpoint_stack
    #
    #     num_gaussians = self.gaussians._xyz.shape[0]
    #
    #     # Assume semantic classes won't exceed 100 (adjust as needed)
    #     num_classes = 100
    #
    #     # Initialize cumulative histogram
    #     counts_total = torch.zeros((num_gaussians, num_classes), device='cuda')
    #     for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="gs_object"):
    #         # Get render result
    #         render_pkg = self.render(viewpoint_cam, self.gaussians)
    #
    #         # Semantic label per pixel in current view (from segmentation)
    #         objects_mask = viewpoint_cam.objects.cuda().long()  # (H, W)
    #         gs_id = render_pkg["gs_id"]                         # (H, W)
    #
    #         # Accumulate semantic histogram for current frame
    #         counts_total = accumulate_semantic_counts(gs_id, objects_mask, counts_total,i)
    #     # Final dominant semantic label per Gaussian
    #     _object = counts_total.argmax(dim=1)
    #     self.gaussians._object = _object.clone()
    #     self.remove_outlier_semantics(self.gaussians)
    #     # self.fill_semantic_with_knn_and_loose_hull(self.gaussians)

    @torch.no_grad()
    def gs_id(self, viewpoint_stack):

        """
        Assign dominant semantic class to each Gaussian from multi-view accumulation.
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack

        num_gaussians = self.gaussians._xyz.shape[0]

        # Assume semantic classes won't exceed 1000 (adjust as needed)
        num_classes = 1000

        # Initialize cumulative histogram
        counts_total = torch.zeros((num_gaussians, num_classes), device='cuda')

        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="gs_object"):
            # Get render result
            render_pkg = self.render(viewpoint_cam, self.gaussians)

            # Semantic label per pixel in current view (from segmentation)
            objects_mask = viewpoint_cam.objects.cuda().long()  # (H, W)
            gs_id = render_pkg["gs_id"]                         # (H, W)

            # Accumulate semantic histogram for current frame
            counts_total = accumulate_semantic_counts(gs_id, objects_mask, counts_total, i)

        # Final dominant semantic label per Gaussian
        _object = counts_total.argmax(dim=1)
        self.gaussians._object = _object.clone()
        self.remove_outlier_semantics(self.gaussians)
        # self.fill_semantic_with_knn_and_loose_hull(self.gaussians)

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            rgb = render_pkg['render']
            alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
            depth = render_pkg['surf_depth']
            depth_normal = render_pkg['surf_normal']
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            # self.alphamaps.append(alpha.cpu())
            # self.normals.append(normal.cpu())
            # self.depth_normals.append(depth_normal.cpu())

        # self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        # self.depthmaps = torch.stack(self.depthmaps, dim=0)
        # self.alphamaps = torch.stack(self.alphamaps, dim=0)
        # self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.estimate_bounding_sphere()



    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.

        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]

            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1,2,0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets.
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp

            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )

        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        def save_depth_percentile(depth_tensor, vis_path, idx, low=2, high=98):
            depth = depth_tensor[0].cpu().numpy().astype(np.float32)

            # Compute percentile thresholds
            vmin, vmax = np.percentile(depth, (low, high))

            # Clip and normalize
            depth = np.clip(depth, vmin, vmax)
            depth = (depth - vmin) / (vmax - vmin + 1e-6)

            save_img_u8(depth, os.path.join(vis_path, f'depth_{idx:05d}.png'))
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        gt_depths_path = os.path.join(path, "gt_depth")
        vis_path = os.path.join(path, "vis")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        os.makedirs(gt_depths_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            gt_depth = viewpoint_cam.depths[0, :, :]
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(gt_depth.cpu().numpy(), os.path.join(gt_depths_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            # save_img_u8(((self.depthmaps[idx][0].cpu() - self.depthmaps[idx][0].cpu().min()) / (self.depthmaps[idx][0].cpu().max() - self.depthmaps[idx][0].cpu().min() + 1e-6)).numpy(),os.path.join(vis_path, 'depth_{0:05d}.png'.format(idx)))
            save_depth_percentile(self.depthmaps[idx], vis_path, idx)
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            # save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            # save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            # save_img_u8((self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5), os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))
