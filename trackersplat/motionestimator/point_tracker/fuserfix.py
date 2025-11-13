import torch
from tqdm import tqdm
from typing import List
from gaussian_splatting import Camera
from trackersplat.utils import ILS_RotationScale, ISVD_Mean3D
from trackersplat.utils.featurefusion import feature_fusion
from .abc import Motion
from .fuser import BaseMotionFuser


class DetectFixMotionFuser(BaseMotionFuser):
    def __init__(self, *args, motion_threshold=0.2, **kwargs):
        '''
        Base class for motion fusion.
        `motion_threshold` is the threshold for detecting static gaussians, unit is pixel.
        '''
        super().__init__(*args, **kwargs)
        self.motion_threshold = motion_threshold

    def _count_fixed_pixels(self, camera: Camera, track: torch.Tensor):
        motion_threshold = self.motion_threshold
        x = torch.arange(camera.image_width, dtype=torch.float, device=self.device)
        y = torch.arange(camera.image_height, dtype=torch.float, device=self.device)
        xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)
        dist = torch.linalg.norm(xy - track, dim=-1)
        is_static = (dist < motion_threshold).type(torch.float).unsqueeze(-1)
        _, features, features_alpha, pixhit, features_idx = feature_fusion(self.gaussians, camera, is_static, 0)
        return features.squeeze(-1), features_alpha, pixhit, features_idx

    def compute_fixed_mask_and_weights(self, fixed_sum, fixed_alpha, fixed_pixhits, viewhits):
        '''Overload this method to make your own mask and weights'''
        # a gaussian should be fixed if it hit by more than 9 pixels and has more than 90% of its pixels fixed in at least 2 views
        hits_in_view_threshold, n_views_threshold, avg_in_view_threshold, alpha_rel_threshold, alpha_abs_threshold = 3*3, 2, 0.9, 0.5, 1.0
        hits_in_view_mask = fixed_pixhits > hits_in_view_threshold
        fixed_avg = torch.zeros_like(fixed_sum)
        fixed_avg[fixed_alpha > 1e-12] = fixed_sum[fixed_alpha > 1e-12] / fixed_alpha[fixed_alpha > 1e-12]
        fixed_avg_mask = fixed_avg > avg_in_view_threshold
        alpha_mask = fixed_alpha >= (fixed_alpha.max(-1).values.unsqueeze(-1) * alpha_rel_threshold)
        alpha_mask |= fixed_alpha > alpha_abs_threshold
        n_views_mask = (hits_in_view_mask & fixed_avg_mask & alpha_mask).sum(-1) > n_views_threshold
        valid_mask = n_views_mask
        return valid_mask, fixed_avg.sum(-1)[valid_mask] / viewhits[valid_mask]

    def compute_motion(self, cameras: List[Camera], tracks: List[torch.Tensor]) -> Motion:
        gaussians = self.gaussians
        fixed_sums = torch.zeros((gaussians.get_xyz.shape[0], len(tracks)), device=self.device, dtype=torch.float32)
        fixed_alphas = torch.zeros((gaussians.get_xyz.shape[0], len(tracks)), device=self.device, dtype=torch.float32)
        fixed_pixhits = torch.zeros((gaussians.get_xyz.shape[0], len(tracks)), device=self.device, dtype=torch.int)
        fixed_viewhits = torch.zeros((gaussians.get_xyz.shape[0],), device=self.device, dtype=torch.int)
        for i, (camera, track) in enumerate(zip(tqdm(cameras, desc="Computing fixed"), tracks)):
            fixed_sum, fixed_alpha, fixed_pixhit, fixed_idx = self._count_fixed_pixels(camera, track)
            fixed_sums[fixed_idx, i] = fixed_sum
            fixed_alphas[fixed_idx, i] = fixed_alpha
            fixed_pixhits[fixed_idx, i] = fixed_pixhit
            fixed_viewhits[fixed_idx] += 1
        # solve fixed mask
        fixed_mask, weights_fixed = self.compute_fixed_mask_and_weights(fixed_sums, fixed_alphas, fixed_pixhits, fixed_viewhits)

        weights = torch.zeros((gaussians.get_xyz.shape[0],), device=self.device, dtype=torch.float64)
        pixhits = torch.zeros((gaussians.get_xyz.shape[0],), device=self.device, dtype=torch.int)
        viewhits = torch.zeros((gaussians.get_xyz.shape[0],), device=self.device, dtype=torch.int)
        isvd = ISVD_Mean3D(batch_size=gaussians.get_xyz.shape[0], device=self.device, k=len(tracks), dtype=torch.float32)
        # isvd = ISVDSelectK_Mean3D(batch_size=gaussians.get_xyz.shape[0], device=self.device, k=3, dtype=torch.float32)
        ils = ILS_RotationScale(batch_size=gaussians.get_xyz.shape[0], k=len(tracks), device=self.device)
        for i, (camera, track) in enumerate(zip(tqdm(cameras, desc="Computing motion"), tracks)):
            X, Y, A, valid_mask, weight, pixhit = self._compute_equations(camera, track)
            # isvd.update(A, valid_mask, weight) # do not use weight for mean3D
            valid2not_fixed_mask = (~fixed_mask)[valid_mask]
            valid_and_not_fixed_mask = valid_mask & (~fixed_mask)
            isvd.update(A[valid2not_fixed_mask], valid_and_not_fixed_mask, torch.ones_like(weight[valid2not_fixed_mask]))
            ils.update(X[valid2not_fixed_mask], Y[valid2not_fixed_mask], valid_and_not_fixed_mask, weight[valid2not_fixed_mask])
            weights[valid_mask] += weight
            pixhits += pixhit
            viewhits[valid_mask] += 1

        # solve cov and mean mask
        U, S, A_count = isvd.U, torch.diagonal(isvd.S, dim1=-2, dim2=-1), isvd.A_count
        v11, v12 = ils.v11, ils.v12
        valid_mask_cov, weights_cov = self.precompute_valid_mask_and_weights_cov3D(v11, v12, viewhits, weights, pixhits)
        valid_mask_mean, weights_mean = self.precompute_valid_mask_and_weights_mean3D(U, S, A_count, viewhits, weights, pixhits)

        weights_cov = weights_cov[(~fixed_mask)[valid_mask_cov]]
        valid_mask_cov &= ~fixed_mask
        weights_mean = weights_mean[(~fixed_mask)[valid_mask_mean]]
        valid_mask_mean &= ~fixed_mask

        translation_vector, valid_mask_mean, weights_mean = self._compute_translation(isvd, valid_mask_mean, weights_mean, viewhits, weights, pixhits)
        rotation_transform, scaling_transform, valid_mask_cov, weights_cov = self._compute_transformation(ils, valid_mask_cov, weights_cov, viewhits, weights, pixhits)
        return Motion(
            fixed_mask=fixed_mask,
            motion_mask_cov=valid_mask_cov,
            motion_mask_mean=valid_mask_mean,
            rotation_quaternion=rotation_transform,
            # scaling_modifier_log=scaling_transform, # change scaling has no benefit but produce some thin and long gaussians, bad
            translation_vector=translation_vector,
            confidence_fix=weights_fixed,
            confidence_cov=weights_cov,
            confidence_mean=weights_mean,
        )
