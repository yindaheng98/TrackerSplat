import math
import torch
from tqdm import tqdm
from typing import List
from itertools import permutations
from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.utils import matrix_to_quaternion, quaternion_raw_multiply, build_rotation
from instantsplatstream.utils import quaternion_invert, ISVD_Mean3D, ILS_RotationScale
from instantsplatstream.utils.featurefusion import feature_fusion
from instantsplatstream.utils.motionfusion import motion_fusion, solve_transform
from .abc import Motion, MotionFuser, PointTrackSequence


class BaseMotionFuser(MotionFuser):
    def __init__(self, gaussians: GaussianModel, motion_threshold=0.2, device=torch.device("cuda")):
        '''
        Base class for motion fusion.
        `motion_threshold` is the threshold for detecting static gaussians, unit is pixel.
        '''
        super().__init__()
        self.gaussians = gaussians
        self.motion_threshold = motion_threshold
        self.to(device)

    def to(self, device: torch.device) -> 'MotionFuser':
        self.gaussians = self.gaussians.to(device)
        self.device = device
        return self

    def update_baseframe(self, frame: GaussianModel) -> 'MotionFuser':
        self.gaussians = frame
        return self.to(self.device)

    def __call__(self, trackviews: List[PointTrackSequence]) -> List[Motion]:
        motions = []
        cameras = [camera.build_camera(device=self.device) for camera in trackviews]
        for frame_idx in range(0, trackviews[0].track.shape[0]):
            tracks = [camera.track[frame_idx, ...] for camera in trackviews]
            motion = self.compute_motion(cameras, tracks)
            motions.append(motion)
        return motions

    def compute_valid_mask_and_weights_2d(self, out, motion2d, motion_alpha, motion_det, pixhit):
        '''Overload this method to make your own mask and weights'''
        valid_mask = (out['radii'] > 0) & (motion_det.abs() > 1e-12) & (motion_alpha > 1e-3) & (pixhit > 1)
        weights = motion_alpha[valid_mask]
        return valid_mask, weights

    def _compute_equations(self, camera: Camera, track: torch.Tensor):
        # TODO: filter static points before motion fusion
        gaussians = self.gaussians
        out, motion2d, motion_alpha, motion_det, pixhit = motion_fusion(gaussians, camera, track)
        valid_mask, weights = self.compute_valid_mask_and_weights_2d(out, motion2d, motion_alpha, motion_det, pixhit)
        assert list(valid_mask.shape) == [gaussians.get_xyz.shape[0]] and list(weights.shape) == [valid_mask.sum().item()]
        mean = gaussians.get_xyz.detach()[valid_mask]
        cov3D = gaussians.covariance_activation(gaussians.get_scaling[valid_mask], 1., gaussians._rotation[valid_mask])
        transform2d = motion2d[valid_mask]
        X, Y, A = solve_transform(mean, cov3D, camera.FoVx, camera.FoVy, camera.image_width, camera.image_height, camera.world_view_transform, camera.full_proj_transform, transform2d)
        return X, Y, A, valid_mask, weights, pixhit

    def _count_fixed_pixels(self, camera: Camera, track: torch.Tensor):
        motion_threshold = self.motion_threshold
        x = torch.arange(camera.image_width, dtype=torch.float, device=self.device)
        y = torch.arange(camera.image_height, dtype=torch.float, device=self.device)
        xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)
        dist = torch.linalg.norm(xy - track, dim=-1)
        is_static = (dist < motion_threshold).type(torch.float).unsqueeze(-1)
        _, features, features_alpha, features_idx = feature_fusion(self.gaussians, camera, is_static, 0)
        return features.squeeze(-1), features_alpha, features_idx

    def compute_fixed_mask_and_weights(self, fixed_sum, fixed_alpha, fixed_pixhits, viewhits, alpha, pixhits):
        '''Overload this method to make your own mask and weights'''
        # a gaussian should be fixed if it hit by more than 6 pixels and has more than 50% of its pixels fixed in any view
        hits_in_view_threshold, avg_in_view_threshold = 6, 0.5
        fixed_avg = torch.zeros_like(fixed_sum)
        hits_in_view_mask = fixed_pixhits > hits_in_view_threshold
        fixed_avg[hits_in_view_mask] = fixed_sum[hits_in_view_mask] / fixed_alpha[hits_in_view_mask]
        valid_mask = (hits_in_view_mask & (fixed_avg > avg_in_view_threshold)).any(dim=-1)
        return valid_mask, fixed_avg.sum(-1)[valid_mask] * viewhits[valid_mask] / viewhits.max()

    def compute_valid_mask_and_weights_3d(self, v11, v12, U, S, A_count, viewhits, alpha, pixhits):
        '''Overload this method to make your own mask and weights'''
        valid_mask_cov, weights_cov = self.compute_valid_mask_and_weights_cov3D(v11, v12, viewhits, alpha, pixhits)
        valid_mask_mean, weights_mean = self.compute_valid_mask_and_weights_mean3D(U, S, A_count, viewhits, alpha, pixhits)
        valid_mask = valid_mask_cov & valid_mask_mean
        weights = weights_cov[valid_mask[valid_mask_cov]] * weights_mean[valid_mask[valid_mask_mean]]
        return valid_mask, weights

    def compute_valid_mask_and_weights_cov3D(self, v11, v12, viewhits, alpha, pixhits):
        '''Overload this method to make your own mask and weights'''
        valid_mask = (viewhits > 2) & (alpha > 1e-3) & (pixhits > 3)
        v11_scaled = v11[valid_mask, ...] / alpha[valid_mask, ...].unsqueeze(-1).unsqueeze(-1)
        det_abs = torch.linalg.det(v11_scaled).abs()
        det_clamp = 1e-9  # ! Solve cov3D may consume a lot of memory
        valid_mask_det = (det_abs > det_clamp)
        valid_mask[valid_mask.clone()] = valid_mask_det
        det_log = torch.log(det_abs[valid_mask_det])-math.log(det_clamp)  # det_log\in(0, +\infty), det=1e-12->det_log=0, det=+\infty->det_log=+\infty
        det_weights = (torch.sigmoid(det_log) - 0.5) * 2  # det_weights\in(0, 1), det=1e-12->det_weights=0, det=+\infty->det_weights=1
        weights = alpha[valid_mask] * det_weights
        return valid_mask, weights

    def compute_valid_mask_and_weights_mean3D(self, U, S, A_count, viewhits, alpha, pixhits):
        '''Overload this method to make your own mask and weights'''
        valid_mask = (viewhits > 2) & (alpha > 1e-3) & (pixhits > 3)
        S_min = S.min(-1).values
        S_clamp = 1e-1
        valid_mask &= (S_min < S_clamp) & (A_count > 2)  # S_min[valid_mask]\in(0, 1e-3)
        S_min_log = -math.log(S_clamp)-torch.log(S_min[valid_mask])  # S_min_log\in(0, +\infty), S_min=1e-3->S_min_log=0, S_min=0->S_min_log=+\infty
        weights = torch.sigmoid(S_min_log)  # weights\in(0.5, 1), S_min=1e-3->weights=0.5, S_min=0->weights=1
        return valid_mask, weights

    def compute_best_order(self, R, S, R_base, S_base):
        '''Overload this method to make your own order'''
        orders = torch.tensor(list(permutations(range(3))), dtype=torch.int64, device=self.device)
        R_diff = (R[:, :, orders].transpose(1, 2) - R_base[:, None, :, :]).abs().sum((-1, -2))
        S_diff = (S[:, orders] - S_base[:, None, :]).abs().sum(-1)
        diff = R_diff + S_diff
        bestorder = orders[diff.argmin(-1), :]
        return bestorder

    def compute_transformation(self, rotation, scale_log, rotation_base, scaling_base_log):
        '''Overload this method to make your own transformation'''
        rotation_transform = torch.nn.functional.normalize(quaternion_raw_multiply(rotation, quaternion_invert(rotation_base)))
        # # verify rotation_transform
        # rotation_ = quaternion_raw_multiply(rotation_transform, rotation_base)
        # diff = rotation_ - rotation
        scaling_transform = scale_log - scaling_base_log
        return rotation_transform, scaling_transform

    def compute_motion(self, cameras: List[Camera], tracks: List[torch.Tensor]) -> Motion:
        gaussians = self.gaussians
        weights = torch.zeros((gaussians.get_xyz.shape[0],), device=self.device, dtype=torch.float64)
        pixhits = torch.zeros((gaussians.get_xyz.shape[0],), device=self.device, dtype=torch.int)
        viewhits = torch.zeros((gaussians.get_xyz.shape[0],), device=self.device, dtype=torch.int)
        fixed_sums = torch.zeros((gaussians.get_xyz.shape[0], len(tracks)), device=self.device, dtype=torch.float32)
        fixed_alphas = torch.zeros((gaussians.get_xyz.shape[0], len(tracks)), device=self.device, dtype=torch.float32)
        fixed_pixhits = torch.zeros((gaussians.get_xyz.shape[0], len(tracks)), device=self.device, dtype=torch.int)
        isvd = ISVD_Mean3D(batch_size=gaussians.get_xyz.shape[0], device=self.device, dtype=torch.float32)
        ils = ILS_RotationScale(batch_size=gaussians.get_xyz.shape[0], device=self.device)
        for i, (camera, track) in enumerate(zip(tqdm(cameras, desc="Computing motion"), tracks)):
            X, Y, A, valid_mask, weight, pixhit = self._compute_equations(camera, track)
            isvd.update(A, valid_mask, weight)
            ils.update(X, Y, valid_mask, weight)
            weights[valid_mask] += weight
            pixhits += pixhit
            viewhits[valid_mask] += 1
            fixed_sum, fixed_alpha, fixed_idx = self._count_fixed_pixels(camera, track)
            fixed_sums[fixed_idx, i] = fixed_sum
            fixed_alphas[fixed_idx, i] = fixed_alpha
            fixed_pixhits[:, i] = pixhit
        U, S, A_count = isvd.U, torch.diagonal(isvd.S, dim1=-2, dim2=-1), isvd.A_count
        v11, v12 = ils.v11, ils.v12
        valid_mask_cov, weights_cov = self.compute_valid_mask_and_weights_cov3D(v11, v12, viewhits, weights, pixhits)
        valid_mask_mean, weights_mean = self.compute_valid_mask_and_weights_mean3D(U, S, A_count, viewhits, weights, pixhits)
        fixed_mask, weights_fixed = self.compute_fixed_mask_and_weights(fixed_sums, fixed_alphas, fixed_pixhits, viewhits, weights, pixhits)

        # solve mean3D
        mean3D, valid_mask_mean_solved = isvd.solve(valid_mask_mean)
        # re select them
        mean3D = mean3D[(valid_mask_mean & (~fixed_mask))[valid_mask_mean_solved]]
        # re select weights
        weights_mean = weights_mean[(valid_mask_mean_solved & (~fixed_mask))[valid_mask_mean]]
        valid_mask_mean &= valid_mask_mean_solved & (~fixed_mask)

        # mean3D motion vector
        translation_vector = mean3D - gaussians.get_xyz[valid_mask_mean]

        # solve R and S matrix
        R, S, valid_mask_cov_solved = ils.solve(valid_mask_cov)  # ! Solve R,S may consume a lot of memory if this is small
        # re select them
        R = R[(valid_mask_cov & (~fixed_mask))[valid_mask_cov_solved]]
        S = S[(valid_mask_cov & (~fixed_mask))[valid_mask_cov_solved]]
        # re select weights
        weights_cov = weights_cov[(valid_mask_cov_solved & (~fixed_mask))[valid_mask_cov]]
        valid_mask_cov &= valid_mask_cov_solved & (~fixed_mask)

        # correct the order
        rotation_base = self.gaussians._rotation[valid_mask_cov, ...]
        scaling_base = self.gaussians._scaling[valid_mask_cov, ...]
        R_base = build_rotation(rotation_base)
        S_base = self.gaussians.scaling_activation(scaling_base)
        bestorder = self.compute_best_order(R, S, R_base, S_base)
        R_best = torch.gather(R, 2, bestorder.unsqueeze(1).expand(-1, 3, -1))
        S_best = torch.gather(S, 1, bestorder)
        rotation_curr = matrix_to_quaternion(R_best) / torch.nn.functional.normalize(rotation_base) * rotation_base
        scale_curr = self.gaussians.scaling_inverse_activation(S_best)
        rotation_transform, scaling_transform = self.compute_transformation(rotation_curr, scale_curr, rotation_base, scaling_base)
        return Motion(
            fixed_mask=fixed_mask,
            motion_mask_cov=valid_mask_cov,
            motion_mask_mean=valid_mask_mean,
            rotation_quaternion=rotation_transform,
            scaling_modifier_log=scaling_transform,
            translation_vector=translation_vector,
            confidence_fix=weights_fixed,
            confidence_cov=weights_cov,
            confidence_mean=weights_mean,
            update_baseframe=False
        )
