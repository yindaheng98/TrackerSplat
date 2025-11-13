import math
import torch
from tqdm import tqdm
from typing import List
from itertools import permutations
from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.utils import matrix_to_quaternion, quaternion_raw_multiply, build_rotation
from trackersplat.utils import quaternion_invert, ILS_RotationScale, ISVD_Mean3D, ISVDSelectK_Mean3D
from trackersplat.utils.motionfusion import motion_fusion, solve_transform
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
        cameras = [camera.build_track_camera(device=self.device) for camera in trackviews]
        for frame_idx in range(0, trackviews[0].track.shape[0]):
            tracks = [camera.track[frame_idx, ...] for camera in trackviews]
            with torch.no_grad():
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

    def precompute_valid_mask_and_weights_cov3D(self, v11, v12, viewhits, alpha, pixhits):
        '''Overload this method to make your own mask and weights'''
        valid_mask = (viewhits > 2) & (alpha > 1e-3) & (pixhits > 3)
        v11_scaled = v11[valid_mask, ...] / alpha[valid_mask, ...].unsqueeze(-1).unsqueeze(-1)
        det_abs = torch.linalg.det(v11_scaled).abs()
        det_clamp = 1e-9  # ! Solve cov3D may consume a lot of memory if this is small
        valid_mask_det = (det_abs > det_clamp)
        valid_mask[valid_mask.clone()] = valid_mask_det
        det_log = torch.log(det_abs[valid_mask_det])-math.log(det_clamp)  # det_log\in(0, +\infty), det=1e-12->det_log=0, det=+\infty->det_log=+\infty
        det_weights = (torch.sigmoid(det_log) - 0.5) * 2  # det_weights\in(0, 1), det=1e-12->det_weights=0, det=+\infty->det_weights=1
        weights = torch.sigmoid(torch.log(alpha[valid_mask])) * det_weights
        return valid_mask, weights

    def postcompute_valid_mask_and_weights_cov3D(self, R, S, error, weight, mask, viewhits, alpha, pixhits):
        '''Overload this method to make your own mask and weights'''
        # return R, S, mask, weight
        error_avg = error.abs().sum(-1)/alpha[mask]
        error_clamp = 0.5
        small_mask = error_avg < error_clamp
        mask = mask.clone()
        mask[mask.clone()] = small_mask
        return R[small_mask], S[small_mask], mask, weight[small_mask] * (1 - error_avg[small_mask])

    def precompute_valid_mask_and_weights_mean3D(self, U, S, A_count, viewhits, alpha, pixhits):
        '''Overload this method to make your own mask and weights'''
        valid_mask = (viewhits > 2) & (alpha > 1e-3) & (pixhits > 3)
        S_min = S.min(-1).values
        S_clamp = 1
        valid_mask &= (S_min < S_clamp) & (A_count > 2)  # S_min[valid_mask]\in(0, 1e-3)
        S_min_log = -math.log(S_clamp)-torch.log(S_min[valid_mask])  # S_min_log\in(0, +\infty), S_min=1e-3->S_min_log=0, S_min=0->S_min_log=+\infty
        weights = torch.sigmoid(S_min_log)  # weights\in(0.5, 1), S_min=1e-3->weights=0.5, S_min=0->weights=1
        return valid_mask, weights

    def postcompute_valid_mask_and_weights_mean3D(self, mean3D, error, weight, mask, viewhits, alpha, pixhits):
        '''Overload this method to make your own mask and weights'''
        # return mean3D, mask, weight
        error_avg = error.abs().sum(-1)  # /alpha[mask] # do not use weight for mean3D
        error_clamp = 2
        small_mask = error_avg < error_clamp
        mask = mask.clone()
        mask[mask.clone()] = small_mask
        return mean3D[small_mask], mask, weight[small_mask] * (1 - error_avg[small_mask])

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
        isvd = ISVD_Mean3D(batch_size=gaussians.get_xyz.shape[0], device=self.device, k=len(tracks), dtype=torch.float32)
        # isvd = ISVDSelectK_Mean3D(batch_size=gaussians.get_xyz.shape[0], device=self.device, k=3, dtype=torch.float32)
        ils = ILS_RotationScale(batch_size=gaussians.get_xyz.shape[0], k=len(tracks), device=self.device)
        for camera, track in zip(tqdm(cameras, desc="Computing motion"), tracks):
            X, Y, A, valid_mask, weight, pixhit = self._compute_equations(camera, track)
            # isvd.update(A, valid_mask, weight) # do not use weight for mean3D
            isvd.update(A, valid_mask, torch.ones_like(weight))
            ils.update(X, Y, valid_mask, weight)
            weights[valid_mask] += weight
            pixhits += pixhit
            viewhits[valid_mask] += 1

        # solve cov and mean mask
        translation_vector, valid_mask_mean, weights_mean = self._solve_translation(isvd, viewhits, weights, pixhits)
        rotation_transform, scaling_transform, valid_mask_cov, weights_cov = self._solve_transformation(ils, viewhits, weights, pixhits)
        return Motion(
            fixed_mask=None,
            motion_mask_cov=valid_mask_cov,
            motion_mask_mean=valid_mask_mean,
            rotation_quaternion=rotation_transform,
            # scaling_modifier_log=scaling_transform, # change scaling has no benefit but produce some thin and long gaussians, bad
            translation_vector=translation_vector,
            confidence_fix=None,
            confidence_cov=weights_cov,
            confidence_mean=weights_mean,
        )

    def _solve_translation(self, isvd: ISVD_Mean3D, viewhits, weights, pixhits):
        valid_mask_mean, weights_mean = self.precompute_valid_mask_and_weights_mean3D(isvd.U, torch.diagonal(isvd.S, dim1=-2, dim2=-1), isvd.A_count, viewhits, weights, pixhits)

        # solve mean3D
        mean3D, mean3Derror, valid_mask_mean_solved = isvd.solve(valid_mask_mean)  # ! Solve mean3D may be time consuming
        # select mean3D
        weights_mean = weights_mean[valid_mask_mean_solved[valid_mask_mean]]
        mean3D, valid_mask_mean, weights_mean = self.postcompute_valid_mask_and_weights_mean3D(mean3D, mean3Derror, weights_mean, valid_mask_mean_solved, viewhits, weights, pixhits)

        # mean3D motion vector
        translation_vector = mean3D - self.gaussians.get_xyz[valid_mask_mean]

        return translation_vector, valid_mask_mean, weights_mean

    def _solve_transformation(self, ils: ILS_RotationScale, viewhits, weights, pixhits):
        valid_mask_cov, weights_cov = self.precompute_valid_mask_and_weights_cov3D(ils.v11, ils.v12, viewhits, weights, pixhits)

        # solve R and S matrix
        R, S, coverror, valid_mask_cov_solved = ils.solve(valid_mask_cov)  # ! Solve R,S may consume a lot of memory
        # select mean3D
        weights_cov = weights_cov[valid_mask_cov_solved[valid_mask_cov]]
        R, S, valid_mask_cov, weights_cov = self.postcompute_valid_mask_and_weights_cov3D(R, S, coverror, weights_cov, valid_mask_cov_solved, viewhits, weights, pixhits)

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
        return rotation_transform, scaling_transform, valid_mask_cov, weights_cov
