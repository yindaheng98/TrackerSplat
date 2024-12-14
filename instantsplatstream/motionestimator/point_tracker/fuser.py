import math
import torch
from tqdm import tqdm
from typing import List
from itertools import permutations
from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.utils import matrix_to_quaternion, quaternion_raw_multiply, build_rotation
from instantsplatstream.utils import quaternion_invert, ISVD, ILS, ILS_RotationScale
from instantsplatstream.utils.motionfusion import motion_fusion, solve_transform, unflatten_symmetry_3x3
from .abc import Motion, MotionFuser, PointTrackSequence


class BaseMotionFuser(MotionFuser):
    def __init__(self, gaussians: GaussianModel, device=torch.device("cuda")):
        super().__init__()
        self.gaussians = gaussians
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
        valid_mask = (out['radii'] > 0) & (motion_det > 1e-12) & (motion_alpha > 1e-3) & (pixhit > 1)
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

    def compute_valid_mask_and_weights_3d(self, v11, v12, U, S, alpha, pixhits, A_count):
        '''Overload this method to make your own mask and weights'''
        valid_mask_cov, weights_cov = self.compute_valid_mask_and_weights_cov3D(v11, v12, alpha, pixhits)
        valid_mask_mean, weights_mean = self.compute_valid_mask_and_weights_mean3D(U, S, A_count)
        valid_mask = valid_mask_cov & valid_mask_mean
        weights = weights_cov[valid_mask[valid_mask_cov]] * weights_mean[valid_mask[valid_mask_mean]]
        return valid_mask, weights

    def compute_valid_mask_and_weights_cov3D(self, v11, v12, alpha, pixhits):
        '''Overload this method to make your own mask and weights'''
        v11_scaled = v11 / alpha.unsqueeze(-1).unsqueeze(-1)
        det = torch.linalg.det(v11_scaled)
        valid_mask = (alpha > 1e-3) & (det > 1e-12)
        weights = alpha[valid_mask]
        return valid_mask, weights

    def compute_valid_mask_and_weights_mean3D(self, U, S, A_count):
        '''Overload this method to make your own mask and weights'''
        S_min = S.min(-1).values
        S_clamp = 1e-3
        valid_mask = (A_count > 3) & (S_min < S_clamp)  # S_min[valid_mask]\in(0, 1e-3)
        S_min_log = -math.log(S_clamp)-torch.log(S_min[valid_mask])  # S_min_log\in(0, +\infty), S_min=1e-3->S_min_log=0, S_min=0->S_min_log=+\infty
        weights = torch.sigmoid(S_min_log)  # weights\in(0.5, 1), S_min=1e-3->weights=0.5, S_min=0->weights=1
        # TODO: add A_count into weights
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
        isvd = ISVD(batch_size=gaussians.get_xyz.shape[0], n=4, device=self.device)
        ils_ = ILS(batch_size=gaussians.get_xyz.shape[0], n=6, dtype=torch.float64, device=self.device)
        ils = ILS_RotationScale(batch_size=gaussians.get_xyz.shape[0], n=3, device=self.device)
        for camera, track in zip(tqdm(cameras, desc="Computing motion"), tracks):
            X, Y, A, valid_mask, weight, pixhit = self._compute_equations(camera, track)
            isvd.update(A, valid_mask, weight)
            ils_.update(X, Y, valid_mask, weight)
            ils.update(X, Y, valid_mask, weight)
            weights[valid_mask] += weight
            pixhits += pixhit
        U, S, A_count = isvd.U, isvd.S, isvd.A_count
        S = torch.diagonal(S, dim1=-2, dim2=-1)
        v11, v12 = ils_.v11, ils_.v12
        valid_mask, weights = self.compute_valid_mask_and_weights_3d(v11, v12, U, S, weights, pixhits, A_count)

        # solve mean3D
        p_hom = torch.gather(U[valid_mask], 2, S[valid_mask].min(-1).indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 4, -1)).squeeze(-1)
        mean3D = p_hom[..., :-1] / p_hom[..., -1:]
        translation_vector = mean3D - gaussians.get_xyz[valid_mask]

        # solve cov3D
        cov3D_flatten = torch.linalg.inv(v11[valid_mask]).bmm(v12[valid_mask]).squeeze(-1)
        # # verify cov3D
        # cov3D_true = gaussians.covariance_activation(gaussians.get_scaling, 1, gaussians._rotation)
        # diff_cov3D = cov3D_flatten - cov3D_true[valid_mask]

        # solve R and S matrix
        cov3D = unflatten_symmetry_3x3(cov3D_flatten)
        L, Q = torch.linalg.eigh(cov3D.type(torch.float32))  # ! random order
        # # verify cov3D
        # diff_cov3D = Q @ (L.unsqueeze(-1) * Q.transpose(1, 2)) - cov3D
        # # we can verify that the order do not influence the result
        # order = [2, 1, 0]
        # diff_cov3D = Q[..., order] @ (L[..., order].unsqueeze(-1) * Q[..., order].transpose(1, 2)) - cov3D
        negative_mask = (L < 0).any(-1)  # drop negative eigen values in L
        R_ = Q[~negative_mask, ...]
        S_ = torch.sqrt(L[~negative_mask, ...])
        valid_positive_mask_ = valid_mask.clone()
        valid_positive_mask_[valid_mask] = ~negative_mask
        R, S, valid_positive_mask = ils.solve(valid_mask)
        # correct the order
        rotation_base = self.gaussians._rotation[valid_positive_mask, ...]
        scaling_base = self.gaussians._scaling[valid_positive_mask, ...]
        R_base = build_rotation(rotation_base)
        S_base = self.gaussians.scaling_activation(scaling_base)
        bestorder = self.compute_best_order(R, S, R_base, S_base)
        R_best = torch.gather(R, 2, bestorder.unsqueeze(1).expand(-1, 3, -1))
        S_best = torch.gather(S, 1, bestorder)
        rotation_curr = matrix_to_quaternion(R_best) / torch.nn.functional.normalize(rotation_base) * rotation_base
        scale_curr = self.gaussians.scaling_inverse_activation(S_best)
        rotation_transform, scaling_transform = self.compute_transformation(rotation_curr, scale_curr, rotation_base, scaling_base)
        return Motion(
            motion_mask=valid_positive_mask,
            rotation_quaternion=rotation_transform,
            scaling_modifier_log=scaling_transform,
            translation_vector=translation_vector,
            confidence=None,  # TODO: implement the confidence
            update_baseframe=False
        )
