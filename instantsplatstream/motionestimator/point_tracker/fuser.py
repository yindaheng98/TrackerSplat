import torch
from typing import List
from itertools import permutations
from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.utils import matrix_to_quaternion, quaternion_raw_multiply, build_rotation
from instantsplatstream.utils import quaternion_invert
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
        conv3D = gaussians.covariance_activation(gaussians.get_scaling[valid_mask], 1., gaussians._rotation[valid_mask])
        transform2d = motion2d[valid_mask]
        X, Y = solve_transform(mean, conv3D, camera.FoVx, camera.FoVy, camera.image_width, camera.image_height, camera.world_view_transform, transform2d)
        return X, Y, valid_mask, weights, pixhit

    def compute_valid_mask_and_weights_3d(self, v11, v12, alpha, pixhits):
        '''Overload this method to make your own mask and weights'''
        v11_scaled = v11 / alpha.unsqueeze(-1).unsqueeze(-1)
        det = torch.linalg.det(v11_scaled)
        valid_mask = (alpha > 1e-3) & (det > 1e-12)
        weights = alpha[valid_mask]
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
        v11 = torch.zeros((gaussians.get_xyz.shape[0], 6, 6), device=self.device, dtype=torch.float64)
        v12 = torch.zeros((gaussians.get_xyz.shape[0], 6, 1), device=self.device, dtype=torch.float64)
        weights = torch.zeros((gaussians.get_xyz.shape[0],), device=self.device, dtype=torch.float64)
        pixhits = torch.zeros((gaussians.get_xyz.shape[0],), device=self.device, dtype=torch.int)
        for camera, track in zip(cameras, tracks):
            X, Y, valid_mask, weight, pixhit = self._compute_equations(camera, track)
            v11valid = X.transpose(1, 2).bmm(X)
            v12valid = X.transpose(1, 2).bmm(Y)
            v11[valid_mask] += v11valid * weight.unsqueeze(-1).unsqueeze(-1)
            v12[valid_mask] += v12valid * weight.unsqueeze(-1).unsqueeze(-1)
            weights[valid_mask] += weight
            pixhits += pixhit
        valid_mask, weights = self.compute_valid_mask_and_weights_3d(v11, v12, weights, pixhits)

        # solve conv3D
        conv3D_flatten = torch.linalg.inv(v11[valid_mask]).bmm(v12[valid_mask]).squeeze(-1)
        # # verify conv3D
        # conv3D_true = gaussians.covariance_activation(gaussians.get_scaling, 1, gaussians._rotation)
        # diff_conv3D = conv3D_flatten - conv3D_true

        # solve R and S matrix
        conv3D = unflatten_symmetry_3x3(conv3D_flatten)
        L, Q = torch.linalg.eigh(conv3D.type(torch.float32))  # ! random order
        # # verify conv3D
        # diff_conv3D = Q @ (L.unsqueeze(-1) * Q.transpose(1, 2)) - conv3D
        # # we can verify that the order do not influence the result
        # order = [2, 1, 0]
        # diff_conv3D = Q[..., order] @ (L[..., order].unsqueeze(-1) * Q[..., order].transpose(1, 2)) - conv3D
        negative_mask = (L < 0).any(-1)  # drop negative eigen values in L
        R = Q[~negative_mask, ...]
        S = torch.sqrt(L[~negative_mask, ...])
        valid_positive_mask = valid_mask.clone()
        valid_positive_mask[valid_mask] = ~negative_mask
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
            translation_vector=None,  # TODO: implement the xyz transformation
            confidence=None,  # TODO: implement the confidence
            update_baseframe=False
        )
