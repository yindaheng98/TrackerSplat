import torch
from typing import List
from gaussian_splatting import Camera, GaussianModel
from instantsplatstream.utils.motionfusion import motion_fusion, solve_transform
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

    def compute_valid_mask_and_weights(self, out, motion2d, motion_alpha, motion_det, pixhit) -> Motion:
        valid_mask = (out['radii'] > 0) & (motion_det > 1e-12) & (motion_alpha > 1e-3) & (pixhit > 1)
        weights = motion_alpha[valid_mask]
        return valid_mask, weights

    def _compute_equations(self, camera: Camera, track: torch.Tensor) -> Motion:
        gaussians = self.gaussians
        out, motion2d, motion_alpha, motion_det, pixhit = motion_fusion(gaussians, camera, track)
        valid_mask, weights = self.compute_valid_mask_and_weights(out, motion2d, motion_alpha, motion_det, pixhit)
        assert list(valid_mask.shape) == [gaussians.get_xyz.shape[0]] and list(weights.shape) == [valid_mask.sum().item()]
        mean = gaussians.get_xyz.detach()[valid_mask]
        conv3D = gaussians.covariance_activation(gaussians.get_scaling[valid_mask], 1., gaussians._rotation[valid_mask])
        transform2d = motion2d[valid_mask]
        X, Y = solve_transform(mean, conv3D, camera.FoVx, camera.FoVy, camera.image_width, camera.image_height, camera.world_view_transform, transform2d)
        return X, Y, valid_mask, weights

    def compute_motion(self, cameras: List[Camera], tracks: List[torch.Tensor]) -> Motion:
        gaussians = self.gaussians
        v11 = torch.zeros((gaussians.get_xyz.shape[0], 6, 6), device=self.device, dtype=torch.float64)
        v12 = torch.zeros((gaussians.get_xyz.shape[0], 6, 1), device=self.device, dtype=torch.float64)
        alpha = torch.zeros((gaussians.get_xyz.shape[0],), device=self.device, dtype=torch.float64)
        for camera, track in zip(cameras, tracks):
            X, Y, valid_mask, weights = self._compute_equations(camera, track)
            v11valid = X.transpose(1, 2).bmm(X)
            v12valid = X.transpose(1, 2).bmm(Y)
            v11[valid_mask] += v11valid * weights.unsqueeze(-1).unsqueeze(-1)
            v12[valid_mask] += v12valid * weights.unsqueeze(-1).unsqueeze(-1)
            alpha[valid_mask] += weights
            pass
            # TODO: implement the rest of the method
