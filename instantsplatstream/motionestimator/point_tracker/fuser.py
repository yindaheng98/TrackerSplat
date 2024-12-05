import torch
from typing import List
from gaussian_splatting import Camera, GaussianModel
from instantsplatstream.utils.motionfusion import motion_fusion
from .abc import Motion, MotionFuser, PointTrackSequence


class BaseMotionFuser(MotionFuser):
    def __init__(self, model: GaussianModel, device=torch.device("cuda")):
        super().__init__()
        self.model = model
        self.to(device)

    def to(self, device: torch.device) -> 'MotionFuser':
        self.model = self.model.to(device)
        self.device = device
        return self

    def update_baseframe(self, frame: GaussianModel) -> 'MotionFuser':
        self.model = frame
        return self.to(self.device)

    def __call__(self, trackviews: List[PointTrackSequence]) -> List[Motion]:
        motions = []
        cameras = [camera.build_camera(device=self.device) for camera in trackviews]
        for frame_idx in range(0, trackviews[0].track.shape[0]):
            tracks = [camera.track[frame_idx, ...] for camera in trackviews]
            motion = self.compute_motion(cameras, tracks)
            motions.append(motion)
        return motions

    def compute_motion(self, cameras: List[Camera], tracks: List[torch.Tensor]) -> Motion:
        for camera, track in zip(cameras, tracks):
            out, motion2d, motion_alpha, motion_det, pixhit = motion_fusion(self.model, camera, track)
            raise NotImplementedError  # TODO: implement the rest of the method
