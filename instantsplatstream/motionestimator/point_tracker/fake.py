import torch
from typing import Tuple
from instantsplatstream.motionestimator import FixedViewFrameSequenceMeta
from .abc import PointTrackSequence, PointTracker, PointTrackMotionEstimator


class FakePointTracker(PointTracker):
    def __init__(
            self,
            rescale_factor=1.0,
            device=torch.device("cuda")):
        self.rescale_factor = rescale_factor
        self.to(device)

    def to(self, device: torch.device) -> 'FakePointTracker':
        self.device = device
        return self

    def compute_rescale(self, frames: FixedViewFrameSequenceMeta) -> Tuple[int, int]:
        return int(frames.image_height * self.rescale_factor) // 8 * 8, int(frames.image_width * self.rescale_factor) // 8 * 8

    def track(self, frames: FixedViewFrameSequenceMeta, height: int, width: int) -> PointTrackSequence:
        x = torch.arange(width, dtype=torch.float, device=self.device)
        y = torch.arange(height, dtype=torch.float, device=self.device)
        xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)
        A = torch.rand((2, 2)).to(self.device) - 0.5
        A = torch.eye(2).to(self.device)
        b = (torch.rand(2).to(self.device) - 0.5) * height
        b = torch.zeros(2).to(self.device)
        solution = torch.cat([b[:, None], A], dim=1).T
        xy_transformed = (xy.view(-1, 2) @ A.T + b).view(xy.shape)
        track = xy_transformed.unsqueeze(0).repeat(len(frames.frames_path), 1, 1, 1)
        mask = torch.ones_like(track[..., 0], dtype=torch.bool)
        return track, mask


def FakeMotionEstimator(fuser, device=torch.device("cuda"), **kwargs):
    return PointTrackMotionEstimator(FakePointTracker(device=device, **kwargs), fuser, device)
