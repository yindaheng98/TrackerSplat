import torch
from typing import Tuple
from cotracker.predictor import CoTrackerPredictor
from trackersplat.motionestimator import FixedViewFrameSequenceMeta
from .abc import PointTrackSequence, PointTracker, PointTrackMotionEstimator


class Cotracker3PointTracker(PointTracker):
    def __init__(
            self,
            checkpoint="./checkpoints/scaled_offline.pth",
            rescale_factor=1.0,
            device=torch.device("cuda")):
        self.model = CoTrackerPredictor(
            checkpoint=checkpoint,
            offline=True,
        )
        self.rescale_factor = rescale_factor
        self.to(device)

    def to(self, device: torch.device) -> 'Cotracker3PointTracker':
        self.model = self.model.to(device)
        self.device = device
        return self

    def compute_rescale(self, frames: FixedViewFrameSequenceMeta) -> Tuple[int, int]:
        grid_size = 80
        W, H = int(frames.image_width * self.rescale_factor), int(frames.image_height * self.rescale_factor)
        grid_step = W // grid_size
        grid_width = W // grid_step
        grid_height = H // grid_step
        return grid_height * grid_step, grid_width * grid_step

    def track(self, video: torch.Tensor) -> PointTrackSequence:
        _, _, height, width = video.shape
        with torch.no_grad():
            pred_tracks, pred_visibility = self.model(video[None].to(self.device))
        track_seq = pred_tracks.squeeze(0)
        mask_seq = pred_visibility.squeeze(0)
        idx_seq = track_seq[0, ..., :2].round().int()
        track = torch.zeros_like(track_seq)
        mask = torch.zeros_like(mask_seq)
        w_idx, h_idx = idx_seq[..., 0], idx_seq[..., 1]
        idx = h_idx * width + w_idx
        track[:, idx, ...] = track_seq
        mask[:, idx, ...] = mask_seq
        track = track.reshape(-1, height, width, 2)
        mask = mask.reshape(-1, height, width)
        return track[1:, ...], mask[1:, ...]  # (tracks, visibility mask)


def Cotracker3MotionEstimator(fuser, device=torch.device("cuda"), **kwargs):
    return PointTrackMotionEstimator(Cotracker3PointTracker(device=device, **kwargs), fuser, device)
