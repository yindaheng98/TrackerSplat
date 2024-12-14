import torch
from typing import Tuple
from cotracker.predictor import CoTrackerPredictor
from dot.utils.io import read_frame
from instantsplatstream.motionestimator import FixedViewFrameSequenceMeta
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
        return int(frames.image_height * self.rescale_factor) // 8 * 8, int(frames.image_width * self.rescale_factor) // 8 * 8

    def track(self, frames: FixedViewFrameSequenceMeta, height: int, width: int) -> PointTrackSequence:
        video = []
        for path in frames.frames_path:
            frame = read_frame(path, resolution=(height, width))
            video.append(frame)
        video = torch.stack(video).to(self.device)
        with torch.no_grad():
            pred_tracks, pred_visibility = self.model(video[None])
        track = pred_tracks.squeeze(0).reshape(-1, height, width, 2)
        mask = pred_visibility.squeeze(0).reshape(-1, height, width)
        return track[1:, ...], mask[1:, ...]


def Cotracker3MotionEstimator(fuser, device=torch.device("cuda"), **kwargs):
    return PointTrackMotionEstimator(Cotracker3PointTracker(device=device, **kwargs), fuser, device)
