import torch
from cotracker.predictor import CoTrackerPredictor
from dot.utils.io import read_frame
from instantsplatstream.motionestimator import FixedViewFrameSequenceMeta
from .abc import FixedViewPointTrackSequence, FixedViewBatchPointTracker, FixedViewBatchPointTrackMotionEstimationFunc


class Cotracker3PointTracker(FixedViewBatchPointTracker):
    def __init__(
            self,
            height: int = 512, width: int = 512,
            checkpoint="./checkpoints/scaled_offline.pth",
            device=torch.device("cuda")):
        self.model = CoTrackerPredictor(
            checkpoint=checkpoint,
            offline=True,
        )
        self.to(device)
        self.height = height
        self.width = width

    def to(self, device: torch.device) -> 'FixedViewBatchPointTracker':
        self.model = self.model.to(device)
        self.device = device
        return self

    def __call__(self, frames: FixedViewFrameSequenceMeta) -> FixedViewPointTrackSequence:
        video = []
        for path in frames.frames_path:
            frame = read_frame(path, resolution=(self.height, self.width))
            video.append(frame)
        video = torch.stack(video).to(self.device)
        with torch.no_grad():
            pred_tracks, pred_visibility = self.model(video[None])
        return FixedViewPointTrackSequence(
            image_height=self.height,
            image_width=self.width,
            FoVx=frames.FoVx,
            FoVy=frames.FoVy,
            R=frames.R,
            T=frames.T,
            track=pred_tracks[0].reshape(-1, self.height, self.width, 2),
            mask=pred_visibility[0].reshape(-1, self.height, self.width),
        )


def Cotracker3MotionEstimationFunc(track2motion, device=torch.device("cuda"), **kwargs):
    return FixedViewBatchPointTrackMotionEstimationFunc(Cotracker3PointTracker(device=device, **kwargs), track2motion, device)
