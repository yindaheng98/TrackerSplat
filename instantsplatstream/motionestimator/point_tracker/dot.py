import torch
from dot.models import DenseOpticalTracker
from dot.utils.io import read_frame
from instantsplatstream.motionestimator import FixedViewFrameSequenceMeta
from .abc import FixedViewPointTrackSequence, FixedViewBatchPointTracker, FixedViewBatchPointTrackMotionEstimationFunc


class DotPointTracker(FixedViewBatchPointTracker):
    def __init__(
            self,
            height: int,
            width: int,
            tracker_config: str,
            tracker_path: str,
            estimator_config: str,
            estimator_path: str,
            refiner_config: str,
            refiner_path: str):
        self.model = DenseOpticalTracker(
            height=height,
            width=width,
            tracker_config=tracker_config,
            tracker_path=tracker_path,
            estimator_config=estimator_config,
            estimator_path=estimator_path,
            refiner_config=refiner_config,
            refiner_path=refiner_path,
        )

    def to(self, device: torch.device) -> 'FixedViewBatchPointTracker':
        self.model = self.model.to(device)
        return self

    def __call__(self, frames: FixedViewFrameSequenceMeta) -> FixedViewPointTrackSequence:
        video = []
        for path in frames.frames_path:
            frame = read_frame(path)
            video.append(frame)
        video = torch.stack(video)
        with torch.no_grad():
            pred = self.model.get_tracks_from_first_to_every_other_frame({"video": video[None]})
        tracks = pred["tracks"][0]
        return FixedViewPointTrackSequence(
            image_height=frames.image_height,
            image_width=frames.image_width,
            FoVx=frames.FoVx,
            FoVy=frames.FoVy,
            R=frames.R,
            T=frames.T,
            track=tracks,
            mask=torch.ones_like(tracks),
        )


def DotMotionEstimationFunc(track2motion, device="cuda", *args, **kwargs):
    return FixedViewBatchPointTrackMotionEstimationFunc(DotPointTracker(*args, **kwargs), track2motion, device)
