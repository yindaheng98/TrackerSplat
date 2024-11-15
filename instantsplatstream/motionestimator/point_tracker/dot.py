from typing import NamedTuple

import torch
from dot.models import create_model
from instantsplatstream.motionestimator import FixedViewFrameSequenceMeta
from .abc import FixedViewPointTrackFrame, FixedViewPointTracker, FixedViewPointTrackerFactory


class DotArgs(NamedTuple):
    model: str
    height: int
    width: int
    tracker_config: str
    tracker_path: str
    estimator_config: str
    estimator_path: str
    refiner_config: str
    refiner_path: str


class DotPointTracker(FixedViewPointTracker):
    def __init__(self, model, frames: FixedViewFrameSequenceMeta, batch_size):
        self.model = model
        self.frames = frames
        self.batch_size = batch_size
        self.curr_initframe_idx = -1
        self.curr_tracks = []

    def __call__(self, prevframe_idx: int) -> FixedViewPointTrackFrame:
        initframe_idx = (prevframe_idx // self.batch_size) * self.batch_size
        if initframe_idx == self.curr_initframe_idx:
            return self.curr_tracks[prevframe_idx % self.batch_size]
        # TODO: Update self.curr_tracks
        return self.curr_tracks[prevframe_idx % self.batch_size]


class DotPointTrackerFactory(FixedViewPointTrackerFactory):
    def __init__(self, batch_size=1, *args, **kwargs):
        self.args = DotArgs(*args, **kwargs)
        self.batch_size = batch_size
        self.model = create_model(self.args)
        self.device_dict = {}

    def __call__(self, frames: FixedViewFrameSequenceMeta, device: torch.device) -> FixedViewPointTracker:
        if str(device) not in self.device_dict:
            self.device_dict[str(device)] = self.model.to(device)
        return DotPointTracker(self.device_dict[str(device)], frames, self.batch_size)
