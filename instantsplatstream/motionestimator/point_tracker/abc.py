from typing import List, NamedTuple
from abc import ABC, abstractmethod
import torch
from instantsplatstream.motionestimator import Motion, FixedViewMotionEstimator, FixedViewFrameSequenceMeta


class FixedViewPointTrackFrame(NamedTuple):
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    R: torch.Tensor  # TODO: quaternion maybe better?
    T: torch.Tensor
    prevframe_path: str
    thisframe_path: str
    track: torch.Tensor
    mask: torch.Tensor


class FixedViewPointTracker(metaclass=ABC):

    @abstractmethod
    def to(self, device: torch.device) -> 'FixedViewPointTracker':
        return self

    @abstractmethod
    def __call__(self, prevframe_idx: int) -> FixedViewPointTrackFrame:
        raise NotImplementedError


class FixedViewPointTrackerFactory(metaclass=ABC):

    @abstractmethod
    def __call__(self, frames: FixedViewFrameSequenceMeta) -> FixedViewPointTracker:
        raise NotImplementedError


class FixedViewPointTracks2Motion(metaclass=ABC):

    @abstractmethod
    def to(self, device: torch.device) -> 'FixedViewPointTracks2Motion':
        return self

    @abstractmethod
    def __call__(self, tracks: List[FixedViewPointTrackFrame]) -> Motion:
        raise NotImplementedError


class FixedViewPointTrackingMotionEstimator(FixedViewMotionEstimator, metaclass=ABC):
    def __init__(self, cameras, tracker_factory: FixedViewPointTrackerFactory, track2motion: FixedViewPointTracks2Motion, device="cuda"):
        super().__init__(cameras)
        self.tracks2motion: FixedViewPointTracks2Motion = track2motion
        self.trackers: List[FixedViewPointTracker] = [tracker_factory(camera) for camera in cameras]
        self.to(device)

    def to(self, device: torch.device) -> 'FixedViewPointTrackingMotionEstimator':
        self.tracks2motion = self.tracks2motion.to(device)
        self.trackers = [tracker.to(device) for tracker in self.trackers]
        return self

    def estimate(self, prevframe_idx: int) -> Motion:
        return self.tracks2motion([tracker(prevframe_idx) for tracker in self.trackers])
