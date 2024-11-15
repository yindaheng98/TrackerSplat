from typing import List, NamedTuple
from abc import ABCMeta, abstractmethod
import torch
from instantsplatstream.motionestimator import Motion, FixedViewBatchMotionEstimationFunc, FixedViewFrameSequenceMeta


class FixedViewPointTrackSequence(NamedTuple):
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    R: torch.Tensor  # TODO: quaternion maybe better?
    T: torch.Tensor
    track: torch.Tensor
    mask: torch.Tensor


class FixedViewBatchPointTracker(metaclass=ABCMeta):

    @abstractmethod
    def to(self, device: torch.device) -> 'FixedViewBatchPointTracker':
        return self

    @abstractmethod
    def __call__(self, frames: FixedViewFrameSequenceMeta) -> FixedViewPointTrackSequence:
        raise NotImplementedError


class FixedViewBatchTracks2Motion(metaclass=ABCMeta):

    @abstractmethod
    def to(self, device: torch.device) -> 'FixedViewBatchTracks2Motion':
        return self

    @abstractmethod
    def __call__(self, tracks: List[FixedViewPointTrackSequence]) -> List[Motion]:
        raise NotImplementedError


class FixedViewBatchPointTrackMotionEstimationFunc(FixedViewBatchMotionEstimationFunc, metaclass=ABCMeta):
    def __init__(self, tracker: FixedViewBatchPointTracker, track2motion: FixedViewBatchTracks2Motion, device=torch.device("cuda")):
        self.tracker = tracker
        self.tracks2motion = track2motion
        self.to(device)

    def to(self, device: torch.device) -> 'FixedViewBatchPointTrackMotionEstimationFunc':
        self.tracker = self.tracker.to(device)
        self.tracks2motion = self.tracks2motion.to(device)
        return self

    def __call__(self, frames: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        return self.tracks2motion([self.tracker(camera) for camera in frames])
