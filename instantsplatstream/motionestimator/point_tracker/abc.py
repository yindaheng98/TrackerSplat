from typing import List, NamedTuple
from abc import ABCMeta, abstractmethod
import torch
from instantsplatstream.motionestimator import Motion, FixedViewBatchMotionEstimationFunc, FixedViewFrameSequenceMeta


class PointTrackSequence(NamedTuple):
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    R: torch.Tensor  # TODO: quaternion maybe better?
    T: torch.Tensor
    track: torch.Tensor
    mask: torch.Tensor


class PointTracker(metaclass=ABCMeta):

    @abstractmethod
    def to(self, device: torch.device) -> 'PointTracker':
        return self

    @abstractmethod
    def __call__(self, frames: FixedViewFrameSequenceMeta) -> PointTrackSequence:
        raise NotImplementedError


class MotionFuser(metaclass=ABCMeta):

    @abstractmethod
    def to(self, device: torch.device) -> 'MotionFuser':
        return self

    @abstractmethod
    def __call__(self, trackviews: List[PointTrackSequence]) -> List[Motion]:
        raise NotImplementedError


class PointTrackMotionEstimationFunc(FixedViewBatchMotionEstimationFunc, metaclass=ABCMeta):
    def __init__(self, tracker: PointTracker, fuser: MotionFuser, device=torch.device("cuda")):
        self.tracker = tracker
        self.fuser = fuser
        self.to(device)

    def to(self, device: torch.device) -> 'PointTrackMotionEstimationFunc':
        self.tracker = self.tracker.to(device)
        self.fuser = self.fuser.to(device)
        return self

    def __call__(self, views: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        return self.fuser([self.tracker(camera) for camera in views])
