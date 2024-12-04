from typing import List, NamedTuple
from abc import ABCMeta, abstractmethod
import torch
from gaussian_splatting import GaussianModel
from instantsplatstream.motionestimator import Motion, FixedViewBatchMotionEstimator, FixedViewFrameSequenceMeta


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

    @abstractmethod
    def update_baseframe(self, frame: GaussianModel) -> 'MotionFuser':
        return self


class PointTrackMotionEstimator(FixedViewBatchMotionEstimator, metaclass=ABCMeta):
    def __init__(self, tracker: PointTracker, fuser: MotionFuser, device=torch.device("cuda")):
        self.tracker = tracker
        self.fuser = fuser
        self.to(device)

    def to(self, device: torch.device) -> 'PointTrackMotionEstimator':
        self.tracker = self.tracker.to(device)
        self.fuser = self.fuser.to(device)
        return self

    def __call__(self, views: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        return self.fuser([self.tracker(camera) for camera in views])

    def update_baseframe(self, frame: GaussianModel) -> 'PointTrackMotionEstimator':
        self.fuser = self.fuser.update_baseframe(frame)
        return self
