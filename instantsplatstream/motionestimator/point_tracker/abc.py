from typing import List
from abc import ABC, abstractmethod
import torch
from instantsplatstream.dataset import CameraMeta
from instantsplatstream.motionestimator import Motion, FixedViewMotionEstimator


class PointTrack(CameraMeta):
    track: torch.Tensor
    mask: torch.Tensor


class PointTrack2Motion(metaclass=ABC):

    @abstractmethod
    def to(self, device: torch.device) -> 'PointTrack2Motion':
        return self

    @abstractmethod
    def __call__(self, tracks: List[PointTrack]) -> Motion:
        raise NotImplementedError


class FixedViewPointTrackingMotionEstimator(FixedViewMotionEstimator, metaclass=ABC):
    def __init__(self, cameras, track2motion: PointTrack2Motion, device="cuda"):
        super().__init__(cameras)
        self.track2motion = track2motion
        self.to(device)

    def to(self, device: torch.device) -> 'FixedViewPointTrackingMotionEstimator':
        self.track2motion = self.track2motion.to(device)
        return self

    @abstractmethod
    def point_track(self, idx: int) -> List[PointTrack]:
        raise NotImplementedError

    def estimate(self, idx: int) -> Motion:
        return self.track2motion(self.point_track(idx))
