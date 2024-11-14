from typing import List
from abc import ABC, abstractmethod
import torch
from gaussian_splatting import GaussianModel
from instantsplatstream.dataset import CameraMeta
from instantsplatstream.motionestimator import Motion, FixedViewMotionEstimator


class CameraPointTrack(CameraMeta):
    track: torch.Tensor
    mask: torch.Tensor


class PointTrack2Motion(metaclass=ABC):

    @abstractmethod
    def __call__(self, tracks: List[CameraPointTrack]) -> Motion:
        raise NotImplementedError


class FixedViewPointTrackingMotionEstimator(FixedViewMotionEstimator, metaclass=ABC):
    def __init__(self, cameras, track2motion: PointTrack2Motion):
        super().__init__(cameras)
        self.track2motion = track2motion

    @abstractmethod
    def point_track(self, idx: int) -> List[CameraPointTrack]:
        raise NotImplementedError

    def estimate(self, idx: int) -> Motion:
        return self.track2motion(self.point_track(idx))
