from typing import List
from abc import ABC, abstractmethod
import torch
from gaussian_splatting import GaussianModel
from instantsplatstream.dataset import CameraMeta
from instantsplatstream.motionestimator import Motion, FixedViewMotionEstimator


class CameraPointTrack(CameraMeta):
    track: torch.Tensor
    mask: torch.Tensor


class FixedViewPointTrackingMotionEstimator(FixedViewMotionEstimator, metaclass=ABC):
    def __init__(self, cameras, model: GaussianModel):
        super().__init__(cameras)
        self.model = model

    @abstractmethod
    def point_track(self, idx: int) -> List[CameraPointTrack]:
        raise NotImplementedError

    def estimate(self, idx: int) -> Motion:
        track = self.point_track(idx)
        pass  # TODO: implement the motion fusion from point tracks
