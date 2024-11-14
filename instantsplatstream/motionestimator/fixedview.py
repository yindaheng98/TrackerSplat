from typing import List
from abc import ABC, abstractmethod
from instantsplatstream.dataset import CameraMeta
from .abc import Motion, MotionEstimator


class FixedViewCameraMetaSequence(CameraMeta):
    image_path: List[str]


class FixedViewMotionEstimator(MotionEstimator, metaclass=ABC):
    def __init__(self, cameras: List[FixedViewCameraMetaSequence]):
        super().__init__()
        for camera in cameras:
            assert len(camera.image_path) == len(cameras[0].image_path)
        self.cameras = cameras
        self.length = len(cameras[0].image_path)
        self.iter_idx = 0

    @abstractmethod
    def estimate(self, prevframe_idx: int) -> Motion:
        raise NotImplementedError

    def gather_views(self, frame_idx: int) -> List[CameraMeta]:
        return [CameraMeta(**{**camera._asdict(), "image_path": camera.image_path[frame_idx]}) for camera in self.cameras]

    def __iter__(self) -> 'FixedViewMotionEstimator':
        return self

    def __next__(self) -> Motion:
        if self.iter_idx+1 >= self.length:
            raise StopIteration
        motion = self.estimate(self.iter_idx)
        self.iter_idx += 1
        return motion


class DynamicViewMotionEstimator(MotionEstimator, metaclass=ABC):
    pass  # TODO: implement the dynamic view motion estimator
