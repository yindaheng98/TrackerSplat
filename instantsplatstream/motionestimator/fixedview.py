from typing import List, Union
from abc import ABC, abstractmethod
from instantsplatstream.dataset import CameraMeta
from .abc import Motion, MotionEstimator


class FixedViewCameraMetaSequence(CameraMeta):
    image_path: List[str]


class ViewCollector:
    def __init__(self, cameras: List[FixedViewCameraMetaSequence]):
        self.cameras = cameras

    def __getitem__(self, frame_idx: Union[int, slice]) -> List[FixedViewCameraMetaSequence]:
        if isinstance(frame_idx, slice):
            return [FixedViewCameraMetaSequence(**{**camera._asdict(), "image_path": camera.image_path[frame_idx]}) for camera in self.cameras]
        if isinstance(frame_idx, int):
            return [FixedViewCameraMetaSequence(**{**camera._asdict(), "image_path": [camera.image_path[frame_idx]]}) for camera in self.cameras]
        raise ValueError("frame_idx must be either an integer or a slice")


class FixedViewMotionEstimator(MotionEstimator, metaclass=ABC):
    def __init__(self, cameras: List[FixedViewCameraMetaSequence]):
        super().__init__()
        self.cameras = cameras
        for camera in self.cameras:
            assert len(camera.image_path) == len(self.cameras[0].image_path)
        self.iter_idx = 0

    @property
    def frames(self) -> ViewCollector:
        '''So you can access the views like this: estimator.frames[0] or estimator.frames[0:10]'''
        return ViewCollector(self.cameras)

    @abstractmethod
    def estimate(self, prevframe_idx: int) -> Motion:
        raise NotImplementedError

    def __iter__(self) -> 'FixedViewMotionEstimator':
        self.iter_idx = 0
        return self

    def __next__(self) -> Motion:
        for camera in self.cameras:
            assert len(camera.image_path) == len(self.cameras[0].image_path)
        if self.iter_idx+1 >= len(self.cameras[0].image_path):
            raise StopIteration
        motion = self.estimate(self.iter_idx)
        self.iter_idx += 1
        return motion


class DynamicViewMotionEstimator(MotionEstimator, metaclass=ABC):
    pass  # TODO: implement the dynamic view motion estimator
