from typing import List, NamedTuple, Union
from abc import ABC, abstractmethod
import torch
from gaussian_splatting.camera import build_camera
from .abc import Motion, MotionEstimator


class FixedViewFrameSequenceMeta(NamedTuple):
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    R: torch.Tensor  # TODO: quaternion maybe better?
    T: torch.Tensor
    frames_path: List[str]

    def build_camera(self, device="cuda"):
        return build_camera(**{**self._asdict(), "image_path": None}, device=device)


class ViewCollector:
    def __init__(self, cameras: List[FixedViewFrameSequenceMeta]):
        self.cameras = cameras

    def __getitem__(self, frame_idx: Union[int, slice]) -> List[FixedViewFrameSequenceMeta]:
        if isinstance(frame_idx, slice):
            return [FixedViewFrameSequenceMeta(**{**camera._asdict(), "frame_paths": camera.frames_path[frame_idx]}) for camera in self.cameras]
        if isinstance(frame_idx, int):
            return [FixedViewFrameSequenceMeta(**{**camera._asdict(), "frame_paths": [camera.frames_path[frame_idx]]}) for camera in self.cameras]
        raise ValueError("frame_idx must be either an integer or a slice")


class FixedViewMotionEstimator(MotionEstimator, metaclass=ABC):
    def __init__(self, cameras: List[FixedViewFrameSequenceMeta]):
        super().__init__()
        self.cameras = cameras
        for camera in self.cameras:
            assert len(camera.frames_path) == len(self.cameras[0].frames_path)
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
            assert len(camera.frames_path) == len(self.cameras[0].frames_path)
        if self.iter_idx+1 >= len(self.cameras[0].frames_path):
            raise StopIteration
        motion = self.estimate(self.iter_idx)
        self.iter_idx += 1
        return motion


class DynamicViewMotionEstimator(MotionEstimator, metaclass=ABC):
    pass  # TODO: implement the dynamic view motion estimator
