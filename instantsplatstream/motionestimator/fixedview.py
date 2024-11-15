from typing import List, NamedTuple, Union
from abc import ABC, abstractmethod
import torch
from .abc import Motion, MotionEstimator


class FixedViewFrameSequenceMeta(NamedTuple):
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    R: torch.Tensor  # TODO: quaternion maybe better?
    T: torch.Tensor
    frames_path: List[str]


class FixedViewBatchMotionEstimationFunc(metaclass=ABC):
    @abstractmethod
    def to(self, device: torch.device) -> 'MotionEstimator':
        return self

    @abstractmethod
    def __call__(self, frames: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        raise NotImplementedError


class FixedViewBatchMotionEstimator(MotionEstimator):
    def __init__(self, cameras: List[FixedViewFrameSequenceMeta], batch_func: FixedViewBatchMotionEstimationFunc, batch_size=2):
        super().__init__()
        self.cameras = cameras
        for camera in self.cameras:
            assert len(camera.frames_path) == len(self.cameras[0].frames_path)
        self.batch_func = batch_func
        self.batch_size = batch_size

    def to(self, device: torch.device) -> 'MotionEstimator':
        self.batch_func = self.batch_func.to(device)
        return self

    @property
    def frames(self) -> List[FixedViewFrameSequenceMeta]:
        '''So you can access the views like this: estimator.frames[0] or estimator.frames[0:10]'''
        class ViewCollector:
            def __init__(self, cameras: List[FixedViewFrameSequenceMeta]):
                self.cameras = cameras
                for camera in self.cameras:
                    assert len(camera.frames_path) == len(self.cameras[0].frames_path)

            def __getitem__(self, frame_idx: Union[int, slice]) -> List[FixedViewFrameSequenceMeta]:
                if isinstance(frame_idx, slice):
                    return [FixedViewFrameSequenceMeta(**{**camera._asdict(), "frame_paths": camera.frames_path[frame_idx]}) for camera in self.cameras]
                if isinstance(frame_idx, int):
                    return [FixedViewFrameSequenceMeta(**{**camera._asdict(), "frame_paths": [camera.frames_path[frame_idx]]}) for camera in self.cameras]
                raise ValueError("frame_idx must be either an integer or a slice")
        return ViewCollector(self.cameras)

    def __iter__(self) -> 'FixedViewBatchMotionEstimator':
        self.frame_idx = 0
        self.curr_initframe_idx = -1
        self.curr_motions = []
        return self

    def __next__(self) -> Motion:
        length = len(self.cameras[0].frames_path)
        for camera in self.cameras:
            assert len(camera.frames_path) == length
        self.frame_idx += 1
        if self.frame_idx >= length:
            raise StopIteration
        prevframe_idx = self.frame_idx - 1
        initframe_idx = (prevframe_idx // self.batch_size) * self.batch_size
        if initframe_idx + self.batch_size > length:
            initframe_idx = length - self.batch_size
        if initframe_idx != self.curr_initframe_idx:
            self.curr_motions = self.batch_func(self.frames[initframe_idx:initframe_idx + self.batch_size])
            self.curr_motions[-1] = self.curr_motions[-1]._replace(update_baseframe=True)
        return self.curr_motions[self.frame_idx - initframe_idx]
