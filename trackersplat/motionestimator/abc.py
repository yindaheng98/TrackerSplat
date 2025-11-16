from abc import ABCMeta, abstractmethod
import torch
from gaussian_splatting import GaussianModel
from .motion import Motion
from .utils import compensate


class MotionEstimator(metaclass=ABCMeta):
    @abstractmethod
    def to(self, device: torch.device) -> 'MotionEstimator':
        return self

    @abstractmethod
    def __iter__(self) -> 'MotionEstimator':
        return self

    @abstractmethod
    def __next__(self) -> Motion:
        raise StopIteration

    @abstractmethod
    def update_baseframe(self, frame: GaussianModel) -> 'MotionEstimator':
        return self


class MotionCompensater(metaclass=ABCMeta):
    def __init__(self, gaussians: GaussianModel, estimator: MotionEstimator, device: torch.device = "cuda"):
        self.initframe = gaussians
        self.estimator = estimator
        self.to(device)

    def to(self, device: torch.device) -> 'MotionCompensater':
        self.initframe = self.initframe.to(device)
        self.estimator = self.estimator.to(device)
        return self

    def update_baseframe(self, frame: GaussianModel) -> 'MotionCompensater':
        self.baseframe = frame
        self.estimator.update_baseframe(frame)
        return self

    def __iter__(self) -> 'MotionCompensater':
        self.estimator = self.estimator.__iter__()
        self.update_baseframe(self.initframe)
        return self

    def compensate(self, baseframe: GaussianModel, motion: Motion) -> GaussianModel:
        '''Overload this method to make your own compensation'''
        return compensate(baseframe, motion)

    def __next__(self) -> GaussianModel:
        motion = self.estimator.__next__()
        motion.validate()
        currframe = self.compensate(self.baseframe, motion)
        if motion.update_baseframe:
            self.update_baseframe(currframe)
        return currframe
