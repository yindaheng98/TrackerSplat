from typing import NamedTuple
from abc import ABCMeta, abstractmethod
import torch
from gaussian_splatting import GaussianModel


class Motion(NamedTuple):
    fixed_mask: torch.Tensor = None
    motion_mask_cov: torch.Tensor = None
    motion_mask_mean: torch.Tensor = None
    rotation_quaternion: torch.Tensor = None
    scaling_modifier_log: torch.Tensor = None
    translation_vector: torch.Tensor = None
    confidence_fix: torch.Tensor = None
    confidence_cov: torch.Tensor = None
    confidence_mean: torch.Tensor = None
    update_baseframe: bool = True


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

    @abstractmethod
    def compensate(baseframe: GaussianModel, motion: Motion) -> GaussianModel:
        return baseframe

    def __next__(self) -> GaussianModel:
        motion = self.estimator.__next__()
        currframe = self.compensate(self.baseframe, motion)
        # TODO: Training the model
        if motion.update_baseframe:
            self.update_baseframe(currframe)
        return currframe
