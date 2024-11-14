from typing import NamedTuple
from abc import ABC, abstractmethod
import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.utils import quaternion_raw_multiply


class Motion(NamedTuple):
    rotation_quaternion: torch.Tensor
    translation_vector: torch.Tensor


class MotionEstimator(metaclass=ABC):
    @abstractmethod
    def to(self, device: torch.device) -> 'MotionEstimator':
        return self

    @abstractmethod
    def __iter__(self) -> 'MotionEstimator':
        return self

    @abstractmethod
    def __next__(self) -> Motion:
        raise StopIteration


class MotionCompensater:
    def __init__(self, gaussians: GaussianModel, estimator: MotionEstimator):
        self.gaussians = gaussians
        self.estimator = estimator

    def to(self, device: torch.device) -> 'MotionCompensater':
        self.gaussians = self.gaussians.to(device)
        self.estimator = self.estimator.to(device)
        return self

    def __iter__(self) -> 'MotionCompensater':
        self.estimator = self.estimator.__iter__()
        return self

    def __next__(self) -> GaussianModel:
        motion = self.estimator.__next__()
        if motion.translation_vector is not None:
            assert motion.translation_vector.shape == self.gaussians._xyz.shape
            with torch.no_grad():
                self.gaussians._xyz += motion.translation_vector
        if motion.rotation_quaternion is not None:
            assert motion.rotation_quaternion.shape == self.gaussians._rotation.shape
            with torch.no_grad():
                self.gaussians._rotation = quaternion_raw_multiply(motion.rotation_quaternion, self.gaussians._rotation)
        return self.gaussians
