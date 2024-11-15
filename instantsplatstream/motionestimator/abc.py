import copy
from typing import NamedTuple, List
from abc import ABC, abstractmethod
import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.utils import quaternion_raw_multiply


class Motion(NamedTuple):
    rotation_quaternion: torch.Tensor
    translation_vector: torch.Tensor
    update_baseframe: bool = True


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
        self.initframe = gaussians
        self.estimator = estimator

    def to(self, device: torch.device) -> 'MotionCompensater':
        self.initframe = self.initframe.to(device)
        self.estimator = self.estimator.to(device)
        return self

    def __iter__(self) -> 'MotionCompensater':
        self.estimator = self.estimator.__iter__()
        self.baseframe = self.initframe
        return self

    def __next__(self) -> GaussianModel:
        currframe = copy.copy(self.baseframe)
        motion = self.estimator.__next__()
        if motion.translation_vector is not None:
            assert motion.translation_vector.shape == self.baseframe._xyz.shape
            with torch.no_grad():
                currframe._xyz = self.baseframe._xyz + motion.translation_vector
        if motion.rotation_quaternion is not None:
            assert motion.rotation_quaternion.shape == self.baseframe._rotation.shape
            with torch.no_grad():
                currframe._rotation = quaternion_raw_multiply(motion.rotation_quaternion, self.baseframe._rotation)
        if motion.update_baseframe:
            self.baseframe = currframe
        return currframe
