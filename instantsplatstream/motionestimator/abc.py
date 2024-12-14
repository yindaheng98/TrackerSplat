import copy
from typing import NamedTuple
from abc import ABCMeta, abstractmethod
import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.utils import quaternion_raw_multiply


class Motion(NamedTuple):
    motion_mask_cov: torch.Tensor = None
    motion_mask_mean: torch.Tensor = None
    rotation_quaternion: torch.Tensor = None
    scaling_modifier_log: torch.Tensor = None
    translation_vector: torch.Tensor = None
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
        self.estimator.update_baseframe(self.baseframe)
        return self

    @staticmethod
    def transform_xyz(baseframe: GaussianModel, motion: Motion) -> torch.Tensor:
        assert motion.translation_vector.shape == baseframe._xyz.shape
        with torch.no_grad():
            return baseframe._xyz + motion.translation_vector

    @staticmethod
    def transform_rotation(baseframe: GaussianModel, motion: Motion) -> torch.Tensor:
        assert motion.rotation_quaternion.shape == baseframe._rotation.shape
        with torch.no_grad():
            return quaternion_raw_multiply(motion.rotation_quaternion, baseframe._rotation)

    @staticmethod
    def transform_scaling(baseframe: GaussianModel, motion: Motion) -> torch.Tensor:
        assert motion.scaling_modifier_log.shape == baseframe._scaling.shape
        with torch.no_grad():
            return motion.scaling_modifier_log + baseframe._scaling

    def __next__(self) -> GaussianModel:
        currframe = copy.copy(self.baseframe)
        motion = self.estimator.__next__()
        if motion.translation_vector is not None:
            currframe._xyz = self.transform_xyz(self.baseframe, motion)
        if motion.rotation_quaternion is not None:
            currframe._rotation = self.transform_rotation(self.baseframe, motion)
        if motion.scaling_modifier_log is not None:
            currframe._scaling = self.transform_scaling(self.baseframe, motion)
        # TODO: Training the model
        if motion.update_baseframe:
            self.baseframe = currframe
            self.estimator.update_baseframe(currframe)
        return currframe
