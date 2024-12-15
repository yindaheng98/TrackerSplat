import copy
from typing import NamedTuple
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from gaussian_splatting import GaussianModel
from gaussian_splatting.utils import quaternion_raw_multiply


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


class MotionCompensater:
    def __init__(self, gaussians: GaussianModel, estimator: MotionEstimator, device: torch.device = "cuda"):
        self.initframe = gaussians
        self.estimator = estimator
        self.to(device)

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
        if motion.translation_vector is None:
            return baseframe._xyz.clone()
        if motion.motion_mask_mean is None:
            with torch.no_grad():
                return baseframe._xyz + motion.translation_vector
        with torch.no_grad():
            xyz = baseframe._xyz.clone()
            xyz[motion.motion_mask_mean] += motion.translation_vector
            return xyz

    @staticmethod
    def transform_rotation(baseframe: GaussianModel, motion: Motion) -> torch.Tensor:
        if motion.rotation_quaternion is None:
            return baseframe._rotation.clone()
        if motion.motion_mask_cov is None:
            with torch.no_grad():
                return quaternion_raw_multiply(motion.rotation_quaternion, baseframe._rotation)
        with torch.no_grad():
            rot = baseframe._rotation.clone()
            rot[motion.motion_mask_cov] = quaternion_raw_multiply(motion.rotation_quaternion, baseframe._rotation[motion.motion_mask_cov])
            return rot

    @staticmethod
    def transform_scaling(baseframe: GaussianModel, motion: Motion) -> torch.Tensor:
        if motion.scaling_modifier_log is None:
            return baseframe._scaling.clone()
        if motion.motion_mask_cov is None:
            with torch.no_grad():
                return motion.scaling_modifier_log + baseframe._scaling
        with torch.no_grad():
            scaling = baseframe._scaling.clone()
            scaling[motion.motion_mask_cov] = motion.scaling_modifier_log + baseframe._scaling[motion.motion_mask_cov]
            return scaling

    @staticmethod
    def compensate(baseframe: GaussianModel, motion: Motion) -> GaussianModel:
        '''Overload this method to make your own compensation'''
        currframe = copy.deepcopy(baseframe)
        currframe._xyz = nn.Parameter(MotionCompensater.transform_xyz(baseframe, motion))
        currframe._rotation = nn.Parameter(MotionCompensater.transform_rotation(baseframe, motion))
        currframe._scaling = nn.Parameter(MotionCompensater.transform_scaling(baseframe, motion))
        return currframe

    def __next__(self) -> GaussianModel:
        motion = self.estimator.__next__()
        currframe = self.compensate(self.baseframe, motion)
        # TODO: Training the model
        if motion.update_baseframe:
            self.baseframe = currframe
            self.estimator.update_baseframe(currframe)
        return currframe
