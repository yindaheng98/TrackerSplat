from typing import NamedTuple
from abc import ABCMeta, abstractmethod
import torch
from gaussian_splatting import GaussianModel
from .utils import compensate


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
    update_baseframe: bool = False

    opacity_modifier_log: torch.Tensor = None
    features_dc_modifier: torch.Tensor = None
    features_rest_modifier: torch.Tensor = None

    def to(self, device: torch.device) -> 'Motion':
        return self._replace(**{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in self._asdict().items()})

    def validate(self):
        if self.fixed_mask is not None:
            assert self.fixed_mask.dtype == torch.bool and self.fixed_mask.dim() == 1
            if self.confidence_fix is not None:
                assert self.confidence_fix.dim() == 1 and self.confidence_fix.size(0) == self.fixed_mask.sum()
        else:
            assert self.confidence_fix is None

        if self.motion_mask_cov is not None:
            assert self.motion_mask_cov.dtype == torch.bool and self.motion_mask_cov.dim() == 1
            if self.confidence_cov is not None:
                assert self.confidence_cov.dim() == 1 and self.confidence_cov.size(0) == self.motion_mask_cov.sum()
            if self.rotation_quaternion is not None:
                assert self.rotation_quaternion.dim() == 2 and self.rotation_quaternion.size(0) == self.motion_mask_cov.sum() and self.rotation_quaternion.size(1) == 4
            elif self.scaling_modifier_log is not None:
                assert self.scaling_modifier_log.dim() == 2 and self.scaling_modifier_log.size(0) == self.motion_mask_cov.sum() and self.scaling_modifier_log.size(1) == 3
        else:
            assert self.confidence_cov is None

        if self.motion_mask_mean is not None:
            assert self.motion_mask_mean.dtype == torch.bool and self.motion_mask_mean.dim() == 1
            if self.confidence_mean is not None:
                assert self.confidence_mean.dim() == 1 and self.confidence_mean.size(0) == self.motion_mask_mean.sum()
            if self.translation_vector is not None:
                assert self.translation_vector.dim() == 2 and self.translation_vector.size(0) == self.motion_mask_mean.sum() and self.translation_vector.size(1) == 3
        else:
            assert self.confidence_mean is None


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
