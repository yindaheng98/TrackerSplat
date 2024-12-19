import copy
import torch
import torch.nn as nn
from gaussian_splatting import GaussianModel
from gaussian_splatting.utils import quaternion_raw_multiply
from instantsplatstream.motionestimator import Motion, MotionCompensater


class BaseMotionCompensater(MotionCompensater):

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

    def compensate(self, baseframe: GaussianModel, motion: Motion) -> GaussianModel:
        '''Overload this method to make your own compensation'''
        currframe = copy.deepcopy(baseframe)
        currframe._xyz = nn.Parameter(self.transform_xyz(baseframe, motion))
        currframe._rotation = nn.Parameter(self.transform_rotation(baseframe, motion))
        currframe._scaling = nn.Parameter(self.transform_scaling(baseframe, motion))
        return currframe
