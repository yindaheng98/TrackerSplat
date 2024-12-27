import copy
import torch
import torch.nn as nn
from gaussian_splatting import GaussianModel
from gaussian_splatting.utils import quaternion_raw_multiply
from instantsplatstream.motionestimator import Motion, MotionCompensater


def transform_xyz(baseframe: GaussianModel, translation_vector: torch.Tensor, motion_mask_mean: torch.Tensor = None) -> torch.Tensor:
    if motion_mask_mean is None:
        with torch.no_grad():
            return baseframe._xyz + translation_vector
    with torch.no_grad():
        xyz = baseframe._xyz.clone()
        xyz[motion_mask_mean] += translation_vector
        return xyz


def transform_rotation(baseframe: GaussianModel, rotation_quaternion: torch.Tensor, motion_mask_cov: torch.Tensor = None) -> torch.Tensor:
    if motion_mask_cov is None:
        with torch.no_grad():
            return quaternion_raw_multiply(rotation_quaternion, baseframe._rotation)
    with torch.no_grad():
        rot = baseframe._rotation.clone()
        rot[motion_mask_cov] = quaternion_raw_multiply(rotation_quaternion, baseframe._rotation[motion_mask_cov])
        return rot


def transform_scaling(baseframe: GaussianModel, scaling_modifier_log: torch.Tensor, motion_mask_cov: torch.Tensor = None) -> torch.Tensor:
    if motion_mask_cov is None:
        with torch.no_grad():
            return scaling_modifier_log + baseframe._scaling
    with torch.no_grad():
        scaling = baseframe._scaling.clone()
        scaling[motion_mask_cov] = scaling_modifier_log + baseframe._scaling[motion_mask_cov]
        return scaling


class BaseMotionCompensater(MotionCompensater):

    def compensate(self, baseframe: GaussianModel, motion: Motion) -> GaussianModel:
        '''Overload this method to make your own compensation'''
        currframe = copy.deepcopy(baseframe)
        if motion.translation_vector is not None:
            currframe._xyz = nn.Parameter(transform_xyz(baseframe, motion.translation_vector, motion.motion_mask_mean))
        if motion.rotation_quaternion is not None:
            currframe._rotation = nn.Parameter(transform_rotation(baseframe, motion.rotation_quaternion, motion.motion_mask_cov))
        if motion.scaling_modifier_log is not None:
            currframe._scaling = nn.Parameter(transform_scaling(baseframe, motion.scaling_modifier_log, motion.motion_mask_cov))
        if motion.opacity_modifier_log is not None:
            with torch.no_grad():
                currframe._opacity = nn.Parameter(motion.opacity_modifier_log + baseframe._opacity)
        if motion.features_dc_modifier is not None:
            with torch.no_grad():
                currframe._features_dc = nn.Parameter(motion.features_dc_modifier + baseframe._features_dc)
        if motion.features_rest_modifier is not None:
            with torch.no_grad():
                currframe._features_rest = nn.Parameter(motion.features_rest_modifier + baseframe._features_rest)
        return currframe
