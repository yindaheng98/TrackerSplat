import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.utils import quaternion_raw_multiply
from trackersplat.utils import quaternion_invert

from .abc import GaussianModel, Motion


def compare(baseframe: GaussianModel, curframe: GaussianModel) -> Motion:
    with torch.no_grad():
        return Motion(
            translation_vector=curframe._xyz - baseframe._xyz,
            rotation_quaternion=torch.nn.functional.normalize(quaternion_raw_multiply(curframe._rotation, quaternion_invert(baseframe._rotation))),
            scaling_modifier_log=curframe._scaling - baseframe._scaling,
            opacity_modifier_log=curframe._opacity - baseframe._opacity,
            features_dc_modifier=curframe._features_dc - baseframe._features_dc,
            features_rest_modifier=curframe._features_rest - baseframe._features_rest
        )


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
