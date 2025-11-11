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
