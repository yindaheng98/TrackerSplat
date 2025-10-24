import copy
import torch
import torch.nn as nn
from gaussian_splatting import GaussianModel
from trackersplat.motionestimator import Motion
from trackersplat.utils import axis_angle_to_quaternion, quaternion_to_axis_angle, propagate

from .base import transform_xyz, transform_rotation, transform_scaling
from .filter import FilteredMotionCompensater


class PropagatedMotionCompensater(FilteredMotionCompensater):

    def compute_neighbor_rotation(self, rotation_quaternion: torch.Tensor, motion_mask_cov: torch.Tensor, confidence_cov: torch.Tensor, fixed_mask: torch.Tensor) -> torch.Tensor:
        assert rotation_quaternion is not None, "Rotation quaternion is required"
        assert motion_mask_cov is not None, "Covariance motion mask is required"
        rotation_axis_angle = quaternion_to_axis_angle(rotation_quaternion)
        prop_axis_angle, prop_confidence = propagate(
            init_mask=motion_mask_cov.clone(),
            init_value_at_mask=rotation_axis_angle,
            init_weight_at_mask=confidence_cov,
            neighbor_indices=self.neighbor_indices, neighbor_weights=self.neighbor_weights
        )
        # prop_axis_angle *= (prop_confidence / (prop_confidence + fix_confidence)).unsqueeze(-1) # TODO: fix 0 division
        rotation_quaternion = axis_angle_to_quaternion(prop_axis_angle)
        return rotation_quaternion

    def compute_neighbor_transformation(self, translation_vector: torch.Tensor, motion_mask_mean: torch.Tensor, confidence_mean: torch.Tensor, fixed_mask: torch.Tensor) -> torch.Tensor:
        prop_translation_vector, prop_confidence = propagate(
            init_mask=motion_mask_mean.clone(),
            init_value_at_mask=translation_vector,
            init_weight_at_mask=confidence_mean,
            neighbor_indices=self.neighbor_indices, neighbor_weights=self.neighbor_weights
        )
        # prop_translation_vector *= (prop_confidence / (prop_confidence + fix_confidence)).unsqueeze(-1) # TODO: fix 0 division
        return prop_translation_vector

    def compute_neighbor_fix(self, fixed_mask: torch.Tensor, confidence_fix: torch.Tensor, n_iter=100) -> torch.Tensor:
        '''If 3/4 neighbors are fixed, the point is fixed'''
        fixed_mask = fixed_mask.clone()
        for _ in range(n_iter):
            new_fixed_mask = fixed_mask[self.neighbor_indices[~fixed_mask]].sum(-1) >= self.k * 3 // 4
            fixed_mask[~fixed_mask] = new_fixed_mask
            if new_fixed_mask.sum() <= 0:
                break
        return fixed_mask

    def compensate(self, baseframe: GaussianModel, motion: Motion) -> GaussianModel:
        '''Overload this method to make your own compensation'''
        currframe = copy.deepcopy(baseframe)
        median_translation_vector = self.median_filter_neighbor_transformation(motion.translation_vector, motion.motion_mask_mean)
        fixed_mask = self.compute_neighbor_fix(motion.fixed_mask, motion.confidence_fix)
        rotation = self.compute_neighbor_rotation(motion.rotation_quaternion, motion.motion_mask_cov, motion.confidence_cov, fixed_mask)
        translation = self.compute_neighbor_transformation(median_translation_vector, motion.motion_mask_mean, motion.confidence_mean, fixed_mask)
        rotation[fixed_mask, 0] = 1
        rotation[fixed_mask, 1:] = 0
        translation[fixed_mask, :] = 0
        currframe._xyz = nn.Parameter(transform_xyz(baseframe, translation))
        currframe._rotation = nn.Parameter(transform_rotation(baseframe, rotation))
        if motion.scaling_modifier_log is not None:
            scaling_modifier_log = motion.scaling_modifier_log
            scaling_modifier_log[fixed_mask[motion.motion_mask_cov], :] = 0
            currframe._scaling = nn.Parameter(transform_scaling(baseframe, scaling_modifier_log, motion.motion_mask_cov))
        # with torch.no_grad():
        #     currframe._opacity[fixed_mask] += currframe.inverse_opacity_activation(torch.tensor(0.05, device=currframe._opacity.device))  # debug
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
