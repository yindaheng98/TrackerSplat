import copy
import torch
import torch.nn as nn
from gaussian_splatting import GaussianModel
from instantsplatstream.motionestimator import Motion
from instantsplatstream.utils import axis_angle_to_quaternion, quaternion_to_axis_angle, propagate, motion_median_filter

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

    def compensate(self, baseframe: GaussianModel, motion: Motion) -> GaussianModel:
        '''Overload this method to make your own compensation'''
        currframe = copy.deepcopy(baseframe)
        median_translation_vector = self.median_filter_neighbor_transformation(motion.translation_vector, motion.motion_mask_mean)
        rotation = self.compute_neighbor_rotation(motion.rotation_quaternion, motion.motion_mask_cov, motion.confidence_cov, motion.fixed_mask)
        translation = self.compute_neighbor_transformation(median_translation_vector, motion.motion_mask_mean, motion.confidence_mean, motion.fixed_mask)
        rotation[motion.fixed_mask, 0] = 1
        rotation[motion.fixed_mask, 1:] = 0
        translation[motion.fixed_mask, :] = 0
        currframe._xyz = nn.Parameter(transform_xyz(baseframe, translation))
        currframe._rotation = nn.Parameter(transform_rotation(baseframe, rotation))
        if motion.scaling_modifier_log is not None:
            scaling_modifier_log = motion.scaling_modifier_log
            scaling_modifier_log[motion.fixed_mask[motion.motion_mask_cov], :] = 0
            currframe._scaling = nn.Parameter(transform_scaling(baseframe, scaling_modifier_log, motion.motion_mask_cov))
        return currframe
