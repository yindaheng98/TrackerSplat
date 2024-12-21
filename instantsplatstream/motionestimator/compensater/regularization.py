import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from gaussian_splatting import GaussianModel
from gaussian_splatting.utils import quaternion_to_matrix
from instantsplatstream.motionestimator import Motion
from instantsplatstream.utils.simple_knn import knn_kernel
from instantsplatstream.utils import axis_angle_to_quaternion, quaternion_to_axis_angle, propagate

from .base import BaseMotionCompensater


class RegularizedMotionCompensater(BaseMotionCompensater):
    def __init__(self, k: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def update_knn(self, gaussians: GaussianModel, k: int) -> 'RegularizedMotionCompensater':
        xyz = gaussians.get_xyz.detach()
        assert k <= xyz.size(0), "k should be less than the number of points"

        # k nearest neighbors of each points
        self.neighbor_indices, dists = knn_kernel(xyz, k)
        self.neighbor_weights = torch.exp(-F.normalize(dists))
        self.neighbor_relative_dists_last = dists

        # vector from each points to their k nearest neighbors (a.k.a. "neighbor offsets")
        self.neighbor_offsets_last = xyz[self.neighbor_indices] - xyz.unsqueeze(-2)

        # rotation matrix of each points
        self.rotation_matrix_last = quaternion_to_matrix(gaussians.get_rotation.detach())
        self.rotation_matrix_inv_last = self.rotation_matrix_last.transpose(2, 1)

        # "neighbor offsets" in the local coordinate system of each points
        self.neighbor_offsets_point_coord_last = (
            self.rotation_matrix_inv_last.unsqueeze(1) @ self.neighbor_offsets_last.unsqueeze(-1)
        ).squeeze(-1)
        return self

    def update_baseframe(self, frame) -> 'RegularizedMotionCompensater':
        return super().update_baseframe(frame).update_knn(frame, self.k)

    def compute_neighbor_rotation(self, motion: Motion) -> torch.Tensor:
        assert motion.rotation_quaternion is not None, "Rotation quaternion is required"
        assert motion.motion_mask_cov is not None, "Covariance motion mask is required"
        rotation_axis_angle = quaternion_to_axis_angle(motion.rotation_quaternion)
        prop_axis_angle, prop_confidence = propagate(
            init_mask=motion.motion_mask_cov.clone(),
            init_value_at_mask=rotation_axis_angle,
            init_weight_at_mask=motion.confidence_cov,
            neighbor_indices=self.neighbor_indices, neighbor_weights=self.neighbor_weights
        )
        rotation_quaternion = axis_angle_to_quaternion(prop_axis_angle)
        return rotation_quaternion, prop_confidence

    def compensate(self, baseframe: GaussianModel, motion: Motion) -> GaussianModel:
        '''Overload this method to make your own compensation'''
        currframe = copy.deepcopy(baseframe)
        rotation, rotation_confidence = self.compute_neighbor_rotation(motion)
        currframe._xyz = nn.Parameter(self.transform_xyz(baseframe, motion))
        currframe._rotation = nn.Parameter(self.transform_rotation(baseframe, motion))
        currframe._scaling = nn.Parameter(self.transform_scaling(baseframe, motion))
        return currframe
