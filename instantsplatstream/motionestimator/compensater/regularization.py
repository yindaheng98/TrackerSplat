import copy
import torch
import torch.nn as nn
from gaussian_splatting import GaussianModel
from gaussian_splatting.utils import quaternion_raw_multiply
from instantsplatstream.motionestimator import Motion, MotionCompensater
from .base import BaseMotionCompensater

from instantsplatstream.utils.simple_knn import knn_kernel


class RegularizedMotionCompensater(BaseMotionCompensater):
    def __init__(self, k: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def update_knn(self, gaussians: GaussianModel, k: int) -> 'RegularizedMotionCompensater':
        xyz = gaussians.get_xyz.detach()
        assert k <= xyz.size(0) // 2, "k should be less than half of the gaussians"
        self.knn_idx, self.knn_dist = knn_kernel(gaussians.get_xyz.detach(), k)
        return self

    def update_baseframe(self, frame) -> 'RegularizedMotionCompensater':
        return super().update_baseframe(frame).update_knn(frame, self.k)

    def compensate(self, baseframe: GaussianModel, motion: Motion) -> GaussianModel:
        '''Overload this method to make your own compensation'''
        currframe = copy.deepcopy(baseframe)
        currframe._xyz = nn.Parameter(self.transform_xyz(baseframe, motion))
        currframe._rotation = nn.Parameter(self.transform_rotation(baseframe, motion))
        currframe._scaling = nn.Parameter(self.transform_scaling(baseframe, motion))
        return currframe
