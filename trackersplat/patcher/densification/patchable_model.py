import torch
import torch.nn as nn
from gaussian_splatting import GaussianModel


class PatchableGaussianModel(GaussianModel):
    def __init__(self, base: GaussianModel):
        super().__init__(sh_degree=base.max_sh_degree)
        self.base = base
        self._xyz = nn.Parameter(torch.empty((0, *base._xyz.shape[1:]), device=base._xyz.device))
        self._features_dc = nn.Parameter(torch.empty((0, *base._features_dc.shape[1:]), device=base._features_dc.device))
        self._features_rest = nn.Parameter(torch.empty((0, *base._features_rest.shape[1:]), device=base._features_rest.device))
        self._opacity = nn.Parameter(torch.empty((0, *base._opacity.shape[1:]), device=base._opacity.device))
        self._scaling = nn.Parameter(torch.empty((0, *base._scaling.shape[1:]), device=base._scaling.device))
        self._rotation = nn.Parameter(torch.empty((0, *base._rotation.shape[1:]), device=base._rotation.device))

    @property
    def get_scaling(self):
        return self.scaling_activation(torch.cat((self.base._scaling, self._scaling), dim=0))

    @property
    def get_rotation(self):
        return self.rotation_activation(torch.cat((self.base._rotation, self._rotation), dim=0))

    @property
    def get_xyz(self):
        return torch.cat((self.base._xyz, self._xyz), dim=0)

    @property
    def get_features_dc(self):
        return torch.cat((self.base._features_dc, self._features_dc), dim=0)

    @property
    def get_features_rest(self):
        return torch.cat((self.base._features_rest, self._features_rest), dim=0)

    @property
    def get_opacity(self):
        return self.opacity_activation(torch.cat((self.base._opacity, self._opacity), dim=0))
