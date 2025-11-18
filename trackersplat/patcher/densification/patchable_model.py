import torch
import torch.nn as nn
from gaussian_splatting import GaussianModel


class PatchableGaussianModel(GaussianModel):
    def __init__(self, base: GaussianModel, base_size: int = None):
        super().__init__(sh_degree=base.max_sh_degree)
        base_size = base_size or base._xyz.shape[0]
        self.base = base
        self.base_size = base_size
        self._xyz = nn.Parameter(base._xyz[base_size:])
        self._features_dc = nn.Parameter(base._features_dc[base_size:])
        self._features_rest = nn.Parameter(base._features_rest[base_size:])
        self._opacity = nn.Parameter(base._opacity[base_size:])
        self._scaling = nn.Parameter(base._scaling[base_size:])
        self._rotation = nn.Parameter(base._rotation[base_size:])
        self.setup_functions()
        self.scale_modifier = 1.0
        self.debug = False
        self.antialiasing = False

    @property
    def get_scaling(self):
        return self.scaling_activation(torch.cat((self.base._scaling[:self.base_size], self._scaling), dim=0))

    @property
    def get_rotation(self):
        return self.rotation_activation(torch.cat((self.base._rotation[:self.base_size], self._rotation), dim=0))

    @property
    def get_xyz(self):
        return torch.cat((self.base._xyz[:self.base_size], self._xyz), dim=0)

    @property
    def get_features(self):
        features_dc = torch.cat((self.base._features_dc[:self.base_size], self._features_dc), dim=0)
        features_rest = torch.cat((self.base._features_rest[:self.base_size], self._features_rest), dim=0)
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return torch.cat((self.base._features_dc[:self.base_size], self._features_dc), dim=0)

    @property
    def get_features_rest(self):
        return torch.cat((self.base._features_rest[:self.base_size], self._features_rest), dim=0)

    @property
    def get_opacity(self):
        return self.opacity_activation(torch.cat((self.base._opacity[:self.base_size], self._opacity), dim=0))

    def load_full_model(self, full_model: GaussianModel) -> GaussianModel:
        full_model._xyz = nn.Parameter(self.get_xyz.detach())
        full_model._features_dc = nn.Parameter(self.get_features_dc.detach())
        full_model._features_rest = nn.Parameter(self.get_features_rest.detach())
        full_model._opacity = nn.Parameter(self.inverse_opacity_activation(self.get_opacity.detach()))
        full_model._scaling = nn.Parameter(self.scaling_inverse_activation(self.get_scaling.detach()))
        full_model._rotation = nn.Parameter(self.get_rotation.detach())
        return full_model
