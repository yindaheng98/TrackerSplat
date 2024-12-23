import torch
import torch.nn.functional as F
from gaussian_splatting.utils import quaternion_to_matrix
from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.trainer import BaseTrainer
from instantsplatstream.motionestimator.fixedview import FixedViewFrameSequenceMetaDataset
from instantsplatstream.utils.simple_knn import knn_kernel
from instantsplatstream.motionestimator.incremental_trainer import TrainerFactory


def quaternion_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def weighted_l2_loss(x, y, w):
    return torch.sqrt((torch.flatten(x - y, start_dim=2) ** 2).sum(-1) * w + 1e-20).mean()


def color_l2_loss(x, y):
    return torch.sqrt((torch.flatten(x - y, start_dim=1) ** 2).sum(-1) + 1e-20).mean()


class RegularizedTrainer(BaseTrainer):

    def __init__(
            self, model: GaussianModel,
            spatial_lr_scale: float,
            neighbors=8,
            stretch_shrink_start=4,
            scaling_max=5,
            loss_weight_overall=0.5,
            loss_weights={'rotation': 10.0, 'rigidity': 1.0, 'isometry': 1.0, 'stretch': 10.0, 'color': 10.0},
            *args, **kwargs):
        super().__init__(model, spatial_lr_scale, *args, **kwargs)
        self.neighbors = neighbors
        self.stretch_shrink_start = stretch_shrink_start
        self.scaling_max = scaling_max
        self.loss_weight_overall = loss_weight_overall
        self.loss_weights = loss_weights
        self.update_knn(model)

    def update_knn(self, last_gaussian: GaussianModel) -> 'RegularizedTrainer':
        self._features_dc_last = last_gaussian._features_dc.detach()
        _xyz_last = last_gaussian._xyz.detach()

        # pre-compute values
        self.neighbor_indices, dists = knn_kernel(_xyz_last, k=self.neighbors)
        self.neighbor_weights = torch.exp(-dists)
        self.neighbor_relative_dists_last = dists
        self.neighbor_offsets_last = _xyz_last[self.neighbor_indices] - _xyz_last.unsqueeze(-2)
        # pre-compute values
        self.rotation_matrix_last = quaternion_to_matrix(last_gaussian.get_rotation.detach())
        self.rotation_matrix_inv_last = self.rotation_matrix_last.transpose(2, 1)
        self.neighbor_offsets_point_coord_last = (
            self.rotation_matrix_inv_last.unsqueeze(1) @ self.neighbor_offsets_last.unsqueeze(-1)
        ).squeeze(-1)
        self._scaling_last = last_gaussian._scaling.detach().clone()
        return self

    def incremental_reg(self):
        loss = {}
        rotation_matrix = quaternion_to_matrix(self.model.get_rotation)
        relative_rotation_matrix = rotation_matrix @ self.rotation_matrix_inv_last
        loss['rotation'] = weighted_l2_loss(
            relative_rotation_matrix.unsqueeze(-3),
            relative_rotation_matrix[self.neighbor_indices],
            self.neighbor_weights
        )

        neighbor_offsets = self.model._xyz[self.neighbor_indices] - self.model._xyz.unsqueeze(-2)
        neighbor_offsets_point_coord = (
            rotation_matrix.transpose(2, 1).unsqueeze(1) @
            neighbor_offsets.unsqueeze(-1)
        ).squeeze(-1)
        loss['rigidity'] = weighted_l2_loss(
            neighbor_offsets_point_coord,
            self.neighbor_offsets_point_coord_last,
            self.neighbor_weights)

        neighbor_relative_dists = torch.norm(neighbor_offsets, p=2, dim=-1)
        loss['isometry'] = weighted_l2_loss(
            neighbor_relative_dists.unsqueeze(-1),
            self.neighbor_relative_dists_last.unsqueeze(-1),
            self.neighbor_weights)

        relative_scaling = self.model._scaling - self._scaling_last
        neighbor_relative_scaling = relative_scaling[self.neighbor_indices]
        loss['stretch'] = weighted_l2_loss(
            relative_scaling.unsqueeze(1),
            neighbor_relative_scaling,
            self.neighbor_weights) + relative_scaling.abs().mean()

        loss['color'] = color_l2_loss(
            self.model._features_dc,
            self._features_dc_last
        )

        weighted_loss = sum([self.loss_weights[k] * v for k, v in loss.items()])
        return weighted_loss * self.loss_weight_overall

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        return super().loss(out, camera) + self.incremental_reg()


class RegularizedTrainerFactory(TrainerFactory):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, model: GaussianModel, dataset: FixedViewFrameSequenceMetaDataset) -> BaseTrainer:
        return RegularizedTrainer(model, dataset.scene_extent(), *self.args, **self.kwargs)
