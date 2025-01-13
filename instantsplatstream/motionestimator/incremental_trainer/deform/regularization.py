import torch
import torch.nn.functional as F
from gaussian_splatting.utils import quaternion_to_matrix
from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.utils.schedular import get_expon_lr_func
from instantsplatstream.motionestimator.fixedview import FixedViewFrameSequenceMetaDataset
from instantsplatstream.utils.simple_knn import knn_kernel
from instantsplatstream.motionestimator.incremental_trainer import TrainerFactory, BaseTrainer

from .base import HexplaneTrainer


def compute_plane_smoothness(t):
    batch_size, c, h, w = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., :h-1, :]  # [batch, c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]  # [batch, c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


def weighted_l2_loss(x, y, w):
    return torch.sqrt((torch.flatten(x - y, start_dim=2) ** 2).sum(-1) * w + 1e-20).mean()


def scale_loss(point_scale, thr):
    return torch.mean(
        torch.max((torch.max(torch.abs(point_scale), dim=1).values / torch.min(torch.abs(point_scale), dim=1).values), torch.tensor([thr], device="cuda")) - thr)


class RegularizedHexplaneTrainer(HexplaneTrainer):

    def __init__(
            self, model: GaussianModel,
            basemodel: GaussianModel,
            spatial_lr_scale: float,
            neighbors=8,
            # https://github.com/wanglids/ST-4DGS/blob/bf0dbb13e76bf41b2c2a4ca64063e5d346db7c74/arguments/__init__.py#L114
            lambda_ani=0.2,
            lambda_loc=0.001,
            # https://github.com/wanglids/ST-4DGS/blob/bf0dbb13e76bf41b2c2a4ca64063e5d346db7c74/arguments/DyNeRF.py#L11C20-L11C29
            time_smoothness_weight=0.001,
            plane_tv_weight=0.0002,
            l1_time_planes=0.001,
            *args, **kwargs):
        super().__init__(model, basemodel, spatial_lr_scale, *args, **kwargs)
        self.neighbors = neighbors
        self.lambda_ani = lambda_ani
        self.lambda_loc = lambda_loc
        self.time_smoothness_weight = time_smoothness_weight
        self.plane_tv_weight = plane_tv_weight
        self.l1_time_planes = l1_time_planes
        self.update_knn(basemodel)

    def update_knn(self, last_gaussian: GaussianModel) -> 'HexplaneTrainer':
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

    def _loc_regulation(self):
        # https://github.com/wanglids/ST-4DGS/blob/bf0dbb13e76bf41b2c2a4ca64063e5d346db7c74/train.py#L200
        rotation_matrix = quaternion_to_matrix(self.model.get_rotation)
        relative_rotation_matrix = rotation_matrix @ self.rotation_matrix_inv_last
        loss_rigid = weighted_l2_loss(
            relative_rotation_matrix.unsqueeze(-3),
            relative_rotation_matrix[self.neighbor_indices],
            self.neighbor_weights
        )

        neighbor_offsets = self.model._xyz[self.neighbor_indices] - self.model._xyz.unsqueeze(-2)
        neighbor_offsets_point_coord = (
            rotation_matrix.transpose(2, 1).unsqueeze(1) @
            neighbor_offsets.unsqueeze(-1)
        ).squeeze(-1)
        loss_rot = weighted_l2_loss(
            neighbor_offsets_point_coord,
            self.neighbor_offsets_point_coord_last,
            self.neighbor_weights)

        return loss_rigid + loss_rot

    def _plane_regulation(self):
        # https://github.com/wanglids/ST-4DGS/blob/bf0dbb13e76bf41b2c2a4ca64063e5d346db7c74/scene/gaussian_model.py#L563
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids = [0, 1, 3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total

    def _time_regulation(self):
        # https://github.com/wanglids/ST-4DGS/blob/bf0dbb13e76bf41b2c2a4ca64063e5d346db7c74/scene/gaussian_model.py#L575
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids = [2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total

    def _l1_regulation(self):
        # https://github.com/wanglids/ST-4DGS/blob/bf0dbb13e76bf41b2c2a4ca64063e5d346db7c74/scene/gaussian_model.py#L587
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total

    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        # https://github.com/wanglids/ST-4DGS/blob/bf0dbb13e76bf41b2c2a4ca64063e5d346db7c74/scene/gaussian_model.py#L601
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        # https://github.com/wanglids/ST-4DGS/blob/bf0dbb13e76bf41b2c2a4ca64063e5d346db7c74/train.py#L139
        tv = self.compute_regulation(
            time_smoothness_weight=self.time_smoothness_weight,
            l1_time_planes_weight=self.l1_time_planes,
            plane_tv_weight=self.plane_tv_weight)
        ani = scale_loss(self.model.get_scaling, thr=10.0)
        loc = self._loc_regulation()
        # https://github.com/wanglids/ST-4DGS/blob/bf0dbb13e76bf41b2c2a4ca64063e5d346db7c74/train.py#L228
        return super().loss(out, camera) + tv + self.lambda_ani*ani + self.lambda_loc*loc


class RegularizedHexplaneTrainerFactory(TrainerFactory):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, model: GaussianModel, basemodel: GaussianModel, dataset: FixedViewFrameSequenceMetaDataset, mask: torch.Tensor) -> RegularizedHexplaneTrainer:
        return RegularizedHexplaneTrainer(model, basemodel, dataset.scene_extent(), *self.args, **self.kwargs)
