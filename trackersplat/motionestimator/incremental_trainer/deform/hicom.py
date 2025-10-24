from trackersplat.motionestimator.incremental_trainer import TrainerFactory
from trackersplat.motionestimator.fixedview import FixedViewFrameSequenceMetaDataset
from gaussian_splatting.utils.schedular import get_expon_lr_func
from gaussian_splatting.utils import l1_loss, ssim
from gaussian_splatting.trainer import BaseTrainer
from gaussian_splatting import GaussianModel, Camera
# https://github.com/gqk/HiCoM/blob/main/pipeline/hicom/model/deformation.py
import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from absl import logging
from jaxtyping import Float
from torch import Tensor, nn


class Grid(nn.Module):
    def __init__(
        self,
        size: Tuple[int, int, int],
        xyz_min: Float[Tensor, "3"],
        xyz_max: Float[Tensor, "3"],
    ):
        super().__init__()

        size = size if torch.is_tensor(size) else torch.tensor(size)
        base = torch.tensor([size[1] * size[2], size[2], 1])

        self.register_buffer("size", size)
        self.register_buffer("base", base)
        self.register_buffer("xyz_min", xyz_min)
        self.register_buffer("xyz_max", xyz_max)

        logging.info(f"Set grid min: {xyz_min.tolist()}, max: {xyz_max.tolist()}")

    def normalize(self, xyz: Float[Tensor, "n 3"], clamp: bool = True):
        xyz = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)
        if clamp:
            return xyz.clamp(0, 1)
        return xyz

    def hash(self, xyz: Float[Tensor, "n 3"]) -> torch.Tensor:
        xyz_normed = self.normalize(xyz)
        index = ((xyz_normed * self.size - 0.5).clamp(0).int() * self.base).sum(dim=-1)
        return index


@dataclass
class DeformConfig:
    quantile: float = 0.05
    lr: float = 0.0005
    num_stage1_steps: int = 100
    max_gs_per_grid: int = 5
    num_grid_levels: int = 3
    grid_level_ratio: int = 2
    momentum: Optional[float] = 0.6
    densify_interval: int = 40
    densify_grad_threshold: float = 1.5e-4
    opacity_threshold: float = 0.01


class Deformation(nn.Module):
    config: DeformConfig
    grids: List[Grid]
    optimizer: Optional[torch.optim.Optimizer] = None

    def __init__(self, config: DeformConfig):
        super().__init__()
        self.config = config
        self.rotation_activation = torch.nn.functional.normalize
        self.grids = []

    @torch.no_grad()
    def create_grids(self, xyz: Float[Tensor, "n 3"]):
        q, max_gs_per_grid = self.config.quantile, self.config.max_gs_per_grid
        xyz_min = xyz.quantile(q, dim=0) * (1 + q)
        xyz_max = xyz.quantile(1 - q, dim=0) * (1 + q)
        n = math.ceil((xyz.shape[0] / max_gs_per_grid) ** (1 / 3))

        grids, level = [], self.config.num_grid_levels
        while n > 0 and level > 0:
            grids.append(Grid((n, n, n), xyz_min, xyz_max).to(xyz.device))
            n, level = n // self.config.grid_level_ratio, level - 1
        return grids

    def setup(self, xyz: Float[Tensor, "n 3"], reset_grid: bool = False):
        if not self.grids or reset_grid:
            self.grids = self.create_grids(xyz)
            num_grids = sum(g.size.prod().item() for g in self.grids)
            delta = torch.zeros(num_grids, 7, device=xyz.device)
            delta[:, 3] = 1  # 0 0 0 1 0 0 0 -> x y z q1 q2 q3 q4
            self.register_parameter("delta", nn.Parameter(delta))

        index, offset = [], 0
        for g in self.grids:
            index.append(g.hash(xyz) + offset)
            offset += g.size.prod().item()
        index = torch.stack(index)
        self.register_buffer("index", index)
        if self.config.momentum is not None:
            self.delta.data *= self.config.momentum

        count = index.flatten().unique().numel()
        logging.info(f"Setup deformation, grids: {offset}, occupied grids: {count}")

        self.reset_optimizer()

    def reg_loss(self):
        identity = torch.zeros_like(self.delta[:1, :])
        identity[:, 3] = 1
        return (self.delta - identity).abs().mean()

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.lr, eps=1e-15
        )

    def capture(self) -> Dict[str, Any]:
        if not self.grids:
            return {}
        return dict(delta=self.delta, index=self.index)  # TODO grids state

    def restore(self, ckpt: Dict[str, Any]):
        if "delta" in ckpt:
            logging.info("Restore deformation from checkpoint")
            self.register_buffer("index", ckpt["index"])
            self.register_parameter("delta", ckpt["delta"])
            self.reset_optimizer()

    def forward(self, xyz: Float[Tensor, "n 3"], normalized: bool = False):
        delta = self.delta[self.index].sum(dim=0)
        delta_xyz = delta[:, :3].contiguous()
        delta_rot = delta[:, 3:].contiguous()
        return delta_xyz, self.rotation_activation(delta_rot)


# https://github.com/Awesome3DGS/libgs/blob/main/libgs/utils/general.py#L105C1-L116C44
def quaternion_multiply(a: torch.Tensor, b: torch.Tensor):
    a_norm = torch.nn.functional.normalize(a)
    b_norm = torch.nn.functional.normalize(b)
    w1, x1, y1, z1 = a_norm[:, 0], a_norm[:, 1], a_norm[:, 2], a_norm[:, 3]
    w2, x2, y2, z2 = b_norm[:, 0], b_norm[:, 1], b_norm[:, 2], b_norm[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack([w, x, y, z], dim=1)


# My own code
class HiCoMTrainer(BaseTrainer):

    def __init__(
            self, model: GaussianModel,
            basemodel: GaussianModel,
            spatial_lr_scale: float,
            # https://github.com/gqk/HiCoM/blob/main/config/dynerf.yaml#L40
            lambda_dssim=0.2,
            lambda_deform=1.0,
            position_lr_init=0.00016,
            position_lr_final=1.6e-06,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=15_000,
            feature_lr=0.0025,
            opacity_lr=0.05,
            scaling_lr=0.005,
            rotation_lr=0.001,
            **kwargs):
        super().__init__(
            model, spatial_lr_scale,
            lambda_dssim=lambda_dssim,
            position_lr_init=position_lr_init,
            position_lr_final=position_lr_final,
            position_lr_delay_mult=position_lr_delay_mult,
            position_lr_max_steps=position_lr_max_steps,
            feature_lr=feature_lr,
            opacity_lr=opacity_lr,
            scaling_lr=scaling_lr,
            rotation_lr=rotation_lr
        )
        # https://github.com/wanglids/ST-4DGS/blob/bf0dbb13e76bf41b2c2a4ca64063e5d346db7c74/scene/gaussian_model.py#L162
        config = DeformConfig(**kwargs)
        self.deform = Deformation(config).to(model._xyz.device)
        self.lambda_deform = lambda_deform
        self._xyz_ori = basemodel.get_xyz.detach()
        self._rotation_ori = basemodel.get_rotation.detach()
        self.deform.setup(self._xyz_ori.detach(), reset_grid=True)
        self.optimizer.add_param_group({'params': self.deform.parameters(), 'lr': self.deform.config.lr, "name": "deformation"})

    @property
    def model(self) -> GaussianModel:
        delta_xyz, delta_rot = self.deform(self._xyz_ori)
        if hasattr(self._model, "_xyz"):
            del self._model._xyz
        if hasattr(self._model, "_rotation"):
            del self._model._rotation
        self._model._xyz = self._xyz_ori + delta_xyz
        self._model._rotation = quaternion_multiply(self._rotation_ori, delta_rot)
        return self._model

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        render = out["render"]
        gt = camera.ground_truth_image
        Ll1 = l1_loss(render, gt)
        ssim_value = ssim(render, gt)
        loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim_value)
        loss_reg = self.deform.reg_loss()
        return loss + self.lambda_deform * loss_reg


class HiCoMTrainerFactory(TrainerFactory):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, model: GaussianModel, basemodel: GaussianModel, dataset: FixedViewFrameSequenceMetaDataset, mask: torch.Tensor) -> HiCoMTrainer:
        return HiCoMTrainer(model, basemodel, dataset.scene_extent(), *self.args, **self.kwargs)
