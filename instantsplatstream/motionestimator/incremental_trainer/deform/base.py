from typing import Callable, Dict
import torch
from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.utils import l1_loss, ssim
from gaussian_splatting.utils.schedular import get_expon_lr_func
from instantsplatstream.motionestimator.fixedview import FixedViewFrameSequenceMetaDataset
from instantsplatstream.motionestimator.incremental_trainer import TrainerFactory, BaseTrainer

from .deformation import DeformNetwork


class HexplaneTrainer(AbstractTrainer):

    def __init__(
            self, model: GaussianModel,
            basemodel: GaussianModel,
            spatial_lr_scale: float,
            lambda_dssim=0.2,
            feature_lr=0.0025,
            position_lr_max_steps=30_000,
            # https://github.com/wanglids/ST-4DGS/blob/bf0dbb13e76bf41b2c2a4ca64063e5d346db7c74/arguments/__init__.py
            deformation_lr_init=0.00016,
            deformation_lr_final=0.000016,
            deformation_lr_delay_mult=0.01,
            grid_lr_init=0.0016,
            grid_lr_final=0.00016,
            **kwargs):
        self._model = GaussianModel(model.max_sh_degree)
        self._model._features_dc = model._features_dc
        self._model._features_rest = model._features_rest
        self.lambda_dssim = lambda_dssim
        params = [
            {'params': [model._features_dc], 'lr': feature_lr, "name": "f_dc"},
            {'params': [model._features_rest], 'lr': feature_lr / 20.0, "name": "f_rest"},
        ]
        self._deformation = DeformNetwork(**kwargs).to(model._xyz.device)

        # https://github.com/wanglids/ST-4DGS/blob/bf0dbb13e76bf41b2c2a4ca64063e5d346db7c74/scene/gaussian_model.py#L162
        params += [
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': deformation_lr_init * spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': grid_lr_init * spatial_lr_scale, "name": "grid"},
        ]
        optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        schedulers = {
            get_expon_lr_func(
                lr_init=deformation_lr_init*spatial_lr_scale,
                lr_final=deformation_lr_final*spatial_lr_scale,
                lr_delay_mult=deformation_lr_delay_mult,
                max_steps=position_lr_max_steps),
            get_expon_lr_func(
                lr_init=grid_lr_init*spatial_lr_scale,
                lr_final=grid_lr_final*spatial_lr_scale,
                lr_delay_mult=deformation_lr_delay_mult,
                max_steps=position_lr_max_steps)
        }
        self._optimizer = optimizer
        self._schedulers = schedulers
        self._curr_step = 0

        self.mean3D_base = basemodel._xyz.detach()
        self.scales_base = basemodel._scaling.detach()
        self.rotations_base = basemodel._rotation.detach()
        self.opacity_base = basemodel._opacity.detach()

        # https://github.com/wanglids/ST-4DGS/blob/bf0dbb13e76bf41b2c2a4ca64063e5d346db7c74/train.py#L161
        # https://github.com/wanglids/ST-4DGS/blob/bf0dbb13e76bf41b2c2a4ca64063e5d346db7c74/scene/dataset_readers.py#L124
        self.per_time = torch.tensor(0).to(model._xyz.device).repeat(model._xyz.shape[0], 1)

    @property
    def curr_step(self) -> int:
        return self._curr_step

    @curr_step.setter
    def curr_step(self, v):
        self._curr_step = v

    @property
    def model(self) -> GaussianModel:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def schedulers(self) -> Dict[str, Callable[[int], float]]:
        return self._schedulers

    def forward_backward(self, camera: Camera):
        means3D_deform, scales_deform, rotations_deform, opacity_deform = self._deformation(self.mean3D_base, self.scales_base, self.rotations_base, self.opacity_base, self.per_time)
        self.model._xyz = means3D_deform
        self.model._scaling = scales_deform
        self.model._rotation = rotations_deform
        self.model._opacity = opacity_deform
        out = self.model(camera)
        loss = self.loss(out, camera)
        loss.backward()
        return loss, out

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        render = out["render"]
        gt = camera.ground_truth_image
        Ll1 = l1_loss(render, gt)
        ssim_value = ssim(render, gt)
        loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim_value)
        return loss


class HexplaneTrainerFactory(TrainerFactory):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, model: GaussianModel, basemodel: GaussianModel, dataset: FixedViewFrameSequenceMetaDataset, mask: torch.Tensor) -> HexplaneTrainer:
        return HexplaneTrainer(model, basemodel, dataset.scene_extent(), *self.args, **self.kwargs)
