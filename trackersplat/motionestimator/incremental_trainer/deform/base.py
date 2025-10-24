import copy
import torch
from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.trainer import BaseTrainer
from gaussian_splatting.utils import l1_loss, ssim
from gaussian_splatting.utils.schedular import get_expon_lr_func
from trackersplat.motionestimator.fixedview import FixedViewFrameSequenceMetaDataset
from trackersplat.motionestimator.incremental_trainer import TrainerFactory

from .deformation import DeformNetwork


class HexplaneTrainer(BaseTrainer):

    def __init__(
            self, model: GaussianModel,
            basemodel: GaussianModel,
            spatial_lr_scale: float,
            # https://github.com/wanglids/ST-4DGS/blob/bf0dbb13e76bf41b2c2a4ca64063e5d346db7c74/arguments/__init__.py
            lambda_dssim=0.2,
            position_lr_init=0.00016,
            position_lr_final=0.0000016,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=20_000,
            feature_lr=0.0025,
            opacity_lr=0.05,
            scaling_lr=0.005,
            rotation_lr=0.001,
            deformation_lr_init=0.00016,
            deformation_lr_final=0.000016,
            deformation_lr_delay_mult=0.01,
            grid_lr_init=0.0016,
            grid_lr_final=0.00016,
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
        self._deformation = DeformNetwork(**kwargs).to(model._xyz.device)
        self.optimizer.add_param_group({'params': list(self._deformation.get_mlp_parameters()), 'lr': deformation_lr_init * spatial_lr_scale, "name": "deformation"})
        self.optimizer.add_param_group({'params': list(self._deformation.get_grid_parameters()), 'lr': grid_lr_init * spatial_lr_scale, "name": "grid"})
        self.schedulers["deformation"] = get_expon_lr_func(
            lr_init=deformation_lr_init*spatial_lr_scale,
            lr_final=deformation_lr_final*spatial_lr_scale,
            lr_delay_mult=deformation_lr_delay_mult,
            max_steps=position_lr_max_steps)
        self.schedulers["grid"] = get_expon_lr_func(
            lr_init=grid_lr_init*spatial_lr_scale,
            lr_final=grid_lr_final*spatial_lr_scale,
            lr_delay_mult=deformation_lr_delay_mult,
            max_steps=position_lr_max_steps)
        self._curr_step = 0

        self.mean3D_base = basemodel._xyz.detach()
        self.scales_base = basemodel._scaling.detach()
        self.rotations_base = basemodel._rotation.detach()
        self.opacity_base = basemodel._opacity.detach()

        # https://github.com/wanglids/ST-4DGS/blob/bf0dbb13e76bf41b2c2a4ca64063e5d346db7c74/train.py#L161
        # https://github.com/wanglids/ST-4DGS/blob/bf0dbb13e76bf41b2c2a4ca64063e5d346db7c74/scene/dataset_readers.py#L124
        self.per_time = torch.tensor(0).to(model._xyz.device).repeat(model._xyz.shape[0], 1)

    @property
    def model(self) -> GaussianModel:
        means3D_deform, scales_deform, rotations_deform, opacity_deform = self._deformation(self.mean3D_base, self.scales_base, self.rotations_base, self.opacity_base, self.per_time)
        if hasattr(self._model, "_xyz"):
            del self._model._xyz
        if hasattr(self._model, "_scaling"):
            del self._model._scaling
        if hasattr(self._model, "_rotation"):
            del self._model._rotation
        if hasattr(self._model, "_opacity"):
            del self._model._opacity
        self._model._xyz = means3D_deform
        self._model._scaling = scales_deform
        self._model._rotation = rotations_deform
        self._model._opacity = opacity_deform
        return self._model

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
