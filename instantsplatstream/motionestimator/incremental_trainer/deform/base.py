import torch
from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.utils.schedular import get_expon_lr_func
from instantsplatstream.motionestimator.fixedview import FixedViewFrameSequenceMetaDataset
from instantsplatstream.motionestimator.incremental_trainer import TrainerFactory, BaseTrainer

from .deformation import DeformNetwork


class HexplaneTrainer(BaseTrainer):

    def __init__(
            self, model: GaussianModel,
            basemodel: GaussianModel,
            spatial_lr_scale: float,
            deformation_lr_init=0.00016,
            deformation_lr_final=0.000016,
            deformation_lr_delay_mult=0.01,
            grid_lr_init=0.0016,
            grid_lr_final=0.00016,
            kwargs_hexplane={},
            position_lr_max_steps=30_000,
            *args, **kwargs):
        super().__init__(
            model, spatial_lr_scale, *args,
            position_lr_init=0, position_lr_final=0,  # Do not train the position, train the deformation
            position_lr_max_steps=position_lr_max_steps,
            **kwargs)
        self._deformation = DeformNetwork(**kwargs_hexplane)
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
        self.basemodel = basemodel

    def forward_backward(self, camera: Camera):
        out = self.model(camera)
        loss = self.loss(out, camera)
        loss.backward()
        return loss, out


class HexplaneTrainerFactory(TrainerFactory):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, model: GaussianModel, basemodel: GaussianModel, dataset: FixedViewFrameSequenceMetaDataset, mask: torch.Tensor) -> HexplaneTrainer:
        return HexplaneTrainer(model, basemodel, dataset.scene_extent(), *self.args, **self.kwargs)
