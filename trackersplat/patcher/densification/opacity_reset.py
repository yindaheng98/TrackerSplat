from typing import Callable
import torch

from gaussian_splatting.gaussian_model import GaussianModel
from gaussian_splatting.trainer import AbstractTrainer, OpacityResetter
from gaussian_splatting.trainer.opacity_reset import replace_tensor_to_optimizer

from .trainer import PatchDensificationTrainer


class PatchOpacityResetter(OpacityResetter):
    def __init__(
            self, base_trainer: PatchDensificationTrainer,
            opacity_reset_from_iter=0,
            opacity_reset_until_iter=2,
            opacity_reset_interval=2,
            patch_opacity_reset_from_iter=500,
            patch_opacity_reset_until_iter=750,
            patch_opacity_reset_interval=250,
    ):
        super().__init__(
            base_trainer,
            opacity_reset_from_iter=patch_opacity_reset_from_iter,
            opacity_reset_until_iter=patch_opacity_reset_until_iter,
            opacity_reset_interval=patch_opacity_reset_interval,
        )
        self.base_opacity_reset_from_iter = opacity_reset_from_iter
        self.base_opacity_reset_until_iter = opacity_reset_until_iter
        self.base_opacity_reset_interval = opacity_reset_interval

    def optim_step(self):
        with torch.no_grad():
            if self.base_opacity_reset_from_iter <= self.curr_step <= self.base_opacity_reset_until_iter and self.curr_step % self.base_opacity_reset_interval == 0:
                opacities = self.model.base.opacity_activation(self.model.base._opacity)
                opacities_new = self.model.base.inverse_opacity_activation(torch.min(opacities, torch.ones_like(opacities)*0.01))
                optimizable_tensors = replace_tensor_to_optimizer(self.optimizer, opacities_new, "base_opacity")
                self.model.base._opacity = optimizable_tensors["base_opacity"]
                torch.cuda.empty_cache()
        return super().optim_step()


def PatchOpacityResetTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: GaussianModel,
        scene_extent: float,
        *args,
        opacity_reset_from_iter=0,
        opacity_reset_until_iter=2,
        opacity_reset_interval=2,
        patch_opacity_reset_from_iter=500,
        patch_opacity_reset_until_iter=750,
        patch_opacity_reset_interval=250,
        **kwargs) -> OpacityResetter:
    return PatchOpacityResetter(
        base_trainer=base_trainer_constructor(model, scene_extent, *args, **kwargs),
        opacity_reset_from_iter=opacity_reset_from_iter,
        opacity_reset_until_iter=opacity_reset_until_iter,
        opacity_reset_interval=opacity_reset_interval,
        patch_opacity_reset_from_iter=patch_opacity_reset_from_iter,
        patch_opacity_reset_until_iter=patch_opacity_reset_until_iter,
        patch_opacity_reset_interval=patch_opacity_reset_interval,
    )
