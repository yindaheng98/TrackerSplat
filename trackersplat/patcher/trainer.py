from typing import Callable
import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import OpacityResetter
from gaussian_splatting.trainer.densifier import AbstractDensifier, SplitCloneDensifier, DensificationInstruct, DensificationTrainer, NoopDensifier


class GradientAttractDensifier(SplitCloneDensifier):
    """Move low-opacity gaussians to high gradient areas."""

    def densify(self) -> DensificationInstruct:
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        clone = self.densify_and_clone(grads, self.densify_grad_threshold, self.scene_extent)
        split = self.densify_and_split(grads, self.densify_grad_threshold, self.scene_extent)

        return DensificationInstruct(
            new_xyz=torch.cat((clone.new_xyz, split.new_xyz), dim=0),
            new_features_dc=torch.cat((clone.new_features_dc, split.new_features_dc), dim=0),
            new_features_rest=torch.cat((clone.new_features_rest, split.new_features_rest), dim=0),
            new_opacities=torch.cat((clone.new_opacities, split.new_opacities), dim=0),
            new_scaling=torch.cat((clone.new_scaling, split.new_scaling), dim=0),
            new_rotation=torch.cat((clone.new_rotation, split.new_rotation), dim=0),
            remove_mask=split.remove_mask
        )


def GradientAttractTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, float], AbstractDensifier],
        model: GaussianModel,
        scene_extent: float,
        *args,
        densify_from_iter=500,
        densify_until_iter=1000,
        densify_interval=100,
        densify_grad_threshold=0.0002,
        densify_percent_dense=0.01,
        densify_percent_too_big=0.8,
        **kwargs):
    densifier = noargs_base_densifier_constructor(model, scene_extent)
    densifier = GradientAttractDensifier(
        densifier,
        scene_extent,
        densify_from_iter=densify_from_iter,
        densify_until_iter=densify_until_iter,
        densify_interval=densify_interval,
        densify_grad_threshold=densify_grad_threshold,
        densify_percent_dense=densify_percent_dense,
        densify_percent_too_big=densify_percent_too_big
    )
    return DensificationTrainer(
        model, scene_extent,
        densifier,
        *args, **kwargs
    )


def GradientAttractDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        *args, **kwargs):
    return GradientAttractTrainerWrapper(
        lambda model, scene_extent: NoopDensifier(model),
        model,
        scene_extent,
        *args, **kwargs
    )


def OpacityResetGradientAttractDensificationTrainer(
        model: GaussianModel, scene_extent: float,
        *args,
        opacity_reset_from_iter=0,
        opacity_reset_until_iter=2,
        opacity_reset_interval=2,
        **kwargs):
    trainer = OpacityResetter(
        base_trainer=GradientAttractDensificationTrainer(model, scene_extent, *args, **kwargs),
        opacity_reset_from_iter=opacity_reset_from_iter,
        opacity_reset_until_iter=opacity_reset_until_iter,
        opacity_reset_interval=opacity_reset_interval,
    )
    return trainer


PatchDensificationTrainer = OpacityResetGradientAttractDensificationTrainer
