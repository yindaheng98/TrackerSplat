from typing import Callable
import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import OpacityResetter
from gaussian_splatting.trainer.densifier import AbstractDensifier, AdaptiveSplitCloneDensifier, DensificationInstruct, DensificationTrainer, NoopDensifier


class GradientAttractDensifier(AdaptiveSplitCloneDensifier):
    """Move low-opacity gaussians to high gradient areas."""

    def densify(self) -> DensificationInstruct:
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        gradscore = torch.norm(grads, dim=-1)
        score_scaling = torch.max(self.model.get_scaling, dim=1).values
        score_opacity = self.model.get_opacity.squeeze(-1)
        score = score_scaling * score_opacity

        too_big_pts_mask = torch.max(self.model.get_scaling, dim=1).values > self.densify_percent_too_big*self.scene_extent
        n_should_select = max(0, self.densify_target_n - grads.shape[0] - too_big_pts_mask.sum().item())
        gradscore = torch.norm(grads, dim=-1)
        gradscore_rest = gradscore[~too_big_pts_mask]
        _, indices = torch.sort(gradscore_rest, descending=True)
        grad_threshold = gradscore_rest[indices[min(n_should_select, gradscore_rest.shape[0]) - 1]].item()
        if n_should_select <= 0:
            grad_threshold = self.densify_grad_threshold
        big_grad_pts_mask = gradscore >= min(grad_threshold, self.densify_grad_threshold)
        pts_mask = torch.logical_or(too_big_pts_mask, big_grad_pts_mask)

        clone = self.densify_and_clone(pts_mask, self.scene_extent)
        split = self.densify_and_split(pts_mask, self.scene_extent)

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
