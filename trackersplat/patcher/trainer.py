from typing import Callable
import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import OpacityResetter
from gaussian_splatting.trainer.densifier import AbstractDensifier, AdaptiveSplitCloneDensifier, DensificationInstruct, DensificationTrainer, NoopDensifier


class GradientAttractDensifier(AdaptiveSplitCloneDensifier):
    """Move low-opacity gaussians to high gradient areas."""

    def __init__(
        self,
        *args,
        densify_target_n=None,
        **kwargs
    ):
        super().__init__(*args, densify_target_n=densify_target_n, **kwargs)

    def densify(self) -> DensificationInstruct:
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        too_big_pts_mask = torch.max(self.model.get_scaling, dim=1).values > self.densify_percent_too_big*self.scene_extent
        n_should_select = max(0, self.densify_target_n - too_big_pts_mask.sum().item()) if self.densify_target_n is not None else 0
        gradscore = torch.norm(grads, dim=-1)
        if n_should_select <= 0:
            grad_threshold = self.densify_grad_threshold
        else:
            gradscore_rest = gradscore[~too_big_pts_mask]
            _, indices = torch.sort(gradscore_rest, descending=True)
            grad_threshold = gradscore_rest[indices[min(n_should_select, gradscore_rest.shape[0]) - 1]].item()
        big_grad_pts_mask = gradscore >= max(grad_threshold, self.densify_grad_threshold)
        pts_mask = torch.logical_or(too_big_pts_mask, big_grad_pts_mask)

        clone = self.densify_and_clone(pts_mask, self.scene_extent)
        split = self.densify_and_split(pts_mask, self.scene_extent)

        score_scaling = torch.max(self.model.get_scaling, dim=1).values
        score_opacity = self.model.get_opacity.squeeze(-1)
        score = score_scaling * score_opacity * (gradscore + gradscore[gradscore > 0].min())  # avoid zero

        # delete pts_mask.sum().item() lowest score gaussians outside the selected ones
        n_remove = min(pts_mask.sum().item(), score.shape[0] - pts_mask.sum().item())
        rest_score = score[~pts_mask]
        _, rest_indices = torch.sort(rest_score, descending=False)
        remove_indices = rest_indices[:n_remove]
        remove_mask = torch.zeros_like(pts_mask, dtype=torch.bool)
        remove_mask[torch.arange(score.shape[0], device=score.device)[~pts_mask][remove_indices]] = True

        return DensificationInstruct(
            new_xyz=torch.cat((clone.new_xyz, split.new_xyz), dim=0),
            new_features_dc=torch.cat((clone.new_features_dc, split.new_features_dc), dim=0),
            new_features_rest=torch.cat((clone.new_features_rest, split.new_features_rest), dim=0),
            new_opacities=torch.cat((clone.new_opacities, split.new_opacities), dim=0),
            new_scaling=torch.cat((clone.new_scaling, split.new_scaling), dim=0),
            new_rotation=torch.cat((clone.new_rotation, split.new_rotation), dim=0),
            remove_mask=split.remove_mask | remove_mask,
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
        densify_target_n=None,  # ! this is different from usual AdaptiveSplitCloneDensifier, densify_target_n here is the target number of new gaussians to add at each densification step
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
        densify_percent_too_big=densify_percent_too_big,
        densify_target_n=densify_target_n,
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
