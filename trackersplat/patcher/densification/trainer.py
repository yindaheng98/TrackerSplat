from typing import Callable
import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.utils import build_rotation
from gaussian_splatting.trainer.densifier import AbstractDensifier, AdaptiveSplitCloneDensifier, DensificationInstruct, DensificationTrainer, NoopDensifier


def select_gradient_patch(n_should_select: int, grads: torch.Tensor, grad_threshold: float):
    gradscore = torch.norm(grads, dim=-1)
    if n_should_select > 0:
        _, indices = torch.sort(gradscore, descending=True)
        grad_threshold = gradscore[indices[min(n_should_select, gradscore.shape[0]) - 1]].item()
    return gradscore >= grad_threshold


class GradientPatchDensifier(AdaptiveSplitCloneDensifier):
    """Move low-opacity gaussians to high gradient areas."""

    def __init__(
        self,
        *args,
        densify_target_n=None,
        densify_donot_increase=False,
        **kwargs
    ):
        super().__init__(*args, densify_target_n=densify_target_n, **kwargs)
        self.densify_donot_increase = densify_donot_increase

    def prune(self, n_remove: int, donot_remove_mask: torch.Tensor, grads: torch.Tensor) -> None:
        # TODO: prune by importance score from reduced_3dgs.pruning and reduced_3dgs.importance
        gradscore = torch.norm(grads, dim=-1)
        score_scaling = torch.max(self.model.get_scaling, dim=1).values
        score_opacity = self.model.get_opacity.squeeze(-1)
        score = score_scaling * score_opacity * (gradscore + gradscore[gradscore > 0].min())  # avoid zero
        rest_score = score[~donot_remove_mask]
        _, rest_indices = torch.sort(rest_score, descending=False)
        remove_indices = rest_indices[:n_remove]
        remove_mask = torch.zeros_like(donot_remove_mask, dtype=torch.bool)
        remove_mask[torch.arange(score.shape[0], device=score.device)[~donot_remove_mask][remove_indices]] = True
        return remove_mask

    def densify_and_split(self, selected_pts_mask, scene_extent, N=2):
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.model.get_scaling, dim=1).values > self.densify_percent_dense*scene_extent)
        # N=selected_pts_mask.sum(), add 2N new points and remove N old points

        stds = self.model.get_scaling[selected_pts_mask]
        means = torch.zeros((stds.size(0), 3), device=self.model._xyz.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.model._rotation[selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.model.get_xyz[selected_pts_mask]
        new_scaling = self.model.scaling_inverse_activation(self.model.get_scaling[selected_pts_mask] / (0.8*N))
        new_rotation = self.model._rotation[selected_pts_mask]
        new_features_dc = self.model._features_dc[selected_pts_mask]
        new_features_rest = self.model._features_rest[selected_pts_mask]
        new_opacity = self.model._opacity[selected_pts_mask]

        return DensificationInstruct(
            new_xyz=new_xyz,
            new_features_dc=new_features_dc,
            new_features_rest=new_features_rest,
            new_opacities=new_opacity,
            new_scaling=new_scaling,
            new_rotation=new_rotation,
        )

    def densify(self) -> DensificationInstruct:
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        pts_mask = select_gradient_patch(self.densify_target_n, grads, self.densify_grad_threshold)

        clone = self.densify_and_clone(pts_mask, self.scene_extent)
        split = self.densify_and_split(pts_mask, self.scene_extent)

        remove_mask = None
        if self.densify_donot_increase:
            # remove pts_mask.sum().item() points or as many as possible (there is no enough points to remove)
            n_remove = min(pts_mask.sum().item(), grads.shape[0] - pts_mask.sum().item())
            remove_mask = self.prune(n_remove, pts_mask, grads)

        return DensificationInstruct(
            new_xyz=torch.cat((clone.new_xyz, split.new_xyz), dim=0),
            new_features_dc=torch.cat((clone.new_features_dc, split.new_features_dc), dim=0),
            new_features_rest=torch.cat((clone.new_features_rest, split.new_features_rest), dim=0),
            new_opacities=torch.cat((clone.new_opacities, split.new_opacities), dim=0),
            new_scaling=torch.cat((clone.new_scaling, split.new_scaling), dim=0),
            new_rotation=torch.cat((clone.new_rotation, split.new_rotation), dim=0),
            remove_mask=remove_mask,
        )


def GradientPatchTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, float], AbstractDensifier],
        model: GaussianModel,
        scene_extent: float,
        *args,
        densify_from_iter=500,
        densify_until_iter=1000,
        densify_interval=100,
        densify_grad_threshold=0.0001,
        densify_percent_dense=0.01,
        densify_percent_too_big=0.8,
        densify_target_n=None,  # ! this is different from usual AdaptiveSplitCloneDensifier, densify_target_n here is the target number of new gaussians to add at each densification step
        **kwargs):
    densifier = noargs_base_densifier_constructor(model, scene_extent)
    densifier = GradientPatchDensifier(
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


def GradientPatchDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        *args, **kwargs):
    return GradientPatchTrainerWrapper(
        lambda model, scene_extent: NoopDensifier(model),
        model,
        scene_extent,
        *args, **kwargs
    )
