from typing import Callable
import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.utils import build_rotation
from gaussian_splatting.trainer.densifier import AbstractDensifier, SplitCloneDensifier, DensificationInstruct, NoopDensifier

from .trainer import PatchDensificationTrainer


def select_gradient_patch(n_should_select: int, grads: torch.Tensor, grad_threshold: float):
    gradscore = torch.norm(grads, dim=-1)
    if n_should_select > 0:
        _, indices = torch.sort(gradscore, descending=True)
        grad_threshold = gradscore[indices[min(n_should_select, gradscore.shape[0]) - 1]].item()
    return gradscore >= grad_threshold


class GradientPatchDensifier(SplitCloneDensifier):
    """Move low-opacity gaussians to high gradient areas."""

    def __init__(
        self, base_densifier: AbstractDensifier,
        scene_extent,
        *args,
        densify_n_every_step=1000,
        densify_n_limit=1500,
        **kwargs
    ):
        super().__init__(base_densifier, scene_extent, *args, densify_percent_too_big=None, densify_limit_n=None, **kwargs)
        self.densify_n_every_step = densify_n_every_step
        self.densify_n_limit = densify_n_limit

    def densify_and_split(self, selected_pts_mask, scene_extent, N=2):
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.model.get_scaling, dim=1).values > self.densify_percent_dense*scene_extent)
        # N=selected_pts_mask.sum(), add 2N new points and remove N old points

        stds = self.model.get_scaling[selected_pts_mask]
        means = torch.zeros((stds.size(0), 3), device=self.model.get_xyz.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.model.get_rotation[selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.model.get_xyz[selected_pts_mask]
        new_scaling = self.model.scaling_inverse_activation(self.model.get_scaling[selected_pts_mask] / (0.8*N))
        new_rotation = self.model.get_rotation[selected_pts_mask]
        new_features_dc = self.model.get_features_dc[selected_pts_mask]
        new_features_rest = self.model.get_features_rest[selected_pts_mask]
        new_opacity = self.model.inverse_opacity_activation(self.model.get_opacity[selected_pts_mask])

        return DensificationInstruct(
            new_xyz=new_xyz,
            new_features_dc=new_features_dc,
            new_features_rest=new_features_rest,
            new_opacities=new_opacity,
            new_scaling=new_scaling,
            new_rotation=new_rotation,
        )

    def densify_and_clone(self, selected_pts_mask, scene_extent):
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.model.get_scaling, dim=1).values <= self.densify_percent_dense*scene_extent)
        # N=selected_pts_mask.sum(), add N new points

        new_xyz = self.model.get_xyz[selected_pts_mask]
        new_features_dc = self.model.get_features_dc[selected_pts_mask]
        new_features_rest = self.model.get_features_rest[selected_pts_mask]
        new_opacities = self.model.inverse_opacity_activation(self.model.get_opacity[selected_pts_mask])
        new_scaling = self.model.scaling_inverse_activation(self.model.get_scaling[selected_pts_mask])
        new_rotation = self.model.get_rotation[selected_pts_mask]

        return DensificationInstruct(
            new_xyz=new_xyz,
            new_features_dc=new_features_dc,
            new_features_rest=new_features_rest,
            new_opacities=new_opacities,
            new_scaling=new_scaling,
            new_rotation=new_rotation,
        )

    def prune(self, n_remove: int, grads: torch.Tensor) -> None:
        if n_remove <= 0:
            return None
        # TODO: prune by importance score from reduced_3dgs.pruning and reduced_3dgs.importance
        gradscore = torch.norm(grads, dim=-1)
        score_scaling = torch.max(self.model.scaling_activation(self.model._scaling), dim=1).values
        score_opacity = self.model.opacity_activation(self.model._opacity).squeeze(-1)
        score = score_scaling * score_opacity * (gradscore + gradscore[gradscore > 0].min())  # avoid zero
        _, indices = torch.sort(score, descending=False)
        remove_indices = indices[:n_remove]
        remove_mask = torch.zeros(score.shape, dtype=torch.bool, device=score.device)
        remove_mask[remove_indices] = True
        return remove_mask

    def densify(self) -> DensificationInstruct:
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        pts_mask = select_gradient_patch(self.densify_n_every_step, grads, self.densify_grad_threshold)

        clone = self.densify_and_clone(pts_mask, self.scene_extent)
        split = self.densify_and_split(pts_mask, self.scene_extent)

        # remove pts_mask.sum().item() points or as many as possible (there is no enough points to remove)
        n_should_remove = pts_mask.sum().item() + self.model._xyz.shape[0] - self.densify_n_limit
        n_remove = min(n_should_remove, self.model._xyz.shape[0])
        remove_mask = self.prune(n_remove, grads[self.model.base._xyz.shape[0]:])

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
        densify_n_every_step=1000,
        densify_n_limit=1500,
        **kwargs):
    return PatchDensificationTrainer.from_base_model(
        lambda model, scene_extent: GradientPatchDensifier(
            noargs_base_densifier_constructor(model, scene_extent),
            scene_extent,
            densify_from_iter=densify_from_iter,
            densify_until_iter=densify_until_iter,
            densify_interval=densify_interval,
            densify_grad_threshold=densify_grad_threshold,
            densify_percent_dense=densify_percent_dense,
            densify_n_every_step=densify_n_every_step,
            densify_n_limit=densify_n_limit,
        ),
        model, scene_extent,
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
