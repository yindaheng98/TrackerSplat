from gaussian_splatting.trainer.densifier import AbstractDensifier, DensificationTrainer, NoopDensifier
from gaussian_splatting.trainer.densifier.trainer import cat_tensors_to_optimizer, mask_tensors_in_optimizer
from gaussian_splatting.utils.schedular import get_expon_lr_func

from .patchable_model import PatchableGaussianModel


class PatchDensificationTrainer(DensificationTrainer):
    '''Train new-added patch gaussians with patch-specific learning rates.'''

    def __init__(
        self,
        model: PatchableGaussianModel,
        scene_extent: float,
        densifier: AbstractDensifier = NoopDensifier(),
        patch_position_lr_init=0.00016,
        patch_position_lr_final=0.0000016,
        patch_position_lr_delay_mult=0.01,
        patch_position_lr_max_steps=30_000,
        patch_feature_lr=0.0025,
        patch_opacity_lr=0.025,
        patch_scaling_lr=0.005,
        patch_rotation_lr=0.001,
        *args,
        **kwargs
    ):
        super().__init__(model.base, scene_extent, densifier, *args, **kwargs)
        self.optimizer.add_param_group({'params': [model._xyz], 'lr': patch_position_lr_init * scene_extent, "name": "patch_xyz"})
        self.optimizer.add_param_group({'params': [model._features_dc], 'lr': patch_feature_lr, "name": "patch_f_dc"})
        self.optimizer.add_param_group({'params': [model._features_rest], 'lr': patch_feature_lr / 20.0, "name": "patch_f_rest"})
        self.optimizer.add_param_group({'params': [model._opacity], 'lr': patch_opacity_lr, "name": "patch_opacity"})
        self.optimizer.add_param_group({'params': [model._scaling], 'lr': patch_scaling_lr, "name": "patch_scaling"})
        self.optimizer.add_param_group({'params': [model._rotation], 'lr': patch_rotation_lr, "name": "patch_rotation"})
        self.schedulers["patch_xyz"] = get_expon_lr_func(
            lr_init=patch_position_lr_init*scene_extent,
            lr_final=patch_position_lr_final*scene_extent,
            lr_delay_mult=patch_position_lr_delay_mult,
            max_steps=patch_position_lr_max_steps,
        )

    def add_points(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        optimizable_tensors = cat_tensors_to_optimizer(self.optimizer, {
            "patch_xyz": new_xyz,
            "patch_f_dc": new_features_dc,
            "patch_f_rest": new_features_rest,
            "patch_opacity": new_opacities,
            "patch_scaling": new_scaling,
            "patch_rotation": new_rotation})

        self.model.update_points_add(
            xyz=optimizable_tensors["patch_xyz"],
            features_dc=optimizable_tensors["patch_f_dc"],
            features_rest=optimizable_tensors["patch_f_rest"],
            opacity=optimizable_tensors["patch_opacity"],
            scaling=optimizable_tensors["patch_scaling"],
            rotation=optimizable_tensors["patch_rotation"],
        )

    def remove_points(self, rm_mask):
        optimizable_tensors = mask_tensors_in_optimizer(self.optimizer, rm_mask, ["patch_xyz", "patch_f_dc", "patch_f_rest", "patch_opacity", "patch_scaling", "patch_rotation"])

        self.model.update_points_remove(
            removed_mask=rm_mask,
            xyz=optimizable_tensors["patch_xyz"],
            features_dc=optimizable_tensors["patch_f_dc"],
            features_rest=optimizable_tensors["patch_f_rest"],
            opacity=optimizable_tensors["patch_opacity"],
            scaling=optimizable_tensors["patch_scaling"],
            rotation=optimizable_tensors["patch_rotation"],
        )
