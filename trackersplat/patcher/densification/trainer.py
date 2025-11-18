import copy
from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer.densifier import AbstractDensifier, DensificationTrainer, NoopDensifier
from gaussian_splatting.utils.schedular import get_expon_lr_func

from .patchable_model import PatchableGaussianModel


class PatchDensificationTrainer(DensificationTrainer):
    '''Train new-added patch gaussians with patch-specific learning rates.'''

    def __init__(
        self,
        model: PatchableGaussianModel,
        scene_extent: float,
        densifier: AbstractDensifier,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30_000,
        feature_lr=0.0025,
        opacity_lr=0.025,
        scaling_lr=0.005,
        rotation_lr=0.001,
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
        super().__init__(
            model, scene_extent, densifier,
            *args,
            position_lr_init=patch_position_lr_init,
            position_lr_final=patch_position_lr_final,
            position_lr_delay_mult=patch_position_lr_delay_mult,
            position_lr_max_steps=patch_position_lr_max_steps,
            feature_lr=patch_feature_lr,
            opacity_lr=patch_opacity_lr,
            scaling_lr=patch_scaling_lr,
            rotation_lr=patch_rotation_lr,
            **kwargs)  # model._xyz etc. are patched model's tensors, so use patch_*_lr here

        # model.base._xyz etc. are base model's tensors, so use *_lr here
        self.optimizer.add_param_group({'params': [model.base._xyz], 'lr': position_lr_init * scene_extent, "name": "base_xyz"})
        self.optimizer.add_param_group({'params': [model.base._features_dc], 'lr': feature_lr, "name": "base_f_dc"})
        self.optimizer.add_param_group({'params': [model.base._features_rest], 'lr': feature_lr / 20.0, "name": "base_f_rest"})
        self.optimizer.add_param_group({'params': [model.base._opacity], 'lr': opacity_lr, "name": "base_opacity"})
        self.optimizer.add_param_group({'params': [model.base._scaling], 'lr': scaling_lr, "name": "base_scaling"})
        self.optimizer.add_param_group({'params': [model.base._rotation], 'lr': rotation_lr, "name": "base_rotation"})
        self.schedulers["base_xyz"] = get_expon_lr_func(
            lr_init=position_lr_init*scene_extent,
            lr_final=position_lr_final*scene_extent,
            lr_delay_mult=position_lr_delay_mult,
            max_steps=position_lr_max_steps,
        )
