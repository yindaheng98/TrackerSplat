from gaussian_splatting import GaussianModel
from .opacity_reset import PatchOpacityResetTrainerWrapper
from .gradient_patch import GradientPatchDensificationTrainer


def OpacityResetGradientPatchDensificationTrainer(
        model: GaussianModel, scene_extent: float,
        *args,
        opacity_reset_from_iter=0,
        opacity_reset_until_iter=2,
        opacity_reset_interval=2,
        patch_opacity_reset_from_iter=500,
        patch_opacity_reset_until_iter=800,
        patch_opacity_reset_interval=150,
        **kwargs):
    return PatchOpacityResetTrainerWrapper(
        GradientPatchDensificationTrainer, model, scene_extent, *args,
        opacity_reset_from_iter=opacity_reset_from_iter,
        opacity_reset_until_iter=opacity_reset_until_iter,
        opacity_reset_interval=opacity_reset_interval,
        patch_opacity_reset_from_iter=patch_opacity_reset_from_iter,
        patch_opacity_reset_until_iter=patch_opacity_reset_until_iter,
        patch_opacity_reset_interval=patch_opacity_reset_interval,
        **kwargs
    )


PatchDensificationTrainer = OpacityResetGradientPatchDensificationTrainer
