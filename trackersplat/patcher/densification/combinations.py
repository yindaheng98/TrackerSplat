from gaussian_splatting import GaussianModel
from .opacity_reset import PatchOpacityResetTrainerWrapper
from .gradientpatch import GradientPatchDensificationTrainer


def OpacityResetGradientPatchDensificationTrainer(
        model: GaussianModel, scene_extent: float,
        *args,
        opacity_reset_from_iter=0,
        opacity_reset_until_iter=2,
        opacity_reset_interval=2,
        patch_opacity_reset_from_iter=3000,
        patch_opacity_reset_until_iter=15000,
        patch_opacity_reset_interval=3000,
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
