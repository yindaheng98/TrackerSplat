from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import OpacityResetter
from .trainer import GradientPatchDensificationTrainer


def OpacityResetGradientPatchDensificationTrainer(
        model: GaussianModel, scene_extent: float,
        *args,
        opacity_reset_from_iter=0,
        opacity_reset_until_iter=2,
        opacity_reset_interval=2,
        **kwargs):
    trainer = OpacityResetter(
        base_trainer=GradientPatchDensificationTrainer(model, scene_extent, *args, **kwargs),
        opacity_reset_from_iter=opacity_reset_from_iter,
        opacity_reset_until_iter=opacity_reset_until_iter,
        opacity_reset_interval=opacity_reset_interval,
    )
    return trainer


PatchDensificationTrainer = OpacityResetGradientPatchDensificationTrainer
