import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import BaseTrainer, OpacityResetDensificationTrainer, OpacityResetAdaptiveDensificationTrainer
from reduced_3dgs.combinations import PrunerInDensifyTrainer, PrunerInAdaptiveDensifyTrainer
from reduced_3dgs.combinations import SHCullingDensificationTrainer, SHCullingAdaptiveDensificationTrainer
from reduced_3dgs.combinations import SHCullingPrunerInDensifyTrainer, SHCullingPrunerInAdaptiveDensifyTrainer
from trackersplat import MotionEstimator
from trackersplat.dataset import VideoCameraDataset
from trackersplat.motionestimator import FixedViewFrameSequenceMetaDataset
from trackersplat.motionestimator.incremental_trainer import TrainerFactory
from .abc import PatchCompensater
from .training import TrainingPatchCompensater, TrainingProcess, BaseTrainingProcess
from .trainer import PatchDensificationTrainer

densification_trainer = {
    "densify": OpacityResetDensificationTrainer,
    "adaptivedensify": OpacityResetAdaptiveDensificationTrainer,
    "densify-pruning": PrunerInDensifyTrainer,
    "adaptivedensify-pruning": PrunerInAdaptiveDensifyTrainer,
    "densify-shculling": SHCullingDensificationTrainer,
    "adaptivedensify-shculling": SHCullingAdaptiveDensificationTrainer,
    "densify-prune-shculling": SHCullingPrunerInDensifyTrainer,
    "adaptivedensify-prune-shculling": SHCullingPrunerInAdaptiveDensifyTrainer,
    "patch-densify": PatchDensificationTrainer,
}


class DensificationTrainerFactory(TrainerFactory):
    def __init__(self, trainer: str, *args, **kwargs):
        self.trainer = densification_trainer[trainer]
        self.input_dataset = trainer not in ["densify", "adaptivedensify", "patch-densify"]
        self.args = args
        self.kwargs = kwargs

    def __call__(self, model: GaussianModel, dataset: FixedViewFrameSequenceMetaDataset) -> BaseTrainer:
        return self.trainer(model, dataset.scene_extent(), dataset, *self.args, **self.kwargs) if self.input_dataset else self.trainer(model, dataset.scene_extent(), *self.args, **self.kwargs)


def build_densification_patcher(
        dataset: VideoCameraDataset, gaussians: GaussianModel, estimator: MotionEstimator,  # 3 of 6 basic args
        *args,
        training_proc: TrainingProcess = BaseTrainingProcess(),
        iteration: int = 1000,
        device: torch.device = "cuda",  # 1 of 6 basic args
        patch_every_n_frames: int = 1, patch_every_n_updates: int = 1,  # 2 of 6 basic args
        **kwargs) -> PatchCompensater:
    return TrainingPatchCompensater(
        trainer_factory=DensificationTrainerFactory(*args, **kwargs),
        dataset=dataset, gaussians=gaussians, estimator=estimator,  # 3 of 6 basic args
        training_proc=training_proc,
        iteration=iteration,
        device=device,  # 1 of 6 basic args
        patch_every_n_frames=patch_every_n_frames, patch_every_n_updates=patch_every_n_updates,  # 2 of 6 basic args
    )
