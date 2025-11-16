from gaussian_splatting import GaussianModel
import torch
from trackersplat import MotionEstimator
from trackersplat.dataset import VideoCameraDataset, FrameCameraDataset
from trackersplat.motionestimator.incremental_trainer import TrainerFactory, TrainingProcess, BaseTrainingProcess
from .abc import PatchCompensater


class TrainingPatchCompensater(PatchCompensater):
    def __init__(
            self,
            trainer_factory: TrainerFactory,
            dataset: VideoCameraDataset, gaussians: GaussianModel, estimator: MotionEstimator,  # 3 of 4 basic args
            training_proc: TrainingProcess = BaseTrainingProcess(),
            iteration=1000,
            device: torch.device = "cuda",  # 1 of 4 basic args
    ):
        super().__init__(dataset=dataset, gaussians=gaussians, estimator=estimator, device=device)  # 4 basic args for PatchCompensater
        self.trainer_factory = trainer_factory
        self.training = training_proc
        self.iteration = iteration

    def patch(self, gaussians: GaussianModel, dataset: FrameCameraDataset, frame_idx: int) -> GaussianModel:
        trainer = self.trainer_factory(gaussians, self.baseframe, dataset, False)
        self.training(dataset, trainer, self.iteration, frame_idx)
        return gaussians
