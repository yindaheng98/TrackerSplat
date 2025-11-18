import random
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
import torch
from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting import GaussianModel
from trackersplat import MotionEstimator
from trackersplat.dataset import VideoCameraDataset, FrameCameraDataset
from .abc import PatchCompensater


class TrainerFactory(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, model: GaussianModel, dataset: FrameCameraDataset) -> AbstractTrainer:
        raise NotImplementedError


class TrainingProcess(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, dataset: CameraDataset, trainer: AbstractTrainer, iteration: int, frame_idx: int):
        raise NotImplementedError


class BaseTrainingProcess(TrainingProcess):
    def __call__(self, dataset: CameraDataset, trainer: AbstractTrainer, iteration: int, frame_idx: int):
        '''Overload this method to make your own training'''
        pbar = tqdm(range(1, iteration+1))
        epoch = list(range(len(dataset)))
        ema_loss_for_log = 0.0
        for step in pbar:
            epoch_idx = step % len(dataset)
            if epoch_idx == 0:
                random.shuffle(epoch)
            idx = epoch[epoch_idx]
            loss, out = trainer.step(dataset[idx])
            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if step % 10 == 0:
                    pbar.set_postfix({'epoch': step // len(dataset), 'loss': ema_loss_for_log, 'n': trainer.model.get_xyz.shape[0]})


class TrainingPatchCompensater(PatchCompensater):
    def __init__(
            self,
            trainer_factory: TrainerFactory,
            dataset: VideoCameraDataset, gaussians: GaussianModel, estimator: MotionEstimator,  # 3 of 6 basic args
            training_proc: TrainingProcess = BaseTrainingProcess(),
            iteration=1000,
            device: torch.device = "cuda",  # 1 of 6 basic args
            patch_every_n_frames: int = 1, patch_every_n_updates: int = 1,  # 2 of 6 basic args
    ):
        super().__init__(
            dataset=dataset, gaussians=gaussians, estimator=estimator, device=device,
            patch_every_n_frames=patch_every_n_frames, patch_every_n_updates=patch_every_n_updates
        )  # 6 basic args for PatchCompensater
        self.trainer_factory = trainer_factory
        self.training = training_proc
        self.iteration = iteration

    def patch(self, gaussians: GaussianModel, dataset: FrameCameraDataset, frame_idx: int) -> GaussianModel:
        trainer = self.trainer_factory(gaussians, dataset)
        self.training(dataset, trainer, self.iteration, frame_idx)
        return gaussians
