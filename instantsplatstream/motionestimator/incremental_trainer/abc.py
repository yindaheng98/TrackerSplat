import copy
import random
from typing import List
from abc import ABCMeta, abstractmethod
import torch
from tqdm import tqdm
from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.utils import psnr
from instantsplatstream.motionestimator import Motion, FixedViewBatchMotionEstimator, FixedViewFrameSequenceMeta, FixedViewFrameSequenceMetaDataset
from gaussian_splatting.utils import quaternion_raw_multiply
from instantsplatstream.utils import quaternion_invert


def compare(baseframe: GaussianModel, curframe: GaussianModel) -> Motion:
    with torch.no_grad():
        return Motion(
            translation_vector=curframe._xyz - baseframe._xyz,
            rotation_quaternion=torch.nn.functional.normalize(quaternion_raw_multiply(curframe._rotation, quaternion_invert(baseframe._rotation))),
            scaling_modifier_log=curframe._scaling - baseframe._scaling,
            opacity_modifier_log=curframe._opacity - baseframe._opacity,
            features_dc_modifier=curframe._features_dc - baseframe._features_dc,
            features_rest_modifier=curframe._features_rest - baseframe._features_rest
        )


class TrainerFactory(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, model: GaussianModel, basemodel: GaussianModel, dataset: FixedViewFrameSequenceMetaDataset) -> AbstractTrainer:
        raise NotImplementedError


class IncrementalTrainingMotionEstimator(FixedViewBatchMotionEstimator):
    def __init__(
            self,
            trainer_factory: TrainerFactory,
            iteration=1000, device=torch.device("cuda")):
        self.trainer_factory = trainer_factory
        self.iteration = iteration
        self.to(device)

    def to(self, device: torch.device) -> 'IncrementalTrainingMotionEstimator':
        self.device = device
        return self

    @staticmethod
    def training(dataset: CameraDataset, trainer: AbstractTrainer, iteration: int):
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
                    pbar.set_postfix({'epoch': step // len(dataset), 'loss': ema_loss_for_log})

    def __call__(self, views: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        motions = []
        for i in range(1, len(views[0].frames_path)):
            curr_frame = copy.deepcopy(self.baseframe)
            dataset = FixedViewFrameSequenceMetaDataset(views, i, self.device)
            trainer = self.trainer_factory(curr_frame, self.baseframe, dataset)
            self.training(dataset, trainer, self.iteration)
            motions.append(compare(self.baseframe, curr_frame))
        return motions

    def update_baseframe(self, frame: GaussianModel) -> 'IncrementalTrainingMotionEstimator':
        self.baseframe = frame
        return self


class IncrementalTrainingMotionEstimatorWrapper(FixedViewBatchMotionEstimator):
    def __init__(self, base: IncrementalTrainingMotionEstimator):
        self.base = base

    @property
    def baseframe(self) -> GaussianModel:
        return self.base.baseframe

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def trainer_factory(self) -> TrainerFactory:
        return self.base.trainer_factory

    @property
    def iteration(self) -> int:
        return self.base.iteration

    def to(self, device: torch.device) -> 'IncrementalTrainingMotionEstimatorWrapper':
        self.base.to(device)
        return self

    def training(self, dataset: CameraDataset, trainer: AbstractTrainer, iteration: int):
        self.base.training(dataset, trainer, iteration)

    def __call__(self, views: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        motions = []
        for i in range(1, len(views[0].frames_path)):
            curr_frame = copy.deepcopy(self.baseframe)
            dataset = FixedViewFrameSequenceMetaDataset(views, i, self.device)
            trainer = self.trainer_factory(curr_frame, self.baseframe, dataset)
            self.training(dataset, trainer, self.iteration)
            motions.append(compare(self.baseframe, curr_frame))
        return motions

    def update_baseframe(self, frame: GaussianModel) -> 'IncrementalTrainingMotionEstimatorWrapper':
        self.base.update_baseframe(frame)
        return self
