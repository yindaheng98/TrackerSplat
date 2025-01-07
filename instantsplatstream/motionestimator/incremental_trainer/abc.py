import copy
import random
from typing import List
from abc import ABCMeta, abstractmethod
import torch
from tqdm import tqdm
from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.dataset import CameraDataset
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
    def __call__(self, model: GaussianModel, basemodel: GaussianModel, dataset: FixedViewFrameSequenceMetaDataset, mask: torch.Tensor) -> AbstractTrainer:
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
                    pbar.set_postfix({'epoch': step // len(dataset), 'loss': ema_loss_for_log})


class IncrementalTrainingMotionEstimator(FixedViewBatchMotionEstimator):
    def __init__(
            self,
            trainer_factory: TrainerFactory,
            training_proc: TrainingProcess = BaseTrainingProcess(),
            iteration=1000, device=torch.device("cuda")):
        self.trainer_factory = trainer_factory
        self.training = training_proc
        self.iteration = iteration
        self.to(device)

    def to(self, device: torch.device) -> 'IncrementalTrainingMotionEstimator':
        self.device = device
        return self

    def __call__(self, views: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        motions = []
        for i in range(1, len(views[0].frames_path)):
            curr_frame = copy.deepcopy(self.baseframe)
            dataset = FixedViewFrameSequenceMetaDataset(views, i, self.device)
            trainer = self.trainer_factory(curr_frame, self.baseframe, dataset, False)
            self.training(dataset, trainer, self.iteration, i)
            motions.append(compare(self.baseframe, curr_frame))
        return motions

    def update_baseframe(self, frame: GaussianModel) -> 'IncrementalTrainingMotionEstimator':
        self.baseframe = frame
        return self


class Incremental1StepTrainingMotionEstimator(IncrementalTrainingMotionEstimator):

    def __call__(self, views: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        motions = []
        last_frame = self.baseframe
        for i in range(1, len(views[0].frames_path)):
            curr_frame = copy.deepcopy(last_frame)
            dataset = FixedViewFrameSequenceMetaDataset(views, i, self.device)
            trainer = self.trainer_factory(curr_frame, self.baseframe, dataset, False)
            self.training(dataset, trainer, self.iteration, i)
            motions.append(compare(self.baseframe, curr_frame))
            last_frame = curr_frame
        return motions
