from typing import Iterable, List, Tuple
import torch
from gaussian_splatting import GaussianModel
from trackersplat.motionestimator import Motion, FixedViewBatchMotionEstimator, MotionCompensater, FixedViewFrameSequenceMeta, FixedViewFrameSequenceMetaDataset
from trackersplat.motionestimator.incremental_trainer import TrainerFactory, TrainingProcess, BaseTrainingProcess, build_trainer_factory
from .abc import MotionRefiner


class IncrementalTrainingRefiner(MotionRefiner):
    def __init__(
            self,
            base_batch_func: FixedViewBatchMotionEstimator, base_compensater: MotionCompensater,  # 2 of 3 basic args
            trainer_factory: TrainerFactory,
            training_proc: TrainingProcess = BaseTrainingProcess(),
            iteration=1000,
            device=torch.device("cuda"),  # 1 of 3 basic args
    ):
        super().__init__(base_batch_func=base_batch_func, base_compensater=base_compensater, device=device)
        self.trainer_factory = trainer_factory
        self.training = training_proc
        self.iteration = iteration

    def refine(self, frames: Iterable[Tuple[Motion, GaussianModel]], views: List[FixedViewFrameSequenceMeta], baseframe: GaussianModel) -> Iterable[GaussianModel]:
        for i, (motion, frame) in zip(range(1, len(views[0].frames_path)), frames):
            dataset = FixedViewFrameSequenceMetaDataset(views, i, self.device)
            trainer = self.trainer_factory(frame, baseframe, dataset, motion.fixed_mask)
            self.training(dataset, trainer, self.iteration, views[0].frame_idx[i])
            yield frame


def build_training_refiner(
        trainer: str,
        base_batch_func: FixedViewBatchMotionEstimator, base_compensater: MotionCompensater,  # 2 of 3 basic args
        training_proc: TrainingProcess = BaseTrainingProcess(),
        iteration=1000,
        device=torch.device("cuda"),  # 1 of 3 basic args
        *args, **kwargs) -> IncrementalTrainingRefiner:
    return IncrementalTrainingRefiner(
        base_batch_func=base_batch_func, base_compensater=base_compensater,
        trainer_factory=build_trainer_factory(trainer, *args, **kwargs),
        training_proc=training_proc,
        iteration=iteration,
        device=device,
    )
