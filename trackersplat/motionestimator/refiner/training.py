from typing import Iterable, List, Tuple
import torch
from gaussian_splatting import GaussianModel
from trackersplat.motionestimator import FixedViewBatchMotionEstimator, FixedViewBatchMotionEstimatorWrapper, FixedViewFrameSequenceMeta, FixedViewFrameSequenceMetaDataset
from trackersplat.motionestimator.incremental_trainer import TrainerFactory, TrainingProcess, BaseTrainingProcess, build_trainer_factory
from trackersplat import Motion, compare, compensate


class IncrementalTrainingRefiner(FixedViewBatchMotionEstimatorWrapper):
    def __init__(
            self,
            base_batch_func: FixedViewBatchMotionEstimator,  # 1 of 2 basic args
            trainer_factory: TrainerFactory,
            training_proc: TrainingProcess = BaseTrainingProcess(),
            iteration=1000,
            device=torch.device("cuda"),  # 1 of 2 basic args
    ):
        super().__init__(base_batch_func=base_batch_func, device=device)  # 2 basic args for FixedViewBatchMotionEstimatorWrapper
        self.trainer_factory = trainer_factory
        self.training = training_proc
        self.iteration = iteration

    def update_baseframe(self, frame: GaussianModel) -> 'FixedViewBatchMotionEstimatorWrapper':
        frame = frame.to(self.device)
        self.baseframe = frame
        return super().update_baseframe(frame)

    def iterate_motion_frame_tuples(self, views: List[FixedViewFrameSequenceMeta]):
        for motion in super().__call__(views):
            motion.validate()
            frame = compensate(self.baseframe, motion)
            yield motion, frame

    def __call__(self, views: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        return [compare(self.baseframe, frame) for frame in self.refine(self.iterate_motion_frame_tuples(views), views, self.baseframe)]

    def refine(self, frames: Iterable[Tuple[Motion, GaussianModel]], views: List[FixedViewFrameSequenceMeta], baseframe: GaussianModel) -> Iterable[GaussianModel]:
        for i, (motion, frame) in zip(range(1, len(views[0].frames_path)), frames):
            dataset = FixedViewFrameSequenceMetaDataset(views, i, self.device)
            trainer = self.trainer_factory(frame, baseframe, dataset, motion.fixed_mask)
            self.training(dataset, trainer, self.iteration, views[0].frame_idx[i])
            yield frame


def build_training_refiner(
        trainer: str,
        base_batch_func: FixedViewBatchMotionEstimator,  # 1 of 2 basic args
        training_proc: TrainingProcess = BaseTrainingProcess(),
        iteration=1000,
        device=torch.device("cuda"),  # 1 of 2 basic args
        *args, **kwargs) -> IncrementalTrainingRefiner:
    return IncrementalTrainingRefiner(
        base_batch_func=base_batch_func,  # 1 of 2 basic args
        trainer_factory=build_trainer_factory(trainer, *args, **kwargs),
        training_proc=training_proc,
        iteration=iteration,
        device=device,  # 1 of 2 basic args
    )
