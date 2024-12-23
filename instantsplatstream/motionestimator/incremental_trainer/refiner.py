import copy
from typing import List
from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import TrainerWrapper
from instantsplatstream.motionestimator import Motion, FixedViewBatchMotionEstimator, MotionCompensater, FixedViewFrameSequenceMeta, FixedViewFrameSequenceMetaDataset

from .abc import training, compare, IncrementalTrainingMotionEstimator


class MaskedTrainer(TrainerWrapper):
    def __init__(self, trainer, mask):
        super().__init__(trainer)
        self.mask = mask

    def optim_step(self):
        if self.mask is not None:
            self.model._xyz.grad[self.mask] = 0
            self.model._features_dc.grad[self.mask] = 0
            self.model._features_rest.grad[self.mask] = 0
            self.model._scaling.grad[self.mask] = 0
            self.model._rotation.grad[self.mask] = 0
            self.model._opacity.grad[self.mask] = 0
        return super().optim_step()


class IncrementalTrainingRefiner(IncrementalTrainingMotionEstimator):
    def __init__(self, base_batch_func: FixedViewBatchMotionEstimator, base_compensater: MotionCompensater, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_batch_func = base_batch_func
        self.base_compensater = base_compensater

    def __call__(self, views: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        motions = []
        for i, motion in zip(range(1, len(views[0].frames_path)), self.base_batch_func(views)):
            motion.validate()
            curr_frame = copy.deepcopy(self.base_compensater.compensate(self.baseframe, motion))
            dataset = FixedViewFrameSequenceMetaDataset(views, i, self.device)
            trainer = MaskedTrainer(self.trainer_factory(curr_frame, self.baseframe, dataset), motion.fixed_mask)
            training(dataset, trainer, self.iteration)
            motions.append(compare(self.baseframe, curr_frame))
        return motions

    def update_baseframe(self, frame: GaussianModel) -> 'IncrementalTrainingMotionEstimator':
        self.baseframe = frame
        self.base_compensater.update_baseframe(self.baseframe)
        return self
