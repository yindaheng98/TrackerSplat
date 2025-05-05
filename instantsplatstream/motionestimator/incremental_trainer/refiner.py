import copy
from typing import List
from gaussian_splatting import GaussianModel
from instantsplatstream.motionestimator import Motion, FixedViewBatchMotionEstimator, MotionCompensater, FixedViewFrameSequenceMeta, FixedViewFrameSequenceMetaDataset

from .abc import compare, IncrementalTrainingMotionEstimator


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
            trainer = self.trainer_factory(curr_frame, self.baseframe, dataset, motion.fixed_mask)
            self.training(dataset, trainer, self.iteration, views[0].frame_idx[i])
            motions.append(compare(self.baseframe, curr_frame))
        return motions

    def update_baseframe(self, frame: GaussianModel) -> 'IncrementalTrainingMotionEstimator':
        self.baseframe = frame
        self.base_compensater.update_baseframe(self.baseframe)
        return self
