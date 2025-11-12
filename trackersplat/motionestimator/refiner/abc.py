import copy
from typing import List
from abc import ABCMeta, abstractmethod
import torch
from gaussian_splatting import GaussianModel
from trackersplat.motionestimator import Motion, FixedViewBatchMotionEstimator, MotionCompensater, FixedViewFrameSequenceMeta, FixedViewFrameSequenceMetaDataset
from trackersplat.motionestimator.utils import compare


class MotionRefiner(FixedViewBatchMotionEstimator, metaclass=ABCMeta):
    def __init__(self, base_batch_func: FixedViewBatchMotionEstimator, base_compensater: MotionCompensater, device=torch.device("cuda")):
        self.base_batch_func = base_batch_func
        self.base_compensater = base_compensater
        self.to(device)

    def to(self, device: torch.device) -> 'MotionRefiner':
        self.device = device
        return self

    def __call__(self, views: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        motions = []
        for motion in self.base_batch_func(views):
            motion.validate()
            motions.append(motion)
        frames = [copy.deepcopy(self.base_compensater.compensate(self.baseframe, motion)) for motion in motions]
        return [compare(self.baseframe, frame) for frame in self.refine(motions, frames, views, self.baseframe)]

    @abstractmethod
    def refine(self, motions: List[Motion], frames: List[GaussianModel], views: List[FixedViewFrameSequenceMeta], baseframe: GaussianModel) -> List[GaussianModel]:
        raise NotImplementedError

    def update_baseframe(self, frame: GaussianModel) -> 'MotionRefiner':
        self.baseframe = frame
        self.base_compensater.update_baseframe(self.baseframe)
        return self
