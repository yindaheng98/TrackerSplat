import copy
from typing import Iterable, List, Tuple
from abc import ABCMeta, abstractmethod
import torch
from gaussian_splatting import GaussianModel
from trackersplat.motionestimator import Motion, FixedViewBatchMotionEstimator, MotionCompensater, FixedViewFrameSequenceMeta, FixedViewFrameSequenceMetaDataset
from trackersplat.motionestimator.utils import compare


class MotionRefiner(FixedViewBatchMotionEstimator, metaclass=ABCMeta):
    def __init__(
        self,
        base_batch_func: FixedViewBatchMotionEstimator, base_compensater: MotionCompensater, device=torch.device("cuda")  # 3 basic args
    ):
        self.base_batch_func = base_batch_func
        self.base_compensater = base_compensater
        self.to(device)

    def to(self, device: torch.device) -> 'MotionRefiner':
        self.device = device
        return self

    def __call__(self, views: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        def make_tuples(views: List[FixedViewFrameSequenceMeta]):
            for motion in self.base_batch_func(views):
                motion.validate()
                frame = copy.deepcopy(self.base_compensater.compensate(self.baseframe, motion))
                yield motion, frame
        return [compare(self.baseframe, frame) for frame in self.refine(make_tuples(views), views, self.baseframe)]

    @abstractmethod
    def refine(self, frames: Iterable[Tuple[Motion, GaussianModel]], views: List[FixedViewFrameSequenceMeta], baseframe: GaussianModel) -> Iterable[GaussianModel]:
        raise NotImplementedError

    def update_baseframe(self, frame: GaussianModel) -> 'MotionRefiner':
        self.baseframe = frame
        self.base_compensater.update_baseframe(self.baseframe)
        return self
