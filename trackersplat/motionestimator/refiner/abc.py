from typing import Iterable, List, Tuple
from abc import ABCMeta, abstractmethod
from gaussian_splatting import GaussianModel
from trackersplat.motionestimator import Motion, FixedViewBatchMotionEstimatorWrapper, FixedViewFrameSequenceMeta
from trackersplat.motionestimator import compare, compensate


class MotionRefiner(FixedViewBatchMotionEstimatorWrapper, metaclass=ABCMeta):

    def update_baseframe(self, frame: GaussianModel) -> 'FixedViewBatchMotionEstimatorWrapper':
        self.baseframe = frame
        return super().update_baseframe(frame)

    def iterate_motion_frame_tuples(self, views: List[FixedViewFrameSequenceMeta]):
        for motion in self.base_batch_func(views):
            motion.validate()
            frame = compensate(self.baseframe, motion)
            yield motion, frame

    def __call__(self, views: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        return [compare(self.baseframe, frame) for frame in self.refine(self.iterate_motion_frame_tuples(views), views, self.baseframe)]

    @abstractmethod
    def refine(self, frames: Iterable[Tuple[Motion, GaussianModel]], views: List[FixedViewFrameSequenceMeta], baseframe: GaussianModel) -> Iterable[GaussianModel]:
        raise NotImplementedError
