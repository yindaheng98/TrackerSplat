from abc import ABCMeta, abstractmethod
from gaussian_splatting import GaussianModel
from trackersplat import MotionCompensater
from trackersplat.dataset import VideoCameraDataset, FrameCameraDataset


class PatchCompensater(MotionCompensater, metaclass=ABCMeta):
    def __init__(self, dataset: VideoCameraDataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset

    def __iter__(self) -> 'MotionCompensater':
        self.estimator = self.estimator.__iter__()
        self.update_baseframe(self.initframe)
        self.frame_idx = 0
        return self

    def __next__(self) -> GaussianModel:
        motion = self.estimator.__next__()
        motion.validate()
        currframe = self.compensate(self.baseframe, motion)

        currframe = self.patch(currframe, self.dataset[self.frame_idx])
        self.frame_idx += 1

        if motion.update_baseframe:
            self.update_baseframe(currframe)
        return currframe

    @abstractmethod
    def patch(self, gaussians: GaussianModel, dataset: FrameCameraDataset) -> GaussianModel:
        pass
