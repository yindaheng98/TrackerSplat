from abc import ABCMeta, abstractmethod
from gaussian_splatting import GaussianModel
import torch
from trackersplat import MotionCompensater
from trackersplat import MotionEstimator
from trackersplat.dataset import VideoCameraDataset, FrameCameraDataset


class PatchCompensater(MotionCompensater, metaclass=ABCMeta):
    def __init__(self, dataset: VideoCameraDataset, gaussians: GaussianModel, estimator: MotionEstimator, device: torch.device = "cuda"):  # 4 basic args
        super().__init__(gaussians=gaussians, estimator=estimator, device=device)  # 3 basic args for MotionCompensater
        self.dataset = dataset

    def to(self, device):
        self.dataset = self.dataset.to(device)
        return super().to(device)

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
