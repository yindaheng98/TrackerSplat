from abc import ABCMeta, abstractmethod
from gaussian_splatting import GaussianModel
import torch
from trackersplat import MotionCompensater, MotionEstimator
from trackersplat.dataset import VideoCameraDataset, FrameCameraDataset


class PatchCompensater(MotionCompensater, metaclass=ABCMeta):
    def __init__(
        self,
        dataset: VideoCameraDataset,
        gaussians: GaussianModel, estimator: MotionEstimator, device: torch.device = "cuda",  # 3 basic args
        patch_every_n_frames: int = 1, patch_every_n_updates: int = 1,
    ):
        self.dataset = dataset
        self.patch_every_n_frames = patch_every_n_frames
        self.patch_every_n_updates = patch_every_n_updates
        super().__init__(gaussians=gaussians, estimator=estimator, device=device)  # 3 basic args for MotionCompensater

    def to(self, device):
        self.dataset = self.dataset.to(device)
        return super().to(device)

    def __iter__(self) -> 'MotionCompensater':
        self.estimator = self.estimator.__iter__()
        self.update_baseframe(self.initframe)
        self.frame_idx = 1
        self.update_idx = 1
        return self

    def __next__(self) -> GaussianModel:
        # TODO: a solution to batch processing:
        # TODO: run self.estimator.__next__() multiple times until motion.update_baseframe is true
        # TODO: collect all the frames to be patched
        motion = self.estimator.__next__()
        motion.validate()
        currframe = self.compensate(self.baseframe, motion)

        if (self.frame_idx % self.patch_every_n_frames == 0) or (motion.update_baseframe and (self.update_idx % self.patch_every_n_updates == 0)):
            currframe = self.patch(currframe, self.dataset[self.frame_idx], self.frame_idx)
        self.frame_idx += 1

        if motion.update_baseframe:
            self.update_baseframe(currframe)
            self.update_idx += 1
        return currframe

    @abstractmethod
    def patch(self, gaussians: GaussianModel, dataset: FrameCameraDataset, frame_idx: int) -> GaussianModel:
        pass
