import copy
import random
from typing import List
from abc import ABCMeta, abstractmethod
import torch
from tqdm import tqdm
from gaussian_splatting import GaussianModel
from gaussian_splatting.camera import build_camera
from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.utils import psnr
from instantsplatstream.motionestimator import Motion, FixedViewBatchMotionEstimator, FixedViewFrameSequenceMeta
from gaussian_splatting.utils import quaternion_raw_multiply
from instantsplatstream.utils import quaternion_invert


class FrameCameraDataset(CameraDataset):
    def __init__(self, views: List[FixedViewFrameSequenceMeta], frame_idx: int, device):
        super().__init__()
        self.raw_cameras = views
        self.frame_idx = frame_idx
        self.to(device)

    def to(self, device):
        self.cameras = [build_camera(
            image_height=cam.image_height, image_width=cam.image_width,
            FoVx=cam.FoVx, FoVy=cam.FoVy,
            R=cam.R.to(device), T=cam.T.to(device),
            image_path=cam.frames_path[self.frame_idx],
            device=device
        ) for cam in self.raw_cameras]
        return self

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx]


def training(dataset: CameraDataset, trainer: AbstractTrainer, iteration: int):
    pbar = tqdm(range(1, iteration+1))
    epoch = list(range(len(dataset)))
    epoch_psnr = torch.empty(3, 0)
    ema_loss_for_log = 0.0
    avg_psnr_for_log = 0.0
    for step in pbar:
        epoch_idx = step % len(dataset)
        if epoch_idx == 0:
            avg_psnr_for_log = epoch_psnr.mean().item()
            epoch_psnr = torch.empty(3, 0)
            random.shuffle(epoch)
        idx = epoch[epoch_idx]
        loss, out = trainer.step(dataset[idx])
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            epoch_psnr = torch.concat([epoch_psnr.to(out["render"].device), psnr(out["render"], dataset[idx].ground_truth_image)], dim=1)
            if step % 10 == 0:
                pbar.set_postfix({'epoch': step // len(dataset), 'loss': ema_loss_for_log, 'psnr': avg_psnr_for_log})


def compare(baseframe: GaussianModel, curframe: GaussianModel) -> Motion:
    with torch.no_grad():
        return Motion(
            translation_vector=curframe._xyz - baseframe._xyz,
            rotation_quaternion=torch.nn.functional.normalize(quaternion_raw_multiply(curframe._rotation, quaternion_invert(baseframe._rotation))),
            scaling_modifier_log=curframe._scaling - baseframe._scaling
        )


class TrainerFactory(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, model: GaussianModel, dataset: FrameCameraDataset) -> AbstractTrainer:
        raise NotImplementedError


class IncrementalTrainingMotionEstimator(FixedViewBatchMotionEstimator, metaclass=ABCMeta):
    def __init__(
            self,
            trainer_factory: TrainerFactory,
            iteration=1000, device=torch.device("cuda")):
        self.trainer_factory = trainer_factory
        self.iteration = iteration
        self.to(device)

    def to(self, device: torch.device) -> 'IncrementalTrainingMotionEstimator':
        self.device = device
        return self

    def __call__(self, views: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        motions = []
        for i in range(1, len(views[0].frames_path)):
            curr_frame = copy.deepcopy(self.baseframe)
            dataset = FrameCameraDataset(views, i, self.device)
            trainer = self.trainer_factory(curr_frame, dataset)
            training(dataset, trainer, self.iteration)
            motions.append(compare(self.baseframe, curr_frame))
        return motions

    def update_baseframe(self, frame: GaussianModel) -> 'IncrementalTrainingMotionEstimator':
        self.baseframe = frame
        return self
