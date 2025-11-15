import torch
from gaussian_splatting import GaussianModel
from trackersplat.dataset import VideoCameraDataset
from trackersplat.motionestimator import FixedViewMotionEstimator, FixedViewBatchMotionEstimator, MotionCompensater
from .training import build_training_refiner


def build_compensater_with_refine(
        type: str, trainer: str, gaussians: GaussianModel, dataset: VideoCameraDataset, batch_size: int,
        base_batch_func: FixedViewBatchMotionEstimator, device=torch.device("cuda"),  # 2 basic args
        *args, **kwargs):
    batch_func = {
        "training": build_training_refiner,
    }[type](
        trainer=trainer,
        base_batch_func=base_batch_func, device=device,  # 2 basic args
        *args, **kwargs)
    motion_estimator = FixedViewMotionEstimator(dataset=dataset, batch_func=batch_func, batch_size=batch_size, device=device)
    motion_compensater = MotionCompensater(gaussians=gaussians, estimator=motion_estimator, device=device)
    return motion_compensater
