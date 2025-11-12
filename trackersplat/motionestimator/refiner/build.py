import torch
from gaussian_splatting import GaussianModel
from trackersplat.dataset import VideoCameraDataset
from trackersplat.motionestimator import FixedViewMotionEstimator, FixedViewBatchMotionEstimator, MotionCompensater
from trackersplat.motionestimator.compensater import BaseMotionCompensater
from .training import build_training_refiner


def build_compensater_with_refine(
        type: str, gaussians: GaussianModel, dataset: VideoCameraDataset, batch_size: int,
        base_batch_func: FixedViewBatchMotionEstimator, base_compensater: MotionCompensater, device=torch.device("cuda"),  # 3 basic args
        *args, **kwargs):
    batch_func = {
        "training": build_training_refiner,
    }[type](
        base_batch_func=base_batch_func, base_compensater=base_compensater, device=device,  # 3 basic args
        *args, **kwargs)
    motion_estimator = FixedViewMotionEstimator(dataset=dataset, batch_func=batch_func, device=device, batch_size=batch_size)
    motion_compensater = BaseMotionCompensater(gaussians=gaussians, estimator=motion_estimator, device=device)
    return motion_compensater
