import torch
from gaussian_splatting import GaussianModel
from trackersplat.dataset import VideoCameraDataset
from trackersplat.motionestimator import FixedViewMotionEstimator, FixedViewBatchMotionEstimator, MotionCompensater
from .training import build_training_refiner
from .filter import FilteredMotionRefiner
from .propogate import PropagatedMotionRefiner


def build_refiner(
        refiner: str,
        base_batch_func: FixedViewBatchMotionEstimator, device=torch.device("cuda"),  # 2 basic args
        *args, **kwargs):
    return {
        "training": build_training_refiner,
        "filter": FilteredMotionRefiner,
        "propagate": PropagatedMotionRefiner,
    }[refiner](
        base_batch_func=base_batch_func, device=device,  # 2 basic args
        *args, **kwargs)


def build_motion_estimator_with_refine(
        refiner: str, dataset: VideoCameraDataset, batch_size: int,
        base_batch_func: FixedViewBatchMotionEstimator, device=torch.device("cuda"),  # 2 basic args
        *args, **kwargs):
    batch_func = build_refiner(refiner=refiner, base_batch_func=base_batch_func, device=device, *args, **kwargs)
    motion_estimator = FixedViewMotionEstimator(dataset=dataset, batch_func=batch_func, batch_size=batch_size, device=device)
    return motion_estimator
