import torch
from trackersplat.motionestimator import FixedViewBatchMotionEstimator, FixedViewBatchMotionEstimatorWrapper
from .training import build_training_refiner
from .filter import FilteredMotionRefiner
from .propogate import PropagatedMotionRefiner


def build_regularization_refiner(
        refiner: str,
        base_batch_func: FixedViewBatchMotionEstimator, device=torch.device("cuda"),  # 2 basic args
        *args, **kwargs):
    return {
        "base": FixedViewBatchMotionEstimatorWrapper,
        "filter": FilteredMotionRefiner,
        "propagate": PropagatedMotionRefiner,
    }[refiner](
        base_batch_func=base_batch_func, device=device,  # 2 basic args
        *args, **kwargs)


def build_refiner(
        refiner: str,
        base_batch_func: FixedViewBatchMotionEstimator, device=torch.device("cuda"),  # 2 basic args
        *args, **kwargs):
    return (build_training_refiner if refiner == "training" else build_regularization_refiner)(
        base_batch_func=base_batch_func, device=device,  # 2 basic args
        *args, **kwargs)
