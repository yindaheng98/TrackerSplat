"""
This module implements motion estimation algorithms based on point tracking.
Core abstract class: PointTrackMotionEstimator
Tool abstract classes: PointTracker, MotionFuser
"""
from .abc import PointTrackSequence, PointTracker, MotionFuser, PointTrackMotionEstimator
from .dot import DotPointTracker, DotMotionEstimator, Cotracker3DotMotionEstimator, TapirDotMotionEstimator, BootsTapirDotMotionEstimator
from .cotracker import Cotracker3PointTracker, Cotracker3MotionEstimator
from .fuser import BaseMotionFuser
from .fuserfix import DetectFixMotionFuser
from .parallel import DataParallelPointTrackMotionEstimator


def build_point_track_batch_motion_estimator(
        estimator: str, fuser: MotionFuser, device="cuda",
        mask_input=True, mask_output=True,
        *args, **kwargs) -> PointTrackMotionEstimator:
    estimator = {
        "dot": DotMotionEstimator,
        "dot-tapir": TapirDotMotionEstimator,
        "dot-bootstapir": BootsTapirDotMotionEstimator,
        "dot-cotracker3": Cotracker3DotMotionEstimator,
        "cotracker3": Cotracker3MotionEstimator,
    }[estimator](fuser=fuser, device=device, *args, **kwargs)
    estimator.mask_input = mask_input
    estimator.mask_output = mask_output
    return estimator
