from .abc import PointTrackSequence, PointTracker, MotionFuser, PointTrackMotionEstimator
from .dot import DotPointTracker, DotMotionEstimator, Cotracker3DotMotionEstimator, TapirDotMotionEstimator, BootsTapirDotMotionEstimator
from .cotracker import Cotracker3PointTracker, Cotracker3MotionEstimator
from .fuser import BaseMotionFuser


def build_point_track_batch_motion_estimator(estimator: str, fuser: MotionFuser, device="cuda", *args, **kwargs) -> PointTrackMotionEstimator:
    return {
        "dot": DotMotionEstimator,
        "dot-tapir": TapirDotMotionEstimator,
        "dot-bootstapir": BootsTapirDotMotionEstimator,
        "dot-cotracker3": Cotracker3DotMotionEstimator,
        "cotracker3": Cotracker3MotionEstimator,
    }[estimator](fuser=fuser, device=device, *args, **kwargs)
