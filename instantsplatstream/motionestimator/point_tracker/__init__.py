from .abc import PointTrackSequence, PointTracker, MotionFuser, PointTrackMotionEstimator
from .dot import DotPointTracker, DotMotionEstimator, Cotracker3DotMotionEstimator, TapirDotMotionEstimator, BootsTapirDotMotionEstimator
from .cotracker import Cotracker3PointTracker, Cotracker3MotionEstimator
from .fuser import BaseMotionFuser


def build_motion_estimator(estimator: str, fuser: MotionFuser, device="cuda", *args, **kwargs) -> PointTrackMotionEstimator:
    match estimator:
        case "dot":
            estimator = DotMotionEstimator(fuser=fuser, device=device, *args, **kwargs)
        case "dot-tapir":
            estimator = TapirDotMotionEstimator(fuser=fuser, device=device, *args, **kwargs)
        case "dot-bootstapir":
            estimator = BootsTapirDotMotionEstimator(fuser=fuser, device=device, *args, **kwargs)
        case "dot-cotracker3":
            estimator = Cotracker3DotMotionEstimator(fuser=fuser, device=device, *args, **kwargs)
        case "cotracker3":
            estimator = Cotracker3MotionEstimator(fuser=fuser, device=device, *args, **kwargs)
        case _:
            raise ValueError(f"Unknown extractor: {estimator}")
    return estimator
