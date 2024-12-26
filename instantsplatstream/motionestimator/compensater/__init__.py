from .base import BaseMotionCompensater
from .filter import FilteredMotionCompensater
from .regularization import PropagatedMotionCompensater


def build_motion_compensater(compensater, **kwargs) -> BaseMotionCompensater:
    return {
        "propagate": PropagatedMotionCompensater,
        "filter": FilteredMotionCompensater,
        "base": BaseMotionCompensater
    }[compensater](**kwargs)
