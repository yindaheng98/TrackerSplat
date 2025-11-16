from .filter import FilteredMotionCompensater
from .regularization import PropagatedMotionCompensater


def build_motion_compensater(compensater, **kwargs):
    return {
        "propagate": PropagatedMotionCompensater,
        "filter": FilteredMotionCompensater,
    }[compensater](**kwargs)
