from .filter import FilteredMotionRefiner
from .propogate import PropagatedMotionRefiner


def build_motion_compensater(compensater, **kwargs):
    return {
        "filter": FilteredMotionRefiner,
        "propagate": PropagatedMotionRefiner,
    }[compensater](**kwargs)
