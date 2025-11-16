"""
This module implements refinement based on incremental training.
Core abstract class: IncrementalTrainingRefiner
"""
from .build import build_motion_estimator_with_refine
from .training import build_training_refiner, IncrementalTrainingRefiner
from .filter import FilteredMotionRefiner
from .propogate import PropagatedMotionRefiner
