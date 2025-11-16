"""
This module implements refinement based on incremental training or regularization.
Core abstract class: FixedViewBatchMotionEstimatorWrapper
"""
from .training import build_training_refiner, IncrementalTrainingRefiner
from .filter import FilteredMotionRefiner
from .propogate import PropagatedMotionRefiner
from .build import build_regularization_refiner, build_refiner
