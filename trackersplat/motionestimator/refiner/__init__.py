"""
This module implements refinement based on incremental training.
Core abstract class: IncrementalTrainingRefiner
"""
from .training import build_training_refiner, IncrementalTrainingRefiner
from .build import build_compensater_with_refine
