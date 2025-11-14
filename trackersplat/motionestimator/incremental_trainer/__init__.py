"""
This module implements motion estimation algorithms based on incremental training.
Core abstract class: IncrementalTrainingMotionEstimator
Tool abstract classes: TrainingProcess, TrainerFactory
"""
from .abc import IncrementalTrainingMotionEstimator, TrainingProcess, BaseTrainingProcess, TrainerFactory
from .abc import TrainerFactory
from .base import BaseTrainer, BaseTrainerFactory
from .regularization import RegularizedTrainerFactory, RegularizedTrainer
from .mask import MaskedBaseTrainerFactory, MaskedRegularizedTrainerFactory, MaskedTrainer
from .deform import HexplaneTrainerFactory, HexplaneTrainer, RegularizedHexplaneTrainerFactory, RegularizedHexplaneTrainer
from .deform import HiCoMTrainer, HiCoMTrainerFactory


def build_trainer_factory(trainer: str, *args, **kwargs) -> TrainerFactory:
    return {
        "base": BaseTrainerFactory,
        "regularized": RegularizedTrainerFactory,
        "masked": MaskedBaseTrainerFactory,
        "maskregularized": MaskedRegularizedTrainerFactory,
        "hexplane": HexplaneTrainerFactory,
        "regularizedhexplane": RegularizedHexplaneTrainerFactory,
        "hicom": HiCoMTrainerFactory,
    }[trainer](*args, **kwargs)
