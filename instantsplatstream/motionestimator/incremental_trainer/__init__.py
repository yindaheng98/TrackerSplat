from .abc import IncrementalTrainingMotionEstimator, TrainerFactory
from .refiner import IncrementalTrainingRefiner
from .abc import TrainerFactory
from .base import BaseTrainerFactory
from .regularization import RegularizedTrainerFactory, RegularizedTrainer


def build_trainer_factory(trainer: str, *args, **kwargs) -> TrainerFactory:
    return {
        "base": BaseTrainerFactory,
        "regularized": RegularizedTrainerFactory,
    }[trainer](*args, **kwargs)
