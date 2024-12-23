
from instantsplatstream.motionestimator.incremental_trainer import TrainerFactory
from .base import BaseTrainerFactory
from .regularization import RegularizedTrainerFactory


def build_trainer_factory(trainer: str, *args, **kwargs) -> TrainerFactory:
    return {
        "base": BaseTrainerFactory,
        "regularized": RegularizedTrainerFactory,
    }[trainer](*args, **kwargs)
