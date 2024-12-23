
from instantsplatstream.motionestimator.incremental_trainer import TrainerFactory
from .base import BaseTrainerFactory


def build_trainer_factory(trainer: str, *args, **kwargs) -> TrainerFactory:
    return {
        "base": BaseTrainerFactory,
    }[trainer](*args, **kwargs)
