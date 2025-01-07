from .abc import IncrementalTrainingMotionEstimator, Incremental1StepTrainingMotionEstimator, TrainingProcess, BaseTrainingProcess, TrainerFactory
from .refiner import IncrementalTrainingRefiner
from .abc import TrainerFactory
from .base import BaseTrainer, BaseTrainerFactory
from .regularization import RegularizedTrainerFactory, RegularizedTrainer
from .mask import MaskedBaseTrainerFactory, MaskedRegularizedTrainerFactory, MaskedTrainer


def build_trainer_factory(trainer: str, *args, **kwargs) -> TrainerFactory:
    return {
        "base": BaseTrainerFactory,
        "regularized": RegularizedTrainerFactory,
        "masked": MaskedBaseTrainerFactory,
        "maskregularized": MaskedRegularizedTrainerFactory,
    }[trainer](*args, **kwargs)
