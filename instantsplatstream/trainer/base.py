from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import BaseTrainer
from instantsplatstream.motionestimator.incremental_trainer import TrainerFactory, FrameCameraDataset


class BaseTrainerFactory(TrainerFactory):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, model: GaussianModel, dataset: FrameCameraDataset) -> BaseTrainer:
        return BaseTrainer(model, dataset.scene_extent(), *self.args, **self.kwargs)
