import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import BaseTrainer
from instantsplatstream.motionestimator import FixedViewFrameSequenceMetaDataset
from .abc import TrainerFactory


class BaseTrainerFactory(TrainerFactory):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, model: GaussianModel, basemodel: GaussianModel, dataset: FixedViewFrameSequenceMetaDataset, mask: torch.Tensor) -> BaseTrainer:
        return BaseTrainer(model, dataset.scene_extent(), *self.args, **self.kwargs)
