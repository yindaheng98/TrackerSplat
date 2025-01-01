import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import BaseTrainer
from gaussian_splatting.trainer import TrainerWrapper
from instantsplatstream.motionestimator import FixedViewFrameSequenceMetaDataset
from .abc import TrainerFactory
from .regularization import RegularizedTrainer


class MaskedTrainer(TrainerWrapper):
    def __init__(self, trainer, mask):
        super().__init__(trainer)
        self.mask = mask

    def optim_step(self):
        if self.mask is not None:
            self.model._xyz.grad[self.mask] = 0
            self.model._features_dc.grad[self.mask] = 0
            self.model._features_rest.grad[self.mask] = 0
            self.model._scaling.grad[self.mask] = 0
            self.model._rotation.grad[self.mask] = 0
            self.model._opacity.grad[self.mask] = 0
        return super().optim_step()


class MaskedBaseTrainerFactory(TrainerFactory):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, model: GaussianModel, basemodel: GaussianModel, dataset: FixedViewFrameSequenceMetaDataset, mask: torch.Tensor) -> BaseTrainer:
        return MaskedTrainer(BaseTrainer(model, dataset.scene_extent(), *self.args, **self.kwargs), mask)


class MaskedRegularizedTrainerFactory(TrainerFactory):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, model: GaussianModel, basemodel: GaussianModel, dataset: FixedViewFrameSequenceMetaDataset, mask: torch.Tensor) -> RegularizedTrainer:
        return MaskedTrainer(RegularizedTrainer(model, dataset.scene_extent(), *self.args, **self.kwargs), mask)
