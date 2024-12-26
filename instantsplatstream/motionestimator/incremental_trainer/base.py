from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import BaseTrainer
from instantsplatstream.motionestimator.fixedview import FixedViewFrameSequenceMetaDataset
from .abc import TrainerFactory


class BaseTrainerNoScale(BaseTrainer):
    def optim_step(self):
        self.model._scaling.grad[...] = 0  # no scaling, or the scene will explode
        return super().optim_step()


class BaseTrainerFactory(TrainerFactory):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, model: GaussianModel, basemodel: GaussianModel, dataset: FixedViewFrameSequenceMetaDataset) -> BaseTrainerNoScale:
        return BaseTrainerNoScale(model, dataset.scene_extent(), *self.args, **self.kwargs)
