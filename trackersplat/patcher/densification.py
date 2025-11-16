from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import BaseTrainer, OpacityResetDensificationTrainer, OpacityResetAdaptiveDensificationTrainer
from reduced_3dgs.combinations import PrunerInDensifyTrainer, PrunerInAdaptiveDensifyTrainer
from reduced_3dgs.combinations import SHCullingDensificationTrainer, SHCullingAdaptiveDensificationTrainer
from reduced_3dgs.combinations import SHCullingPrunerInDensifyTrainer, SHCullingPrunerInAdaptiveDensifyTrainer
from trackersplat.motionestimator import FixedViewFrameSequenceMetaDataset
from trackersplat.motionestimator.incremental_trainer import TrainerFactory

densification_trainer = {
    "densify": OpacityResetDensificationTrainer,
    "adaptivedensify": OpacityResetAdaptiveDensificationTrainer,
    "densify-pruning": PrunerInDensifyTrainer,
    "adaptivedensify-pruning": PrunerInAdaptiveDensifyTrainer,
    "densify-shculling": SHCullingDensificationTrainer,
    "adaptivedensify-shculling": SHCullingAdaptiveDensificationTrainer,
    "densify-prune-shculling": SHCullingPrunerInDensifyTrainer,
    "adaptivedensify-prune-shculling": SHCullingPrunerInAdaptiveDensifyTrainer,
}


class DensificationTrainerFactory(TrainerFactory):
    def __init__(self, trainer: str, *args, **kwargs):
        self.trainer = densification_trainer[trainer]
        self.input_dataset = trainer in ["densify", "adaptivedensify"]
        self.args = args
        self.kwargs = kwargs

    def __call__(self, model: GaussianModel, dataset: FixedViewFrameSequenceMetaDataset) -> BaseTrainer:
        return self.trainer(model, dataset.scene_extent(), dataset, *self.args, **self.kwargs) if self.input_dataset else self.trainer(model, dataset, *self.args, **self.kwargs)
