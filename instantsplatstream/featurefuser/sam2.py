import torch
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from .fuser import FeatureExtractor


class SAM2FeatureExtractor(FeatureExtractor):
    def __init__(
            self,
            configfile: str = "./configs/sam2.1/sam2.1_hiera_l.yaml",
            checkpoint="./checkpoints/sam2.1_hiera_large.pt",
            device: torch.device = torch.device("cuda")):
        super().__init__()
        self.model = build_sam2(configfile, checkpoint, device=device)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)
        self.original_reset_predictor = self.mask_generator.predictor.__class__.reset_predictor
        self.img_embed = {}

        def custom_reset_predictor(cls):
            self.img_embed = cls._features
            self.original_reset_predictor(cls)

        self.mask_generator.predictor.reset_predictor = custom_reset_predictor.__get__(self.mask_generator.predictor)

    @property
    def n_features(self) -> int:
        return 32

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        return self

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        _ = self.mask_generator.generate((image.permute(1, 2, 0) * 255).cpu().numpy())
        features = F.interpolate(self.img_embed['high_res_feats'][0], size=image.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        return features
