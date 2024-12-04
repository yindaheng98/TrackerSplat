import torch
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from .fuser import FeatureExtractor


class SAM2FeatureExtractor(FeatureExtractor):
    def __init__(
            self,
            configfile: str = "./configs/sam2.1/sam2.1_hiera_l.yaml",
            checkpoint="./checkpoints/sam2.1_hiera_large.pt",
            device: torch.device = torch.device("cuda")):
        super().__init__()
        self.model = build_sam2(configfile, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(self.model)

    @property
    def n_features(self) -> int:
        return self.model.sam_mask_decoder.conv_s0.weight.shape[0] + self.model.sam_mask_decoder.conv_s1.weight.shape[0]

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        return self

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        _, h, w = image.shape
        self.predictor.set_image((image.permute(1, 2, 0) * 255).cpu().numpy())
        feat_s0, feat_s1 = self.predictor._features['high_res_feats']
        features = torch.cat([
            F.interpolate(feat_s0, (h, w), mode="bilinear", align_corners=False),
            F.interpolate(feat_s1, (h, w), mode="bilinear", align_corners=False),
        ], dim=1).squeeze(0)
        return features
