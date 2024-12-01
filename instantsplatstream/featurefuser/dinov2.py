from typing import Union, List
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import torchvision
from dinov2.models import build_model_from_cfg
from dinov2.utils.utils import load_pretrained_weights
from .fuser import FeatureExtractor


class Dinov2FeatureExtractor(FeatureExtractor):
    def __init__(self, configfile: Union[List[str], str], checkpoint, device: torch.device = torch.device("cuda")):
        super().__init__()
        if isinstance(configfile, list):
            config = OmegaConf.merge(*[OmegaConf.create(OmegaConf.load(c)) for c in configfile])
        else:
            config = OmegaConf.create(OmegaConf.load(configfile))
        self.model, self.embed_dim = build_model_from_cfg(config, only_teacher=True)
        load_pretrained_weights(self.model, checkpoint, "teacher")
        self.patch_size = config.student.patch_size
        self.to(device)
        self.norm = torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        self.output_indices = [1]

    @property
    def n_features(self) -> int:
        return self.embed_dim

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        return self

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        _, h, w = image.shape
        pad_h = 0 if h % self.patch_size == 0 else self.patch_size - (h % self.patch_size)
        pad_w = 0 if w % self.patch_size == 0 else self.patch_size - (w % self.patch_size)
        image_in = F.pad(self.norm(image), (0, pad_w, 0, pad_h), mode='constant', value=0).transpose(1, 2).unsqueeze(0)
        with torch.no_grad():
            features_out = self.model.get_intermediate_layers(image_in, n=self.output_indices, reshape=True)
            features_out = [feature.squeeze(0).transpose(1, 2) for feature in features_out]
            features_pad = F.interpolate(features_out[0].unsqueeze(0), size=(h + pad_h, w + pad_w), mode='bilinear', align_corners=False).squeeze(0)
            for feature in features_out[1:]:
                features_pad += F.interpolate(feature.unsqueeze(0), size=(h + pad_h, w + pad_w), mode='bilinear', align_corners=False).squeeze(0)
            features = features_pad[:, :h, :w]
            return features
