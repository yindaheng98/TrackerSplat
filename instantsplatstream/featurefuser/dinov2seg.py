import math
import torch
import torch.nn.functional as F
from functools import partial
from itertools import chain
from omegaconf import OmegaConf
import torchvision
import mmcv
from mmseg.apis import init_segmentor
from mmcv.runner import load_checkpoint
from mmseg.ops import resize
from dinov2.models import build_model_from_cfg
from dinov2.utils.utils import load_pretrained_weights
from .fuser import FeatureExtractor
import dinov2.eval.segmentation.models


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}

HEAD_SCALE_COUNT = 3  # more scales: slower but better results, in (1,2,3,4,5)


class Dinov2SegFeatureExtractor(FeatureExtractor):
    def __init__(
            self,
            BACKBONE_SIZE="large",  # in ("small", "base", "large" or "giant")
            HEAD_SCALE_COUNT=3,  # more scales: slower but better results, in (1,2,3,4,5)
            HEAD_DATASET="ade20k",  # in ("ade20k", "voc2012")
            HEAD_TYPE="ms",  # in ("ms, "linear")
            device: torch.device = torch.device("cuda")):
        super().__init__()

        # init backbone model
        backbone_arch = backbone_archs[BACKBONE_SIZE]
        config = OmegaConf.merge(
            OmegaConf.create(OmegaConf.load("./configs/dinov2/ssl_default_config.yaml")),
            OmegaConf.create(OmegaConf.load(f"./configs/dinov2/{backbone_arch}_reg4_pretrain.yaml")),
        )
        self.model, self.embed_dim = build_model_from_cfg(config, only_teacher=True)
        backbone_name = f"dinov2_{backbone_arch}"
        load_pretrained_weights(self.model, f"./checkpoints/{backbone_name}_reg4_pretrain.pth", "teacher")
        self.patch_size = config.student.patch_size
        head_config_url = f"./configs/dinov2/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
        cfg = mmcv.Config.fromfile(head_config_url)
        if HEAD_TYPE == "ms":
            cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
        self.backbone = partial(
            self.model.get_intermediate_layers,
            n=cfg.model.backbone.out_indices,
            reshape=True,
        )

        # init decode head
        model = init_segmentor(cfg)
        model.init_weights()
        head_checkpoint_url = f"./checkpoints/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"
        load_checkpoint(model, head_checkpoint_url, map_location="cpu")
        self.head = model.decode_head
        self.to(device)
        self.norm = torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

    @property
    def n_features(self) -> int:
        return 150

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        self.head = self.head.to(device)
        return self

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        _, h, w = image.shape
        pad_h = 0 if h % self.patch_size == 0 else self.patch_size - (h % self.patch_size)
        pad_w = 0 if w % self.patch_size == 0 else self.patch_size - (w % self.patch_size)
        image_in = F.pad(self.norm(image), (0, pad_w, 0, pad_h), mode='constant', value=0).transpose(1, 2).unsqueeze(0)
        with torch.no_grad():
            features_out = self.backbone(image_in)
            features_head = self.head(features_out)
            features_resize = F.interpolate(features_head, size=(h + pad_h, w + pad_w), mode='bilinear', align_corners=False).squeeze(0)
            features = features_resize[:, :h, :w]
            return features

    def postprocess_features(self, features):
        return F.softmax(features, dim=1).argmax(dim=1)
