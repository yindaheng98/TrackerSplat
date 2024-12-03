import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from omegaconf import OmegaConf
import torchvision
import mmengine
from mmseg.apis import init_model
from mmengine.runner import load_checkpoint
from dinov2.models import build_model_from_cfg
from dinov2.utils.utils import load_pretrained_weights
from mmengine.model import BaseModule
from mmseg.models.utils import resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.builder import HEADS
from mmseg.models.builder import BACKBONES
from .fuser import FeatureExtractor


@BACKBONES.register_module()
class DinoVisionTransformer(BaseModule):
    """Vision Transformer."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()


@HEADS.register_module()
class BNHead(BaseDecodeHead):
    """Just a batchnorm."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.resize_factors = resize_factors

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # print("inputs", [i.shape for i in inputs])
        x = self._transform_inputs(inputs)
        # print("x", x.shape)
        feats = self.bn(x)
        # print("feats", feats.shape)
        return feats

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            # select indices
            inputs = [inputs[i] for i in self.in_index]
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
                inputs = [
                    resize(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
                    for x, f in zip(inputs, self.resize_factors)
                ]
                # print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
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
            OmegaConf.create(OmegaConf.load(f"./configs/dinov2/{backbone_arch}_pretrain.yaml")),
        )
        self.model, self.embed_dim = build_model_from_cfg(config, only_teacher=True)
        backbone_name = f"dinov2_{backbone_arch}"
        load_pretrained_weights(self.model, f"./checkpoints/{backbone_name}_pretrain.pth", "teacher")
        self.patch_size = config.student.patch_size
        head_config_url = f"./configs/dinov2/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
        cfg = mmengine.Config.fromfile(head_config_url)
        if HEAD_TYPE == "ms":
            cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
        self.backbone = partial(
            self.model.get_intermediate_layers,
            n=cfg.model.backbone.out_indices,
            reshape=True,
        )

        # init decode head
        model = init_model(cfg)
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
        image_in = F.pad(self.norm(image), (0, pad_w, 0, pad_h), mode='constant', value=0).unsqueeze(0)
        with torch.no_grad():
            features_out = self.backbone(image_in)
            features_head = self.head(features_out)
            features_resize = F.interpolate(features_head, size=(h + pad_h, w + pad_w), mode='bilinear', align_corners=False).squeeze(0)
            features = features_resize[:, :h, :w]
            return features

    def postprocess_features(self, features):
        return F.softmax(features, dim=1)
