from instantsplatstream.featurefuser import FeatureFuser, Dinov2SegFeatureExtractor
from mmseg.ops import resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.builder import HEADS
import torch.nn as nn
import torchvision
import dinov2.eval.segmentation.utils.colormaps as colormaps
import numpy as np
from PIL import Image
from mmcv.runner import load_checkpoint
import mmcv
import urllib
import math
import itertools
from functools import partial

import torch
import torch.nn.functional as F
from mmseg.apis import init_segmentor, inference_segmentor


from mmcv.runner import BaseModule
from mmseg.models.builder import BACKBONES
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
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


BACKBONE_SIZE = "small"  # in ("small", "base", "large" or "giant")


backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"

backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
backbone_model.eval()
backbone_model.cuda()


HEAD_SCALE_COUNT = 3  # more scales: slower but better results, in (1,2,3,4,5)
HEAD_DATASET = "ade20k"  # in ("ade20k", "voc2012")
HEAD_TYPE = "ms"  # in ("ms, "linear")


DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

with urllib.request.urlopen(head_config_url) as f:
    cfg_str = f.read().decode()
cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
if HEAD_TYPE == "ms":
    cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
    print("scales:", cfg.data.test.pipeline[1]["img_ratios"])

model = init_segmentor(cfg)
model.backbone.forward = partial(
    backbone_model.get_intermediate_layers,
    n=cfg.model.backbone.out_indices,
    reshape=True,
)
if hasattr(backbone_model, "patch_size"):
    model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
model.init_weights()
load_checkpoint(model, head_checkpoint_url, map_location="cpu")
model.cuda()
model.eval()

with open("./data/truck/images/000001.jpg", "rb") as f:
    image = Image.open(f).convert("RGB")


DATASET_COLORMAPS = {
    "ade20k": colormaps.ADE20K_COLORMAP,
    "voc2012": colormaps.VOC2012_COLORMAP,
}

array = np.array(image)[:, :, ::-1]  # BGR
segmentation_logits = inference_segmentor(model, array)[0]
colormap = DATASET_COLORMAPS[HEAD_DATASET]
colormap_array = np.array(colormap, dtype=np.uint8)
segmentation_values = colormap_array[segmentation_logits + 1]
segmented_image = Image.fromarray(segmentation_values)
segmented_image.show()


image = torch.from_numpy(array.copy()).float().permute(2, 0, 1)
norm = torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])(image)
pad = CenterPadding(backbone_model.patch_size)(norm)
with torch.no_grad():
    features = backbone_model.get_intermediate_layers(
        pad.unsqueeze(0).cuda(),
        n=cfg.model.backbone.out_indices,
        reshape=True,)
    head = model.decode_head(features)
    print(head)
    out = resize(
        input=head,
        size=pad.shape[1:],
        mode='bilinear',
        align_corners=False)
    out = F.softmax(out, dim=1).argmax(dim=1).squeeze(0)

segmentation_values = colormap_array[out.cpu().numpy() + 1]
segmented_image = Image.fromarray(segmentation_values)
segmented_image.show()

extractor = Dinov2SegFeatureExtractor(device='cuda')
with torch.no_grad():
    out = extractor.extract_features(image.cuda())
    out = F.softmax(out.unsqueeze(0), dim=1).argmax(dim=1).squeeze(0)

segmentation_values = colormap_array[out.cpu().numpy() + 1]
segmented_image = Image.fromarray(segmentation_values)
segmented_image.show()
