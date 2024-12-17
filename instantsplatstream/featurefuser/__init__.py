from .fuser import FeatureExtractor, FeatureFuser
from .dinov2 import Dinov2FeatureExtractor
from .dinov2seg import Dinov2SegFeatureExtractor
from .sam2 import SAM2FeatureExtractor


def build_feature_extractor(extractor: str, configfile: str, checkpoint: str, device: str) -> FeatureExtractor:
    match extractor:
        case "sam2":
            extractor = SAM2FeatureExtractor(configfile, checkpoint, device=device)
        case "dinov2":
            extractor = Dinov2FeatureExtractor(["configs/dinov2/ssl_default_config.yaml", configfile], checkpoint, device=device)
        case "dinov2seg":
            extractor = Dinov2SegFeatureExtractor(BACKBONE_SIZE=configfile, device=device)
        case _:
            raise ValueError(f"Unknown extractor: {extractor}")
    return extractor
