import copy
from typing import List
import torch
from torch import nn
from abc import ABCMeta, abstractmethod
from gaussian_splatting import Camera, GaussianModel
from instantsplatstream.utils.featurefusion import feature_fusion


class FeatureExtractor(metaclass=ABCMeta):
    @property
    @abstractmethod
    def n_features(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def to(self, device: torch.device) -> 'FeatureExtractor':
        raise NotImplementedError

    @abstractmethod
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extract_features_batch(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.extract_features(image) for image in images]

    def postprocess_features(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def assign_colors(self, features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        colormap = torch.rand((self.n_features, 3), dtype=torch.float, device=features.device)
        linspace = torch.linspace(0, 1, steps=self.n_features, device=features.device)
        colormap = torch.stack([
            linspace,
            linspace[torch.randperm(self.n_features, device=features.device)],
            torch.flip(linspace, dims=(0,)),
        ]).T
        valid_idx = weights > 1e-5
        sum_weights = features[valid_idx, ...].sum(dim=1)
        sum_colors = (features[valid_idx, ...].unsqueeze(-1) * colormap.unsqueeze(0)).sum(dim=1)
        valid_colors = sum_colors / sum_weights.unsqueeze(-1)
        valid_colors[sum_weights < 1e-5, ...] = 0
        colors = torch.zeros((features.shape[0], 3), dtype=valid_colors.dtype, device=valid_colors.device)
        colors[valid_idx, ...] = valid_colors
        return colors


class FeatureFuser(metaclass=ABCMeta):
    def __init__(self, gaussians: GaussianModel, extractor: FeatureExtractor, fusion_alpha_threshold=0., device: torch.device = torch.device("cuda")):
        self.gaussians = gaussians
        self.extractor = extractor
        self.fusion_alpha_threshold = fusion_alpha_threshold
        n_gaussians = gaussians.get_xyz.shape[0]
        self.features = torch.zeros(size=(n_gaussians, extractor.n_features), dtype=torch.float64)
        self.weights = torch.zeros(size=(n_gaussians,), dtype=torch.float64)
        self.to(device)

    def to(self, device: torch.device) -> 'FeatureFuser':
        self.gaussians = self.gaussians.to(device)
        self.features = self.features.to(device)
        self.weights = self.weights.to(device)
        self.device = device
        return self

    def splat_feature_map(self, camera: Camera, feature_map: torch.Tensor) -> torch.Tensor:
        assert feature_map.shape[0] == self.extractor.n_features and feature_map.shape[1:] == (camera.image_height, camera.image_width)
        _, features, features_alpha, features_idx = feature_fusion(self.gaussians, camera, feature_map.permute(1, 2, 0), self.fusion_alpha_threshold)
        self.features[features_idx] += features
        self.weights[features_idx] += features_alpha

    def fuse(self, camera: Camera):
        assert camera.ground_truth_image is not None and camera.ground_truth_image.dim() == 3
        feature_map = self.extractor.extract_features(camera.ground_truth_image)
        self.splat_feature_map(camera, feature_map)

    def fuse_batch(self, cameras: List[Camera]):
        images = []
        for camera in cameras:
            assert camera.ground_truth_image is not None and camera.ground_truth_image.dim() == 3
            images.append(camera.ground_truth_image)
        feature_maps = self.extractor.extract_features_batch(images)
        assert len(cameras) == len(feature_maps)
        for camera, feature_map in zip(cameras, feature_maps):
            self.splat_feature_map(camera, feature_map)

    def get_features(self) -> torch.Tensor:
        features = self.features / self.weights.unsqueeze(-1)
        features[self.weights < 1e-5, ...] = 0
        return self.extractor.postprocess_features(features)

    def visualize_features(self) -> GaussianModel:
        colors = self.extractor.assign_colors(self.get_features(), self.weights)
        gaussians = copy.copy(self.gaussians)
        gaussians._opacity = nn.Parameter(gaussians._opacity.clone())
        gaussians._opacity[self.weights < 1] += gaussians.inverse_opacity_activation(self.weights[self.weights < 1].unsqueeze(-1))
        gaussians._opacity[gaussians.get_opacity < 0.05] = gaussians.inverse_opacity_activation(torch.tensor(0.05)).to(gaussians._opacity.device)
        gaussians._features_dc = nn.Parameter(gaussians._features_dc.clone())
        gaussians._features_dc[:, 0, :] = colors
        gaussians._features_rest = nn.Parameter(gaussians._features_rest.clone())
        gaussians._features_rest[...] = 0
        return gaussians
