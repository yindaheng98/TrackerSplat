from typing import List
import torch
from abc import ABCMeta, abstractmethod
from gaussian_splatting import Camera, GaussianModel
from instantsplatstream.utils.featurefusion import feature_fusion


class FeatureFuser(metaclass=ABCMeta):
    def __init__(self, gaussians: GaussianModel, n_features: int, device: torch.device = torch.device("cuda")):
        self.gaussians = gaussians
        self.n_features = n_features
        n_gaussians = gaussians.get_xyz.shape[0]
        self.features = torch.zeros(size=(n_gaussians, n_features))
        self.weights = torch.zeros(size=(n_gaussians,))
        self.to(device)

    def to(self, device: torch.device):
        self.gaussians = self.gaussians.to(device)
        self.features = self.features.to(device)
        self.weights = self.weights.to(device)
        return self

    def splat_feature_map(self, camera: Camera, feature_map: torch.Tensor) -> torch.Tensor:
        assert feature_map.shape[0] == self.n_features and feature_map.shape[1:] == (camera.image_height, camera.image_width)
        _, features, features_alpha = feature_fusion(self.gaussians, camera, feature_map.permute(1, 2, 0))
        self.features += features
        self.weights += features_alpha

    @abstractmethod
    def compute_feature_map(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fuse(self, camera: Camera):
        assert camera.ground_truth_image is not None and camera.ground_truth_image.dim() == 3
        feature_map = self.compute_feature_map(camera.ground_truth_image)
        self.splat_feature_map(camera, feature_map)

    def compute_feature_map_batch(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.compute_feature_map(image) for image in images]

    def fuse_batch(self, cameras: List[Camera]):
        images = []
        for camera in cameras:
            assert camera.ground_truth_image is not None and camera.ground_truth_image.dim() == 3
            images.append(camera.ground_truth_image)
        feature_maps = self.compute_feature_map_batch(images)
        for camera, feature_map in zip(cameras, feature_maps):
            self.splat_feature_map(camera, feature_map)
