from typing import List, NamedTuple
import torch

from gaussian_splatting.camera import build_camera
from torch.utils.data import Dataset


class DatasetCameraMeta(NamedTuple):
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    R: torch.Tensor
    T: torch.Tensor
    image_path: str = None

    def build_camera(self, device=torch.device("cuda")):
        return build_camera(**self._asdict(), device=device)


class FrameCameraDataset(Dataset):
    def __init__(self, DatasetCameraMetas: List[DatasetCameraMeta], device=torch.device("cuda")):
        super().__init__()
        self.DatasetCameraMetas = [DatasetCameraMeta(**camera._asdict()) for camera in DatasetCameraMetas]
        self.to(device)

    def to(self, device) -> 'FrameCameraDataset':
        self.cameras = [camera.build_camera(device=device) for camera in self.DatasetCameraMetas]
        return self

    def __getitem__(self, idx):
        return self.cameras[idx]

    def __len__(self):
        return len(self.cameras)


MetaFrame = List[DatasetCameraMeta]


class VideoCameraDataset(Dataset):
    def __init__(self, frames: List[MetaFrame], device=torch.device("cuda")):
        super().__init__()
        self.framemetas = [[DatasetCameraMeta(**camera._asdict()) for camera in frame] for frame in frames]
        self.to(device)

    def to(self, device) -> 'VideoCameraDataset':
        self.device = device
        return self

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) == 1:
            idx = idx[0]
        if isinstance(idx, int):  # a frame contains multiple cameras
            return FrameCameraDataset(self.framemetas[idx], device=self.device)
        if isinstance(idx, slice) or isinstance(idx, list):  # a video contains multiple frames
            return VideoCameraDataset(self.framemetas[idx], device=self.device)
        if isinstance(idx, tuple) and len(idx) == 2 and isinstance(idx[0], int):
            frame = self.framemetas[idx[0]]
            if isinstance(idx[1], int):  # a camera
                return frame[idx[1]].build_camera(device=self.device)
            if isinstance(idx[1], slice) or isinstance(idx[1], list):  # a frame contains multiple cameras
                return FrameCameraDataset(frame[idx[1]], device=self.device)
        raise ValueError("Invalid index")

    def __len__(self):
        return len(self.framemetas)

    def get_metas(self):
        return [[DatasetCameraMeta(**camera._asdict()) for camera in frame] for frame in self.framemetas]
