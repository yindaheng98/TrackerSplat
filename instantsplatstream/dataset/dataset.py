from typing import List, NamedTuple
import torch

from gaussian_splatting.camera import build_camera
from torch.utils.data import Dataset


class MetaCamera(NamedTuple):
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    R: torch.Tensor
    T: torch.Tensor
    image_path: str


class FrameCameraDataset(Dataset):
    def __init__(self, metacameras: List[MetaCamera], device="cuda"):
        super().__init__()
        self.metacameras = [MetaCamera(**camera._asdict()) for camera in metacameras]
        self.to(device)

    def to(self, device) -> 'FrameCameraDataset':
        self.cameras = [build_camera(**camera._asdict(), device=device) for camera in self.metacameras]
        return self

    def __getitem__(self, idx):
        return self.cameras[idx]

    def __len__(self):
        return len(self.cameras)


MetaFrame = List[MetaCamera]


class VideoCameraDataset(Dataset):
    def __init__(self, frames: List[MetaFrame], device="cuda"):
        super().__init__()
        self.frames = [[MetaCamera(**camera._asdict()) for camera in frame] for frame in frames]
        self.to(device)

    def to(self, device) -> 'VideoCameraDataset':
        self.device = device
        return self

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) == 1:
            idx = idx[0]
        if isinstance(idx, int):  # a frame contains multiple cameras
            return FrameCameraDataset(self.frames[idx], device=self.device)
        if isinstance(idx, slice) or isinstance(idx, list):  # a video contains multiple frames
            return VideoCameraDataset(self.frames[idx], device=self.device)
        if isinstance(idx, tuple) and len(idx) == 2 and isinstance(idx[0], int):
            frame = self.frames[idx[0]]
            if isinstance(idx[1], int):  # a camera
                return build_camera(**frame[idx[1]]._asdict(), device=self.device)
            if isinstance(idx[1], slice) or isinstance(idx[1], list):  # a frame contains multiple cameras
                return FrameCameraDataset(frame[idx[1]], device=self.device)
        raise ValueError("Invalid index")

    def __len__(self):
        return len(self.frames)
