import os
from tqdm import tqdm
import torch
from gaussian_splatting.dataset.colmap.dataset import read_colmap_cameras

from .dataset import DatasetCameraMeta, VideoCameraDataset


def read_colmap_framemetas(video_folder: str, frame_folder_fmt: str = "frame%d", start_frame=1, n_frames=None):
    """
    Load a video dataset from a sequence of COLMAP workspaces.
    """
    framemetas = []
    frame_idx = start_frame
    pbar = tqdm(total=n_frames, desc="Loading camera from COLMAP workspaces")
    cameras_count = 0
    while n_frames is None or frame_idx <= n_frames:
        frame_folder = os.path.join(video_folder, frame_folder_fmt % frame_idx)
        if not os.path.exists(frame_folder):
            break
        framemeta = [DatasetCameraMeta(
            image_height=camera.image_height,
            image_width=camera.image_width,
            FoVx=camera.FoVx,
            FoVy=camera.FoVy,
            R=camera.R,
            T=camera.T,
            image_path=camera.image_path
        ) for camera in read_colmap_cameras(frame_folder)]
        framemeta = sorted(framemeta, key=lambda x: x.image_path)
        framemetas.append(framemeta)
        cameras_count += len(framemeta)
        pbar.set_postfix({'total frames': frame_idx, 'total cameras': cameras_count})
        frame_idx += 1
        pbar.update(1)
    return framemetas


def ColmapVideoCameraDataset(*args, device=torch.device("cuda"), **kwargs) -> VideoCameraDataset:
    '''Read a video dataset from a sequence of COLMAP workspaces.'''
    return VideoCameraDataset(frames=read_colmap_framemetas(*args, **kwargs), device=device)


def fixedview_validate(framemetas):
    for framemeta in tqdm(framemetas[1:], desc="Validating camera from COLMAP workspaces"):
        assert len(framemeta) == len(framemetas[0])
        for camera0, camera in zip(framemetas[0], framemeta):
            assert camera0.image_height == camera.image_height
            assert camera0.image_width == camera.image_width
            assert camera0.FoVx == camera.FoVx
            assert camera0.FoVy == camera.FoVy
            assert torch.equal(camera0.R, camera.R)
            assert torch.equal(camera0.T, camera.T)


def FixedViewColmapVideoCameraDataset(*args, device=torch.device("cuda"), **kwargs) -> VideoCameraDataset:
    '''Read a video dataset from a sequence of COLMAP workspaces and validate.'''
    framemetas = read_colmap_framemetas(*args, **kwargs)
    fixedview_validate(framemetas)
    return VideoCameraDataset(frames=framemetas, device=device)
