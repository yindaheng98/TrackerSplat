import os
from tqdm import tqdm
from gaussian_splatting.dataset.colmap.dataset import read_colmap_cameras

from .dataset import CameraMeta, VideoCameraDataset


def ColmapVideoCameraDataset(video_folder: str, frame_folder_fmt: str = "frame%d", start_frame=1, n_frames=None, device="cuda") -> VideoCameraDataset:
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
        framemeta = [CameraMeta(
            image_height=camera.image_height,
            image_width=camera.image_width,
            FoVx=camera.FoVx,
            FoVy=camera.FoVy,
            R=camera.R,
            T=camera.T,
            image_path=camera.image_path
        ) for camera in read_colmap_cameras(frame_folder)]
        framemetas.append(framemeta)
        cameras_count += len(framemeta)
        pbar.set_postfix({'total frames': frame_idx, 'total cameras': cameras_count})
        frame_idx += 1
        pbar.update(1)
    return VideoCameraDataset(framemetas, device=device)
