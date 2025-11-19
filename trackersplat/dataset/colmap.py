import os
from tqdm import tqdm
import torch
from gaussian_splatting.dataset import JSONCameraDataset
from gaussian_splatting.dataset.colmap.dataset import read_colmap_cameras

from .dataset import DatasetCameraMeta, VideoCameraDataset


def read_colmap_framemetas(video_folder: str, frame_folder_fmt: str = "frame%d", start_frame=1, n_frames=None, load_mask=True, load_depth=True):
    """
    Load a video dataset from a sequence of COLMAP workspaces.
    """
    framemetas = []
    frame_idx = start_frame
    pbar = tqdm(total=n_frames, desc="Loading camera from COLMAP workspaces")
    cameras_count = 0
    while n_frames is None or frame_idx <= start_frame + n_frames:
        frame_folder = os.path.join(video_folder, frame_folder_fmt % frame_idx)
        if not os.path.exists(frame_folder):
            break
        framemeta = [DatasetCameraMeta(
            **camera._asdict(),
            frame_idx=frame_idx,
        ) for camera in read_colmap_cameras(frame_folder, load_mask=load_mask, load_depth=load_depth)]
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
            assert abs(camera.FoVx - camera0.FoVx) < 1e-8
            assert abs(camera.FoVy - camera0.FoVy) < 1e-8
            assert torch.isclose(camera0.R, camera.R).all()
            assert torch.isclose(camera0.T, camera.T).all()
            assert camera.frame_idx == framemeta[0].frame_idx


def FixedViewColmapVideoCameraDataset(*args, device=torch.device("cuda"), **kwargs) -> VideoCameraDataset:
    '''Read a video dataset from a sequence of COLMAP workspaces and validate.'''
    framemetas = read_colmap_framemetas(*args, **kwargs)
    fixedview_validate(framemetas)
    return VideoCameraDataset(frames=framemetas, device=device)


def FixedViewColmapVideoCameraDataset_from_json(*args, jsonpath: str, load_mask=True, load_depth=True, device=torch.device("cuda"), **kwargs) -> VideoCameraDataset:
    framemetas = read_colmap_framemetas(*args, load_mask=load_mask, load_depth=load_depth, **kwargs)
    jsoncameras = JSONCameraDataset(jsonpath, load_mask=load_mask, load_depth=load_depth)
    cam_idx_in_json = [None] * len(jsoncameras)
    assert len(framemetas[0]) == len(jsoncameras)
    for i, framemeta in enumerate(framemetas[0]):  # should load the first camera
        for j, jsoncamera in enumerate(jsoncameras):
            if os.path.basename(framemeta.image_path) == os.path.basename(jsoncamera.ground_truth_image_path):
                assert framemeta.image_height == jsoncamera.image_height
                assert framemeta.image_width == jsoncamera.image_width
                cam_idx_in_json[i] = j
    assert None not in cam_idx_in_json
    framemetas = [[DatasetCameraMeta(
        **{
            **camera._asdict(),
            **dict(  # use loaded camera pose
                FoVx=jsoncameras[idx].FoVx,
                FoVy=jsoncameras[idx].FoVy,
                R=jsoncameras[idx].R,
                T=jsoncameras[idx].T,
            )}
    ) for idx, camera in zip(cam_idx_in_json, framemeta)] for framemeta in framemetas]
    fixedview_validate(framemetas)
    return VideoCameraDataset(frames=framemetas, device=device)
