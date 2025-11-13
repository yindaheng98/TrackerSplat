from .dataset import DatasetCameraMeta, FrameCameraDataset, VideoCameraDataset
from .colmap import ColmapVideoCameraDataset, FixedViewColmapVideoCameraDataset, FixedViewColmapVideoCameraDataset_from_json


def prepare_fixedview_dataset(source: str, frame_folder_fmt: str, start_frame: int, n_frames=None, load_camera: str = None, load_mask=True, load_depth=True, device="cuda") -> VideoCameraDataset:
    kwargs = dict(
        video_folder=source,
        frame_folder_fmt=frame_folder_fmt,
        start_frame=start_frame,
        n_frames=n_frames,
        load_mask=load_mask,
        load_depth=load_depth,
    )
    return FixedViewColmapVideoCameraDataset_from_json(jsonpath=load_camera, device=device, **kwargs) if load_camera else FixedViewColmapVideoCameraDataset(device=device, **kwargs)
