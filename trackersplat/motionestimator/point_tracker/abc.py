from typing import List, NamedTuple, Tuple
from abc import ABCMeta, abstractmethod
import torch
from dot.utils.io import read_frame
from gaussian_splatting import GaussianModel
from gaussian_splatting.camera import build_camera
from trackersplat.motionestimator import Motion, FixedViewBatchMotionEstimator, FixedViewFrameSequenceMeta


def read_mask(*args, **kwargs):
    frame = read_frame(*args, **kwargs)
    return frame[0, ...] * 299/1000 + frame[1, ...] * 587/1000 + frame[2, ...] * 114/1000  # ITU-R 601-2 luma transform


class PointTrackSequence(NamedTuple):
    # Same as trackersplat.motionestimator.FixedViewFrameSequenceMeta
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    R: torch.Tensor
    T: torch.Tensor
    frames_path: List[str]
    frame_masks_path: List[str]
    depths_path: List[int]
    depth_masks_path: List[int]
    frame_idx: List[int]

    # Point tracking results
    track_height: int
    track_width: int
    track: torch.Tensor
    visibility: torch.Tensor

    def build_track_camera(self, device=torch.device("cuda")):
        return build_camera(
            image_height=self.track_height,
            image_width=self.track_width,
            FoVx=self.FoVx,
            FoVy=self.FoVy,
            R=self.R,
            T=self.T,
            device=device
        )


class PointTracker(metaclass=ABCMeta):

    @abstractmethod
    def to(self, device: torch.device) -> 'PointTracker':
        return self

    def compute_rescale(self, frames: FixedViewFrameSequenceMeta) -> Tuple[int, int]:
        return frames.image_height, frames.image_width

    def read_frames(self, frames: FixedViewFrameSequenceMeta, height: int, width: int, apply_mask=True) -> List[torch.Tensor]:
        video = []
        for path, mask_path in zip(frames.frames_path, frames.frame_masks_path):
            frame = read_frame(path, resolution=(height, width))
            if apply_mask and None not in frames.frame_masks_path:
                mask = read_frame(mask_path, resolution=(height, width))
                frame *= (mask > 0.5).float()
            video.append(frame)
        return torch.stack(video)

    # If you only mask frame before tracking, there will be more inaccurate tracks, mask after tracking is necessary
    def __call__(self, frames: FixedViewFrameSequenceMeta, mask_input=True, mask_output=True) -> PointTrackSequence:
        height, width = self.compute_rescale(frames)
        video = self.read_frames(frames, height, width, apply_mask=mask_input)
        track, visibility = self.track(video)
        n, h, w, c = track.shape
        assert h == height and w == width and c == 2
        assert visibility.shape == (n, h, w)
        if mask_output and frames.frame_masks_path[0] is not None:
            mask = read_mask(frames.frame_masks_path[0], resolution=(height, width)).to(track.device).unsqueeze(0).unsqueeze(-1)
            pix = torch.stack(torch.meshgrid(torch.arange(track.shape[1]), torch.arange(track.shape[2]), indexing='ij'), dim=-1)[..., [1, 0]].to(device=track.device, dtype=track.dtype).unsqueeze(0)
            track = track * mask + pix * (1 - mask)
        return PointTrackSequence(
            **frames._asdict(),
            track_height=height,
            track_width=width,
            track=track,
            visibility=visibility,
        )

    @abstractmethod
    def track(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError  # return (tracks, visibility mask)


class MotionFuser(metaclass=ABCMeta):

    @abstractmethod
    def to(self, device: torch.device) -> 'MotionFuser':
        return self

    @abstractmethod
    def __call__(self, trackviews: List[PointTrackSequence]) -> List[Motion]:
        raise NotImplementedError

    @abstractmethod
    def update_baseframe(self, frame: GaussianModel) -> 'MotionFuser':
        return self


class PointTrackMotionEstimator(FixedViewBatchMotionEstimator):
    mask_input: bool = True
    mask_output: bool = True

    def __init__(self, tracker: PointTracker, fuser: MotionFuser, device=torch.device("cuda")):
        self.tracker = tracker
        self.fuser = fuser
        self.to(device)

    def to(self, device: torch.device) -> 'PointTrackMotionEstimator':
        self.tracker = self.tracker.to(device)
        self.fuser = self.fuser.to(device)
        return self

    def __call__(self, views: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        trackviews = [self.tracker(view, mask_input=self.mask_input, mask_output=self.mask_output) for view in views]
        for view in trackviews:
            n, h, w, c = view.track.shape
            assert c == 2
            assert list(view.visibility.shape) == [n, h, w]
            assert view.track_height == h and view.track_width == w
        return self.fuser(trackviews)

    def update_baseframe(self, frame: GaussianModel) -> 'PointTrackMotionEstimator':
        self.fuser = self.fuser.update_baseframe(frame)
        return self
