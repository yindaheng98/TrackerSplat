from typing import List, NamedTuple, Tuple
from abc import ABCMeta, abstractmethod
import torch
from dot.utils.io import read_frame
from gaussian_splatting import GaussianModel
from gaussian_splatting.camera import build_camera
from trackersplat.motionestimator import Motion, FixedViewBatchMotionEstimator, FixedViewFrameSequenceMeta


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

    def read_frames(self, frames: FixedViewFrameSequenceMeta, height: int, width: int) -> List[torch.Tensor]:
        return torch.stack([read_frame(path, resolution=(height, width)) for path in frames.frames_path])

    def __call__(self, frames: FixedViewFrameSequenceMeta) -> PointTrackSequence:
        height, width = self.compute_rescale(frames)
        video = self.read_frames(frames, height, width)
        track, visibility = self.track(video)
        n, h, w, c = track.shape
        assert h == height and w == width and c == 2
        assert visibility.shape == (n, h, w)
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
    def __init__(self, tracker: PointTracker, fuser: MotionFuser, device=torch.device("cuda")):
        self.tracker = tracker
        self.fuser = fuser
        self.to(device)

    def to(self, device: torch.device) -> 'PointTrackMotionEstimator':
        self.tracker = self.tracker.to(device)
        self.fuser = self.fuser.to(device)
        return self

    def __call__(self, views: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        trackviews = [self.tracker(view) for view in views]
        for view in trackviews:
            n, h, w, c = view.track.shape
            assert c == 2
            assert list(view.visibility.shape) == [n, h, w]
            assert view.track_height == h and view.track_width == w
        return self.fuser(trackviews)

    def update_baseframe(self, frame: GaussianModel) -> 'PointTrackMotionEstimator':
        self.fuser = self.fuser.update_baseframe(frame)
        return self
