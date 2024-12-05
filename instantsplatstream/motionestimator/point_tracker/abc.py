from typing import List, NamedTuple, Tuple
from abc import ABCMeta, abstractmethod
import torch
from gaussian_splatting import GaussianModel
from instantsplatstream.dataset import DatasetCameraMeta
from instantsplatstream.motionestimator import Motion, FixedViewBatchMotionEstimator, FixedViewFrameSequenceMeta


class PointTrackSequence(NamedTuple):
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    R: torch.Tensor  # TODO: quaternion maybe better?
    T: torch.Tensor
    track: torch.Tensor
    mask: torch.Tensor

    def build_camera(self, device=torch.device("cuda")):
        return DatasetCameraMeta(
            image_height=self.image_height,
            image_width=self.image_width,
            FoVx=self.FoVx,
            FoVy=self.FoVy,
            R=self.R,
            T=self.T,
        ).build_camera(device=device)


class PointTracker(metaclass=ABCMeta):

    @abstractmethod
    def to(self, device: torch.device) -> 'PointTracker':
        return self

    def __call__(self, frames: FixedViewFrameSequenceMeta) -> PointTrackSequence:
        height, width = self.compute_rescale(frames)
        track, mask = self.track(frames, height, width)
        n, h, w, c = track.shape
        assert h == height and w == width and c == 2
        assert mask.shape == (n, h, w)
        return PointTrackSequence(
            image_height=height,
            image_width=width,
            FoVx=frames.FoVx,
            FoVy=frames.FoVy,
            R=frames.R,
            T=frames.T,
            track=track,
            mask=mask,
        )

    def compute_rescale(self, frames: FixedViewFrameSequenceMeta) -> Tuple[int, int]:
        return frames.image_height, frames.image_width

    @abstractmethod
    def track(self, frames: FixedViewFrameSequenceMeta, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


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


class PointTrackMotionEstimator(FixedViewBatchMotionEstimator, metaclass=ABCMeta):
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
            assert view.track.shape == trackviews[0].track.shape
            n, c, h, w = view.track.shape
            assert view.mask.shape == (n, h, w)
            assert view.image_height == h and view.image_width == w
        return self.fuser(trackviews)

    def update_baseframe(self, frame: GaussianModel) -> 'PointTrackMotionEstimator':
        self.fuser = self.fuser.update_baseframe(frame)
        return self
