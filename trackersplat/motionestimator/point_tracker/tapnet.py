import torch
from typing import Tuple
from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint
from trackersplat.motionestimator import FixedViewFrameSequenceMeta
from .abc import PointTrackSequence, PointTracker, PointTrackMotionEstimator


class TAPNextPointTracker(PointTracker):
    def __init__(
            self,
            checkpoint="./checkpoints/bootstapnext_ckpt.npz",
            rescale_factor=1.0,
            device=torch.device("cuda")):
        self.model = TAPNext(image_size=(256, 256))
        self.model = restore_model_from_jax_checkpoint(self.model, checkpoint)
        self.rescale_factor = rescale_factor
        self.to(device)

    def to(self, device: torch.device) -> 'TAPNextPointTracker':
        self.model = self.model.to(device)
        self.device = device
        return self

    def compute_rescale(self, frames: FixedViewFrameSequenceMeta) -> Tuple[int, int]:
        return int(frames.image_height * self.rescale_factor), int(frames.image_width * self.rescale_factor)

    def track(self, video: torch.Tensor) -> PointTrackSequence:
        _, _, height, width = video.shape
        with torch.no_grad():
            pred_tracks, pred_visibility = self.model(video[None].to(self.device))
        track = pred_tracks.squeeze(0).reshape(-1, height, width, 2)
        mask = pred_visibility.squeeze(0).reshape(-1, height, width)
        return track[1:, ...], mask[1:, ...]  # (tracks, visibility mask)


def TAPNextMotionEstimator(fuser, device=torch.device("cuda"), **kwargs):
    return PointTrackMotionEstimator(TAPNextPointTracker(device=device, **kwargs), fuser, device)
