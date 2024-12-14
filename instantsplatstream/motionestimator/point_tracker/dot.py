from typing import Tuple
import torch
from dot.models import DenseOpticalTracker
from dot.utils.io import read_frame, read_config
from dot.utils.torch import get_grid
from instantsplatstream.motionestimator import FixedViewFrameSequenceMeta
from .abc import PointTrackSequence, PointTracker, PointTrackMotionEstimator


def resize_model(model: DenseOpticalTracker, height: int, width: int, estimator_patch_size: int, refiner_patch_size: int, device: torch.device) -> DenseOpticalTracker:
    model.resolution = [height, width]
    coarse_height, coarse_width = height // estimator_patch_size, width // estimator_patch_size
    model.point_tracker.optical_flow_estimator.register_buffer("coarse_grid", get_grid(coarse_height, coarse_width, device=device))
    coarse_height, coarse_width = height // refiner_patch_size, width // refiner_patch_size
    model.optical_flow_refiner.register_buffer("coarse_grid", get_grid(coarse_height, coarse_width, device=device))
    return model


class DotPointTracker(PointTracker):
    def __init__(
            self,
            tracker_config: str = "submodules/dot/configs/cotracker2_patch_4_wind_8.json",
            tracker_path: str = "checkpoints/movi_f_cotracker2_patch_4_wind_8.pth",
            estimator_config: str = "submodules/dot/configs/raft_patch_8.json",
            estimator_path: str = "checkpoints/cvo_raft_patch_8.pth",
            refiner_config: str = "submodules/dot/configs/raft_patch_4_alpha.json",
            refiner_path: str = "checkpoints/movi_f_raft_patch_4_alpha.pth",
            n_tracks_total=1024,
            n_tracks_batch=1024,
            rescale_factor=1.0,
            device=torch.device("cuda")):
        self.model = DenseOpticalTracker(
            tracker_config=tracker_config,
            tracker_path=tracker_path,
            estimator_config=estimator_config,
            estimator_path=estimator_path,
            refiner_config=refiner_config,
            refiner_path=refiner_path,
        )
        self.to(device)
        self.n_tracks_total = n_tracks_total
        self.n_tracks_batch = n_tracks_batch
        self.rescale_factor = rescale_factor
        self.estimator_patch_size = read_config(estimator_config).patch_size
        self.refiner_patch_size = read_config(refiner_config).patch_size

    def to(self, device: torch.device) -> 'DotPointTracker':
        self.model = self.model.to(device)
        self.device = device
        return self

    def compute_rescale(self, frames: FixedViewFrameSequenceMeta) -> Tuple[int, int]:
        return int(frames.image_height * self.rescale_factor) // 8 * 8, int(frames.image_width * self.rescale_factor) // 8 * 8

    def track(self, frames: FixedViewFrameSequenceMeta, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model = resize_model(self.model, height, width, self.estimator_patch_size, self.refiner_patch_size, self.device)
        video = []
        for path in frames.frames_path:
            frame = read_frame(path, resolution=(height, width))
            video.append(frame)
        video = torch.stack(video).to(self.device)
        with torch.no_grad():
            pred = self.model.get_tracks_from_first_to_every_other_frame(
                data={"video": video[None]},
                num_tracks=self.n_tracks_total,
                sim_tracks=self.n_tracks_batch,
            )
        tracks = pred["tracks"].squeeze(0)
        return tracks[1:, ..., :2], tracks[1:, ..., 2]


def DotMotionEstimator(fuser, device=torch.device("cuda"), **kwargs):
    return PointTrackMotionEstimator(DotPointTracker(device=device, **kwargs), fuser, device)


def Cotracker3DotMotionEstimator(
        fuser, device=torch.device("cuda"),
        tracker_config: str = "submodules/dot/configs/cotracker2_patch_4_wind_8.json",
        tracker_path: str = "checkpoints/movi_f_cotracker2_patch_4_wind_8.pth",
        **kwargs):
    return PointTrackMotionEstimator(DotPointTracker(device=device, tracker_config=tracker_config, tracker_path=tracker_path, **kwargs), fuser, device)


def TapirDotMotionEstimator(
        fuser, device=torch.device("cuda"),
        tracker_config: str = "submodules/dot/configs/tapir.json",
        tracker_path: str = "checkpoints/panning_movi_e_tapir.pth",
        **kwargs):
    return PointTrackMotionEstimator(DotPointTracker(device=device, tracker_config=tracker_config, tracker_path=tracker_path, **kwargs), fuser, device)


def BootsTapirDotMotionEstimator(
        fuser, device=torch.device("cuda"),
        tracker_config: str = "submodules/dot/configs/bootstapir.json",
        tracker_path: str = "checkpoints/panning_movi_e_plus_bootstapir.pth",
        **kwargs):
    return PointTrackMotionEstimator(DotPointTracker(device=device, tracker_config=tracker_config, tracker_path=tracker_path, **kwargs), fuser, device)
