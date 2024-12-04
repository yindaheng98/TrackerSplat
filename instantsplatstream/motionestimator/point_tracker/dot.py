import torch
from dot.models import DenseOpticalTracker
from dot.utils.io import read_frame
from instantsplatstream.motionestimator import FixedViewFrameSequenceMeta
from .abc import PointTrackSequence, PointTracker, PointTrackMotionEstimator


class DotPointTracker(PointTracker):
    def __init__(
            self,
            height: int = 512, width: int = 512,
            tracker_config: str = "submodules/dot/configs/cotracker2_patch_4_wind_8.json",
            tracker_path: str = "checkpoints/movi_f_cotracker2_patch_4_wind_8.pth",
            estimator_config: str = "submodules/dot/configs/raft_patch_8.json",
            estimator_path: str = "checkpoints/cvo_raft_patch_8.pth",
            refiner_config: str = "submodules/dot/configs/raft_patch_4_alpha.json",
            refiner_path: str = "checkpoints/movi_f_raft_patch_4_alpha.pth",
            n_tracks_total=1024,
            n_tracks_batch=1024,
            device=torch.device("cuda")):
        self.model = DenseOpticalTracker(
            height=height,
            width=width,
            tracker_config=tracker_config,
            tracker_path=tracker_path,
            estimator_config=estimator_config,
            estimator_path=estimator_path,
            refiner_config=refiner_config,
            refiner_path=refiner_path,
        )
        self.to(device)
        self.height = height
        self.width = width
        self.n_tracks_total = n_tracks_total
        self.n_tracks_batch = n_tracks_batch

    def to(self, device: torch.device) -> 'DotPointTracker':
        self.model = self.model.to(device)
        self.device = device
        return self

    def __call__(self, frames: FixedViewFrameSequenceMeta) -> PointTrackSequence:
        video = []
        for path in frames.frames_path:
            frame = read_frame(path, resolution=(self.height, self.width))
            video.append(frame)
        video = torch.stack(video).to(self.device)
        with torch.no_grad():
            pred = self.model.get_tracks_from_first_to_every_other_frame(
                data={"video": video[None]},
                num_tracks=self.n_tracks_total,
                sim_tracks=self.n_tracks_batch,
            )
        tracks = pred["tracks"][0]
        return PointTrackSequence(
            image_height=self.height,
            image_width=self.width,
            FoVx=frames.FoVx,
            FoVy=frames.FoVy,
            R=frames.R,
            T=frames.T,
            track=tracks[..., :2],
            mask=tracks[..., 2]
        )


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
