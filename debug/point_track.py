import torch
from typing import List, Tuple

from dot.utils.io import read_frame
from instantsplatstream.dataset import prepare_fixedview_dataset, VideoCameraDataset
from instantsplatstream.motionestimator import Motion, FixedViewFrameSequenceMeta
from instantsplatstream.motionestimator.point_tracker import PointTrackSequence, MotionFuser, build_motion_estimator
from instantsplatstream.motionestimator.point_tracker.visualizer import Visualizer


class FakeFuser(MotionFuser):

    def to(self, device: torch.device) -> 'MotionFuser':
        return self

    def __call__(self, trackviews: List[PointTrackSequence]):
        pass

    def update_baseframe(self, frame):
        return self


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--estimator", choices=["dot", "dot-tapir", "dot-bootstapir", "dot-cotracker3", "cotracker3"], default="dot-cotracker3")
    parser.add_argument("-f", "--frame_folder_fmt", default="frame%d", type=str, help="frame folder format string")
    parser.add_argument("-n", "--n_frames", default=None, type=int, help="number of frames to process")
    parser.add_argument("--start_frame", default=1, type=int, help="start from which frame")
    parser.add_argument("--tracking_rescale", default=1.0, type=float)
    args = parser.parse_args()
    dataset = prepare_fixedview_dataset(
        source=args.source, device=args.device,
        frame_folder_fmt=args.frame_folder_fmt, start_frame=args.start_frame, n_frames=args.n_frames)
    estimator = build_motion_estimator(estimator=args.estimator, fuser=FakeFuser(), device=args.device, rescale_factor=args.tracking_rescale)
    cameras = dataset.get_metas()
    for frame in cameras:
        assert len(frame) == len(cameras[0])
    views = [FixedViewFrameSequenceMeta.from_datasetcameras(frame) for frame in zip(*cameras)]
    for view in views:
        track = estimator.tracker(view)
        n, h, w, c = track.track.shape
        x = torch.arange(w, dtype=torch.float, device=track.track.device)
        y = torch.arange(h, dtype=torch.float, device=track.track.device)
        xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)
        track = track._replace(
            track=torch.cat([xy.unsqueeze(0), track.track], dim=0),
            mask=torch.cat([torch.ones((1, h, w), device=track.mask.device), track.mask], dim=0)
        )
        n += 1
        video = []
        for path in view.frames_path:
            frame = read_frame(path, resolution=estimator.tracker.compute_rescale(view))
            video.append(frame)
        video = torch.stack(video)
        print(video)
