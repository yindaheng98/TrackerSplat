import os
import torch
from typing import List

from dot.utils.io import read_frame
from instantsplatstream.dataset import prepare_fixedview_dataset
from instantsplatstream.motionestimator import FixedViewFrameSequenceMeta
from instantsplatstream.motionestimator.point_tracker import PointTrackSequence, MotionFuser, build_point_track_batch_motion_estimator
from instantsplatstream.motionestimator.point_tracker.visualizer import Visualizer
from dot.utils.io import write_video


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
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--estimator", choices=["dot", "dot-tapir", "dot-bootstapir", "dot-cotracker3", "cotracker3"], default="dot-cotracker3")
    parser.add_argument("-f", "--frame_folder_fmt", default="frame%d", type=str, help="frame folder format string")
    parser.add_argument("-n", "--n_frames", default=None, type=int, help="number of frames to process")
    parser.add_argument("--start_frame", default=1, type=int, help="start from which frame")
    parser.add_argument("--tracking_rescale", default=1.0, type=float)

    parser.add_argument("--spaghetti_radius", type=float, default=1.5)
    parser.add_argument("--spaghetti_length", type=int, default=40)
    parser.add_argument("--spaghetti_grid", type=int, default=30)
    parser.add_argument("--spaghetti_scale", type=float, default=2)
    parser.add_argument("--spaghetti_every", type=int, default=10)
    parser.add_argument("--spaghetti_dropout", type=float, default=0)
    args = parser.parse_args()
    dataset = prepare_fixedview_dataset(
        source=args.source, device=args.device,
        frame_folder_fmt=args.frame_folder_fmt, start_frame=args.start_frame, n_frames=args.n_frames,
        load_camera=args.load_camera)
    estimator = build_point_track_batch_motion_estimator(estimator=args.estimator, fuser=FakeFuser(), device=args.device, rescale_factor=args.tracking_rescale)
    frame_dirname = args.frame_folder_fmt % args.start_frame + "-" + ((args.frame_folder_fmt % (args.start_frame + args.n_frames - 1)) if args.n_frames is not None else "")
    result_path = os.path.join(args.destination, "tracks", frame_dirname)
    visualizer = Visualizer(
        "image", 0.75,
        args.spaghetti_radius, args.spaghetti_length, args.spaghetti_grid, args.spaghetti_scale, args.spaghetti_every, args.spaghetti_dropout
    ).to(args.device)

    cameras = dataset.get_metas()
    for frame in cameras:
        assert len(frame) == len(cameras[0])
    views = [FixedViewFrameSequenceMeta.from_datasetcameras(frame) for frame in zip(*cameras)]
    for idx, view in enumerate(views):
        track = estimator.tracker(view)
        n, h, w, c = track.track.shape
        x = torch.arange(w, dtype=torch.float, device=track.track.device)
        y = torch.arange(h, dtype=torch.float, device=track.track.device)
        xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)
        track = torch.cat([
            torch.cat([
                xy.unsqueeze(0),
                track.track], dim=0),
            torch.cat([
                torch.ones((1, h, w), device=track.mask.device),
                track.mask], dim=0).unsqueeze(-1)
        ], dim=-1)
        mask = torch.ones(h, w).bool().to(track.device)
        n += 1
        video = []
        for path in view.frames_path:
            frame = read_frame(path, resolution=estimator.tracker.compute_rescale(view))
            video.append(frame)
        video = torch.stack(video).to(track.device)
        for mode in ["overlay", "spaghetti_last_static"]:
            visualizer({
                "video": video,
                "tracks": track.permute(0, 2, 1, 3),
                "mask": mask.permute(1, 0),
            }, mode=mode, result_path=os.path.join(result_path, "view_%d" % idx))
        torch.save(track, os.path.join(result_path, "view_%dtrack.pt" % idx))
        write_video(video, os.path.join(result_path, "view_%dvideo" % idx))
