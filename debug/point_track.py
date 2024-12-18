import torch
from typing import List, Tuple

from instantsplatstream.dataset import prepare_fixedview_dataset, VideoCameraDataset
from instantsplatstream.motionestimator import Motion, FixedViewFrameSequenceMeta
from instantsplatstream.motionestimator.point_tracker import PointTrackSequence, MotionFuser, build_motion_estimator
from instantsplatstream.motionestimator.point_tracker.visualizer import Visualizer


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
        frame_folder_fmt=args.frame_folder_fmt, start_frame=args.start_frame, n_frames=None)
    cameras = dataset.get_metas()
    for frame in cameras:
        assert len(frame) == len(cameras[0])
    views = [FixedViewFrameSequenceMeta.from_datasetcameras(frame) for frame in zip(*cameras)]
    for view in views:
        print(view)
    print("Views:", views)
