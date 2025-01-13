import itertools
import torch
import os
from itertools import islice
from functools import partial
from gaussian_splatting import GaussianModel
from instantsplatstream.dataset import prepare_fixedview_dataset, VideoCameraDataset
from instantsplatstream.motionestimator import FixedViewMotionEstimator, MotionCompensater
from instantsplatstream.motionestimator.point_tracker import BaseMotionFuser, build_point_track_batch_motion_estimator
from instantsplatstream.motionestimator.compensater import build_motion_compensater
from instantsplatstream.motionestimation import prepare_gaussians, save_cfg_args


estimator_choices = ["dot", "dot-tapir", "dot-bootstapir", "dot-cotracker3", "cotracker3"]
compensater_choices = ["base", "propagate", "filter"]

pipeline_choices = [compensater + "-" + estimator for compensater, estimator in itertools.product(compensater_choices, estimator_choices)]


def build_pipeline(estimator: str, gaussians: GaussianModel, dataset: VideoCameraDataset, device: torch.device, batch_size: int, **kwargs) -> MotionCompensater:
    compensater, estimator = estimator.split("-", 1)
    batch_func = build_point_track_batch_motion_estimator(estimator=estimator, fuser=BaseMotionFuser(gaussians), device=device, **kwargs)
    motion_estimator = FixedViewMotionEstimator(dataset=dataset, batch_func=batch_func, device=device, batch_size=batch_size)
    motion_compensater = build_motion_compensater(compensater=compensater, gaussians=gaussians, estimator=motion_estimator, device=device)
    return motion_compensater


def motion_compensate(motion_compensater: MotionCompensater, n_frames: int):
    for i, frame_gaussians in enumerate(islice(motion_compensater, n_frames)):
        print("frame", i)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--pipeline", choices=pipeline_choices, default="base-dot-cotracker3")
    parser.add_argument("--iteration_init", required=True, type=str, help="iteration of the initial gaussians")
    parser.add_argument("-f", "--frame_folder_fmt", default="frame%d", type=str, help="frame folder format string")
    parser.add_argument("-n", "--n_frames", default=None, type=int, help="number of frames to process")
    parser.add_argument("-b", "--batch_size", default=3, type=int, help="batch size of point tracking")
    parser.add_argument("--start_frame", default=1, type=int, help="start from which frame")
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    args = parser.parse_args()
    save_frame_cfg_args = partial(save_cfg_args, sh_degree=args.sh_degree, source=args.source, destination=os.path.join(args.destination, args.pipeline), frame_folder_fmt=args.frame_folder_fmt)
    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    load_ply = os.path.join(args.destination, args.frame_folder_fmt % args.start_frame, "point_cloud", "iteration_" + str(args.iteration_init), "point_cloud.ply")
    gaussians = prepare_gaussians(
        sh_degree=args.sh_degree, device=args.device,
        load_ply=load_ply)
    dataset = prepare_fixedview_dataset(
        source=args.source, device=args.device,
        frame_folder_fmt=args.frame_folder_fmt, start_frame=args.start_frame, n_frames=None,
        load_camera=args.load_camera)
    motion_compensater = build_pipeline(args.pipeline, gaussians, dataset, args.device, args.batch_size, **configs)
    motion_compensate(motion_compensater, args.n_frames)
