import torch
import os
from tqdm import tqdm
from os import makedirs
from itertools import islice
from functools import partial
from gaussian_splatting import GaussianModel
import gaussian_splatting.train
from instantsplatstream.dataset import prepare_fixedview_dataset, VideoCameraDataset
from instantsplatstream.motionestimator import FixedViewMotionEstimator, MotionCompensater
from instantsplatstream.motionestimator.point_tracker import BaseMotionFuser, build_motion_estimator
from instantsplatstream.motionestimator.compensater.base import BaseMotionCompensater


def prepare_gaussians(sh_degree: int, device: str, load_ply: str) -> GaussianModel:
    gaussians = GaussianModel(sh_degree).to(device)
    gaussians.load_ply(load_ply)
    return gaussians


def save_cfg_args(sh_degree: int, source: str, destination: str, frame_folder_fmt: str, frame: int) -> str:
    frame_str = frame_folder_fmt % frame
    destination_folder = os.path.join(destination, frame_str)
    source_folder = os.path.join(source, frame_str)
    gaussian_splatting.train.save_cfg_args(destination_folder, sh_degree, source_folder)
    return destination_folder


def build_motion_compensater(estimator: str, gaussians: GaussianModel, dataset: VideoCameraDataset, device: torch.device, batch_size: int, **kwargs) -> MotionCompensater:
    batch_func = build_motion_estimator(estimator=estimator, fuser=BaseMotionFuser(gaussians), device=device, **kwargs)
    motion_estimator = FixedViewMotionEstimator(dataset=dataset, batch_func=batch_func, device=device, batch_size=batch_size)
    motion_compensater = BaseMotionCompensater(gaussians=gaussians, estimator=motion_estimator, device=device)
    return motion_compensater


def motion_compensate(motion_compensater: MotionCompensater, dataset: VideoCameraDataset, save_frame_cfg_args, iteration: int, start_frame: int, n_frames: int):
    for i, frame_gaussians in enumerate(islice(motion_compensater, n_frames)):
        destination_folder = save_frame_cfg_args(frame=start_frame + i + 1)
        save_path = os.path.join(destination_folder, "point_cloud", "iteration_" + str(iteration))
        makedirs(save_path, exist_ok=True)
        frame_gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))
        dataset[i + 1].save_cameras(os.path.join(destination_folder, "cameras.json"))


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--estimator", choices=["dot", "dot-tapir", "dot-bootstapir", "dot-cotracker3", "cotracker3"], default="dot-cotracker3")
    parser.add_argument("--iteration_init", required=True, type=str, help="iteration of the initial gaussians")
    parser.add_argument("-f", "--frame_folder_fmt", default="frame%d", type=str, help="frame folder format string")
    parser.add_argument("-n", "--n_frames", default=None, type=int, help="number of frames to process")
    parser.add_argument("-b", "--batch_size", default=3, type=int, help="batch size of point tracking")
    parser.add_argument("--start_frame", default=1, type=int, help="start from which frame")
    parser.add_argument("--tracking_rescale", default=1.0, type=float)
    args = parser.parse_args()
    save_frame_cfg_args = partial(save_cfg_args, sh_degree=args.sh_degree, source=args.source, destination=args.destination, frame_folder_fmt=args.frame_folder_fmt)
    load_ply = os.path.join(args.destination, args.frame_folder_fmt % args.start_frame, "point_cloud", "iteration_" + str(args.iteration_init), "point_cloud.ply")
    with torch.no_grad():
        gaussians = prepare_gaussians(
            sh_degree=args.sh_degree, device=args.device,
            load_ply=load_ply)
        dataset = prepare_fixedview_dataset(
            source=args.source, device=args.device,
            frame_folder_fmt=args.frame_folder_fmt, start_frame=args.start_frame, n_frames=None,
            load_camera=args.load_camera)
        motion_compensater = build_motion_compensater(args.estimator, gaussians, dataset, args.device, args.batch_size, rescale_factor=args.tracking_rescale)
        motion_compensate(motion_compensater, dataset, save_frame_cfg_args, args.iteration, args.start_frame, args.n_frames)
