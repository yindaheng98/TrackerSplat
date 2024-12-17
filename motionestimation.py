from typing import Tuple
import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser, Namespace
from gaussian_splatting import GaussianModel
from gaussian_splatting.train import save_cfg_args
from instantsplatstream.dataset import prepare_fixedview_dataset, VideoCameraDataset
from instantsplatstream.motionestimator import FixedViewMotionEstimator, MotionCompensater
from instantsplatstream.motionestimator.point_tracker import Cotracker3DotMotionEstimator, Cotracker3MotionEstimator, BaseMotionFuser, build_motion_estimator

parser = ArgumentParser()
parser.add_argument("--sh_degree", default=3, type=int)
parser.add_argument("-s", "--source", required=True, type=str)
parser.add_argument("-d", "--destination", required=True, type=str)
parser.add_argument("-i", "--iteration", required=True, type=int)
parser.add_argument("--iteration_init", required=True, type=str, help="iteration of the initial gaussians")
parser.add_argument("-f", "--frame_folder_fmt", default="frame%d", type=str, help="frame folder format string")
parser.add_argument("-n", "--n_frames", default=None, type=int, help="number of frames to process")
parser.add_argument("-b", "--batch_size", default=3, type=int, help="batch size of point tracking")
parser.add_argument("--load_camera", default=None, type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--start_frame", default=1, type=int, help="start from which frame")
parser.add_argument("--tracking_rescale", default=1.0, type=float)


def prepare_gaussians(sh_degree: int, device: str, load_ply: str) -> GaussianModel:
    gaussians = GaussianModel(sh_degree).to(device)
    gaussians.load_ply(load_ply)
    return gaussians


def save_frame_cfg_args(sh_degree: int, source: str, destination: str, frame_folder_fmt: str, frame: int) -> str:
    frame_str = frame_folder_fmt % frame
    gaussians_folder = os.path.join(destination, frame_str)
    source_folder = os.path.join(source, frame_str)
    save_cfg_args(gaussians_folder, sh_degree, source_folder)
    return gaussians_folder


def build_motion_compensater(gaussians: GaussianModel, dataset: VideoCameraDataset, device, batch_size, **kwargs) -> MotionCompensater:
    batch_func = build_motion_estimator(estimator="dot-cotracker3", fuser=BaseMotionFuser(gaussians), device=device, **kwargs)
    motion_estimator = FixedViewMotionEstimator(dataset=dataset, batch_func=batch_func, device=device, batch_size=batch_size)
    motion_compensater = MotionCompensater(gaussians=gaussians, estimator=motion_estimator, device=device)
    return motion_compensater


def main(motion_compensater: MotionCompensater, dataset: VideoCameraDataset, gaussians_folder, sh_degree: int, source: str, destination: str, iteration: int, args):
    dataset[0].save_cameras(os.path.join(gaussians_folder, "cameras.json"))
    for i, frame_gaussians in enumerate(motion_compensater):
        gaussians_folder = save_frame_cfg_args(sh_degree, source, destination, args.frame_folder_fmt, args.start_frame + i + 1)
        save_path = os.path.join(gaussians_folder, "point_cloud", "iteration_" + str(iteration))
        os.makedirs(save_path, exist_ok=True)
        frame_gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))
        dataset[i + 1].save_cameras(os.path.join(gaussians_folder, "cameras.json"))
        if not i < args.n_frames:
            break


if __name__ == "__main__":
    args = parser.parse_args()
    gaussians_folder = save_frame_cfg_args(args.sh_degree, args.source, args.destination, args.frame_folder_fmt, args.start_frame)
    with torch.no_grad():
        gaussians = prepare_gaussians(
            sh_degree=args.sh_degree, device=args.device,
            load_ply=os.path.join(gaussians_folder, "point_cloud", "iteration_" + str(args.iteration_init), "point_cloud.ply"))
        dataset = prepare_fixedview_dataset(
            source=args.source, device=args.device,
            frame_folder_fmt=args.frame_folder_fmt, start_frame=args.start_frame, n_frames=None,
            load_camera=args.load_camera)
        motion_compensater = build_motion_compensater(gaussians, dataset, args.device, args.batch_size, rescale_factor=args.tracking_rescale)
        main(motion_compensater, dataset, gaussians_folder, args.sh_degree, args.source, args.destination, args.iteration, args)
