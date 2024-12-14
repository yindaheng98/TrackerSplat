from typing import Tuple
import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser, Namespace
from gaussian_splatting import GaussianModel
from instantsplatstream.dataset import VideoCameraDataset, ColmapVideoCameraDataset, FixedViewColmapVideoCameraDataset_from_json
from instantsplatstream.motionestimator import FixedViewMotionEstimator, MotionCompensater
from instantsplatstream.motionestimator.point_tracker import Cotracker3DotMotionEstimator, Cotracker3MotionEstimator, BaseMotionFuser

parser = ArgumentParser()
parser.add_argument("--sh_degree", default=3, type=int)
parser.add_argument("-s", "--source", required=True, type=str)
parser.add_argument("-d", "--destination", required=True, type=str)
parser.add_argument("-i", "--iteration", required=True, type=int)
parser.add_argument("--iteration_init", required=True, type=str, help="iteration of the initial gaussians")
parser.add_argument("-f", "--frame_folder_fmt", default="frame%d", type=str, help="frame folder format string")
parser.add_argument("--load_camera", default=None, type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--start_frame", default=1, type=int, help="start from which frame")
parser.add_argument("--tracking_rescale", default=1.0, type=float)


def init_gaussians(sh_degree: int, device: str, load_ply: str) -> Tuple[VideoCameraDataset, GaussianModel]:
    gaussians = GaussianModel(sh_degree).to(device)
    gaussians.load_ply(load_ply)
    return gaussians


def init_dataset(source: str, device: str, frame_folder_fmt: str, start_frame: int, n_frames=None, load_camera: str = None) -> Tuple[VideoCameraDataset, GaussianModel]:
    kwargs = dict(
        video_folder=source,
        frame_folder_fmt=frame_folder_fmt,
        start_frame=start_frame,
        n_frames=n_frames
    )
    dataset = (FixedViewColmapVideoCameraDataset_from_json(jsonpath=load_camera, **kwargs) if load_camera else ColmapVideoCameraDataset(**kwargs)).to(device)
    return dataset


def init_cfg_args(sh_degree: int, source: str, destination: str, frame_folder_fmt, frame):
    frame_str = frame_folder_fmt % frame
    gaussians_folder = os.path.join(destination, frame_str)
    source_folder = os.path.join(source, frame_str)
    os.makedirs(gaussians_folder, exist_ok=True)
    with open(os.path.join(gaussians_folder, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(sh_degree=sh_degree, source_path=source_folder)))
    return gaussians_folder


def main(sh_degree: int, source: str, destination: str, iteration: int, device: str, args):
    gaussians_folder = init_cfg_args(sh_degree, source, destination, args.frame_folder_fmt, args.start_frame)
    gaussians = init_gaussians(
        sh_degree=sh_degree, device=device,
        load_ply=os.path.join(gaussians_folder, "point_cloud", "iteration_" + str(args.iteration_init), "point_cloud.ply"))
    dataset = init_dataset(
        source=source, device=device,
        frame_folder_fmt=args.frame_folder_fmt, start_frame=args.start_frame, n_frames=None,
        load_camera=args.load_camera)
    batch_func = Cotracker3DotMotionEstimator(fuser=BaseMotionFuser(gaussians), device=device, rescale_factor=args.tracking_rescale)
    motion_estimator = FixedViewMotionEstimator(dataset, batch_func, batch_size=3, device=device)
    motion_compensater = MotionCompensater(gaussians, motion_estimator, device=device)
    for i, frame_gaussians in enumerate(motion_compensater):
        gaussians_folder = init_cfg_args(sh_degree, source, destination, args.frame_folder_fmt, args.start_frame + i + 1)
        save_path = os.path.join(gaussians_folder, "point_cloud", "iteration_" + str(iteration))
        os.makedirs(save_path, exist_ok=True)
        frame_gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))


if __name__ == "__main__":
    args = parser.parse_args()
    with torch.no_grad():
        main(args.sh_degree, args.source, args.destination, args.iteration, args.device, args)
