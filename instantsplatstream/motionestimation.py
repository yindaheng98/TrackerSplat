import itertools
import torch
import os
import random
from tqdm import tqdm
from os import makedirs
from itertools import islice
from functools import partial
from typing import Callable
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.utils import psnr, ssim
from gaussian_splatting.utils.lpipsPyTorch import LPIPS
import gaussian_splatting.train
from instantsplatstream.dataset import prepare_fixedview_dataset, VideoCameraDataset
from instantsplatstream.motionestimator import FixedViewMotionEstimator, MotionCompensater
from instantsplatstream.motionestimator.point_tracker import BaseMotionFuser, build_point_track_batch_motion_estimator
from instantsplatstream.motionestimator.compensater import BaseMotionCompensater, build_motion_compensater
from instantsplatstream.motionestimator.incremental_trainer import IncrementalTrainingMotionEstimator, IncrementalTrainingRefiner, build_trainer_factory, IncrementalTrainingMotionEstimatorWrapper


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


class ITLogger(IncrementalTrainingMotionEstimatorWrapper):
    def __init__(self, base: IncrementalTrainingMotionEstimator, log_path: Callable[[int], str] = None):
        super().__init__(base)
        if log_path is None:
            def log_path(_):
                raise ValueError("log_path is not set")
        self.log_path = log_path
        self.frame = 0

    def training(self, dataset: CameraDataset, trainer: AbstractTrainer, iteration: int):
        '''Overload this method to make your own training'''
        log_path = self.log_path(self.frame)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"step,psnr,ssim,lpips\n")
        lpips = LPIPS(net_type='alex', version='0.1').to(self.device)
        pbar = tqdm(range(iteration), desc=f"Training frame {self.frame + 1}")
        epoch = list(range(len(dataset)))
        random.shuffle(epoch)
        avg_psnr_for_log = 0.0
        avg_ssim_for_log = 0.0
        avg_lpips_for_log = 0.0
        epoch_psnr = torch.zeros(len(dataset), 3, device=self.device)
        epoch_ssim = torch.zeros(len(dataset), device=self.device)
        epoch_lpips = torch.zeros(len(dataset), device=self.device)
        ema_loss_for_log = 0.0
        for step in pbar:
            epoch_idx = step % len(dataset)
            idx = epoch[epoch_idx]
            loss, out = trainer.step(dataset[idx])
            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                epoch_psnr[epoch_idx] = psnr(out["render"], dataset[idx].ground_truth_image).squeeze(-1).to(self.device)
                epoch_ssim[epoch_idx] = ssim(out["render"], dataset[idx].ground_truth_image).to(self.device)
                epoch_lpips[epoch_idx] = lpips(out["render"], dataset[idx].ground_truth_image).to(self.device)
                if step % 10 == 0:
                    pbar.set_postfix({'epoch': step // len(dataset), 'loss': ema_loss_for_log, 'psnr': avg_psnr_for_log, 'ssim': avg_ssim_for_log, 'lpips': avg_lpips_for_log})
            if epoch_idx + 1 == len(dataset):
                random.shuffle(epoch)
                avg_psnr_for_log = epoch_psnr.mean().item()
                avg_ssim_for_log = epoch_ssim.mean().item()
                avg_lpips_for_log = epoch_lpips.mean().item()
                with open(log_path, "a") as f:
                    f.write(f"{step // len(dataset)},{avg_psnr_for_log},{avg_ssim_for_log},{avg_lpips_for_log}\n")
                epoch_psnr = torch.zeros(len(dataset), 3, device=self.device)
                epoch_ssim = torch.zeros(len(dataset), device=self.device)
                epoch_lpips = torch.zeros(len(dataset), device=self.device)
        self.frame += 1


estimator_choices = ["dot", "dot-tapir", "dot-bootstapir", "dot-cotracker3", "cotracker3"]
compensater_choices = ["base", "propagate", "filter"]
trainer_choices = ["base", "regularized"]

train_choices = ["train/" + trainer for trainer in trainer_choices]
refine_choices = ["refine/" + trainer + "-" + compensater + "-" + estimator for trainer, compensater, estimator in itertools.product(trainer_choices, compensater_choices, estimator_choices)]
track_choices = ["track/" + compensater + "-" + estimator for compensater, estimator in itertools.product(compensater_choices, estimator_choices)]

pipeline_choices = train_choices + refine_choices + track_choices


def build_pipeline(pipeline: str, gaussians: GaussianModel, dataset: VideoCameraDataset, device: torch.device, batch_size: int, **kwargs) -> MotionCompensater:
    mode, estimator = pipeline.split("/", 1)
    itlogger = None
    if mode == "train":
        trainer = estimator
        batch_func = IncrementalTrainingMotionEstimator(trainer_factory=build_trainer_factory(trainer), iteration=1000, device=device)
        itlogger = ITLogger(batch_func)
        motion_estimator = FixedViewMotionEstimator(dataset=dataset, batch_func=itlogger, device=device, batch_size=batch_size)
        motion_compensater = BaseMotionCompensater(gaussians=gaussians, estimator=motion_estimator, device=device)
    elif mode == "track":
        compensater, estimator = estimator.split("-", 1)
        batch_func = build_point_track_batch_motion_estimator(estimator=estimator, fuser=BaseMotionFuser(gaussians), device=device, **kwargs)
        motion_estimator = FixedViewMotionEstimator(dataset=dataset, batch_func=batch_func, device=device, batch_size=batch_size)
        motion_compensater = build_motion_compensater(compensater=compensater, gaussians=gaussians, estimator=motion_estimator, device=device)
    elif mode == "refine":
        trainer, compensater, estimator = estimator.split("-", 2)
        batch_func = build_point_track_batch_motion_estimator(estimator=estimator, fuser=BaseMotionFuser(gaussians), device=device, **kwargs)
        motion_estimator = FixedViewMotionEstimator(dataset=dataset, batch_func=batch_func, device=device, batch_size=batch_size)
        motion_compensater = build_motion_compensater(compensater=compensater, gaussians=gaussians, estimator=motion_estimator, device=device)
        batch_func = IncrementalTrainingRefiner(base_batch_func=batch_func, base_compensater=motion_compensater, trainer_factory=build_trainer_factory(trainer), iteration=1000, device=device)
        itlogger = ITLogger(batch_func)
        motion_estimator = FixedViewMotionEstimator(dataset=dataset, batch_func=itlogger, device=device, batch_size=batch_size)
        motion_compensater = BaseMotionCompensater(gaussians=gaussians, estimator=motion_estimator, device=device)
    else:
        ValueError(f"Unknown estimator: {estimator}")
    return motion_compensater, itlogger


def motion_compensate(motion_compensater: MotionCompensater, itlogger: ITLogger, dataset: VideoCameraDataset, save_frame_cfg_args, iteration: int, start_frame: int, n_frames: int):
    log_subpath = os.path.join("log", "iteration_" + str(iteration), "log.csv")
    if itlogger is not None:
        itlogger.log_path = lambda frame: os.path.join(save_frame_cfg_args(frame=start_frame + frame + 1), log_subpath)
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

    parser.add_argument("--pipeline", choices=pipeline_choices, default="track/base-dot-cotracker3")
    parser.add_argument("--iteration_init", required=True, type=str, help="iteration of the initial gaussians")
    parser.add_argument("-f", "--frame_folder_fmt", default="frame%d", type=str, help="frame folder format string")
    parser.add_argument("-n", "--n_frames", default=None, type=int, help="number of frames to process")
    parser.add_argument("-b", "--batch_size", default=3, type=int, help="batch size of point tracking")
    parser.add_argument("--start_frame", default=1, type=int, help="start from which frame")
    parser.add_argument("--tracking_rescale", default=1.0, type=float)
    args = parser.parse_args()
    save_frame_cfg_args = partial(save_cfg_args, sh_degree=args.sh_degree, source=args.source, destination=os.path.join(args.destination, args.pipeline), frame_folder_fmt=args.frame_folder_fmt)
    load_ply = os.path.join(args.destination, args.frame_folder_fmt % args.start_frame, "point_cloud", "iteration_" + str(args.iteration_init), "point_cloud.ply")
    gaussians = prepare_gaussians(
        sh_degree=args.sh_degree, device=args.device,
        load_ply=load_ply)
    dataset = prepare_fixedview_dataset(
        source=args.source, device=args.device,
        frame_folder_fmt=args.frame_folder_fmt, start_frame=args.start_frame, n_frames=None,
        load_camera=args.load_camera)
    motion_compensater, itlogger = build_pipeline(args.pipeline, gaussians, dataset, args.device, args.batch_size, rescale_factor=args.tracking_rescale)
    motion_compensate(motion_compensater, itlogger, dataset, save_frame_cfg_args, args.iteration, args.start_frame, args.n_frames)
