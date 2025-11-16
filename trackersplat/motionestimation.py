import itertools
import torch
import os
import shutil
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
from trackersplat.dataset import prepare_fixedview_dataset, VideoCameraDataset
from trackersplat import MotionCompensater
from trackersplat.motionestimator import FixedViewMotionEstimator
from trackersplat.motionestimator.point_tracker import DetectFixMotionFuser, build_point_track_batch_motion_estimator
from trackersplat.motionestimator.incremental_trainer import IncrementalTrainingMotionEstimator, build_trainer_factory, TrainingProcess, BaseTrainingProcess
from trackersplat.motionestimator.refiner import build_training_refiner, build_regularization_refiner


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


class LoggerTrainingProcess(BaseTrainingProcess):
    def __init__(self, log_path: Callable[[int], str], device: torch.device = torch.device("cuda")):
        self.log_path = log_path
        self.device = device

    def __call__(self, dataset: CameraDataset, trainer: AbstractTrainer, iteration: int, frame_idx: int):
        '''Overload this method to make your own training'''
        log_path = self.log_path(frame_idx)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"epoch,camera,psnr,ssim,lpips,masked_psnr,masked_ssim,masked_lpips\n")
        lpips = LPIPS(net_type='alex', version='0.1').to(self.device)
        pbar = tqdm(range(iteration), desc=f"Training frame {frame_idx}", dynamic_ncols=True)
        epoch = list(range(len(dataset)))
        random.shuffle(epoch)
        avg_psnr_for_log = 0.0
        avg_ssim_for_log = 0.0
        avg_lpips_for_log = 0.0
        avg_maskpsnr_for_log = 0.0
        avg_maskssim_for_log = 0.0
        avg_masklpips_for_log = 0.0
        epoch_psnr = torch.zeros(len(dataset), 3, device=self.device)
        epoch_ssim = torch.zeros(len(dataset), device=self.device)
        epoch_lpips = torch.zeros(len(dataset), device=self.device)
        epoch_maskpsnr = torch.zeros(len(dataset), 3, device=self.device)
        epoch_maskssim = torch.zeros(len(dataset), 3, device=self.device)
        epoch_masklpips = torch.zeros(len(dataset), 3, device=self.device)
        epoch_camids = []
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
                if dataset[idx].ground_truth_image_mask is not None:
                    ground_truth_maskimage = dataset[idx].ground_truth_image * dataset[idx].ground_truth_image_mask
                    rendered_maskimage = out["render"] * dataset[idx].ground_truth_image_mask
                    epoch_maskpsnr[epoch_idx] = psnr(rendered_maskimage, ground_truth_maskimage).squeeze(-1).to(self.device)
                    epoch_maskssim[epoch_idx] = ssim(rendered_maskimage, ground_truth_maskimage).to(self.device)
                    epoch_masklpips[epoch_idx] = lpips(rendered_maskimage, ground_truth_maskimage).to(self.device)
                epoch_camids.append(idx)
                if step % 10 == 0:
                    pbar.set_postfix({
                        'epoch': step // len(dataset),
                        'loss': ema_loss_for_log,
                        'psnr': f"{avg_psnr_for_log:.2f} (masked: {avg_maskpsnr_for_log:.2f})",
                        'ssim': f"{avg_ssim_for_log:.2f} (masked: {avg_maskssim_for_log:.2f})",
                        'lpips': f"{avg_lpips_for_log:.4f} (masked: {avg_masklpips_for_log:.4f})",
                    })
            if epoch_idx + 1 == len(dataset):
                random.shuffle(epoch)
                avg_psnr_for_log = epoch_psnr.mean().item()
                avg_ssim_for_log = epoch_ssim.mean().item()
                avg_lpips_for_log = epoch_lpips.mean().item()
                avg_maskpsnr_for_log = epoch_maskpsnr.mean().item()
                avg_maskssim_for_log = epoch_maskssim.mean().item()
                avg_masklpips_for_log = epoch_masklpips.mean().item()
                with open(log_path, "a") as f:
                    for i in range(len(dataset)):
                        data = f"{step // len(dataset) + 1},{epoch_camids[i] + 1},"
                        data += f"{epoch_psnr[i].mean().item()},{epoch_ssim[i].item()},{epoch_lpips[i].item()},"
                        data += f"{epoch_maskpsnr[i].mean().item()},{epoch_maskssim[i].mean().item()},{epoch_masklpips[i].mean().item()}\n"
                        f.write(data)
                epoch_psnr = torch.zeros(len(dataset), 3, device=self.device)
                epoch_ssim = torch.zeros(len(dataset), device=self.device)
                epoch_lpips = torch.zeros(len(dataset), device=self.device)
                epoch_maskpsnr = torch.zeros(len(dataset), 3, device=self.device)
                epoch_maskssim = torch.zeros(len(dataset), 3, device=self.device)
                epoch_masklpips = torch.zeros(len(dataset), 3, device=self.device)


estimator_choices = ["dot", "dot-tapir", "dot-bootstapir", "dot-cotracker3", "cotracker3"]
compensater_choices = ["base", "propagate", "filter"]
trainer_choices = ["base", "regularized", "masked", "maskregularized", "hexplane", "regularizedhexplane", "hicom"]

train_choices = ["train/" + trainer for trainer in trainer_choices]
refine_choices = ["refine/" + trainer + "-" + compensater + "-" + estimator for trainer, compensater, estimator in itertools.product(trainer_choices, compensater_choices, estimator_choices)]
track_choices = ["track/" + compensater + "-" + estimator for compensater, estimator in itertools.product(compensater_choices, estimator_choices)]

pipeline_choices = train_choices + refine_choices + track_choices


def build_pipeline(pipeline: str, gaussians: GaussianModel, dataset: VideoCameraDataset, training_proc: TrainingProcess, device: torch.device, batch_size: int, iteration: int, configs_refining: dict, **kwargs) -> MotionCompensater:
    mode, estimator = pipeline.split("/", 1)
    if mode[:5] == "train":
        trainer = estimator
        batch_func = IncrementalTrainingMotionEstimator(trainer_factory=build_trainer_factory(trainer, **kwargs), training_proc=training_proc, iteration=iteration, device=device)
    elif mode == "track":
        refiner, estimator = estimator.split("-", 1)
        batch_func = build_point_track_batch_motion_estimator(estimator=estimator, fuser=DetectFixMotionFuser(gaussians), device=device, **kwargs)
        batch_func = build_regularization_refiner(refiner=refiner, base_batch_func=batch_func, device=device)
    elif mode == "refine":
        trainer, refiner, estimator = estimator.split("-", 2)
        batch_func = build_point_track_batch_motion_estimator(estimator=estimator, fuser=DetectFixMotionFuser(gaussians), device=device, **kwargs)
        batch_func = build_regularization_refiner(refiner=refiner, base_batch_func=batch_func, device=device)
        batch_func = build_training_refiner(trainer=trainer, base_batch_func=batch_func, device=device, training_proc=training_proc, iteration=iteration, **configs_refining)
    else:
        ValueError(f"Unknown estimator: {estimator}")
    motion_estimator = FixedViewMotionEstimator(dataset=dataset, batch_func=batch_func, device=device, batch_size=batch_size)
    motion_compensater = MotionCompensater(gaussians=gaussians, estimator=motion_estimator, device=device)
    return motion_compensater


def motion_compensate(motion_compensater: MotionCompensater, dataset: VideoCameraDataset, save_frame_cfg_args, iteration: int, start_frame: int, n_frames: int):
    for i, frame_gaussians in enumerate(islice(motion_compensater, n_frames)):
        destination_folder = save_frame_cfg_args(frame=start_frame + i + 1)
        shutil.rmtree(os.path.join(destination_folder, "point_cloud"), ignore_errors=True)  # remove the previous point cloud
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
    parser.add_argument("--with_image_mask", action="store_true")
    parser.add_argument("--with_depth_data", action="store_true")
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--pipeline", choices=pipeline_choices, default="track/base-dot-cotracker3")
    parser.add_argument("--iteration_init", required=True, type=str, help="iteration of the initial gaussians")
    parser.add_argument("-f", "--frame_folder_fmt", default="frame%d", type=str, help="frame folder format string")
    parser.add_argument("-n", "--n_frames", default=None, type=int, help="number of frames to process")
    parser.add_argument("-b", "--batch_size", default=3, type=int, help="batch size of point tracking")
    parser.add_argument("--start_frame", default=1, type=int, help="start from which frame")
    parser.add_argument("-o", "--option_estimation", default=[], action='append', type=str)
    parser.add_argument("-r", "--option_refining", default=[], action='append', type=str)
    args = parser.parse_args()
    save_frame_cfg_args = partial(save_cfg_args, sh_degree=args.sh_degree, source=args.source, destination=os.path.join(args.destination, args.pipeline), frame_folder_fmt=args.frame_folder_fmt)
    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option_estimation}
    configs_refining = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option_refining}
    load_ply = os.path.join(args.destination, args.frame_folder_fmt % args.start_frame, "point_cloud", "iteration_" + str(args.iteration_init), "point_cloud.ply")
    gaussians = prepare_gaussians(
        sh_degree=args.sh_degree, device=args.device,
        load_ply=load_ply)
    dataset = prepare_fixedview_dataset(
        source=args.source, device=args.device,
        frame_folder_fmt=args.frame_folder_fmt, start_frame=args.start_frame, n_frames=args.n_frames,
        load_camera=args.load_camera,
        load_mask=args.with_image_mask, load_depth=args.with_depth_data)
    training_proc = LoggerTrainingProcess(lambda frame: os.path.join(save_frame_cfg_args(frame=frame), os.path.join("log", "iteration_" + str(args.iteration), "log.csv")), device=args.device)
    motion_compensater = build_pipeline(args.pipeline, gaussians, dataset, training_proc, args.device, args.batch_size, args.iteration, configs_refining, **configs)
    motion_compensate(motion_compensater, dataset, save_frame_cfg_args, args.iteration, args.start_frame, args.n_frames)
