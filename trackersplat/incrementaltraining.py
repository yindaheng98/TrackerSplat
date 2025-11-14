import os
from typing import Tuple
import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import AbstractTrainer, BaseTrainer
from gaussian_splatting.train import save_cfg_args, training
from gaussian_splatting.prepare import prepare_dataset, prepare_gaussians
from trackersplat.motionestimator.incremental_trainer import BaseTrainer, RegularizedTrainer


def prepare_training(
        sh_degree: int, source: str, device: str, mode: str, load_ply: str,
        load_mask=True,
        configs={}) -> Tuple[CameraDataset, GaussianModel, AbstractTrainer]:
    # do not support load_camera:
    # if we need to load, we typically load from frame1/cameras.json, since frame9/cameras.json is currently not exists
    # but frame1/cameras.json contains the image path for frame1 rather than frame9
    dataset = prepare_dataset(source=source, device=device, trainable_camera=False, load_camera=None, load_mask=load_mask, load_depth=False)
    gaussians = prepare_gaussians(sh_degree=sh_degree, source=source, device=device, trainable_camera=False, load_ply=load_ply)
    match mode:
        case "base":
            trainer = BaseTrainer(
                gaussians,
                spatial_lr_scale=dataset.scene_extent(),
                **configs
            )
        case "regularized":
            base_gaussians = GaussianModel(sh_degree).to(device)
            base_gaussians.load_ply(load_ply)
            trainer = RegularizedTrainer(
                gaussians, base_gaussians,
                spatial_lr_scale=dataset.scene_extent(),
                **configs
            )
        case _:
            raise ValueError(f"Unknown mode: {mode}")
    return dataset, gaussians, trainer


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", default=1000, type=int)
    parser.add_argument("--with_image_mask", action="store_true")
    parser.add_argument("--mode", choices=["base", "regularized"], default="base")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--destination_base", required=True, type=str)
    parser.add_argument("--iteration_base", default=None, type=int)
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    args = parser.parse_args()
    save_cfg_args(args.destination, args.sh_degree, args.source)
    torch.autograd.set_detect_anomaly(False)

    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    load_ply_base = os.path.join(args.destination_base, "point_cloud", "iteration_" + str(args.iteration_base), "point_cloud.ply")
    dataset, gaussians, trainer = prepare_training(
        sh_degree=args.sh_degree, source=args.source, device=args.device, mode=args.mode, load_ply=load_ply_base,
        load_mask=args.with_image_mask, configs=configs)
    dataset.save_cameras(os.path.join(args.destination, "cameras.json"))
    torch.cuda.empty_cache()
    training(
        dataset=dataset, gaussians=gaussians, trainer=trainer,
        destination=args.destination, iteration=args.iteration, save_iterations=[],
        device=args.device)
