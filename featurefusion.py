from typing import Tuple
import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser, Namespace
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.dataset.colmap import ColmapTrainableCameraDataset
from gaussian_splatting.dataset import CameraDataset, JSONCameraDataset
from gaussian_splatting.dataset.colmap import ColmapCameraDataset
from instantsplatstream.featurefuser import FeatureFuser, FeatureExtractor, Dinov2FeatureExtractor, Dinov2SegFeatureExtractor, SAM2FeatureExtractor

parser = ArgumentParser()
parser.add_argument("--sh_degree", default=3, type=int)
parser.add_argument("-s", "--source", required=True, type=str)
parser.add_argument("-d", "--destination", required=True, type=str)
parser.add_argument("-i", "--iteration", required=True, type=int)
parser.add_argument("--load_camera", default=None, type=str)
parser.add_argument("--mode", choices=["pure", "densify", "camera"], default="pure")
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--extractor", choices=["sam2", "dinov2", "dinov2seg"], default="sam2")
parser.add_argument("--extractor_configfile", type=str, default="./configs/sam2.1/sam2.1_hiera_l.yaml")
parser.add_argument("--extractor_checkpoint", type=str, default="./checkpoints/sam2.1_hiera_large.pt")
parser.add_argument("--extractor_device", default="cuda", type=str)
parser.add_argument("--save_featuremap", action="store_true")
parser.add_argument("--colorify_algo", choices=["kmeans", "weightedsum"], default="kmeans", type=str)


def init_gaussians(sh_degree: int, source: str, device: str, mode: str, load_ply: str, load_camera: str = None) -> Tuple[CameraDataset, GaussianModel]:
    match mode:
        case "pure" | "densify":
            gaussians = GaussianModel(sh_degree).to(device)
            gaussians.load_ply(load_ply)
            dataset = (JSONCameraDataset(load_camera) if load_camera else ColmapCameraDataset(source)).to(device)
        case "camera":
            gaussians = CameraTrainableGaussianModel(sh_degree).to(device)
            gaussians.load_ply(load_ply)
            dataset = (TrainableCameraDataset.from_json(load_camera) if load_camera else ColmapTrainableCameraDataset(source)).to(device)
        case _:
            raise ValueError(f"Unknown mode: {mode}")
    return dataset, gaussians


def init_extractor(extractor: str, configfile: str, checkpoint: str, device: str) -> FeatureExtractor:
    match extractor:
        case "sam2":
            extractor = SAM2FeatureExtractor(configfile, checkpoint, device=device)
        case "dinov2":
            extractor = Dinov2FeatureExtractor(["configs/dinov2/ssl_default_config.yaml", configfile], checkpoint, device=device)
        case "dinov2seg":
            extractor = Dinov2SegFeatureExtractor(BACKBONE_SIZE=configfile, device=device)
        case _:
            raise ValueError(f"Unknown extractor: {extractor}")
    return extractor


def main(sh_degree: int, source: str, destination: str, iteration: int, device: str, args):
    with open(os.path.join(destination, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(sh_degree=sh_degree, source_path=source)))
    dataset, gaussians = init_gaussians(
        sh_degree=sh_degree, source=source, device=device, mode=args.mode,
        load_ply=os.path.join(destination, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply"),
        load_camera=args.load_camera)
    fusion_save_path = os.path.join(os.path.join(destination, f"featurefusion"))
    render_path = os.path.join(fusion_save_path, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)
    extractor = init_extractor(args.extractor, args.extractor_configfile, args.extractor_checkpoint, device=args.extractor_device)
    fuser = FeatureFuser(gaussians=gaussians, extractor=extractor, fusion_alpha_threshold=0.01, device=device)
    pbar = tqdm(dataset, desc="Rendering progress")
    for idx, camera in enumerate(pbar):
        feature_map = fuser.fuse(camera)
        if args.save_featuremap:
            alpha = 0.5
            color = extractor.assign_colors_to_feature_map(feature_map, algo=args.colorify_algo)
            gt = camera.ground_truth_image
            torchvision.utils.save_image(color * alpha + gt * (1 - alpha), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    # Save the features
    fusion_features_save_path = os.path.join(fusion_save_path, "features", args.extractor)
    makedirs(fusion_features_save_path, exist_ok=True)
    torch.save(fuser.get_features(), os.path.join(fusion_features_save_path, "iteration_" + str(iteration) + ".pt"))
    # Save the visualized point cloud
    makedirs(fusion_save_path, exist_ok=True)
    with open(os.path.join(fusion_save_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(sh_degree=sh_degree, source_path=source)))
    fusion_pcd_save_path = os.path.join(fusion_save_path, "point_cloud", "iteration_" + str(iteration))
    makedirs(fusion_pcd_save_path, exist_ok=True)
    fuser.visualize_features(colorify_algo=args.colorify_algo).save_ply(os.path.join(fusion_pcd_save_path, "point_cloud.ply"))


if __name__ == "__main__":
    args = parser.parse_args()
    with torch.no_grad():
        main(args.sh_degree, args.source, args.destination, args.iteration, args.device, args)
