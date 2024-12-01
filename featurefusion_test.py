import math
import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser, Namespace
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.dataset.colmap import ColmapTrainableCameraDataset
from gaussian_splatting.utils import psnr
from gaussian_splatting.dataset import JSONCameraDataset
from gaussian_splatting.dataset.colmap import ColmapCameraDataset
from instantsplatstream.utils.featurefusion import feature_fusion
from instantsplatstream.featurefuser import FeatureFuser, Dinov2FeatureExtractor, Dinov2SegFeatureExtractor, SAM2FeatureExtractor

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


def init_gaussians(sh_degree: int, source: str, device: str, mode: str, load_ply: str, load_camera: str = None):
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


def init_extractor(extractor: str, configfile: str, checkpoint: str, device: str):
    match extractor:
        case "sam2":
            extractor = SAM2FeatureExtractor(configfile, checkpoint, device=device)
        case "dinov2":
            extractor = Dinov2FeatureExtractor(["configs/dinov2/ssl_default_config.yaml", configfile], checkpoint, device=device)
        case "dinov2seg":
            extractor = Dinov2SegFeatureExtractor(device=device)
        case _:
            raise ValueError(f"Unknown extractor: {extractor}")
    return extractor


def assign_color_block(BLOCK_SIZE, camera):
    x = torch.arange(0, math.ceil(camera.image_width / BLOCK_SIZE)).repeat(BLOCK_SIZE, 1).T.reshape(-1)[:camera.image_width]
    y = torch.arange(0, math.ceil(camera.image_height / BLOCK_SIZE)).repeat(BLOCK_SIZE, 1).T.reshape(-1)[:camera.image_height]
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    tile_id = (x[-1] + 1) * grid_y + grid_x
    colors = torch.rand(((x[-1] + 1) * (y[-1] + 1), 3))
    feature_map = colors[tile_id]
    return feature_map


def main(sh_degree: int, source: str, destination: str, iteration: int, device: str, args):
    with open(os.path.join(destination, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(sh_degree=sh_degree, source_path=source)))
    dataset, gaussians = init_gaussians(
        sh_degree=sh_degree, source=source, device=device, mode=args.mode,
        load_ply=os.path.join(destination, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply"),
        load_camera=args.load_camera)
    render_path = os.path.join(destination, "ours_{}".format(iteration), "renders")
    gt_path = os.path.join(destination, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
    extractor = init_extractor(args.extractor, args.extractor_configfile, args.extractor_checkpoint, device=args.extractor_device)
    fuser = FeatureFuser(gaussians=gaussians, extractor=extractor, fusion_alpha_threshold=0.01, device=device)
    opacity_backup = gaussians._opacity.clone()
    features_dc_backup = gaussians._features_dc.clone()
    features_rest_backup = gaussians._features_rest.clone()
    pbar = tqdm(dataset, desc="Rendering progress")
    for idx, camera in enumerate(pbar):
        fuser.fuse(camera)
        feature_map = assign_color_block(16, camera)
        out, features, features_alpha = feature_fusion(gaussians, camera, feature_map.to(device))
        features = features / features_alpha.unsqueeze(-1)
        features[features_alpha < 1e-5, ...] = 0
        rendering = out["render"]
        gt = camera.ground_truth_image
        pbar.set_postfix({"PSNR": psnr(rendering, gt).mean().item()})
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))
        fusion_save_path = os.path.join(os.path.join(destination, f"fusion_{idx+1}"))
        makedirs(os.path.join(fusion_save_path, "point_cloud", "iteration_" + str(iteration)), exist_ok=True)
        with open(os.path.join(fusion_save_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(sh_degree=sh_degree, source_path=source)))
        gaussians._opacity[features_alpha < 1] += gaussians.inverse_opacity_activation(features_alpha[features_alpha < 1].unsqueeze(-1))
        gaussians._opacity[gaussians.get_opacity < 0.05] = gaussians.inverse_opacity_activation(torch.tensor(0.05)).to(device)
        gaussians._features_dc[:, 0, :] = features
        gaussians._features_rest[...] = 0
        gaussians.save_ply(os.path.join(fusion_save_path, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply"))
        gaussians._opacity[...] = opacity_backup
        gaussians._features_dc[...] = features_dc_backup
        gaussians._features_rest[...] = features_rest_backup


if __name__ == "__main__":
    args = parser.parse_args()
    with torch.no_grad():
        main(args.sh_degree, args.source, args.destination, args.iteration, args.device, args)
