import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.train import save_cfg_args
from gaussian_splatting.render import prepare_rendering
from instantsplatstream.featurefuser import FeatureFuser, build_feature_extractor


def feature_fusion(dataset: CameraDataset, fuser: FeatureFuser, save: str, iteration: int, save_featuremap: bool = False, colorify_algo: str = "kmeans", colorify_alpha: float = 0.5):
    render_path = os.path.join(save, "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    pbar = tqdm(dataset, desc="Rendering progress")
    for idx, camera in enumerate(pbar):
        feature_map = fuser.fuse(camera)
        if save_featuremap:
            color = extractor.assign_colors_to_feature_map(feature_map, algo=colorify_algo)
            gt = camera.ground_truth_image
            colorify_alpha = max(0, min(1, colorify_alpha))
            torchvision.utils.save_image(color * colorify_alpha + gt * (1 - colorify_alpha), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    # Save the features
    features_save = os.path.join(save, "features")
    makedirs(features_save, exist_ok=True)
    torch.save(fuser.get_features(), os.path.join(features_save, "iteration_" + str(iteration) + ".pt"))
    # Save the visualized point cloud
    pcd_save = os.path.join(save, "point_cloud", "iteration_" + str(iteration))
    makedirs(pcd_save, exist_ok=True)
    fuser.visualize_features(colorify_algo=colorify_algo).save_ply(os.path.join(pcd_save, "point_cloud.ply"))


if __name__ == "__main__":
    from argparse import ArgumentParser
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
    parser.add_argument("--colorify_algo", choices=["kmeans", "kmeans-pca", "weightedsum"], default="kmeans", type=str)
    parser.add_argument("--colorify_alpha", default=0.5, type=float)
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    save = os.path.join(os.path.join(args.destination, f"featurefusion-{args.extractor}"))
    save_cfg_args(save, args.sh_degree, args.source)
    with torch.no_grad():
        dataset, gaussians = prepare_rendering(
            sh_degree=args.sh_degree, source=args.source, device=args.device, mode=args.mode,
            load_ply=load_ply, load_camera=args.load_camera)
        extractor = build_feature_extractor(args.extractor, args.extractor_configfile, args.extractor_checkpoint, args.extractor_device)
        fuser = FeatureFuser(gaussians=gaussians, extractor=extractor, fusion_alpha_threshold=0.01, device=args.device)
        feature_fusion(dataset, fuser, save, args.iteration, args.save_featuremap, args.colorify_algo, args.colorify_alpha)
