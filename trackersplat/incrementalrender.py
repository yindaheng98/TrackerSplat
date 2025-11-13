import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.utils import psnr, ssim
from gaussian_splatting.utils.lpipsPyTorch import lpips
from gaussian_splatting.render import prepare_rendering
from gaussian_splatting.prepare import prepare_gaussians
from trackersplat.utils.motionfusion import compute_mean2D
import matplotlib.pyplot as plt
import numpy as np


def draw_motion(rendering, point_image, point_image_after, save_path, threshold=16):
    mask = ((point_image_after - point_image).abs() > threshold).any(-1)
    rendering = (rendering * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    point_image_motion = (point_image_after[mask] - point_image[mask]).cpu().numpy()
    point_image = point_image[mask].cpu().numpy().astype(np.int32)
    fig = plt.figure(figsize=(rendering.shape[1], rendering.shape[0]), dpi=2)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.imshow(rendering)
    ax.quiver(
        point_image[:, 0], point_image[:, 1], point_image_motion[:, 0], point_image_motion[:, 1],
        angles='xy', scale_units='xy', scale=1, width=0.0001, headwidth=1, minshaft=1, minlength=1,
        color='r', alpha=0.25
    )
    ax.set_xlim(0, rendering.shape[1])
    ax.set_ylim(rendering.shape[0], 0)
    ax.axis('off')
    fig.savefig(save_path)
    plt.close(fig)


def rendering(dataset: CameraDataset, gaussians: GaussianModel, gaussians_base: GaussianModel, save: str):
    os.makedirs(save, exist_ok=True)
    dataset.save_cameras(os.path.join(save, "cameras.json"))
    render_path = os.path.join(save, "renders")
    gt_path = os.path.join(save, "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
    motion_path = os.path.join(save, "motion")
    makedirs(motion_path, exist_ok=True)
    pbar = tqdm(dataset, dynamic_ncols=True, desc="Rendering progress")
    with open(os.path.join(save, "log.csv"), "w") as f:
        f.write(f"cam,psnr,ssim,lpips\n")
    for idx, camera in enumerate(pbar):
        out = gaussians(camera)
        rendering = out["render"]
        gt = camera.ground_truth_image
        if camera.ground_truth_image_mask is not None:
            gt_nomask, rendering_nomask = gt, rendering
            gt = gt * camera.ground_truth_image_mask
            rendering = rendering * camera.ground_truth_image_mask
        psnr_log = psnr(rendering, gt).mean().item()
        ssim_log = ssim(rendering, gt).mean().item()
        lpips_log = lpips(rendering, gt).mean().item()
        pbar.set_postfix({"PSNR": psnr_log, "SSIM": ssim_log, "LPIPS": lpips_log})
        with open(os.path.join(save, "log.csv"), "a") as f:
            f.write(f"{idx},{psnr_log},{ssim_log},{lpips_log}\n")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))
        if camera.ground_truth_image_mask is not None:
            torchvision.utils.save_image(gt_nomask, os.path.join(gt_path, '{0:05d}'.format(idx) + "_nomask.png"))
            torchvision.utils.save_image(rendering_nomask, os.path.join(render_path, '{0:05d}'.format(idx) + "_nomask.png"))
        point_image_base = compute_mean2D(camera.full_proj_transform, camera.image_width, camera.image_height, gaussians_base.get_xyz.detach())
        point_image = compute_mean2D(camera.full_proj_transform, camera.image_width, camera.image_height, gaussians.get_xyz.detach())
        valid_mask = (out['radii'] > 0) & (0 < point_image).all(-1) & (point_image[:, 0] < camera.image_width) & (point_image[:, 1] < camera.image_height)
        draw_motion(rendering, point_image_base[valid_mask], point_image[valid_mask], os.path.join(motion_path, '{0:05d}'.format(idx) + ".png"))
        if camera.ground_truth_image_mask is not None:
            draw_motion(rendering_nomask, point_image_base[valid_mask], point_image[valid_mask], os.path.join(motion_path, '{0:05d}'.format(idx) + "_nomask.png"))


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--mode", choices=["base", "camera"], default="base")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--with_image_mask", action="store_true")
    parser.add_argument("--destination_base", required=True, type=str)
    parser.add_argument("--iteration_base", default=None, type=int)
    args = parser.parse_args()
    args.iteration_base = args.iteration_base or args.iteration
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    load_ply_base = os.path.join(args.destination_base, "point_cloud", "iteration_" + str(args.iteration_base), "point_cloud.ply")
    save = os.path.join(args.destination, "ours_{}".format(args.iteration))
    with torch.no_grad():
        dataset, gaussians = prepare_rendering(
            sh_degree=args.sh_degree, source=args.source, device=args.device, trainable_camera=args.mode == "camera",
            load_ply=load_ply, load_camera=args.load_camera,
            load_mask=args.with_image_mask, load_depth=False)
        gaussians_base = prepare_gaussians(sh_degree=args.sh_degree, source=args.source, device=args.device, trainable_camera=args.mode == "camera", load_ply=load_ply_base)
        rendering(dataset, gaussians, gaussians_base, save)
