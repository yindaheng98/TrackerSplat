from typing import Tuple
import math
import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser, Namespace
from gaussian_splatting import GaussianModel
from gaussian_splatting.utils import psnr
from gaussian_splatting.dataset import JSONCameraDataset
from gaussian_splatting.dataset.colmap import ColmapCameraDataset
from instantsplatstream.dataset import VideoCameraDataset, ColmapVideoCameraDataset, FixedViewColmapVideoCameraDataset_from_json
from instantsplatstream.motionestimator import FixedViewMotionEstimator
from instantsplatstream.motionestimator.point_tracker import Cotracker3DotMotionEstimator, BaseMotionFuser, PointTrackSequence
from instantsplatstream.utils.motionfusion import motion_fusion
from instantsplatstream.utils.motionfusion.diff_gaussian_rasterization.motion_utils import solve_cov3D, compute_T, compute_Jacobian, compute_cov2D, transform_cov2D, unflatten_symmetry_3x3

parser = ArgumentParser()
parser.add_argument("--sh_degree", default=3, type=int)
parser.add_argument("-s", "--source", required=True, type=str)
parser.add_argument("-d", "--destination", required=True, type=str)
parser.add_argument("-i", "--iteration", required=True, type=int)
parser.add_argument("--load_camera", default=None, type=str)
parser.add_argument("--mode", choices=["pure", "densify", "camera"], default="pure")
parser.add_argument("--device", default="cuda", type=str)
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


def transform2d_pixel(H, W, device="cuda"):
    x = torch.arange(W, dtype=torch.float, device=device)
    y = torch.arange(H, dtype=torch.float, device=device)
    xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)
    A = torch.rand((2, 2)).to(device) - 0.5
    # A = torch.eye(2).to(device)
    b = (torch.rand(2).to(device) - 0.5) * H
    # b = torch.zeros(2).to(device)
    solution = torch.cat([b[:, None], A], dim=1).T
    xy_transformed = (xy.view(-1, 2) @ A.T + b).view(xy.shape)
    # X = torch.cat([torch.ones((xy.view(-1, 2).shape[0], 1)).to(device=xy.device), xy.view(-1, 2)], dim=1)
    # Y = xy_transformed.view(-1, 2)
    # diff = solution - torch.linalg.lstsq(X, Y).solution
    return xy_transformed, solution


def main(sh_degree: int, source: str, destination: str, iteration: int, device: str, args):
    with open(os.path.join(destination, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(sh_degree=sh_degree, source_path=source)))
    gaussians = init_gaussians(
        sh_degree=sh_degree, device=device,
        load_ply=os.path.join(destination, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply"))
    dataset = init_dataset(
        source=source, device=device,
        frame_folder_fmt="frame%d", start_frame=1, n_frames=None,
        load_camera=args.load_camera)
    render_path = os.path.join(destination, "ours_{}".format(iteration), "renders")
    gt_path = os.path.join(destination, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
    pbar = tqdm(dataset[0], desc="Rendering progress")
    batch_func = Cotracker3DotMotionEstimator(fuser=BaseMotionFuser(gaussians), device=device, rescale_factor=args.tracking_rescale) # This make things wrong
    motion_estimator = FixedViewMotionEstimator(dataset, batch_func, batch_size=8, device=device)
    for idx, camera in enumerate(pbar):
        camera = camera._replace(image_height=int(camera.image_height * 0.25) // 8 * 8, image_width=int(camera.image_width * 0.25) // 8 * 8, ground_truth_image=None)
        xy_transformed, solution = transform2d_pixel(camera.image_height, camera.image_width, device=device)
        batch_func.fuser(trackviews=[PointTrackSequence(
            image_height=camera.image_height,
            image_width=camera.image_width,
            FoVx=camera.FoVx,
            FoVy=camera.FoVy,
            R=camera.R,
            T=camera.T,
            track=xy_transformed.unsqueeze(0),
            mask=torch.ones_like(xy_transformed[..., 0], dtype=torch.bool).unsqueeze(0)
        )])
        motion_estimator.batch_func(motion_estimator.frames[0:3])
        out, motion2d, motion_alpha, motion_det, pixhit = motion_fusion(gaussians, camera, xy_transformed)
        rendering = out["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

        print("\nframe", idx)
        valid_idx = (out['radii'] > 0) & (motion_det > 1e-3) & (motion_alpha > 1e-3) & (pixhit > 1)
        motion_det = motion_det[valid_idx]
        motion_alpha = motion_alpha[valid_idx]
        pixhit = pixhit[valid_idx]
        # verify exported data
        B = motion2d[valid_idx]
        # T = motion2d[..., 6:15].reshape(-1, 3, 3)[valid_idx]
        # conv3D0 = motion2d[..., 6:12][valid_idx]
        conv3D = gaussians.get_covariance()[valid_idx]
        # print("conv3D", (conv3D - conv3D0).abs().max())
        J = compute_Jacobian(gaussians.get_xyz.detach(), camera.FoVx, camera.FoVy, camera.image_width, camera.image_height, camera.world_view_transform)
        T = compute_T(J, camera.world_view_transform)[valid_idx]
        # print("T", (T[:, :2, :] - T0[valid_idx]).abs().max())
        A2D, b2D = B[..., :-1], B[..., -1]
        conv2D = compute_cov2D(T, unflatten_symmetry_3x3(conv3D))
        conv2D_transformed = transform_cov2D(A2D, conv2D)

        # solve underdetermined system of equations
        X, Y = solve_cov3D(gaussians.get_xyz.detach()[valid_idx], camera.FoVx, camera.FoVy, camera.image_width, camera.image_height, camera.world_view_transform, conv2D_transformed)
        rank = torch.linalg.matrix_rank(X)
        valid_idx = (rank == 3)
        qr = torch.linalg.qr(X[valid_idx].transpose(1, 2))
        sigma_flatten = qr.Q.bmm(torch.linalg.inv(qr.R).transpose(1, 2)).bmm(Y[valid_idx]).squeeze(-1)
        print("A_{T} \Sigma_{3D} - b_{T}", (X.bmm(sigma_flatten.unsqueeze(-1)) - Y).abs().mean())  # !large value in Y will cause error in solving sigma

        sigma = torch.zeros((sigma_flatten.shape[0], 3, 3), device=sigma_flatten.device)
        sigma[:, 0, 0] = sigma_flatten[:, 0]
        sigma[:, 0, 1] = sigma_flatten[:, 1]
        sigma[:, 0, 2] = sigma_flatten[:, 2]
        sigma[:, 1, 0] = sigma_flatten[:, 1]
        sigma[:, 1, 1] = sigma_flatten[:, 3]
        sigma[:, 1, 2] = sigma_flatten[:, 4]
        sigma[:, 2, 0] = sigma_flatten[:, 2]
        sigma[:, 2, 1] = sigma_flatten[:, 4]
        sigma[:, 2, 2] = sigma_flatten[:, 5]

        motion_det = motion_det[valid_idx]
        motion_alpha = motion_alpha[valid_idx]
        pixhit = pixhit[valid_idx]
        # verify equations
        B = B[valid_idx]
        T = T[valid_idx]
        A2D, b2D = B[..., :-1], B[..., -1]
        conv2D_transformed = conv2D_transformed[valid_idx]
        print("T \Sigma'_{3D} T^\\top - \Sigma'_{2D}", (T.bmm(sigma).bmm(T.transpose(1, 2))[:, :2, :2] - conv2D_transformed).abs().mean())
        pass


if __name__ == "__main__":
    args = parser.parse_args()
    with torch.no_grad():
        main(args.sh_degree, args.source, args.destination, args.iteration, args.device, args)
