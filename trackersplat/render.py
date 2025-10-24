import os
import shutil

import torch
import torchvision
import tqdm
from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.dataset.dataset import CameraDataset
from gaussian_splatting.prepare import prepare_gaussians, prepare_dataset
from extrinterp import ExtrinsicInterpolationDataset


def load_cameras(
        path: str, device: str, n: int, window_size: int,
        trainable_camera: bool = False,
        use_intrinsics: dict = dict(image_width=1600, FoVx=1.4749, image_height=1200, FoVy=1.1990),
) -> CameraDataset:
    dataset = prepare_dataset(source=None, device=device, trainable_camera=trainable_camera, load_camera=path, load_depth=False)
    return ExtrinsicInterpolationDataset(dataset=dataset, n=n, window_size=window_size, **use_intrinsics).to(device)


def prepare_frame(sh_degree: int, device: str, mode: str, destination: str, iteration: int) -> GaussianModel:
    load_ply = os.path.join(destination, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply")
    gaussians = prepare_gaussians(sh_degree=sh_degree, source=None, device=device, trainable_camera=mode == "camera", load_ply=load_ply)
    return gaussians


def render_frame(camera: Camera, gaussians: GaussianModel, render_path: str, idx: int):
    torch.cuda.empty_cache()
    out = gaussians(camera)
    rendering = out["render"]
    torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


def render_video(
        cameras: CameraDataset,
        sh_degree, data_dir,
        destination, iteration,
        destination_init, iteration_init,
        device, mode,
        frame_format, start_frame, end_frame,
):
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(data_dir, exist_ok=True)
    gaussians = prepare_frame(sh_degree=sh_degree, device=device, mode=mode, destination=destination_init, iteration=iteration_init)
    render_frame(cameras[0], gaussians, data_dir, start_frame - 1)
    for i in tqdm.tqdm(range(start_frame, end_frame + 1), desc="Rendering frames"):
        frame = frame_format % i
        frame_destination = os.path.join(destination, frame)
        gaussians = prepare_frame(sh_degree=sh_degree, device=device, mode=mode, destination=frame_destination, iteration=iteration)
        render_frame(cameras[i - start_frame + 1], gaussians, data_dir, i)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", default=30000, type=int)
    parser.add_argument("--destination_init", required=True, type=str)
    parser.add_argument("--iteration_init", default=30000, type=int)
    parser.add_argument("--frame_format", default="frame%d", type=str)
    parser.add_argument("--frame_start", required=True, type=int)
    parser.add_argument("--frame_end", required=True, type=int)
    parser.add_argument("--load_camera", required=True, type=str)
    parser.add_argument("--mode", choices=["base", "camera"], default="base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--interp_n", type=int, default=None)
    parser.add_argument("--interp_window_size", type=int, default=3)
    parser.add_argument("--use_intrinsics", type=str,
                        default="dict(image_width=1600,FoVx=1.4749,image_height=1200,FoVy=1.1990)",
                        help="Use intrinsics for rendering, can be an integer index or a dict with keys: image_height, image_width, FoVx, FoVy")
    parser.add_argument("--downsample", default=4, type=int)
    parser.add_argument("--data_dir", required=True, type=str)
    args = parser.parse_args()
    with torch.no_grad():
        cameras = load_cameras(
            path=args.load_camera, device=args.device, n=args.interp_n if args.interp_n is not None else (args.frame_end - args.frame_start + 2),
            window_size=args.interp_window_size,
            trainable_camera=args.mode == "camera",
            use_intrinsics=eval(args.use_intrinsics),
        )
        render_video(
            cameras=cameras, sh_degree=args.sh_degree, data_dir=args.data_dir,
            destination=args.destination, iteration=args.iteration,
            destination_init=args.destination_init, iteration_init=args.iteration_init,
            device=args.device, mode=args.mode,
            frame_format=args.frame_format, start_frame=args.frame_start, end_frame=args.frame_end,
        )
