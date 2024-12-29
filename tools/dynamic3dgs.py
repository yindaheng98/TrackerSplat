import shutil
import subprocess
import os
from tqdm import tqdm
import argparse
import json
import torch
import torch.nn.functional as F


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    # if torch.is_grad_enabled():
    #     ret[positive_mask] = torch.sqrt(x[positive_mask])
    # else:
    #     ret = torch.where(positive_mask, torch.sqrt(x), ret)
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


parser = argparse.ArgumentParser()
parser.add_argument("--colmap_executable", type=str, required=True, help="path to colmap executable")
parser.add_argument("--path", type=str, required=True, help="path to the video folder")
parser.add_argument("--n_frames", type=int, default=150, help="number of frames")
parser.add_argument("--use_gpu", type=str, default="1", help="path to colmap executable")


def read_camera_meta(path):
    with open(path) as f:
        return json.load(f)


def execute(cmd):
    proc = subprocess.Popen(cmd, shell=False)
    proc.communicate()
    return proc.returncode


def feature_extractor(args, folder):
    cmd = [
        args.colmap_executable, "feature_extractor",
        "--database_path", os.path.join(folder, "database.db"),
        "--image_path", os.path.join(folder, "images"),
        "--ImageReader.camera_model", "PINHOLE",
        "--SiftExtraction.use_gpu", args.use_gpu,
        "--ImageReader.single_camera_per_image", "1",
    ]
    return execute(cmd)


def exhaustive_matcher(args, folder):
    cmd = [
        args.colmap_executable, "exhaustive_matcher",
        "--database_path", os.path.join(folder, "database.db"),
        "--SiftMatching.use_gpu", args.use_gpu,
    ]
    return execute(cmd)


def point_triangulator(args, folder, mapper_input_path):
    cmd = [
        args.colmap_executable, "point_triangulator",
        "--database_path", os.path.join(folder, "database.db"),
        "--input_path", mapper_input_path,
        "--output_path", mapper_input_path,
        "--image_path", os.path.join(folder, "images")
    ]
    return execute(cmd)


def mapper(args, folder, mapper_input_path):
    os.makedirs(os.path.join(folder, "sparse"), exist_ok=True)
    cmd = [
        args.colmap_executable, "mapper",
        "--database_path", os.path.join(folder, "database.db"),
        "--image_path", os.path.join(folder, "images"),
        "--Mapper.ba_global_function_tolerance=0.000001",
        "--input_path", mapper_input_path,
        "--output_path", os.path.join(folder, "sparse")
    ]
    return execute(cmd)


def model_converter_txt(folder, colmap_executable):
    mapper_input_path = os.path.join(folder, "sparse")
    os.makedirs(mapper_input_path, exist_ok=True)
    cmd = [
        colmap_executable, "model_converter",
        "--input_path", mapper_input_path,
        "--output_path", mapper_input_path,
        "--output_type=TXT",
    ]
    return execute(cmd)


def model_converter_bin(folder, colmap_executable):
    mapper_input_path = os.path.join(folder, "sparse")
    os.makedirs(mapper_input_path, exist_ok=True)
    cmd = [
        colmap_executable, "model_converter",
        "--input_path", mapper_input_path,
        "--output_path", mapper_input_path,
        "--output_type=BIN",
    ]
    return execute(cmd)


if __name__ == "__main__":
    args = parser.parse_args()
    camera_meta = read_camera_meta(os.path.join(args.path, "train_meta.json"))
    for frame in tqdm(range(args.n_frames), desc="Linking frames"):
        folder = os.path.join(args.path, "frame%d" % (frame + 1))
        os.makedirs(os.path.join(folder, "images"), exist_ok=True)
        width, height = camera_meta["w"], camera_meta["h"]
        cameras, images = [], []
        for i, (k, w2c, fn) in enumerate(zip(camera_meta["k"][frame], camera_meta["w2c"][frame], camera_meta["fn"][frame])):
            cam_id = i + 1

            fx, fy = k[0][0], k[1][1]
            cx, cy = k[0][2], k[1][2]
            cameras.append(f"{cam_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}")

            R, T = torch.tensor(w2c)[:3, :3], torch.tensor(w2c)[:3, 3]
            q, t = matrix_to_quaternion(R), T
            img_name = f"cam%02d{os.path.splitext(fn)[1]}" % (i+1)
            img_src = os.path.join(args.path, "ims", fn)
            img_dst = os.path.join(folder, "images", img_name)
            images.append(f"{cam_id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {cam_id} {img_name}")
            if os.path.isfile(img_dst):
                os.remove(img_dst)
            os.link(img_src, img_dst)

        mapper_input_path = os.path.join(folder, "sparse", "loading")
        os.makedirs(mapper_input_path, exist_ok=True)
        with open(os.path.join(mapper_input_path, "cameras.txt"), "w") as f:
            f.write("\n".join(cameras))
        with open(os.path.join(mapper_input_path, "images.txt"), "w") as f:
            f.write("\n\n".join(images))
        open(os.path.join(mapper_input_path, "points3D.txt"), "w").close()

        args.colmap_executable = os.path.abspath(args.colmap_executable)
        if feature_extractor(args, folder) != 0:
            raise RuntimeError("Feature extraction failed")
        if exhaustive_matcher(args, folder) != 0:
            raise RuntimeError("Feature matching failed")
        if point_triangulator(args, folder, mapper_input_path) != 0:
            raise RuntimeError("Triangulation failed")
        if mapper(args, folder, mapper_input_path) != 0:
            raise RuntimeError("Mapping failed")

        # Fixed: wrong number of cameras in images.txt
        if model_converter_txt(folder, args.colmap_executable) != 0:
            raise RuntimeError("Model conversion failed")
        os.remove(os.path.join(folder, "sparse", "cameras.bin"))
        os.remove(os.path.join(folder, "sparse", "images.bin"))
        os.remove(os.path.join(folder, "sparse", "points3D.bin"))
        if model_converter_bin(folder, args.colmap_executable) != 0:
            raise RuntimeError("Model conversion failed")

        # To fit sparse init in instantsplat
        distorted_folder = os.path.join(folder, "distorted")
        if os.path.isdir(distorted_folder):
            shutil.rmtree(distorted_folder)
        distorted_sparse_folder = os.path.join(distorted_folder, "sparse", "0")
        os.makedirs(distorted_sparse_folder, exist_ok=True)
        os.link(os.path.join(folder, "sparse", "cameras.bin"), os.path.join(distorted_sparse_folder, "cameras.bin"))
        os.link(os.path.join(folder, "sparse", "images.bin"), os.path.join(distorted_sparse_folder, "images.bin"))
        os.link(os.path.join(folder, "sparse", "points3D.bin"), os.path.join(distorted_sparse_folder, "points3D.bin"))
        shutil.copy(os.path.join(folder, "database.db"), os.path.join(folder, "distorted", "database.db"))
        input_folder = os.path.join(folder, "input")
        if os.path.isdir(input_folder):
            shutil.rmtree(input_folder)
        os.makedirs(input_folder, exist_ok=True)
        for entry in os.scandir(os.path.join(folder, "images")):
            os.link(entry.path, os.path.join(input_folder, entry.name))
