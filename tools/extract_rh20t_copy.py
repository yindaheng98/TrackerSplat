"""
Copy from: https://github.com/rh20t/rh20t_api/blob/aa3124434729ed622109a29b2cbb9f3bbb1c5eeb/rh20t_api/extract.py
Scripts and sample usages to convert RH20T to image version.
This script should be executed after unzipped the file if you want to use RH20T_api functions.
"""
import os
import cv2
import numpy as np


################################## Convert RH20T to image version ##################################

def convert_color(color_file, color_timestamps, dest_color_dir):
    """
    Args:
    - color_file: the color video file;
    - color_timestamps: the color timestamps;
    - dest_color_dir: the destination color directory.
    """
    cap = cv2.VideoCapture(color_file)
    cnt = 0
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(dest_color_dir, '{}.jpg'.format(color_timestamps[cnt])), frame)
            cnt += 1
        else:
            break
    cap.release()


def convert_dir(color_file, timestamps_file, dest_dir):
    """
    Args:
    - color_file: the color video file;
    - timestamps_file: the timestamps file;
    - dest_dir: the destination directory;
    - depth_file: the depth video file (special encoded), set to None if no depth usage;
    - size: the size of the depth map ( (640, 360) for resized version ).
    """
    assert os.path.exists(color_file)
    assert os.path.exists(timestamps_file)
    meta = np.load(timestamps_file, allow_pickle=True).item()
    dest_color_dir = os.path.join(dest_dir, 'color')
    if not os.path.exists(dest_color_dir):
        os.makedirs(dest_color_dir)
    convert_color(color_file=color_file, color_timestamps=meta['color'], dest_color_dir=dest_color_dir)


def convert_scene(scene_root_dir, dest_scene_dir, scene_depth_dir=None, size=(1280, 720)):
    """
    Args:
    - scene_root_dir: the root directory for the current scene;
    - dest_scene_dir: the destination root directory for the current scene (set to the same as scene_root_dir to extract into the original directory);
    - dest_dir: the destination scene directory;
    - depth_file: the depth video file (special encoded), set to None if no depth usage;
    - size: the size of the depth map ( (640, 360) for resized version ).
    """
    assert os.path.exists(scene_root_dir)
    for cam_folder in os.listdir(scene_root_dir):
        if "cam_" not in cam_folder:
            continue
        try:
            convert_dir(
                color_file=os.path.join(scene_root_dir, cam_folder, 'color.mp4'),
                timestamps_file=os.path.join(scene_root_dir, cam_folder, 'timestamps.npy'),
                dest_dir=os.path.join(dest_scene_dir, cam_folder),
            )
        except Exception as e:
            print(f"Error processing {cam_folder}: {e}")
            continue

################################## Sample Usage ##################################


if __name__ == '__main__':
    # 1. For full version (or ignore depth_dir if no depth usage)
    convert_scene(
        scene_root_dir="data/RH20T_cfg3/task_0001_user_0016_scene_0002_cfg_0003",
        dest_scene_dir="data/RH20T/task_0001_user_0016_scene_0002_cfg_0003"
    )
