import os
import re
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="path to the video folder")
parser.add_argument("--max_interval", type=int, default=100, help="maximum interval between timestamps")


def videos_in_dir(root):
    """
    List all video files in the given directory.
    """
    videos = []
    for entry in os.scandir(root):
        if not os.path.exists(os.path.join(root, entry.name, "color.mp4")):
            continue
        if not os.path.exists(os.path.join(root, entry.name, "timestamps.npy")):
            continue
        if not re.match(r"cam_[a-z0-9]+", entry.name):
            continue
        videos.append(entry.name[4:])
    return videos


def frame_timestamps(videos):
    """
    Extract and align frame timestamps from multiple camera videos in a given directory.
    """
    timestamps_videos = {}
    for video in videos:
        timestamps_videos[video] = np.load(os.path.join(root, f"cam_{video}", "timestamps.npy"), allow_pickle=True).item()['color']
    timestamps_name_by_length = sorted(timestamps_videos.keys(), key=lambda x: len(timestamps_videos[x]), reverse=True)
    anchor_timestamps_name = timestamps_name_by_length[0]
    frames = []
    for anchor_timestamp in timestamps_videos[anchor_timestamps_name]:
        frame = [anchor_timestamp]
        for timestamps_name in timestamps_name_by_length[1:]:
            frame_timestamp, max_interval = None, args.max_interval
            timestamps = timestamps_videos[timestamps_name]
            for timestamp in timestamps:
                if abs(timestamp - anchor_timestamp) < max_interval:
                    frame_timestamp = timestamp
                    max_interval = abs(timestamp - anchor_timestamp)
            frame.append(frame_timestamp)
        frames.append(frame)
    return frames


def combination_idx(frame):
    idx = 0
    for i, f in enumerate(frame):
        if f is not None:
            idx += 1 << i
    return idx


def filter_frames(frames):
    """
    Filter frames based on the number of valid timestamps.
    """
    frame_valid_count = [len([f for f in frame if f is not None]) for frame in frames]
    valid_threshold = np.bincount(frame_valid_count).argmax()
    filtered_frames = []
    tmp, last_ci = [], combination_idx(frames[0])
    for valid_count, frame in zip(frame_valid_count, frames):
        if valid_count >= valid_threshold and combination_idx(frame) == last_ci:
            tmp.append(frame)
        else:
            if len(tmp) > len(filtered_frames):
                filtered_frames = tmp
            tmp = [frame]
        last_ci = combination_idx(frame)
    if len(tmp) > len(filtered_frames):
        filtered_frames = tmp
    tmp = []
    return filtered_frames


if __name__ == "__main__":
    args = parser.parse_args()
    root = args.path
    videos = videos_in_dir(root)
    frames = filter_frames(frame_timestamps(videos))
    print(frames)
