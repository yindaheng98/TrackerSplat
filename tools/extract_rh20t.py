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
    videos_timestamps = {}
    for video in videos:
        videos_timestamps[video] = np.load(os.path.join(root, f"cam_{video}", "timestamps.npy"), allow_pickle=True).item()['color']
    anchor_video = sorted(videos_timestamps.keys(), key=lambda x: len(videos_timestamps[x]), reverse=True)[0]
    frames = []
    for anchor_timestamp in videos_timestamps[anchor_video]:
        frame = [anchor_timestamp]
        for video in videos:
            if video == anchor_video:
                continue
            frame_timestamp, max_interval = None, args.max_interval
            for timestamp in videos_timestamps[video]:
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


def build_frame_dst(videos, frames):
    frame_dst = {video: {} for video in videos}
    for i, frame in enumerate(frames):
        for video, timestamp in zip(videos, frame):
            if timestamp is not None:
                if timestamp not in frame_dst[video]:
                    frame_dst[video][timestamp] = []
                frame_dst[video][timestamp].append(i + 1)
    return frame_dst


def convert_color(root, video, frame_dst):
    color_file = os.path.join(root, f"cam_{video}", "color.mp4")
    color_timestamps = np.load(os.path.join(root, f"cam_{video}", "timestamps.npy"), allow_pickle=True).item()['color']
    cap = cv2.VideoCapture(color_file)
    cnt = 0
    while True:
        ret, frame = cap.read()
        if ret:
            color_timestamp = color_timestamps[cnt]
            if color_timestamp in frame_dst:
                for dst in frame_dst[color_timestamp]:
                    os.makedirs(os.path.join(root, f"frame{dst}"), exist_ok=True)
                    cv2.imwrite(os.path.join(root, f"frame{dst}", f'{video}.png'), frame)
            cnt += 1
        else:
            break
    cap.release()


if __name__ == "__main__":
    args = parser.parse_args()
    root = args.path
    videos = videos_in_dir(root)
    frames = frame_timestamps(videos)
    frames = filter_frames(frames)
    frame_dst = build_frame_dst(videos, frames)
    for video in videos:
        print(f"Converting video cam_{video} ...")
        convert_color(root, video, frame_dst[video])
    print("Done.")
