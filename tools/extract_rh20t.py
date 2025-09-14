import os
import re
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="path to the video folder")
parser.add_argument("--max_interval", type=int, default=100, help="maximum interval between timestamps")

def frame_timestamps(root):
    """
    Extract and align frame timestamps from multiple camera videos in a given directory.
    """
    timestamps_videos = {}
    for entry in os.scandir(root):
        if not os.path.exists(os.path.join(root, entry.name, "color.mp4")):
            continue
        if not os.path.exists(os.path.join(root, entry.name, "timestamps.npy")):
            continue
        if not re.match(r"cam_[a-z0-9]+", entry.name):
            continue
        timestamps_name = entry.name[4:]
        timestamps_videos[timestamps_name] = np.load(os.path.join(root, entry.name, "timestamps.npy"), allow_pickle=True).item()['color']
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

if __name__ == "__main__":
    args = parser.parse_args()
    root = args.path
    frames = frame_timestamps(root)
    valid_count = np.bincount([len([f for f in frame if f is not None]) for frame in frames])
    valid_threshold = valid_count.argmax()
    print(valid_count, valid_threshold)
