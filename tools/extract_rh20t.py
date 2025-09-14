import os
import re
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="path to the video folder")
parser.add_argument("--fmt", type=str, default=r"cam_[a-z0-9]+", help="re format of mp4 file")
parser.add_argument("--max_interval", type=int, default=30, help="maximum interval between timestamps")

if __name__ == "__main__":
    args = parser.parse_args()
    root = args.path

    timestamps_videos = {}
    for entry in os.scandir(root):
        if not re.match(args.fmt, entry.name):
            continue
        if not os.path.exists(os.path.join(root, entry.name, "color.mp4")):
            continue
        if not os.path.exists(os.path.join(root, entry.name, "timestamps.npy")):
            continue
        timestamps_videos[entry.name] = np.load(os.path.join(root, entry.name, "timestamps.npy"), allow_pickle=True).item()['color']
    timestamps_name_by_length = sorted(timestamps_videos.keys(), key=lambda x: len(timestamps_videos[x]), reverse=True)
    anchor_timestamps_name = timestamps_name_by_length[0]
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
        print(frame)
