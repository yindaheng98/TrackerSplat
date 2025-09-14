import os
import re
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="path to the video folder")
parser.add_argument("--max_interval", type=int, default=100, help="maximum interval between timestamps")

if __name__ == "__main__":
    args = parser.parse_args()
    root = args.path
    high_freq_data = np.load(os.path.join(root, "transformed/high_freq_data.npy"), allow_pickle=True).item()
    high_freq_base = [i['timestamp'] for i in high_freq_data['base']]

    high_freq_timestamps_videos, low_freq_timestamps_videos = {}, {}
    for entry in os.scandir(root):
        if not os.path.exists(os.path.join(root, entry.name, "color.mp4")):
            continue
        if not os.path.exists(os.path.join(root, entry.name, "timestamps.npy")):
            continue
        if not re.match(r"cam_[a-z0-9]+", entry.name):
            continue
        timestamps_name = entry.name[4:]
        low_freq_timestamps_videos[timestamps_name] = np.load(os.path.join(root, entry.name, "timestamps.npy"), allow_pickle=True).item()['color']
        high_freq_timestamps_videos[timestamps_name] = [i['timestamp'] for i in high_freq_data[timestamps_name]]
    timestamps_name_by_length = sorted(low_freq_timestamps_videos.keys(), key=lambda x: len(low_freq_timestamps_videos[x]), reverse=True)
    anchor_timestamps_name = timestamps_name_by_length[0]
    for anchor_timestamp in low_freq_timestamps_videos[anchor_timestamps_name]:
        frame = [anchor_timestamp]
        for timestamps_name in timestamps_name_by_length[1:]:
            frame_timestamp, max_interval = None, args.max_interval
            timestamps = low_freq_timestamps_videos[timestamps_name]
            for timestamp in timestamps:
                if abs(timestamp - anchor_timestamp) < max_interval:
                    frame_timestamp = timestamp
                    max_interval = abs(timestamp - anchor_timestamp)
            frame.append(frame_timestamp)
        print(frame)
