import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exec", type=str, required=True, help="path to the ffmpeg")
parser.add_argument("--path", type=str, required=True, help="path to the video folder")
parser.add_argument("--n_frames", type=int, required=True, help="number of frames")
parser.add_argument("--fmt", type=str, default=r"cam_[a-z0-9]+[/]color.mp4", help="re format of mp4 file")
parser.add_argument("--maxt", type=int, default=2, help="max num of thread")

if __name__ == "__main__":
    args = parser.parse_args()
    root = args.path
    t = args.maxt
    names = []
    for top, dirnames, filenames in os.walk(root):
        for filename in filenames:
            names.append(os.path.relpath(os.path.join(top, filename), root).replace("\\", "/"))
    for name in names:
        if not re.match(args.fmt, name):
            continue
        cam = os.path.splitext(name)[0]
        imgs_dir = root + "/frame%d/input/" + re.sub(r"[\\\\/]", "_", cam) + ".png"
        for i in range(1, args.n_frames + 1):
            os.makedirs(root + "/frame%d/input" % i, exist_ok=True)
        cmd = f"{args.exec} -i {root + '/' + name} {imgs_dir} -y &"
        print(cmd)
        t -= 1
        if t <= 0:
            print("wait")
            t = args.maxt
    print("wait")
