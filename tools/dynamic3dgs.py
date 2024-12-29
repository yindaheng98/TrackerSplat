import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="path to the video folder")
parser.add_argument("--n_frames", type=int, default=150, help="number of frames")

if __name__ == "__main__":
    args = parser.parse_args()
    for frame in tqdm(range(args.n_frames), desc="Linking frames"):
        os.makedirs(os.path.join(args.path + "frame%d/images" % (frame + 1)), exist_ok=True)
        for camera in range(31):
            img_src = os.path.join(args.path, "ims/%d/%06d.jpg" % (camera, frame))
            img_dst = os.path.join(args.path, "frame%d/images/cam%02d.jpg" % (frame + 1, camera))
            if os.path.isfile(img_dst):
                os.remove(img_dst)
            os.makedirs(os.path.dirname(img_dst), exist_ok=True)
            os.link(img_src, img_dst)
