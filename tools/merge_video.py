import os
import sys
import cv2
import numpy as np

src_root = sys.argv[1]
videoset = sys.argv[2]
dataset = sys.argv[3]
n_frames = int(sys.argv[4])
dst_root = sys.argv[5]

# 输入视频路径列表
videosetname = f"\"{videoset}\" in {dataset} dataset"
video_paths = [
    'refine/base-propagate-dot-cotracker3.mp4',
    'refine/base-base-dot-cotracker3.mp4',
    'train/hicom.mp4',
    'train/regularized.mp4',
    'train/hexplane.mp4',
    'train/regularizedhexplane.mp4',
]
video_names = [
    'ours',
    'ours (w/o reg)',
    'Parallel HiCoM',
    'Parallel Dynamic3DGS',
    'Parallel 4DGS',
    'Parallel ST-4DGS',
]

# 打开所有视频
caps = [cv2.VideoCapture(os.path.join(src_root, videoset, p)) for p in video_paths]
frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

# 输出视频路径
os.makedirs(os.path.join(dst_root, videoset), exist_ok=True)


for i in range(n_frames):
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        frames.append(frame)
    for j in range(len(frames)):
        video_name = video_names[j]
        if frames[j] is None:
            frames[j] = np.zeros_like(frames[0])
            video_name += '  training failed'
        # 在左上角标上名字
        cv2.putText(
            frames[j],
            video_name,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            4,
            cv2.LINE_AA
        )
    # 横向拼接
    row1 = np.hstack(frames[:3])
    row2 = np.hstack(frames[3:])
    merged_frame = np.vstack([row1, row2])
    title = np.zeros((140, merged_frame.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        title,
        videosetname,
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (0, 0, 255),
        4,
        cv2.LINE_AA
    )
    merged_frame = np.vstack([title, merged_frame])
    print("writing %s frame %d" % (videoset, i))
    cv2.imwrite(os.path.join(dst_root, videoset, "%05d.png" % i), merged_frame)
