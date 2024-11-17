import torch
from instantsplatstream.dataset import ColmapVideoCameraDataset
from instantsplatstream.motionestimator import FixedViewBatchMotionEstimator
from instantsplatstream.motionestimator.point_tracker import Cotracker3DotMotionEstimationFunc, FixedViewBatchTracks2Motion


class TestTrack2Motion(FixedViewBatchTracks2Motion):

    def to(self, device: torch.device) -> 'FixedViewBatchTracks2Motion':
        self.device = device
        return self

    def __call__(self, tracks):
        raise NotImplementedError


device = torch.device("cuda")
dataset = ColmapVideoCameraDataset("data/coffee_martini", device=device)
batch_func = Cotracker3DotMotionEstimationFunc(track2motion=TestTrack2Motion(), height=496, width=664, device=device)
motion_estimator = FixedViewBatchMotionEstimator(dataset, batch_func, batch_size=8, device=device)
for motion in motion_estimator:
    print(motion)
