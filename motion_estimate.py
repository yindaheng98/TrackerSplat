import torch
from instantsplatstream.dataset import ColmapVideoCameraDataset
from instantsplatstream.motionestimator import FixedViewBatchMotionEstimator
from instantsplatstream.motionestimator.point_tracker import Cotracker3DotMotionEstimationFunc, MotionFuser


class TestMotionFuser(MotionFuser):

    def to(self, device: torch.device) -> 'TestMotionFuser':
        self.device = device
        return self

    def __call__(self, trackviews):
        raise NotImplementedError


device = torch.device("cuda")
dataset = ColmapVideoCameraDataset("data/coffee_martini", device=device)
batch_func = Cotracker3DotMotionEstimationFunc(fuser=TestMotionFuser(), height=496, width=664, device=device)
motion_estimator = FixedViewBatchMotionEstimator(dataset, batch_func, batch_size=8, device=device)
for motion in motion_estimator:
    print(motion)
