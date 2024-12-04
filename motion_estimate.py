import torch
from instantsplatstream.dataset import ColmapVideoCameraDataset
from instantsplatstream.motionestimator import FixedViewMotionEstimator
from instantsplatstream.motionestimator.point_tracker import Cotracker3DotMotionEstimator, MotionFuser


class TestMotionFuser(MotionFuser):

    def to(self, device: torch.device) -> 'TestMotionFuser':
        self.device = device
        return self

    def __call__(self, trackviews):
        raise NotImplementedError

    def update_baseframe(self, frame) -> 'TestMotionFuser':
        return self


device = torch.device("cuda")
dataset = ColmapVideoCameraDataset("data/coffee_martini", device=device)
batch_func = Cotracker3DotMotionEstimator(fuser=TestMotionFuser(), height=496, width=664, device=device)
motion_estimator = FixedViewMotionEstimator(dataset, batch_func, batch_size=8, device=device)
for motion in motion_estimator:
    print(motion)
