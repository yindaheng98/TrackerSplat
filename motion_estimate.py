import torch
from instantsplatstream.dataset import FixedViewColmapVideoCameraDataset, FixedViewColmapVideoCameraDataset_from_json
from instantsplatstream.motionestimator import FixedViewMotionEstimator
from instantsplatstream.motionestimator.point_tracker import MotionFuser, Cotracker3DotMotionEstimator, Cotracker3MotionEstimator


class TestMotionFuser(MotionFuser):

    def to(self, device: torch.device) -> 'TestMotionFuser':
        self.device = device
        return self

    def __call__(self, trackviews):
        raise NotImplementedError

    def update_baseframe(self, frame) -> 'TestMotionFuser':
        return self


device = torch.device("cuda")
dataset = FixedViewColmapVideoCameraDataset_from_json("data/coffee_martini", jsonpath="output/coffee_martini/frame1/cameras.json", device=device)
# batch_func = Cotracker3DotMotionEstimator(fuser=TestMotionFuser(), device=device, rescale_factor=0.25)
batch_func = Cotracker3MotionEstimator(fuser=TestMotionFuser(), device=device, rescale_factor=0.25)
motion_estimator = FixedViewMotionEstimator(dataset, batch_func, batch_size=8, device=device)
for motion in motion_estimator:
    print(motion)
