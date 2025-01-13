import itertools
from typing import Callable, List
import torch
import torch.multiprocessing as mp
import os
from itertools import islice
from functools import partial
from gaussian_splatting import GaussianModel
from instantsplatstream.dataset import prepare_fixedview_dataset, VideoCameraDataset
from instantsplatstream.motionestimator import Motion, FixedViewMotionEstimator, FixedViewBatchMotionEstimator, FixedViewFrameSequenceMeta, MotionCompensater
from instantsplatstream.motionestimator.point_tracker import BaseMotionFuser, PointTrackMotionEstimator, build_point_track_batch_motion_estimator
from instantsplatstream.motionestimator.compensater import build_motion_compensater
from instantsplatstream.motionestimation import prepare_gaussians, save_cfg_args


torch.cuda.set_device(int(os.environ.get("LOCAL_DEVICE_ID", "0")))


def parallel_worker(
        estimator: str, gaussians: GaussianModel, kwargs,
        queue_in: mp.Queue, queue_out_result: mp.Queue):
    base_estimator = build_point_track_batch_motion_estimator(estimator=estimator, fuser=BaseMotionFuser(gaussians.to("cuda")), device="cuda", **kwargs)
    tracker = base_estimator.tracker
    fuser = base_estimator.fuser
    while True:
        view = queue_in.get()
        if view is None:
            break
        track = tracker(view)
        queue_out_result.put(track)


class DataParallelPointTrackMotionEstimator(FixedViewBatchMotionEstimator):
    def __init__(self, estimator: str, gaussians: GaussianModel, device_ids=[0], max_size=10, **estimator_kwargs):
        self.estimator = estimator
        self.gaussians = gaussians
        self.estimator_kwargs = estimator_kwargs
        self.device_ids = device_ids

        self.queues_in = [mp.Queue(max_size) for _ in self.device_ids]
        self.queues_out = [mp.Queue(max_size) for _ in self.device_ids]
        self.processes = None

    def start(self):
        processes = []
        for queue_in, queue_out, device_id in zip(self.queues_in, self.queues_out, self.device_ids):
            os.environ["LOCAL_DEVICE_ID"] = str(device_id)
            process = mp.Process(
                target=parallel_worker, args=(
                    self.estimator, self.gaussians, self.estimator_kwargs,
                    queue_in, queue_out))
            process.start()
            processes.append(process)
        del os.environ["LOCAL_DEVICE_ID"]
        self.processes = processes

    def join(self):
        processes = self.processes
        for queue_in in self.queues_in:
            queue_in.put(None)
        for process in processes:
            process.join()
        self.processes = None

    def to(self, device: torch.device) -> 'DataParallelPointTrackMotionEstimator':
        return self

    def __call__(self, views: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        for queue_in, view in zip(itertools.cycle(self.queues_in), views):
            queue_in.put(view)
        tracks = []
        for queue_out, _ in zip(itertools.cycle(self.queues_out), views):
            track = queue_out.get()
            tracks.append(track)
        # TODO: put the track into the fuser
        trackviews = [tracker(view) for view, tracker in zip(views, itertools.cycle(self.trackers))]
        for view in trackviews:
            n, h, w, c = view.track.shape
            assert c == 2
            assert list(view.mask.shape) == [n, h, w]
            assert view.image_height == h and view.image_width == w
        device = self.fusers[0].device
        trackviews = [view._replace(track=view.track.to(device), mask=view.mask.to(device)) for view in trackviews]
        return self.fusers[0](trackviews)

    def update_baseframe(self, frame: GaussianModel) -> 'PointTrackMotionEstimator':
        return self


estimator_choices = ["dot", "dot-tapir", "dot-bootstapir", "dot-cotracker3", "cotracker3"]
compensater_choices = ["base", "propagate", "filter"]

pipeline_choices = [compensater + "-" + estimator for compensater, estimator in itertools.product(compensater_choices, estimator_choices)]


def motion_compensate(estimator: str, gaussians: GaussianModel, dataset: VideoCameraDataset, n_frames: int, device: torch.device, device_ids: List[torch.device], batch_size: int, **kwargs) -> MotionCompensater:
    compensater, estimator = estimator.split("-", 1)
    batch_func = DataParallelPointTrackMotionEstimator(estimator=estimator, gaussians=gaussians, device_ids=device_ids, **kwargs)
    batch_func.start()
    motion_estimator = FixedViewMotionEstimator(dataset=dataset, batch_func=batch_func, device=device, batch_size=batch_size)
    motion_compensater = build_motion_compensater(compensater=compensater, gaussians=gaussians, estimator=motion_estimator, device=device)
    for i, frame_gaussians in enumerate(islice(motion_compensater, n_frames)):
        print("frame", i)
    batch_func.join()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--pipeline", choices=pipeline_choices, default="base-dot-cotracker3")
    parser.add_argument("--parallel_device", required=True, action='append', type=int)
    parser.add_argument("--iteration_init", required=True, type=str, help="iteration of the initial gaussians")
    parser.add_argument("-f", "--frame_folder_fmt", default="frame%d", type=str, help="frame folder format string")
    parser.add_argument("-n", "--n_frames", default=None, type=int, help="number of frames to process")
    parser.add_argument("-b", "--batch_size", default=3, type=int, help="batch size of point tracking")
    parser.add_argument("--start_frame", default=1, type=int, help="start from which frame")
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    args = parser.parse_args()
    save_frame_cfg_args = partial(save_cfg_args, sh_degree=args.sh_degree, source=args.source, destination=os.path.join(args.destination, args.pipeline), frame_folder_fmt=args.frame_folder_fmt)
    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    load_ply = os.path.join(args.destination, args.frame_folder_fmt % args.start_frame, "point_cloud", "iteration_" + str(args.iteration_init), "point_cloud.ply")
    gaussians = prepare_gaussians(
        sh_degree=args.sh_degree, device=args.device,
        load_ply=load_ply)
    dataset = prepare_fixedview_dataset(
        source=args.source, device=args.device,
        frame_folder_fmt=args.frame_folder_fmt, start_frame=args.start_frame, n_frames=None,
        load_camera=args.load_camera)
    motion_compensater = motion_compensate(args.pipeline, gaussians, dataset, args.n_frames, args.device, args.parallel_device, args.batch_size, **configs)
