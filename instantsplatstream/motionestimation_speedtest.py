import itertools
from typing import List
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
        estimator: str, gaussians: GaussianModel, estimator_kwargs,
        queue_in_sync: mp.Queue,
        queue_in: mp.Queue, queue_in_fuser: mp.Queue, queue_out: mp.Queue, *queues_out_fuser: List[mp.Queue]):
    base_estimator = build_point_track_batch_motion_estimator(estimator=estimator, fuser=BaseMotionFuser(gaussians.to("cuda")), device="cuda", **estimator_kwargs)
    tracker = base_estimator.tracker
    fuser = base_estimator.fuser
    while True:
        # start the process, get the number of views
        sync_msg = queue_in_sync.get()
        if sync_msg is None:
            break
        (n_views, n_frames, n_track_task, n_fuse_task) = sync_msg

        # stage 1: track each views
        for i in range(n_track_task):
            # stage 1.1: get and track
            view = queue_in.get()
            trackview = tracker(view)
            # stage 1.1+: validate and get number of frames
            view = trackview
            n, h, w, c = view.track.shape
            assert c == 2
            assert list(view.mask.shape) == [n, h, w]
            assert view.image_height == h and view.image_width == w
            assert n == n_frames
            # stage 1.2: split frames an send to fuser
            for queue_out_fuser, frame_idx in zip(itertools.cycle(queues_out_fuser), range(n)):
                queue_out_fuser.put((frame_idx, view._replace(track=view.track[i:i+1].cpu(), mask=view.mask[i:i+1].cpu())))

        # stage 2: fuse each frames
        trackviews = {}
        for i in range(n_fuse_task * n_views):
            # stage 1.1: gather views
            frame_idx, view = queue_in_fuser.get()
            if frame_idx not in trackviews:
                trackviews[frame_idx] = []
            trackviews[frame_idx].append(view._replace(track=view.track.to('cuda'), mask=view.mask.to('cuda')))
            # stage 1.2: fuse if all views are gathered
            if len(trackviews[frame_idx]) == n_views:
                motions = fuser(trackviews[frame_idx])
                queue_out.put((frame_idx, motions[0]))


def start_parallel_worker(
        estimator: str, gaussians: GaussianModel, estimator_kwargs, device_ids: List[int],
        queues_in_sync: List[mp.Queue],
        queues_in: List[mp.Queue], queues_out: List[mp.Queue], queues_fuser: List[mp.Queue]):
    processes = []
    for queue_in_sync, queue_in, queue_in_fuser, queue_out, device_id in zip(queues_in_sync, queues_in, queues_fuser, queues_out, device_ids):
        os.environ["LOCAL_DEVICE_ID"] = str(device_id)
        process = mp.Process(
            target=parallel_worker, args=(
                estimator, gaussians, estimator_kwargs,
                queue_in_sync,
                queue_in, queue_in_fuser, queue_out, *queues_fuser))
        process.start()
        processes.append(process)
    del os.environ["LOCAL_DEVICE_ID"]
    return processes


def task_io(views: List[FixedViewFrameSequenceMeta], queues_in_sync: List[mp.Queue], queues_in: List[mp.Queue], queues_out: List[mp.Queue]):
    assert len(queues_in_sync) == len(queues_in)

    # stage 1: count tasks
    n_track_task, n_fuse_task = [0]*len(queues_in), [0]*len(queues_in)
    # stage 1.1: count track tasks
    n_views = len(views)
    for proc_idx, view_idx in zip(itertools.cycle(range(len(queues_in))), range(n_views)):
        n_track_task[proc_idx] += 1
    # stage 1.2: count fuse tasks
    n_frames = len(views[0].frames_path) - 1
    for proc_idx, frame_idx in zip(itertools.cycle(range(len(queues_in))), range(n_frames)):
        n_fuse_task[proc_idx] += 1

    # stage 2: inject sync messages
    for queue_in_sync, ntt, nft in zip(queues_in_sync, n_track_task, n_fuse_task):
        queue_in_sync.put((n_views, n_frames, ntt, nft))

    # stage 2: inject tasks
    for queue_in, view in zip(itertools.cycle(queues_in), views):
        queue_in.put(view)

    motions = [None] * n_frames
    for queue_out, frame_idx in zip(itertools.cycle(queues_out), range(n_frames)):
        frame_idx, motion = queue_out.get()
        motions[frame_idx] = motion
    return motions


def join_parallel_worker(processes: List[mp.Process], queues_in_sync: List[mp.Queue]):
    for queue_in in queues_in_sync:
        queue_in.put(None)
    for process in processes:
        process.join()


class DataParallelPointTrackMotionEstimator(FixedViewBatchMotionEstimator):
    def __init__(self, estimator: str, gaussians: GaussianModel, master_device='cuda', slave_device_ids=[0], max_size=100, **estimator_kwargs):
        self.estimator = estimator
        self.gaussians = gaussians
        self.estimator_kwargs = estimator_kwargs
        self.device = master_device
        self.device_ids = slave_device_ids

        self.queues_in_sync = [mp.Queue(1) for _ in self.device_ids]
        self.queues_in = [mp.Queue(max_size) for _ in self.device_ids]
        self.queues_out = [mp.Queue(max_size) for _ in self.device_ids]

        self.manager = mp.Manager()
        self.queues_fuser = [self.manager.Queue(max_size) for _ in self.device_ids]

        self.processes = None

    def start(self):
        self.processes = start_parallel_worker(
            estimator=self.estimator, gaussians=self.gaussians, estimator_kwargs=self.estimator_kwargs, device_ids=self.device_ids,
            queues_in_sync=self.queues_in_sync,
            queues_in=self.queues_in, queues_out=self.queues_out, queues_fuser=self.queues_fuser)

    def join(self):
        self.processes = join_parallel_worker(self.processes, self.queues_in_sync)

    def to(self, device: torch.device) -> 'DataParallelPointTrackMotionEstimator':
        self.device = device
        return self

    def __call__(self, views: List[FixedViewFrameSequenceMeta]) -> List[Motion]:
        motions = task_io(views, self.queues_in_sync, self.queues_in, self.queues_out)
        motions = [motion.to(self.device) for motion in motions]
        return motions

    def update_baseframe(self, frame: GaussianModel) -> 'PointTrackMotionEstimator':
        return self


estimator_choices = ["dot", "dot-tapir", "dot-bootstapir", "dot-cotracker3", "cotracker3"]
compensater_choices = ["base", "propagate", "filter"]

pipeline_choices = [compensater + "-" + estimator for compensater, estimator in itertools.product(compensater_choices, estimator_choices)]


def motion_compensate(estimator: str, gaussians: GaussianModel, dataset: VideoCameraDataset, n_frames: int, device: torch.device, device_ids: List[torch.device], batch_size: int, **kwargs) -> MotionCompensater:
    compensater, estimator = estimator.split("-", 1)
    batch_func = DataParallelPointTrackMotionEstimator(estimator=estimator, gaussians=gaussians, master_device=device, slave_device_ids=device_ids, **kwargs)
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
