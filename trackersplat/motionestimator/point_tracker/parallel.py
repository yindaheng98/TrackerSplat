import itertools
from typing import Any, Callable, Dict, List
import torch
import torch.multiprocessing as mp
import os
from gaussian_splatting import GaussianModel
from trackersplat import Motion
from trackersplat.motionestimator import FixedViewBatchMotionEstimator, FixedViewFrameSequenceMeta
from .abc import PointTrackMotionEstimator


torch.cuda.set_device(int(os.environ.get("LOCAL_DEVICE_ID", "0")))


def parallel_worker(
        build_estimator: Callable[..., PointTrackMotionEstimator], build_estimator_kwargs: Dict[str, Any],
        base_gaussians: GaussianModel,
        queue_out_init_done: mp.Queue, queue_in_sync: mp.Queue,
        queue_in: mp.Queue, queue_in_fuser: mp.Queue, queue_out: mp.Queue, *queues_out_fuser: List[mp.Queue]):
    base_estimator = build_estimator(**build_estimator_kwargs)
    tracker = base_estimator.tracker
    fuser = base_estimator.fuser
    fuser.update_baseframe(base_gaussians)
    queue_out_init_done.put(True)
    while True:
        # start the process, get the number of views
        sync_msg = queue_in_sync.get()
        if sync_msg is None:
            break
        (n_views, n_frames, n_track_task, n_fuse_task) = sync_msg

        # stage 1: track each views
        for _ in range(n_track_task):
            # stage 1.1: get and track
            view = queue_in.get()
            trackview = tracker(view)
            # stage 1.1+: validate and get number of frames
            view = trackview
            n, h, w, c = view.track.shape
            assert c == 2
            assert list(view.visibility.shape) == [n, h, w]
            assert view.track_height == h and view.track_width == w
            assert n == n_frames
            # stage 1.2: split frames an send to fuser
            for queue_out_fuser, frame_idx in zip(itertools.cycle(queues_out_fuser), range(n)):
                queue_out_fuser.put((frame_idx, view._replace(track=view.track[frame_idx:frame_idx+1].cpu(), visibility=view.visibility[frame_idx:frame_idx+1].cpu())))

        # stage 2: fuse each frames
        trackviews = {}
        for i in range(n_fuse_task * n_views):
            # stage 1.1: gather views
            frame_idx, view = queue_in_fuser.get()
            if frame_idx not in trackviews:
                trackviews[frame_idx] = []
            trackviews[frame_idx].append(view._replace(track=view.track.to('cuda'), visibility=view.visibility.to('cuda')))
            # stage 1.2: fuse if all views are gathered
            if len(trackviews[frame_idx]) == n_views:
                motions = fuser(trackviews[frame_idx])
                queue_out.put((frame_idx, motions[0]))


def start_parallel_worker(
        build_estimator: Callable[..., PointTrackMotionEstimator], build_estimator_kwargs: Dict[str, Any],
        base_gaussians: GaussianModel,
        device_ids: List[int],
        queues_in_sync: List[mp.Queue],
        queues_in: List[mp.Queue], queues_out: List[mp.Queue], queues_fuser: List[mp.Queue]):
    queues_out_init_done = [mp.Queue(1) for _ in device_ids]
    processes = []
    for queue_out_init_done, queue_in_sync, queue_in, queue_in_fuser, queue_out, device_id in zip(queues_out_init_done, queues_in_sync, queues_in, queues_fuser, queues_out, device_ids):
        os.environ["LOCAL_DEVICE_ID"] = str(device_id)
        process = mp.Process(
            target=parallel_worker, args=(
                build_estimator, build_estimator_kwargs,
                base_gaussians,
                queue_out_init_done, queue_in_sync,
                queue_in, queue_in_fuser, queue_out, *queues_fuser))
        process.start()
        processes.append(process)
    del os.environ["LOCAL_DEVICE_ID"]
    for queue_out_init_done in queues_out_init_done:
        queue_out_init_done.get()
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
        queue_in.put(view._replace(R=view.R.cpu(), T=view.T.cpu()))

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
    def __init__(
            self,
            build_estimator: Callable[..., PointTrackMotionEstimator], build_estimator_kwargs: Dict[str, Any],
            base_gaussians: GaussianModel,
            master_device='cuda', slave_device_ids=[0], max_size=0):
        self.build_estimator = build_estimator
        self.build_estimator_kwargs = build_estimator_kwargs
        self.base_gaussians = base_gaussians
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
            build_estimator=self.build_estimator, build_estimator_kwargs=self.build_estimator_kwargs,
            base_gaussians=self.base_gaussians,
            device_ids=self.device_ids,
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
        self.base_gaussians = frame
        return self
