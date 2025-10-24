import itertools
from typing import List
import torch
import torch.multiprocessing as mp
import os
from os import makedirs
from itertools import islice
from functools import partial
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import AbstractTrainer
from trackersplat.dataset import prepare_fixedview_dataset, VideoCameraDataset
from trackersplat.motionestimator import FixedViewMotionEstimator, MotionCompensater
from trackersplat.motionestimator.point_tracker import BaseMotionFuser, build_point_track_batch_motion_estimator, DataParallelPointTrackMotionEstimator
from trackersplat.motionestimator.compensater import BaseMotionCompensater, build_motion_compensater
from trackersplat.motionestimator.incremental_trainer import IncrementalTrainingMotionEstimator, Incremental1StepTrainingMotionEstimator, IncrementalTrainingRefiner, build_trainer_factory, TrainingProcess, BaseTrainingProcess
from trackersplat.motionestimation import save_cfg_args, prepare_gaussians


class TimingDataParallelPointTrackMotionEstimator(DataParallelPointTrackMotionEstimator):
    log_path: str

    def __call__(self, views):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        motions = super().__call__(views)
        end.record()
        torch.cuda.synchronize()
        print("tracking timing: ", start.elapsed_time(end))
        with open(self.log_path, "a") as f:
            f.write(f"tracking timing: {start.elapsed_time(end)}\n")
        return motions


class TimingTrainingProcess(BaseTrainingProcess):
    log_path: str

    def __call__(self, dataset: CameraDataset, trainer: AbstractTrainer, iteration: int, frame_idx: int):
        '''Overload this method to make your own training'''
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        super().__call__(dataset, trainer, iteration, frame_idx)
        end.record()
        torch.cuda.synchronize()
        print("training timing: ", start.elapsed_time(end))
        with open(self.log_path, "a") as f:
            f.write(f"training timing: {start.elapsed_time(end)}\n")


estimator_choices = ["dot", "dot-tapir", "dot-bootstapir", "dot-cotracker3", "cotracker3"]
compensater_choices = ["base", "propagate", "filter"]
trainer_choices = ["base", "regularized", "masked", "maskregularized", "hexplane", "regularizedhexplane", "hicom"]

train_choices = [traintype + "/" + trainer for traintype, trainer in itertools.product(["train", "train1step"], trainer_choices)]
refine_choices = ["refine/" + trainer + "-" + compensater + "-" + estimator for trainer, compensater, estimator in itertools.product(trainer_choices, compensater_choices, estimator_choices)]
track_choices = ["track/" + compensater + "-" + estimator for compensater, estimator in itertools.product(compensater_choices, estimator_choices)]

pipeline_choices = train_choices + refine_choices + track_choices


def point_track_builder(estimator: str, gaussians: GaussianModel, **kwargs):
    return build_point_track_batch_motion_estimator(estimator=estimator, fuser=BaseMotionFuser(gaussians.to("cuda")), device="cuda", **kwargs)


def incremental_trainer_builder(builder, trainer: str, iteration: int, **kwargs):
    return builder(trainer_factory=build_trainer_factory(trainer, **kwargs), iteration=iteration, device="cuda")


def build_pipeline(pipeline: str, gaussians: GaussianModel, dataset: VideoCameraDataset, log_path: str, device: torch.device, device_ids: List[torch.device], batch_size: int, iteration: int, **kwargs) -> MotionCompensater:
    training_proc = TimingTrainingProcess()
    training_proc.log_path = log_path
    batch_func_return = None
    mode, estimator = pipeline.split("/", 1)
    if mode[:5] == "train":
        trainer = estimator
        if mode[5:] == "1step":
            batch_func = Incremental1StepTrainingMotionEstimator(trainer_factory=build_trainer_factory(trainer, **kwargs), training_proc=training_proc, iteration=iteration, device=device)
        else:
            batch_func = IncrementalTrainingMotionEstimator(trainer_factory=build_trainer_factory(trainer, **kwargs), training_proc=training_proc, iteration=iteration, device=device)
        motion_estimator = FixedViewMotionEstimator(dataset=dataset, batch_func=batch_func, device=device, batch_size=batch_size)
        motion_compensater = BaseMotionCompensater(gaussians=gaussians, estimator=motion_estimator, device=device)
    elif mode == "track":
        compensater, estimator = estimator.split("-", 1)
        batch_func = TimingDataParallelPointTrackMotionEstimator(
            build_estimator=point_track_builder, build_estimator_kwargs=dict(estimator=estimator, gaussians=gaussians, **kwargs),
            base_gaussians=gaussians,
            master_device=device, slave_device_ids=device_ids)
        batch_func.log_path = log_path
        batch_func_return = batch_func
        motion_estimator = FixedViewMotionEstimator(dataset=dataset, batch_func=batch_func, device=device, batch_size=batch_size)
        motion_compensater = build_motion_compensater(compensater=compensater, gaussians=gaussians, estimator=motion_estimator, device=device)
    elif mode == "refine":
        trainer, compensater, estimator = estimator.split("-", 2)
        batch_func = TimingDataParallelPointTrackMotionEstimator(
            build_estimator=point_track_builder, build_estimator_kwargs=dict(estimator=estimator, gaussians=gaussians, **kwargs),
            base_gaussians=gaussians,
            master_device=device, slave_device_ids=device_ids)
        batch_func.log_path = log_path
        batch_func_return = batch_func
        motion_estimator = FixedViewMotionEstimator(dataset=dataset, batch_func=batch_func, device=device, batch_size=batch_size)
        motion_compensater = build_motion_compensater(compensater=compensater, gaussians=gaussians, estimator=motion_estimator, device=device)
        batch_func = IncrementalTrainingRefiner(base_batch_func=batch_func, base_compensater=motion_compensater, trainer_factory=build_trainer_factory(trainer), training_proc=training_proc, iteration=iteration, device=device)
        motion_estimator = FixedViewMotionEstimator(dataset=dataset, batch_func=batch_func, device=device, batch_size=batch_size)
        motion_compensater = BaseMotionCompensater(gaussians=gaussians, estimator=motion_estimator, device=device)
    else:
        ValueError(f"Unknown estimator: {estimator}")
    return batch_func_return, motion_compensater


def motion_compensate(batch_func: TimingDataParallelPointTrackMotionEstimator, motion_compensater: MotionCompensater, dataset: VideoCameraDataset, save_frame_cfg_args, iteration: int, start_frame: int, n_frames: int, save_frames: bool):
    if batch_func is not None:
        batch_func.start()
    for i, frame_gaussians in enumerate(islice(motion_compensater, n_frames)):
        print(f"Frame {start_frame + i + 1}")
        if not save_frames:
            continue
        destination_folder = save_frame_cfg_args(frame=start_frame + i + 1)
        save_path = os.path.join(destination_folder, "point_cloud", "iteration_" + str(iteration))
        makedirs(save_path, exist_ok=True)
        frame_gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))
        dataset[i + 1].save_cameras(os.path.join(destination_folder, "cameras.json"))
    if batch_func is not None:
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

    parser.add_argument("--pipeline", choices=pipeline_choices, default="track/base-dot-cotracker3")
    parser.add_argument("--parallel_device", required=True, action='append', type=int)
    parser.add_argument("--iteration_init", required=True, type=str, help="iteration of the initial gaussians")
    parser.add_argument("-f", "--frame_folder_fmt", default="frame%d", type=str, help="frame folder format string")
    parser.add_argument("-n", "--n_frames", default=None, type=int, help="number of frames to process")
    parser.add_argument("-b", "--batch_size", default=3, type=int, help="batch size of point tracking")
    parser.add_argument("--start_frame", default=1, type=int, help="start from which frame")
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    parser.add_argument("--save_frames", action='store_true')
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

    frame_str = (args.frame_folder_fmt % args.start_frame) + f"plus{args.batch_size - 1}to{args.n_frames + 1}"
    log_path = os.path.join(os.path.join(args.destination, args.pipeline), "timelog", frame_str + ".txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        pass
    batch_func, motion_compensater = build_pipeline(args.pipeline, gaussians, dataset, log_path, args.device, args.parallel_device, args.batch_size, args.iteration, **configs)
    motion_compensate(batch_func, motion_compensater, dataset, save_frame_cfg_args, args.iteration, args.start_frame, args.n_frames, args.save_frames)
