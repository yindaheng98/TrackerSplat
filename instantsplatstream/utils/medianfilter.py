import torch
import taichi as ti
ti.init(arch=ti.gpu, kernel_profiler=True)


@ti.kernel
def motion_median_filter_kernel(
        points: ti.i32, knns: ti.i32,
        indices: ti.types.ndarray(ndim=2), mask: ti.types.ndarray(ndim=1), masked_indices: ti.types.ndarray(ndim=1),
        motion_src: ti.types.ndarray(ndim=2), motion_dst: ti.types.ndarray(ndim=2),
        sorting_space: ti.types.ndarray(ndim=3)):
    for motion_idx, channel in ti.ndrange(points, 3):
        total_knn: ti.i32 = 0
        for i_knn in range(knns):
            knn_idx = indices[motion_idx, i_knn]
            if mask[knn_idx]:
                knn_masked_idx = masked_indices[knn_idx]
                motion = motion_src[knn_masked_idx, channel]
                for i in range(total_knn):
                    if motion < sorting_space[motion_idx, i, channel]:
                        temp = sorting_space[motion_idx, i, channel]
                        sorting_space[motion_idx, i, channel] = motion
                        motion = temp
                sorting_space[motion_idx, total_knn, channel] = motion
                total_knn += 1
        motion_dst[motion_idx, channel] = sorting_space[motion_idx, total_knn // 2, channel]


def motion_median_filter(
        mask: torch.Tensor, motion: torch.Tensor,
        neighbor_indices: torch.Tensor, neighbor_weights: torch.Tensor):
    # replace the value with the median of the neighbors
    assert mask.dim() == 1
    assert motion.dim() == neighbor_indices.dim() == neighbor_weights.dim() == 2
    points, knns = neighbor_indices.size()
    assert mask.size(0) == neighbor_weights.size(0) == points
    valid_points = mask.sum().item()
    assert valid_points == motion.size(0)
    assert motion.size(1) == 3

    motions_median = motion.clone()
    sorting_space = motion.unsqueeze(1).repeat(1, knns, 1).clone()
    masked_neighbor_indices = torch.full((points,), -1, device=mask.device, dtype=torch.int32)
    masked_neighbor_indices[mask] = torch.arange(valid_points, device=mask.device, dtype=torch.int32)
    motion_median_filter_kernel(valid_points, knns, neighbor_indices[mask], mask, masked_neighbor_indices, motion, motions_median, sorting_space)
    return motions_median
