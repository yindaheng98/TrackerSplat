import torch
import taichi as ti
ti.init(arch=ti.gpu, kernel_profiler=True)


@ti.kernel
def propagation_kernel(
        points: ti.i32, knns: ti.i32, channels: ti.i32,
        indices: ti.types.ndarray(ndim=2), decays: ti.types.ndarray(ndim=2),
        values_src: ti.types.ndarray(ndim=2), values_dst: ti.types.ndarray(ndim=2),
        weights_src: ti.types.ndarray(ndim=1), weights_dst: ti.types.ndarray(ndim=1),
        finish_mask_src: ti.types.ndarray(ndim=1), finish_mask_dst: ti.types.ndarray(ndim=1),
        counts: ti.types.ndarray(ndim=1)):
    for i_src, i_knn in ti.ndrange(points, knns):
        i_dst: ti.i32 = indices[i_src, i_knn]
        if finish_mask_src[i_src] and not finish_mask_src[i_dst]:
            weights_dst[i_dst] += weights_src[i_src] * decays[i_src, i_knn]
            for channel in range(channels):
                values_dst[i_dst, channel] += values_src[i_src, channel] * weights_src[i_src] * decays[i_src, i_knn]
            finish_mask_dst[i_dst] = True
            counts[i_dst] += 1


def propagate(
        init_mask: torch.Tensor, init_value_at_mask: torch.Tensor, init_weight_at_mask: torch.Tensor,
        neighbor_indices: torch.Tensor, neighbor_weights: torch.Tensor,
        n_iter=100):
    assert init_mask.dim() == init_weight_at_mask.dim() == 1
    assert init_value_at_mask.dim() == neighbor_indices.dim() == neighbor_weights.dim() == 2
    points, knns = neighbor_indices.size()
    channels = init_value_at_mask.size(1)
    assert init_mask.size(0) == neighbor_weights.size(0) == points

    init_value_at_mask = init_value_at_mask.type_as(neighbor_weights)
    init_weight_at_mask = init_weight_at_mask.type_as(neighbor_weights)

    finish_mask = init_mask.clone()  # used to tag the points that have been propagated (average up all the valid values and weights from its neighbors)
    weights = torch.zeros(points, device=init_mask.device, dtype=init_weight_at_mask.dtype)
    weights[init_mask] = init_weight_at_mask
    values = torch.zeros(points, channels, device=init_mask.device, dtype=init_value_at_mask.dtype)
    values[init_mask, :] = init_value_at_mask

    for _ in range(n_iter):
        values_sum = torch.zeros_like(values)
        weights_sum = torch.zeros_like(weights)
        finish_mask_dst = torch.zeros_like(finish_mask, dtype=torch.bool)
        counts = torch.zeros(points, dtype=torch.int32, device=init_mask.device)
        propagation_kernel((~finish_mask).sum().item(), knns, channels,
                           neighbor_indices, neighbor_weights,
                           values, values_sum,
                           weights, weights_sum,
                           finish_mask, finish_mask_dst,
                           counts)
        ti.sync()
        values[finish_mask_dst] = values_sum[finish_mask_dst] / weights_sum[finish_mask_dst].unsqueeze(-1)
        weights[finish_mask_dst] = weights_sum[finish_mask_dst] / counts[finish_mask_dst].to(dtype=weights.dtype)
        finish_mask |= finish_mask_dst
        if finish_mask_dst.sum() <= 0 or not torch.any(~finish_mask):
            break
    return values, weights
