import torch
import tqdm
import math


def knn(points: torch.Tensor, k=8, batch=4096):
    """Get k neighbor points for each point."""
    neighbors_idx = torch.zeros(points.shape[0], k, dtype=torch.int32, device="cuda")
    progress_bar = tqdm.tqdm(range(points.shape[0]), desc="Init points center K nearest")
    dists = torch.zeros(points.shape[0], k, dtype=torch.float32, device="cuda")
    for i in range(math.ceil(points.shape[0]/batch)):
        dist = torch.norm(points[i*batch:i*batch+batch, ...].unsqueeze(-2) - points, p=2, dim=-1)
        knn = dist.topk(k + 1, largest=False)
        dists[i*batch:i*batch+batch, ...] = knn.values[:, 1:]
        neighbors_idx[i*batch:i*batch+batch, ...] = knn.indices[:, 1:]
        progress_bar.update(min(i*batch+batch, points.shape[0])-i*batch)
    return neighbors_idx, dists
