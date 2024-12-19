import torch
import tqdm
import math
from ._C import KNN


def knn(points: torch.Tensor, k=8):
    """Get k neighbor points for each point."""
    best, indices = KNN(points, k)
    return indices, best.sqrt()
