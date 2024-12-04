import torch
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def weightedsum(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    n_features = features.shape[1]
    colormap = torch.rand((n_features, 3), dtype=torch.float, device=features.device)
    linspace = torch.linspace(0, 1, steps=n_features, device=features.device)
    colormap = torch.stack([
        linspace,
        linspace[torch.randperm(n_features, device=features.device)],
        torch.flip(linspace, dims=(0,)),
    ]).T
    valid_idx = weights > 1e-5
    sum_weights = features[valid_idx, ...].sum(dim=1)
    sum_colors = (features[valid_idx, ...].unsqueeze(-1) * colormap.unsqueeze(0)).sum(dim=1)
    valid_colors = sum_colors / sum_weights.unsqueeze(-1)
    valid_colors[sum_weights < 1e-5, ...] = 0
    colors = torch.zeros((features.shape[0], 3), dtype=valid_colors.dtype, device=valid_colors.device)
    colors[valid_idx, ...] = valid_colors
    return colors


def kmeans_random(features: torch.Tensor, weights: torch.Tensor, n_colors=128) -> torch.Tensor:
    kmeans = KMeans(n_clusters=n_colors, init='random', random_state=0, n_init="auto", verbose=1, batch_size=n_colors * 2)
    labels = kmeans.fit_predict(features.cpu(), sample_weight=weights.cpu())
    linspace = torch.linspace(0, 1, steps=n_colors, device=features.device)
    colormap = torch.stack([
        linspace,
        linspace[torch.randperm(n_colors, device=features.device)],
        torch.flip(linspace, dims=(0,)),
    ]).T
    colors = colormap[torch.from_numpy(labels).to(features.device), ...]
    colors[weights < 1e-5, ...] = 0
    return colors


def kmeans_pca(features: torch.Tensor, weights: torch.Tensor, n_colors=128) -> torch.Tensor:
    kmeans = KMeans(n_clusters=n_colors, init='random', random_state=0, n_init="auto", verbose=1, batch_size=n_colors * 2)
    labels = kmeans.fit_predict(features.cpu(), sample_weight=weights.cpu())
    pca = PCA(n_components=3).fit(kmeans.cluster_centers_)
    scale = StandardScaler().fit(pca.transform(kmeans.cluster_centers_))
    colors = torch.tensor(scale.transform(pca.transform(kmeans.cluster_centers_[labels])), dtype=features.dtype, device=features.device)
    scale_std = 1
    colors = ((colors + scale_std) / (2 * scale_std)).clamp(0, 1)
    return colors


def assign_colors(features: torch.Tensor, weights: torch.Tensor, algo='kmeans', **kwargs):
    match algo:
        case 'weightedsum':
            return weightedsum(features, weights, **kwargs)
        case 'kmeans':
            return kmeans_random(features, weights, **kwargs)
        case 'kmeans-pca':
            return kmeans_pca(features, weights, **kwargs)
        case _:
            raise ValueError(f"Unknown algorithm {algo}")
