import torch
from .motionfusion import unflatten_symmetry_3x3


class ILS:
    def __init__(self, batch_size: int, n: int, *args, **kwargs):
        self.v11 = torch.zeros((batch_size, n, n), *args, **kwargs)
        self.v12 = torch.zeros((batch_size, n, 1), *args, **kwargs)

    def update(self, X, Y, valid_mask, weight):
        v11valid = X.transpose(1, 2).bmm(X)
        v12valid = X.transpose(1, 2).bmm(Y)
        self.v11[valid_mask] += v11valid * weight.unsqueeze(-1).unsqueeze(-1)
        self.v12[valid_mask] += v12valid * weight.unsqueeze(-1).unsqueeze(-1)


class ILS_Cov3D(ILS):
    def __init__(self, batch_size: int, n: int, *args, **kwargs):
        super(ILS_Cov3D, self).__init__(batch_size, n*(n+1)//2, *args, dtype=torch.float64, **kwargs)

    def solve(self, valid_mask):
        cov3D_flatten = torch.linalg.inv(self.v11[valid_mask]).bmm(self.v12[valid_mask]).squeeze(-1)
        cov3D = unflatten_symmetry_3x3(cov3D_flatten)
        return cov3D, valid_mask


class ILS_RotationScale(ILS_Cov3D):

    def solve(self, valid_mask):
        cov3D, valid_mask = super(ILS_RotationScale, self).solve(valid_mask)
        L, Q = torch.linalg.eigh(cov3D.type(torch.float32))
        # # verify cov3D
        # diff_cov3D = Q @ (L.unsqueeze(-1) * Q.transpose(1, 2)) - cov3D
        # # we can verify that the order do not influence the result
        # order = [2, 1, 0]
        # diff_cov3D = Q[..., order] @ (L[..., order].unsqueeze(-1) * Q[..., order].transpose(1, 2)) - cov3D
        negative_mask = (L < 0).any(-1)  # drop negative eigen values in L
        R = Q[~negative_mask, ...]
        S = torch.sqrt(L[~negative_mask, ...])
        valid_positive_mask = valid_mask.clone()
        valid_positive_mask[valid_mask] = ~negative_mask
        return R, S, valid_positive_mask
