import torch


def SVD(A):
    U, S, Vh = torch.svd(A)
    return U, torch.diag_embed(S)


def SVD_withV(A):
    U, S, Vh = torch.svd(A)
    return U, torch.diag_embed(S), Vh


def IncrementalSVD(U, S, A):
    F = torch.cat([S, U.transpose(-1, -2) @ A], dim=-1)
    Uf, Sf, Vhf = torch.svd(F)
    U = U @ Uf
    S = torch.diag_embed(Sf)
    return U, S


def block_diag_identical(a, n):
    I = torch.eye(n).unsqueeze(0).expand(*a.shape[:-2], -1, -1)
    top_right = torch.zeros(*a.shape[:-2], a.shape[-2], n)
    top = torch.cat([a, top_right], dim=-1)
    bottom_left = torch.zeros(*a.shape[:-2], n, a.shape[-1])
    bottom = torch.cat([bottom_left, I], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def IncrementalSVD_withV(U, S, Vh, A):
    F = torch.cat([S, U.transpose(-1, -2) @ A], dim=-1)
    Uf, Sf, Vhf = torch.svd(F)
    U = U @ Uf
    S = torch.diag_embed(Sf)
    Vh = block_diag_identical(Vh, A.shape[-1]) @ Vhf
    return U, S, Vh


if __name__ == '__main__':
    B = 10
    N = 50
    A = torch.rand(B, 4, N)
    U_, S_, V_ = SVD_withV(A)
    U, S, V = SVD_withV(A[..., :4])
    for i in range(4, N):
        U, S, V = IncrementalSVD_withV(U, S, V, A[..., i:i+1])
    print((U @ S @ V.transpose(-2, -1) - A).abs().max())
    print((U_ @ S_ @ V_.transpose(-2, -1) - A).abs().max())
    print(U + U_)  # TODO: WTF?
    print(S - S_)
    print(V + V_)  # TODO: WTF?
    pass
