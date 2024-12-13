import torch


def SVD(A):
    U, S, Vh = torch.svd(A)
    return U, torch.diag_embed(S)


def IncrementalSVD(U, S, A):
    F = torch.cat([S, U.transpose(-1, -2) @ A], dim=-1)
    Uf, Sf, Vhf = torch.svd(F)
    U = U @ Uf
    S = torch.diag_embed(Sf)
    return U, S

if __name__ == '__main__':
    N = 50
    A = torch.rand(100, 4, N)
    U_, S_ = SVD(A)
    U, S = SVD(A[..., :4])
    for i in range(4, N):
        U, S = IncrementalSVD(U, S, A[..., i:i+1])
    print(U + U_) # TODO: WTF?
    print(S - S_)