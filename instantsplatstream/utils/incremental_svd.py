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


class ISVD:
    def __init__(self, batch_size, n, device):
        self.U = torch.zeros((batch_size, n, n), device=device, dtype=torch.float32)
        self.S = torch.zeros((batch_size, n, n), device=device, dtype=torch.float32)
        self.A_init = torch.zeros((batch_size, n, n), device=device, dtype=torch.float32)
        self.A_count = torch.zeros((batch_size,), device=device, dtype=torch.int)

    def update(self, A, mask, weights):
        '''A and weights are the value after mask, has smaller size than other tensors'''
        self.A_count[mask] += 1
        # Initialize those in first step
        A_init_masked, A_count_masked = self.A_init[mask, ...], self.A_count[mask]
        # TODO: take weights into account
        A_init_masked[A_count_masked == 1, 0:2] = A[A_count_masked == 1, ...]
        A_init_masked[A_count_masked == 2, 2:4] = A[A_count_masked == 2, ...]
        self.A_init[mask, ...] = A_init_masked
        # Compute first step
        init_mask = mask & (self.A_count == 2)
        if init_mask.any():
            self.U[init_mask], self.S[init_mask] = SVD(self.A_init[init_mask].transpose(-2, -1))
        # Compute incremental step
        step_mask = mask & (self.A_count > 2)
        if step_mask.any():
            # TODO: take weights into account
            self.U[step_mask], self.S[step_mask] = IncrementalSVD(self.U[step_mask], self.S[step_mask], A[self.A_count[mask] > 2].transpose(-2, -1))


if __name__ == '__main__':
    B = 3
    N = 8
    A = torch.rand(B, 4, N)
    U_, S_, V_ = SVD_withV(A)
    print("USV.T GT", (U_ @ S_ @ V_.transpose(-2, -1) - A).abs().max())
    U, S, V = SVD_withV(A[..., :4])
    print("USV.T step 0-4", (U @ S @ V.transpose(-2, -1) - A[..., :4]).abs().max())
    step = 2
    for i in range(4, N, step):
        Ui, Si, Vi = SVD_withV(A[..., :i+step])
        print("USV.T GT step", i, (Ui @ Si @ Vi.transpose(-2, -1) - A[..., :i+step]).abs().max())
        U, S, V = IncrementalSVD_withV(U, S, V, A[..., i:i+step])
        print("U abs diff step", i, (U.abs() - Ui.abs()).abs().max())  # TODO: WTF?
        print("S diff step", i, (S - Si).abs().max())
        print("USV.T step", i, (U @ S @ V.transpose(-2, -1) - A[..., :i+step]).abs().max())
    print("U abs diff", (U.abs() - U_.abs()).abs().max())  # TODO: WTF?
    print("U/U[-1] abs diff", (U[..., :-1, :]/U[..., -1:, :] - U_[..., :-1, :]/U_[..., -1:, :]).abs().max())
    print("S diff", (S - S_).abs().max())
    print("V abs diff", (V.abs() - V_.abs()).abs().max())  # TODO: WTF?
    print("USV.T", (U @ S @ V.transpose(-2, -1) - A).abs().max())
    Uv, Sv = U, S
    U, S = SVD(A[..., :4])
    for i in range(4, N, step):
        U, S = IncrementalSVD(U, S, A[..., i:i+step])
    print("U abs diff", (U - Uv).abs().max())  # TODO: WTF?
    print("S diff", (S - Sv).abs().max())
    print("USV.T", (U @ S @ V.transpose(-2, -1) - A).abs().max())
    pass
