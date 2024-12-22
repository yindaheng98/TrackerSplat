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


class ISVD42:
    def __init__(self, batch_size, device, k=None, *args, **kwargs):
        self.U = torch.zeros((batch_size, 4, 4), device=device, *args, **kwargs)
        self.S = torch.zeros((batch_size, 4, 4), device=device, *args, **kwargs)
        self.A_init = torch.zeros((batch_size, 4, 4), device=device, *args, **kwargs)
        self.A_count = torch.zeros((batch_size,), device=device, dtype=torch.int)
        self.A_saved = None
        if k is not None:
            self.A_saved = torch.zeros((batch_size, 4, k*2), device=device, *args, **kwargs)
            self.k = k

    def update(self, A, mask, weights):
        '''A and weights are the value after mask, has smaller size than other tensors'''
        A = A.type(self.A_init.dtype)
        self.A_count[mask] += 1
        # Initialize those in first step
        A_init_masked, A_count_masked = self.A_init[mask, ...], self.A_count[mask]
        # TODO: take weights into account
        A_init_masked[A_count_masked == 1, ..., 0:2] = A[A_count_masked == 1, ...]
        A_init_masked[A_count_masked == 2, ..., 2:4] = A[A_count_masked == 2, ...]
        if self.A_saved is not None:
            A_saved_masked = self.A_saved[mask, ...]
            for k in range(self.k):
                A_saved_masked[A_count_masked == (k + 1), ..., 2*k:2*(k + 1)] = A[A_count_masked == (k + 1), ...]
            self.A_saved[mask, ...] = A_saved_masked
        self.A_init[mask, ...] = A_init_masked
        # Compute first step
        init_mask = mask & (self.A_count == 2)
        if init_mask.any():
            self.U[init_mask], self.S[init_mask] = SVD(self.A_init[init_mask])
        # Compute incremental step
        step_mask = mask & (self.A_count > 2)
        if step_mask.any():
            # TODO: take weights into account
            self.U[step_mask], self.S[step_mask] = IncrementalSVD(self.U[step_mask], self.S[step_mask], A[self.A_count[mask] > 2])


class ISVD_Mean3D(ISVD42):

    def update(self, A, mask, weights):
        super(ISVD_Mean3D, self).update(A.transpose(-2, -1), mask, weights)

    def solve(self, valid_mask):
        valid_mask &= self.A_count >= 2
        S = torch.diagonal(self.S, dim1=-2, dim2=-1)
        p_hom = torch.gather(self.U[valid_mask], 2, S[valid_mask].min(-1).indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 4, -1)).squeeze(-1)
        mean3D = p_hom / p_hom[..., -1:]
        if self.A_saved is not None:
            error = (mean3D.unsqueeze(-2) @ self.A_saved[valid_mask]).squeeze(-2)
            return mean3D[..., :-1], error, valid_mask
        return mean3D[..., :-1], valid_mask


class ISVD4SelectK2:
    """Keep only the K sets of equations with the smallest solving error"""

    def __init__(self, batch_size, device, k=3, *args, **kwargs):
        assert k >= 2
        self.k = k
        self.A_selected = torch.zeros((batch_size, 4, k*2), device=device, *args, **kwargs)
        self.A_count = torch.zeros((batch_size,), device=device, dtype=torch.int)

    def update(self, A, mask, weights):
        A = A.type(self.A_selected.dtype)
        self.A_count[mask] += 1
        # Initialize those in first step
        A_selected_masked, A_count_masked = self.A_selected[mask, ...], self.A_count[mask]
        # TODO: take weights into account
        for k in range(self.k):
            A_selected_masked[A_count_masked == (k + 1), ..., 2*k:2*(k + 1)] = A[A_count_masked == (k + 1), ...]
        self.A_selected[mask, ...] = A_selected_masked
        # Compute incremental step
        step_mask = mask & (self.A_count > self.k)
        if not step_mask.any():
            return
        A_selected = self.A_selected[step_mask, ...]
        A_new = A[self.A_count[mask] > self.k, ...]
        _, error_min = ISVD4SelectK2.solve(A_selected)  # TODO: time consuming
        error_min = error_min.abs().mean(-1)
        error_min_idx = torch.zeros((A_selected.shape[0]), dtype=torch.int, device=step_mask.device) + self.k
        for k in range(self.k):
            A_selected_ = A_selected.clone()
            A_selected_[..., 2*k:2*(k + 1)] = A_new
            _, error = ISVD4SelectK2.solve(A_selected_)  # TODO: time consuming
            error = error.abs().mean(-1)
            less_mask = error < error_min
            error_min[less_mask] = error[less_mask]
            error_min_idx[less_mask] = k
        for k in range(self.k):
            assign_mask = error_min_idx == k
            A_selected[assign_mask, ..., 2*k:2*(k + 1)] = A_new[assign_mask, ...]
        self.A_selected[step_mask, ...] = A_selected
        self.U, self.S = SVD(self.A_selected)

    @staticmethod
    def solve(A):
        U, S = SVD(A)
        S = torch.diagonal(S, dim1=-2, dim2=-1)
        p_hom = torch.gather(U, 2, S.min(-1).indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 4, -1)).squeeze(-1)
        mean3D = p_hom / p_hom[..., -1:]
        error = (mean3D.unsqueeze(-2) @ A).squeeze(-2)
        return mean3D[..., :-1], error


class ISVDSelectK_Mean3D(ISVD4SelectK2):

    def update(self, A, mask, weights):
        super(ISVDSelectK_Mean3D, self).update(A.transpose(-2, -1), mask, weights)

    def solve(self, valid_mask):
        valid_mask &= self.A_count >= 2
        mean3D, error = super().solve(self.A_selected[valid_mask])
        return mean3D, error, valid_mask


if __name__ == '__main__':
    B = 100
    n = 4
    N = 10
    step = 2
    A = torch.rand(B, n, N)
    U_, S_, V_ = SVD_withV(A)
    print("USV.T GT", (U_ @ S_ @ V_.transpose(-2, -1) - A).abs().max())
    U, S, V = SVD_withV(A[..., :n])
    print(f"USV.T step 0-{n}", (U @ S @ V.transpose(-2, -1) - A[..., :n]).abs().max())
    for i in range(n, N, step):
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
    U, S = SVD(A[..., :n])
    for i in range(n, N, step):
        U, S = IncrementalSVD(U, S, A[..., i:i+step])
    print("U abs diff", (U - Uv).abs().max())  # TODO: WTF?
    print("S diff", (S - Sv).abs().max())
    print("USV.T", (U @ S @ V.transpose(-2, -1) - A).abs().max())
    pass
    isvd = ISVD42(B, 'cpu')
    isvd.update(A[..., 0:2], torch.ones(B, dtype=torch.bool, device='cpu'), None)
    isvd.update(A[..., 2:4], torch.ones(B, dtype=torch.bool, device='cpu'), None)
    U, S = isvd.U, isvd.S
    _, _, Vi = SVD_withV(A[..., :4])
    print(f"USV.T step 0-4", (U @ S @ Vi.transpose(-2, -1) - A[..., :4]).abs().max())
    for i in range(4, N, 2):
        isvd.update(A[..., i:i+2], torch.ones(B, dtype=torch.bool, device='cpu'), None)
        U, S = isvd.U, isvd.S
        Ui, Si, _ = SVD_withV(A[..., :i+2])
        print("U abs diff step", i, (U.abs() - Ui.abs()).abs().max())  # TODO: WTF?
        print("S diff step", i, (S - Si).abs().max())
    pass
    isvd = ISVD4SelectK2(4, B, 'cpu')
    isvd.update(A[..., 0:2], torch.ones(B, dtype=torch.bool, device='cpu'), None)
    isvd.update(A[..., 2:4], torch.ones(B, dtype=torch.bool, device='cpu'), None)
    for i in range(4, N, 2):
        isvd.update(A[..., i:i+2], torch.ones(B, dtype=torch.bool, device='cpu'), None)
