import torch


class ILS:
    def __init__(self, batch_size, n, device):
        self.v11 = torch.zeros((batch_size, n, n), device=device, dtype=torch.float64)
        self.v12 = torch.zeros((batch_size, n, 1), device=device, dtype=torch.float64)

    def update(self, X, Y, valid_mask, weight):
        v11valid = X.transpose(1, 2).bmm(X)
        v12valid = X.transpose(1, 2).bmm(Y)
        self.v11[valid_mask] += v11valid * weight.unsqueeze(-1).unsqueeze(-1)
        self.v12[valid_mask] += v12valid * weight.unsqueeze(-1).unsqueeze(-1)
