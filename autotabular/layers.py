import torch
import torch.nn as nn


class FM(nn.Module):
    def __init__(self, n=None, k=None, **kwargs):
        super(FM, self).__init__(**kwargs)
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(n, k), requires_grad=True)
        self.lin = nn.Linear(n, 1)

    def forward(self, x, kwargs):
        if x.shape[0] != 3:
            raise ValueError(f'Wrong dimensions of inputs, expeted 3 but input {x.shape}.')
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True) #S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True) # S_2

        out_inter = 0.5 *( out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin

        return out