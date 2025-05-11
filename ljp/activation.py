"""
        PAPER: Qian Xiang, Xiaodan Wang, Yafei Song, and Lei Lei. 2025.
        ISONet: Reforming 1DCNN for aero-engine system inter-shaft bearing
        fault diagnosis via input spatial over-parameterization,
        Expert Systems with Applications: 12724
        https://doi.org/10.1016/j.eswa.2025.127248
        Email: qianxljp@126.com
"""
import torch


class AReLU(torch.nn.Module):
    def __init__(self, alpha=0.90, beta=2.0):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor([alpha]))
        self.beta = torch.nn.Parameter(torch.tensor([beta]))

    def forward(self, input):
        alpha = torch.clamp(self.alpha, min=0.01, max=0.99)
        beta = 1 + torch.sigmoid(self.beta)
        return torch.nn.functional.relu(input) * beta - torch.nn.functional.relu(-input) * alpha


class TanhExp(torch.nn.Module):
    """
    Xinyu Liu, Xiaoguang Di
    TanhExp: A Smooth Activation Function
    with High Convergence Speed for
    Lightweight Neural Networks
    https://arxiv.org/pdf/2003.09855v1.pdf
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.exp(x))


class Serf(torch.nn.Module):
    """
    Nag, S. and Bhattacharyya, M.,
    “SERF: Towards better training of deep neural networks
    using log-Softplus ERror activator Function”,
    https://arxiv.org/pdf/2108.09598.pdf
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.erf(torch.nn.functional.softplus(x))


class Smish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(torch.log(1 + torch.nn.functional.sigmoid(x)))
