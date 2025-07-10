import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # x: [B, C, T]
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x_norm = (x - mean) / (std + self.eps)
        return self.gamma.unsqueeze(0).unsqueeze(2) * x_norm + self.beta.unsqueeze(0).unsqueeze(2)
