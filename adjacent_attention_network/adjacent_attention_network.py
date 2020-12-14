import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

class AdjacentAttentionNetwork(nn.Module):
    def __init__(
        self,
        *,
        dim
    ):
        super().__init__()

    def forward(self, x, adjacency_mat = None):
        return x
