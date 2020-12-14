import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

# adjacent attention class

class AdjacentAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 4
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.null_k = nn.Parameter(torch.randn(heads, dim_head))
        self.null_v = nn.Parameter(torch.randn(heads, dim_head))

    def forward(
        self,
        x,
        adj_kv_indices,
        mask
    ):
        b, n, d, h = *x.shape, self.heads
        flat_indices = rearrange(adj_kv_indices, 'b n a -> b (n a)')

        # select the neighbors for every individual token. "a" dimension stands for 'adjacent neighbor'
        kv_x = batched_index_select(x, flat_indices)
        kv_x = rearrange(kv_x, 'b (n a) d -> b n a d', n = n)

        # derive query, key, value
        q, k, v = self.to_q(x), *self.to_kv(kv_x).chunk(2, dim = -1)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        k, v = map(lambda t: rearrange(t, 'b n a (h d) -> b h n a d',  h = h), (k, v))

        # add null key / value, so a node can attend to nothing
        # have come across this in GNN literature as some other name
        nk, nv = map(lambda t: rearrange(t, 'h d -> () h () () d').expand(b, -1, n, 1, -1), (self.null_k, self.null_v))
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)
        mask = F.pad(mask, (1, 0), value = 1)

        # similarity of each node to its neighbors
        sim = einsum('b h n d, b h n a d -> b h n a', q, k) * self.scale

        # mask out neighbors that are just padding
        mask_value = -torch.finfo(sim.dtype).max
        mask = rearrange(mask.bool(), 'b n a -> b () n a')
        sim.masked_fill_(~mask.bool(), mask_value)

        # attention
        attn = sim.softmax(dim = -1)

        # get weighted average of the values of all neighbors
        out = einsum('b h n a, b h n a d -> b h n d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # combine output
        return self.to_out(out)

# adjacent network (layers of adjacent attention)

class AdjacentAttentionNetwork(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, AdjacentAttention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads
                ))),
                Residual(PreNorm(dim, FeedForward(
                    dim = dim
                )))
            ]))

    def forward(self, x, adjacency_mat, mask = None):
        device = x.device

        diag = torch.eye(adjacency_mat.shape[-1], device = device).bool()
        adjacency_mat |= diag # nodes should pay attention itself (self-interacting)

        # zero out points on adjacency matrix
        # where the nodes are just padding
        if exists(mask):
            mask = mask[:, :, None] * mask[:, None, :]
            adjacency_mat &= mask

        adj_mat = adjacency_mat.float()

        # get the maximum number of neighbors
        # todo - get distribution of number of neighbors, and strategically break up attention (message passing) to multiple steps
        max_neighbors = int(adj_mat.sum(dim = -1).max())

        # use topk to get all the neighbors
        # also pass the mask into the attention, as some neighbors will be just padding and not actually neighbors
        mask, adj_kv_indices = adj_mat.topk(dim = -1, k = max_neighbors)

        for attn, ff in self.layers:
            x = attn(
                x,
                adj_kv_indices = adj_kv_indices,
                mask = mask
            )

            x = ff(x)

        return x
