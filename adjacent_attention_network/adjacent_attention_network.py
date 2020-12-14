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

        # similarity of each node to its neighbors
        dots = einsum('b h i d, b h i j d -> b h i j', q, k) * self.scale

        # mask out neighbors that are just padding
        mask_value = -torch.finfo(dots.dtype).max
        mask = rearrange(mask.bool(), 'b i j -> b () i j')
        dots.masked_fill_(~mask.bool(), mask_value)

        # attention
        attn = dots.softmax(dim = -1)

        # get weighted average of the values of all neighbors
        out = einsum('b h i j, b h i j d -> b h i d', attn, v)
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
            layer = AdjacentAttention(
                dim = dim,
                dim_head = dim_head,
                heads = heads
            )
            self.layers.append(layer)

    def forward(self, x, adjacency_mat):
        adj_mat = adjacency_mat.float()

        # get the maximum number of neighbors
        # todo - get distribution of number of neighbors, and strategically break up attention (message passing) to multiple steps
        max_neighbors = int(adj_mat.sum(dim = -1).max())

        # use topk to get all the neighbors
        # also pass the mask into the attention, as some neighbors will be just padding and not actually neighbors
        mask, adj_kv_indices = adj_mat.topk(dim = -1, k = max_neighbors)

        for layer in self.layers:
            attn_out = layer(
                x,
                adj_kv_indices,
                mask = mask
            )
            x =  attn_out + x
        return x
