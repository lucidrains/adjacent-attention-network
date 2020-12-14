import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from isab_pytorch import ISAB

# helpers

def exists(val):
    return val is not None

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
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
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
        heads = 4,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.null_k = nn.Parameter(torch.randn(heads, dim_head))
        self.null_v = nn.Parameter(torch.randn(heads, dim_head))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        adj_kv_indices,
        mask
    ):
        b, n, d, h = *x.shape, self.heads
        flat_indices = repeat(adj_kv_indices, 'b n a -> (b h) (n a)', h = h)

        # derive query, key, value
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # gather keys and values according to adjacency matrix
        k, v = map(lambda t: rearrange(t, 'b h n d -> (b h) n d'), (k, v))
        k = batched_index_select(k, flat_indices)
        v = batched_index_select(v, flat_indices)
        k, v = map(lambda t: rearrange(t, '(b h) (n a) d -> b h n a d', h = h, n = n), (k, v))

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

        # dropout
        attn = self.dropout(attn)

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
        heads = 4,
        num_neighbors_cutoff = None,
        num_global_nodes = 0,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        self.num_neighbors_cutoff = num_neighbors_cutoff
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            global_attn = PreNorm(dim, ISAB(
                dim = dim,
                heads = heads,
                num_induced_points = num_global_nodes
            )) if num_global_nodes > 0 else None

            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, AdjacentAttention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    dropout = attn_dropout
                ))),
                global_attn,
                Residual(PreNorm(dim, FeedForward(
                    dim = dim,
                    dropout = ff_dropout
                )))
            ]))

    def forward(self, x, adjacency_mat, mask = None):
        device, n = x.device, x.shape[1]

        diag = torch.eye(adjacency_mat.shape[-1], device = device).bool()
        adjacency_mat |= diag # nodes should pay attention itself (self-interacting)

        # zero out points on adjacency matrix
        # where the nodes are just padding
        if exists(mask):
            adjacency_mat &= (mask[:, :, None] * mask[:, None, :])

        adj_mat = adjacency_mat.float()

        # if we don't set a hard limit to the number of neighbors:
        #   - get the maximum number of neighbors and pad the rest of the nodes with less than that number of neighbors
        # else:
        #   - randomly sample the cutoff number of neighbors for any node that exceeds the max
        #   - this would be similar to random sparse attention (bigbird)

        # get the maximum number of neighbors
        max_neighbors = int(adj_mat.sum(dim = -1).max())

        if exists(self.num_neighbors_cutoff) and max_neighbors > self.num_neighbors_cutoff:
            # to randomly sample the neighbors, add a small uniform noise to the mask and topk
            noise = torch.empty((n, n), device = device).uniform_(-0.01, 0.01)
            adj_mat = adj_mat + noise

            adj_mask, adj_kv_indices = adj_mat.topk(dim = -1, k = self.num_neighbors_cutoff)

            # cast the mask back to 0s and 1s
            adj_mask = (adj_mask > 0.5).float()
        else:
            # todo - get distribution of number of neighbors, and strategically break up attention (message passing) to multiple steps
            #      - start with a bimodal num neighbors test case, then generalize

            # use topk to get all the neighbors
            # also pass the mask into the attention, as some neighbors will be just padding and not actually neighbors
            adj_mask, adj_kv_indices = adj_mat.topk(dim = -1, k = max_neighbors)


        for attn, global_attn, ff in self.layers:
            x = attn(
                x,
                adj_kv_indices = adj_kv_indices,
                mask = adj_mask
            )

            if exists(global_attn):
                out, _ = global_attn(x, mask = mask)
                x = x + out

            x = ff(x)

        return x
